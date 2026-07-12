"""Torch-free scoresheet -> game reconstruction (onnxruntime + numpy).

This is the single inference core shared by the CLI (`read_game.py`) and the
Gradio demo (`app.py`). It deliberately imports **no torch**: the move and
board-diagram classifiers run as ONNX sessions, and every array op is numpy. The
heavy lifting of cell extraction (`togyz.sheet`) and the rules engine
(`togyz.rules`) are already torch-free and imported as-is.

The scoring/beam logic here is a straight port of the torch version whose
weights and variants are benchmarked against hand-transcribed truth in the
git history (sheet 1 = 36/39 exact plies). Keep it numerically equivalent.
"""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .classes import DASH, EMPTY, TUZDYK
from .preprocess import preprocess_pil
from .rules import Game
from .sheet import clean_cell, extract_cells, extract_diagram_cells, render_overlay

MIN_CELL_HEIGHT = 35  # px; below this, resolution is low (surfaced as a warning)
MIN_INK_RATIO = 0.02  # cells with less handwriting ink than this are empty
MIN_KAZAN_INK = 0.028  # kazan boxes are small; box-edge remnants score higher

TTA_SHIFTS = [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)]

EXACT_WEIGHT = 0.5  # weight of the exact class-string probability
FACTOR_WEIGHT = 0.5  # weight of the factorized digit-marginal probability
KAZAN_FLOOR = 1e-4  # a misread checkpoint must not single-handedly kill truth
PIT_FLOOR = 1e-3  # pit reads are noisier than kazans: weaker veto power
PIT_WEIGHT = 0.5  # 18 pit terms per checkpoint must not swamp the rest
RESULT_CODES = {"1-0": 0, "0-1": 1, "draw": -1}

# Memory guard for user-selectable beam widths (the UI offers up to 2^30):
# a late-game hypothesis (copied move lists etc.) costs roughly this many
# bytes, and the expanded pool peaks at beam_width * per_ply hypotheses.
HYPOTHESIS_BYTES = 25_000
MAX_POOL_BYTES = 2 * 2**30  # ~2 GB hypothesis budget


# --------------------------------------------------------------------------- #
# ONNX session loading
# --------------------------------------------------------------------------- #
@dataclass
class Classifier:
    """An ONNX classifier plus the class list its output indices map to."""

    session: ort.InferenceSession
    classes: list[str]
    input_name: str = field(default="")

    def __post_init__(self):
        self.input_name = self.session.get_inputs()[0].name

    def run(self, batch: np.ndarray) -> np.ndarray:
        """batch [N,1,H,W] float32 -> logits [N, num_classes] float32."""
        return self.session.run(None, {self.input_name: batch})[0]


def load_classifier(onnx_path: str | Path) -> Classifier:
    """Load a .onnx and its `<stem>.classes.json` sidecar (written by export)."""
    onnx_path = Path(onnx_path)
    classes_path = onnx_path.with_suffix(".classes.json")
    classes = json.loads(classes_path.read_text())
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return Classifier(session, classes)


# --------------------------------------------------------------------------- #
# Cell classification (numpy + ONNX, with test-time augmentation)
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _tta_views(img: Image.Image):
    """Slightly shifted crops of one cell - averaging their predictions
    smooths out crop-alignment luck (measured: sheet 1 29 -> 36/39 exact)."""
    w, h = img.size
    views = []
    for dx, dy in TTA_SHIFTS:
        views.append(img.crop((max(0, dx), max(0, dy), w + min(0, dx), h + min(0, dy))))
    return views


def classify_cells(images, clf: Classifier, temperature=1.0, tta=True, batch_size=64):
    """Class probabilities per cell, averaged over TTA views."""
    views_per_cell = len(TTA_SHIFTS) if tta else 1
    tensors = []
    for img in images:
        for view in _tta_views(img) if tta else [img]:
            tensors.append(preprocess_pil(view))
    if not tensors:
        return np.empty((0, len(clf.classes)), dtype=np.float32)

    probs_chunks = []
    for i in range(0, len(tensors), batch_size):
        batch = np.stack(tensors[i : i + batch_size]).astype(np.float32)
        probs_chunks.append(_softmax(clf.run(batch), temperature))
    stacked = np.concatenate(probs_chunks, axis=0)
    return stacked.reshape(len(images), views_per_cell, -1).mean(axis=1)


# --------------------------------------------------------------------------- #
# Legal-move scoring, kazan evidence, beam search  (numpy port)
# --------------------------------------------------------------------------- #
def _ply_marginals(probs, classes):
    """Source/landing digit marginals and the x-mark mass for one ply.

    Depends only on the ply's class distribution, so the beam computes it
    once per ply and shares it across every hypothesis (this used to be
    recomputed per hypothesis - the dominant cost at large beam widths).
    """
    source_marginal = [0.0] * 9
    landing_marginal = [0.0] * 9
    x_marginal = 0.0
    for c, p in zip(classes, probs.tolist()):
        if c == "empty":
            continue
        source_marginal[int(c[0]) - 1] += p
        landing_marginal[int(c[1]) - 1] += p
        if c.endswith("x"):
            x_marginal += p
    return source_marginal, landing_marginal, x_marginal


def _legal_scores(probs, legal_moves, class_idx, marginals):
    """Redistribute the 162-class distribution over the current legal moves.

    A legal move earns credit from (a) its exact class string and (b) a
    factorized term P(source digit) * P(landing digit), times agreement with
    the written x-mark probability. Scores stay UNNORMALIZED for the beam so a
    hypothesis whose legal set explains the observation poorly accumulates a
    genuinely low joint likelihood (measured: 24/39 vs 29/39 when normalized).
    """
    source_marginal, landing_marginal, x_marginal = marginals
    scored = []
    for move in legal_moves:
        exact = float(probs[class_idx[move.notation]])
        factorized = source_marginal[move.action] * landing_marginal[int(move.notation[1]) - 1] * 9
        x_agreement = x_marginal if move.makes_tuzdyk else 1.0 - x_marginal
        scored.append(
            ((EXACT_WEIGHT * exact + FACTOR_WEIGHT * factorized) * x_agreement, move)
        )
    total = sum(s for s, _ in scored) or 1.0
    return sorted(((s, s / total, m) for s, m in scored), key=lambda t: -t[0])


def _result_consistency(game: Game, result: str | None) -> int:
    """How well a final hypothesis state matches how the game really ended."""
    if result is None:
        return 1 if game.is_over else 0
    want = RESULT_CODES[result]
    if game.is_over and game.winner() == want:
        return 2
    kazans = game.kazans
    leader = -1 if kazans[0] == kazans[1] else (0 if kazans[0] > kazans[1] else 1)
    return 1 if leader == want else 0


def _filter_probs(probs: np.ndarray, classes: list[str], allowed: set[str]):
    """Zero out contextually impossible classes and renormalize.

    Returns None when nothing observable remains (e.g. a blank crop whose
    mass sat entirely on 'empty') - the cell is then treated as unobserved.
    """
    filtered = np.array(
        [p if c in allowed else 0.0 for c, p in zip(classes, probs)], np.float64
    )
    total = float(filtered.sum())
    if total < 1e-6:
        return None
    return filtered / total


def _diagram_logp(game: Game, cp) -> float:
    """Log-likelihood of a hypothesis' board state under one checkpoint's
    diagram reads (kazan boxes + observed pit cells)."""
    logp = 0.0
    for side, player in (("W", 0), ("B", 1)):
        probs = cp["kazan"].get(side)
        if probs is None:
            continue
        value = game.kazans[player]
        p = float(probs[value]) if value < len(probs) else 0.0
        logp += math.log(max(p, KAZAN_FLOOR))

    for (side, index), (values, p_x, p_dash) in cp["pits"].items():
        player = 0 if side == "W" else 1
        pit_pos = player * 9 + (index - 1)
        # a pit owned as tuzdyk by the opponent is written as 'x'
        if game.tuzdyks[1 - player] == index - 1:
            p = p_x
        else:
            count = game.pits[pit_pos]
            if count == 0:
                p = p_dash + float(values[0])
            else:
                p = float(values[count]) if count < len(values) else 0.0
        logp += PIT_WEIGHT * math.log(max(p, PIT_FLOOR))
    return logp


def safe_beam_width(beam_width: int, per_ply: int = 9) -> int:
    """Largest beam width that keeps the expanded pool inside the RAM budget."""
    return max(1, min(beam_width, MAX_POOL_BYTES // (HYPOTHESIS_BYTES * max(per_ply, 1))))


def beam_decode(ply_probs, classes, beam_width=1024, per_ply=9, result=None,
                checkpoints=None, progress_cb=None):
    """Longest fully-legal move sequence with the highest joint probability,
    exploring all `per_ply` legal continuations per ply; the final pool is
    re-ranked by diagram-checkpoint and known-result consistency.

    `beam_width` is capped by `safe_beam_width` so user-requested widths up
    to 2^30 degrade into a narrower beam instead of exhausting RAM.
    `progress_cb(ply_index, total_plies)` is called once per ply if given.

    Returns (annotated_moves, log_prob, per_ply_share_list).
    """
    beam_width = safe_beam_width(beam_width, per_ply)
    checkpoints = checkpoints or {}
    class_idx = {c: i for i, c in enumerate(classes)}
    active = [(Game(), [], [], 0.0)]  # (game, annotated_moves, chosen_shares, logp)
    finished = []
    total_plies = len(ply_probs)

    for ply_index, probs in enumerate(ply_probs):
        if progress_cb:
            progress_cb(ply_index, total_plies)
        # marginals are identical across all hypotheses this ply - compute once
        marginals = _ply_marginals(probs, classes)
        expanded = []
        for game, moves, chosen, logp in active:
            candidates = _legal_scores(probs, game.legal_moves(), class_idx, marginals)
            for raw, share, move in candidates[:per_ply]:
                nxt = game.copy()
                played = nxt.play(move.action)
                annotated = played.notation + ("+" if played.captured else "")
                new_logp = logp + math.log(max(raw, 1e-12))
                if len(moves) + 1 in checkpoints:
                    new_logp += _diagram_logp(nxt, checkpoints[len(moves) + 1])
                hyp = (nxt, moves + [annotated], chosen + [share], new_logp)
                (finished if nxt.is_over else expanded).append(hyp)
        if not expanded:
            break
        # nlargest instead of a full sort: wide beams stay tractable
        active = heapq.nlargest(beam_width, expanded, key=lambda h: h[3])

    pool = active + finished
    best = max(pool, key=lambda h: (len(h[1]), _result_consistency(h[0], result), h[3]))
    return best[1], best[3], best[2]


def format_pgn(plies: list[str]) -> str:
    """plies -> '1. 54+ 42x' lines (one White/Black pair per line)."""
    lines = []
    for i in range(0, len(plies), 2):
        pair = " ".join(plies[i : i + 2])
        lines.append(f"{i // 2 + 1}. {pair}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# The single entry point
# --------------------------------------------------------------------------- #
def run_pipeline(image, moves_clf: Classifier, diagram_clf: Classifier | None = None,
                 result=None, topk=5, beam_width=8192, per_ply=9,
                 temperature=1.0, tta=True, save_cells_dir=None, progress_cb=None):
    """Read one scoresheet image into game records.

    `image` is a path or a PIL.Image. Returns a dict with the reconstructed
    PGNs and diagnostics; no files are written unless `save_cells_dir` is set
    (the CLI uses it, the Gradio app does not).

    `progress_cb(frac, desc)` - if given - is called with a 0..1 fraction and
    a human-readable phase description as the read progresses.

    Keys: beam_pgn, raw_pgn, legal_pgn, game_json (dict), stopped,
    plies_scanned, legal_plies, beam_plies, checkpoint_report, warnings,
    annotated_image (PIL), median_cell_height, low_resolution (bool).
    """
    def report(frac, desc):
        if progress_cb:
            progress_cb(frac, desc)

    classes = moves_clf.classes
    empty_idx = classes.index(EMPTY)
    warnings: list[str] = []
    effective_width = safe_beam_width(beam_width, per_ply)
    if effective_width < beam_width:
        warnings.append(
            f"beam width {beam_width} trimmed to {effective_width} to stay "
            "inside the memory budget"
        )

    report(0.0, "detecting tables and cells")
    cells, sheet, gridlines = extract_cells(image, with_gridlines=True)
    cells.sort(key=lambda c: (c.move_no, c.side != "W"))  # game order: 1W 1B 2W ...
    median_h = sorted(c.bbox[3] for c in cells)[len(cells) // 2]

    report(0.05, f"classifying {len(cells)} move cells")
    cleaned = [clean_cell(c.image) for c in cells]  # (image, ink_ratio) pairs
    all_probs = classify_cells(
        [img for img, _ in cleaned], moves_clf, temperature=temperature, tta=tta
    )

    game = Game()
    plies, raw_pgn, legal_pgn = [], [], []
    ply_prob_arrays = []
    labels = {}
    stopped = None
    in_sync = True

    cells_dir = Path(save_cells_dir) if save_cells_dir else None
    if cells_dir:
        cells_dir.mkdir(parents=True, exist_ok=True)

    for cell, (clean_img, ink_ratio), probs in zip(cells, cleaned, all_probs):
        if ink_ratio < MIN_INK_RATIO or int(np.argmax(probs)) == empty_idx:
            stopped = {"move": cell.move_no, "side": cell.side, "reason": "empty cell"}
            break

        move_probs = probs.copy()
        move_probs[empty_idx] = 0.0
        raw = classes[int(np.argmax(move_probs))]

        legal = {m.notation: m for m in game.legal_moves()} if in_sync else {}
        topk_idx = np.argsort(-probs)[:topk]
        entry = {
            "move": cell.move_no,
            "side": cell.side,
            "bbox": list(cell.bbox),
            "raw": raw,
            "raw_is_legal": raw in legal if in_sync else None,
            "legal": sorted(legal),
            "top5": [[classes[int(i)], round(float(probs[int(i)]), 4)] for i in topk_idx],
            "probs": {c: round(float(p), 6) for c, p in zip(classes, probs)},
        }
        plies.append(entry)
        ply_prob_arrays.append(probs)
        if cells_dir:
            clean_img.save(cells_dir / f"{cell.move_no:02d}_{cell.side}.png")

        if in_sync and raw in legal:
            played = game.play_notation(raw)
            annotated = played.notation + ("+" if played.captured else "")
            raw_pgn.append(annotated)
            legal_pgn.append(annotated)
            if game.is_over:
                stopped = {"move": cell.move_no, "side": cell.side, "reason": "game over",
                           "winner": ["P1", "P2", "draw"][game.winner()]}
                in_sync = False
        else:
            if in_sync:
                entry["legal_alternatives"] = sorted(
                    ((n, round(float(probs[classes.index(n)]), 4)) for n in legal),
                    key=lambda t: -t[1],
                )
                stopped = stopped or {"move": cell.move_no, "side": cell.side,
                                      "reason": f"illegal move {raw!r}"}
                in_sync = False
            raw_pgn.append(raw)

    # board-diagram evidence from the summary strips (every 10 moves):
    # kazan boxes (values 10-81) plus the 2x9 pit grid ('x'/'-'/0-81)
    checkpoints, checkpoint_report = {}, []
    diagram_cells, diagram_labels = [], {}
    if diagram_clf is not None:
        report(0.30, "reading board diagrams")
        dg_classes = diagram_clf.classes
        value_names = [str(v) for v in range(82)]
        kazan_allowed = set(str(v) for v in range(10, 82))
        pit_allowed = set(value_names) | {TUZDYK, DASH}
        x_idx = dg_classes.index(TUZDYK)
        dash_idx = dg_classes.index(DASH)

        dg_cells = [(c, *clean_cell(c.image)) for c in extract_diagram_cells(sheet)]
        dg_cells = [
            (c, img) for c, img, ink in dg_cells
            if ink >= (MIN_KAZAN_INK if c.kind == "kazan" else MIN_INK_RATIO)
        ]
        if dg_cells:
            dg_probs = classify_cells(
                [img for _, img in dg_cells], diagram_clf,
                temperature=temperature, tta=tta,
            )
            for (cell, img), probs in zip(dg_cells, dg_probs):
                if dg_classes[int(np.argmax(probs))] == EMPTY:
                    continue
                allowed = kazan_allowed if cell.kind == "kazan" else pit_allowed
                filtered = _filter_probs(probs, dg_classes, allowed)
                if filtered is None:
                    continue
                cp = checkpoints.setdefault(
                    cell.move_no * 2, {"kazan": {}, "pits": {}}
                )
                values = filtered[:82]
                top_idx = int(np.argmax(filtered))
                read = dg_classes[top_idx]
                if cell.kind == "kazan":
                    # renormalized over 10-81 only, value-indexed for scoring
                    cp["kazan"][cell.side] = values / max(float(values.sum()), 1e-9)
                else:
                    cp["pits"][(cell.side, cell.pit_index)] = (
                        values, float(filtered[x_idx]), float(filtered[dash_idx])
                    )
                diagram_cells.append(cell)
                diagram_labels[(cell.move_no, cell.kind, cell.side, cell.pit_index)] = read
                entry = {"move": cell.move_no, "kind": cell.kind, "side": cell.side,
                         "read": read, "prob": round(float(filtered[top_idx]), 4)}
                if cell.kind == "pit":
                    entry["index"] = cell.pit_index
                checkpoint_report.append(entry)
                if cells_dir:
                    suffix = f"_{cell.pit_index}" if cell.kind == "pit" else ""
                    img.save(cells_dir /
                             f"{cell.kind}_{cell.move_no:02d}_{cell.side}{suffix}.png")

    def beam_progress(ply_index, total):
        report(0.45 + 0.53 * (ply_index / max(total, 1)),
               f"reconstructing game (ply {ply_index + 1}/{total})")

    beam_moves, beam_logp, beam_shares = beam_decode(
        ply_prob_arrays, classes, effective_width, per_ply,
        result=result, checkpoints=checkpoints, progress_cb=beam_progress,
    )
    report(1.0, "done")
    beam_detail = [
        {"move": p["move"], "side": p["side"], "chosen": m,
         "prob": round(q, 4), "agrees_with_raw": m.rstrip("+") == p["raw"]}
        for p, m, q in zip(plies, beam_moves, beam_shares)
    ]
    for p, m in zip(plies, beam_moves):
        labels[(p["move"], p["side"])] = m

    annotated = render_overlay(
        sheet, cells[: len(plies)], labels,
        diagram_cells=diagram_cells, diagram_labels=diagram_labels,
        gridlines=gridlines,
    )

    game_json = {
        "plies_scanned": len(plies),
        "legal_plies": len(legal_pgn),
        "beam": {"plies": len(beam_moves), "log_prob": round(beam_logp, 3), "moves": beam_detail},
        "checkpoints": checkpoint_report,
        "stopped": stopped or {"reason": "sheet exhausted"},
        "warnings": warnings,
        "plies": plies,
    }
    return {
        "beam_pgn": format_pgn(beam_moves),
        "raw_pgn": format_pgn(raw_pgn),
        "legal_pgn": format_pgn(legal_pgn),
        "game_json": game_json,
        "stopped": stopped or {"reason": "sheet exhausted"},
        "plies_scanned": len(plies),
        "legal_plies": len(legal_pgn),
        "beam_plies": len(beam_moves),
        "checkpoint_report": checkpoint_report,
        "warnings": warnings,
        "annotated_image": annotated,
        "median_cell_height": int(median_h),
        "low_resolution": median_h < MIN_CELL_HEIGHT,
    }
