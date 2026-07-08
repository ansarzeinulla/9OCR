"""Glyph pools: per-character handwriting samples used by the synthesizer.

Sources (all optional, at least one digit source must exist):
  * ingredients/<char>/*.png        EMNIST-style masks, white ink on black
  * glyph_data/emnist/<char>/*.png  same, regenerated via scripts/get_glyphs.py --emnist
  * glyph_data/ardis/<digit>/*.png  European handwriting (ARDIS), pre-normalized
                                    to white-on-black masks by scripts/get_glyphs.py

EMNIST glyphs are American-style; real Togyzkumalak scoresheets use
European/Kazakh styles (crossed 7, serif 1, cursive 9). ARDIS provides real
European glyphs; on top of that, procedural edits (crossbar on 7, flag on 1)
convert a share of EMNIST glyphs to European style, and 'x' marks are also
drawn procedurally for extra variety.
"""

import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DIGITS = "123456789"  # move notation digits; '0' appears only in kazan numbers
CHARS = "0" + DIGITS + "x"
INK_THRESHOLD = 0.15  # mask values above this count as ink


def _list_images(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        str(p) for p in directory.iterdir() if p.suffix.lower() in exts
    )


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > INK_THRESHOLD)
    if len(ys) == 0:
        return mask
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _load_mask(path: str) -> np.ndarray:
    """Load a white-on-black glyph image as float mask in [0, 1]."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return _tight_crop(arr)


def add_crossbar_to_7(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """European 7: horizontal bar through the stem."""
    h, w = mask.shape
    if h < 8 or w < 4:
        return mask
    y = int(h * rng.uniform(0.40, 0.60))
    x0 = int(w * rng.uniform(-0.05, 0.15))
    x1 = int(w * rng.uniform(0.85, 1.05))
    dy = int(w * rng.uniform(-0.10, 0.10))
    thickness = max(2, int(round(h * 0.05 * rng.uniform(0.8, 1.6))))
    out = mask.copy()
    cv2.line(out, (x0, y - dy // 2), (x1, y + dy // 2), 1.0, thickness, cv2.LINE_AA)
    return np.clip(out, 0.0, 1.0)


def add_flag_to_1(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """European 1: diagonal flag from the top of the stem down-left."""
    h, w = mask.shape
    if h < 8:
        return mask
    ink_rows = np.where(mask.max(axis=1) > INK_THRESHOLD)[0]
    if len(ink_rows) == 0:
        return mask
    y_top = int(ink_rows[0])
    row = mask[min(y_top + 1, h - 1)]
    x_top = int(np.argmax(row))
    length = h * rng.uniform(0.22, 0.42)
    angle = math.radians(rng.uniform(25, 55))
    x_end = int(x_top - length * math.cos(angle))
    y_end = int(y_top + length * math.sin(angle))
    thickness = max(2, int(round(h * 0.05 * rng.uniform(0.8, 1.4))))
    # give the flag room on the left if it would leave the canvas
    pad = max(0, -x_end + 2)
    out = np.pad(mask, ((0, 0), (pad, 0)))
    cv2.line(out, (x_top + pad, y_top), (x_end + pad, y_end), 1.0, thickness, cv2.LINE_AA)
    return np.clip(out, 0.0, 1.0)


def _bezier_points(p0, p1, p2, n=24):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
    return pts.astype(np.int32)


def procedural_x(rng: random.Random) -> np.ndarray:
    """Two crossing, slightly curved pen strokes."""
    s = 64
    canvas = np.zeros((s, s), np.float32)
    for (ax, ay), (bx, by) in [((0.15, 0.12), (0.85, 0.88)), ((0.85, 0.10), (0.15, 0.90))]:
        p0 = np.array([ax + rng.uniform(-0.08, 0.08), ay + rng.uniform(-0.08, 0.08)]) * s
        p2 = np.array([bx + rng.uniform(-0.08, 0.08), by + rng.uniform(-0.08, 0.08)]) * s
        mid = (p0 + p2) / 2 + np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)]) * s
        pts = _bezier_points(p0, mid, p2)
        thickness = rng.randint(3, 6)
        cv2.polylines(canvas, [pts], False, 1.0, thickness, cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
    return _tight_crop(np.clip(canvas, 0.0, 1.0))


class GlyphSampler:
    """Samples a glyph mask (float [0,1], white ink, tight-cropped) per character."""

    def __init__(self, project_root: str | os.PathLike = "."):
        root = Path(project_root)
        self.pools: dict[str, dict[str, list[str]]] = {c: {} for c in CHARS}
        for char in CHARS:
            emnist = _list_images(root / "ingredients" / char) + _list_images(
                root / "glyph_data" / "emnist" / char
            )
            if emnist:
                self.pools[char]["emnist"] = emnist
            if char != "x":
                ardis = _list_images(root / "glyph_data" / "ardis" / char)
                if ardis:
                    self.pools[char]["ardis"] = ardis

        missing = [c for c in DIGITS if not self.pools[c]]
        if missing:
            raise FileNotFoundError(
                f"No glyph images found for digits {missing}. "
                "Run `python scripts/get_glyphs.py` (optionally with --emnist) first."
            )

    def describe(self) -> str:
        lines = []
        for char in CHARS:
            sources = ", ".join(
                f"{name}:{len(paths)}" for name, paths in self.pools[char].items()
            ) or "procedural only"
            lines.append(f"  '{char}': {sources}")
        return "\n".join(lines)

    def sample(self, char: str, rng: random.Random) -> np.ndarray:
        if char == "x" and (not self.pools["x"] or rng.random() < 0.35):
            return procedural_x(rng)

        sources = self.pools[char]
        if not sources:
            raise FileNotFoundError(
                f"No glyphs for {char!r} - re-run scripts/get_glyphs.py "
                "(digit 0 needs the ARDIS pool or --emnist)"
            )
        if "ardis" in sources and "emnist" in sources:
            source = "ardis" if rng.random() < 0.55 else "emnist"
        else:
            source = next(iter(sources))
        mask = _load_mask(rng.choice(sources[source]))

        # Convert a share of EMNIST glyphs to European style; ARDIS already is.
        euro_prob = 0.15 if source == "ardis" else 0.45
        if char == "7" and rng.random() < euro_prob:
            mask = add_crossbar_to_7(mask, rng)
        if char == "1" and source == "emnist" and rng.random() < 0.35:
            mask = add_flag_to_1(mask, rng)
        return mask
