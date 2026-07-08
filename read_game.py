"""Read an entire Togyzkumalak scoresheet photo into game records (ONNX).

    python read_game.py "data/2026-07-06 00.00.20.jpg" --out out/sheet1 --result 0-1

Outputs in --out:
    game.json      per ply: bbox, probabilities for all 163 classes, top-k,
                   raw argmax, legal move set, legality flag
    raw.pgn        pure classifier argmax for every ply (even if illegal)
    legal.pgn      replayed under the rules; STOPS at the first illegal argmax,
                   the first empty cell, or when the game is over
    beam.pgn       best fully-legal reconstruction (beam search + kazan/result
                   evidence)
    annotated.jpg  sheet with cell boxes and the beam reconstruction labels
    cells/         every scanned cell crop, e.g. 07_W.png

Inference runs on the exported ONNX models (torch-free). Regenerate them with
`python scripts/export_onnx.py` after training. PGN move annotations: '+' =
capture, 'x' = tuzdyk creation; strip '+' to feed the moves to the 9Q engine.
"""

import argparse
import json
from pathlib import Path

from togyz.pipeline import RESULT_CODES, load_classifier, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="scoresheet photo")
    parser.add_argument("--onnx", default="checkpoints/best.onnx",
                        help="move classifier ONNX (with a .classes.json sidecar)")
    parser.add_argument("--kazan-onnx", default="checkpoints/kazan/best.onnx",
                        help="kazan-number ONNX; checkpoint matching is skipped "
                             "if the file does not exist")
    parser.add_argument("--out", default=None, help="output dir (default: out/<image stem>)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=1024,
                        help="hypotheses kept during beam decoding")
    parser.add_argument("--beam-top", type=int, default=9,
                        help="legal continuations considered per ply (9 = all)")
    parser.add_argument("--result", choices=sorted(RESULT_CODES),
                        help="known game result from the sheet footer "
                             "(1-0 = Bast./White won); re-ranks the beam pool")
    parser.add_argument("--no-tta", action="store_true",
                        help="disable test-time augmentation (7 shifted views/cell)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="softmax temperature; >1 softens overconfident cells")
    args = parser.parse_args()

    out_dir = Path(args.out or Path("out") / Path(args.image).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    moves_clf = load_classifier(args.onnx)
    kazan_clf = None
    if Path(args.kazan_onnx).exists():
        kazan_clf = load_classifier(args.kazan_onnx)
    else:
        print(f"No kazan classifier at {args.kazan_onnx} - checkpoint matching off.")

    print(f"Reading {args.image} ...")
    out = run_pipeline(
        args.image, moves_clf, kazan_clf,
        result=args.result, topk=args.topk,
        beam_width=args.beam_width, per_ply=args.beam_top,
        temperature=args.temperature, tta=not args.no_tta,
        save_cells_dir=out_dir / "cells",
    )

    if out["low_resolution"]:
        print(f"WARNING: median cell height is only {out['median_cell_height']}px - "
              "accuracy suffers at this resolution; re-photograph at full camera "
              "resolution if possible.")
    if out["kazan_report"]:
        print("Kazan checkpoints read: "
              f"{[(r['move'], r['side'], r['read']) for r in out['kazan_report']]}")

    result = {"image": args.image, "onnx": args.onnx, **out["game_json"]}
    (out_dir / "game.json").write_text(json.dumps(result, indent=1))
    (out_dir / "raw.pgn").write_text(out["raw_pgn"])
    (out_dir / "legal.pgn").write_text(out["legal_pgn"])
    (out_dir / "beam.pgn").write_text(out["beam_pgn"])
    out["annotated_image"].save(out_dir / "annotated.jpg", quality=90)

    beam = out["game_json"]["beam"]
    agree = sum(d["agrees_with_raw"] for d in beam["moves"])
    print(f"Scanned {out['plies_scanned']} plies; strict legal replay covers {out['legal_plies']}.")
    print(f"Beam decode: {out['beam_plies']} fully legal plies "
          f"(log-prob {beam['log_prob']:.1f}, agrees with raw argmax on {agree}/{out['beam_plies']}).")
    print(f"Stopped: {out['stopped']}")
    print(f"Outputs in {out_dir}/: game.json, raw.pgn, legal.pgn, beam.pgn, annotated.jpg, cells/")


if __name__ == "__main__":
    main()
