"""Classify cropped cell images.

    python predict.py data/real_crops/96_a.jpg
    python predict.py "data/real_crops/*.jpg" --topk 3
    python predict.py cell.jpg --allowed "12,34x,56"   # restrict to legal moves

--allowed renormalizes probabilities over the given moves - at any game state
at most 9 moves are legal, so an upstream game tracker can pass them here.
"""

import argparse
import glob

import numpy as np
import torch

from togyz.classes import CLASS_TO_IDX
from togyz.model import auto_device, load_checkpoint
from togyz.preprocess import preprocess_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="image paths or globs")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--allowed", default=None,
                        help="comma-separated legal moves, e.g. '12,34x,56'")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = []
    for pattern in args.images:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    device = torch.device(args.device) if args.device else auto_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    classes = ckpt["classes"]

    allowed_idx = None
    if args.allowed:
        moves = [m.strip() for m in args.allowed.split(",") if m.strip()]
        unknown = [m for m in moves if m not in CLASS_TO_IDX]
        if unknown:
            parser.error(f"unknown moves in --allowed: {unknown}")
        allowed_idx = torch.tensor([CLASS_TO_IDX[m] for m in moves])

    batch = torch.from_numpy(np.stack([preprocess_file(p) for p in paths]))
    with torch.no_grad():
        probs = torch.softmax(model(batch.to(device)).cpu(), dim=1)

    for path, p in zip(paths, probs):
        topk = p.topk(min(args.topk, len(classes)))
        guesses = ", ".join(
            f"{classes[i]} {v:.1%}" for i, v in zip(topk.indices.tolist(), topk.values.tolist())
        )
        line = f"{path}: {guesses}"
        if allowed_idx is not None:
            legal = p[allowed_idx]
            legal = legal / legal.sum()
            best = int(legal.argmax())
            line += f"  | legal pick: {classes[int(allowed_idx[best])]} {legal[best]:.1%}"
        print(line)


if __name__ == "__main__":
    main()
