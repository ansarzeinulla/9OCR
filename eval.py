"""Evaluate a checkpoint on the synthetic validation set and the real crops.

    python eval.py --ckpt checkpoints/best.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader

from togyz.dataset import RealCropDataset, SyntheticCellDataset
from togyz.model import auto_device, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else auto_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    classes = ckpt["classes"]
    print(f"Loaded {args.ckpt} (epoch {ckpt['epoch'] + 1}, synth val {ckpt['val_acc']:.2%})")

    # synthetic validation (same fixed seed as train.py)
    val_loader = DataLoader(
        SyntheticCellDataset(args.val_size, seed=1234),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    print(f"Synthetic val accuracy: {correct / total:.2%} ({total} samples)")

    # real crops with per-file report
    real = RealCropDataset()
    if len(real) == 0:
        print("No real crops found in data/real_crops - skipping.")
        return
    images, labels, names = real.batch()
    with torch.no_grad():
        probs = torch.softmax(model(images.to(device)).cpu(), dim=1)
    topk = probs.topk(3, dim=1)

    print(f"\nReal crops ({len(real)} files):")
    top1 = top3 = 0
    for i, name in enumerate(names):
        truth = classes[labels[i]]
        guesses = [
            f"{classes[idx]} {p:.1%}"
            for idx, p in zip(topk.indices[i].tolist(), topk.values[i].tolist())
        ]
        hit1 = topk.indices[i, 0] == labels[i]
        hit3 = (topk.indices[i] == labels[i]).any()
        top1 += int(hit1)
        top3 += int(hit3)
        marker = "OK " if hit1 else ("~3 " if hit3 else "MISS")
        print(f"  [{marker}] {name:<14} truth={truth:<4} top3: {', '.join(guesses)}")
    print(f"Real top-1: {top1}/{len(real)} ({top1 / len(real):.0%})  "
          f"top-3: {top3}/{len(real)} ({top3 / len(real):.0%})")


if __name__ == "__main__":
    main()
