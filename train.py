"""Train the Togyzkumalak move classifier on on-the-fly synthetic cells.

Local smoke test (Mac, ~minutes):
    python train.py --epochs 1 --samples-per-epoch 4000 --batch-size 64

Full run (Colab GPU or any CUDA machine):
    python train.py --epochs 30

Prints synthetic validation accuracy AND accuracy on the real labeled crops
(data/real_crops) every epoch; saves checkpoints/best.pt and last.pt.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from togyz.classes import CLASSES, DIAGRAM_CLASSES
from togyz.dataset import RealCropDataset, SyntheticCellDataset
from togyz.model import auto_device, build_model, save_checkpoint


def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def evaluate_real(model, real: RealCropDataset, device) -> tuple[float, float]:
    """Top-1 and top-3 accuracy on the real crops."""
    if len(real) == 0:
        return float("nan"), float("nan")
    images, labels, _ = real.batch()
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device)).cpu()
    top3 = logits.topk(3, dim=1).indices
    top1_acc = (top3[:, 0] == labels).float().mean().item()
    top3_acc = (top3 == labels[:, None]).any(dim=1).float().mean().item()
    return top1_acc, top3_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["moves", "diagram"], default="moves",
                        help="moves: 163-class cell classifier; diagram: unified "
                             "board-diagram reader (0-81, 'x', '-') for the kazan "
                             "boxes and pit cells of the summary strips "
                             "(replaces the old 'kazan' task)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None,
                        help="default: checkpoints (moves) / checkpoints/diagram")
    parser.add_argument("--resume", default=None, help="path to last.pt to continue")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Device: {device}, task: {args.task}")

    classes = DIAGRAM_CLASSES if args.task == "diagram" else CLASSES
    out_dir = Path(args.out or ("checkpoints/diagram" if args.task == "diagram" else "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.json").write_text(json.dumps(classes))

    train_set = SyntheticCellDataset(args.samples_per_epoch, seed=None,
                                     classes=classes, task=args.task)
    val_set = SyntheticCellDataset(args.val_size, seed=1234,
                                   classes=classes, task=args.task)
    # the labeled real crops are move cells; other tasks have no real eval set
    real_set = RealCropDataset() if args.task == "moves" else RealCropDataset("/nonexistent")
    print(f"Glyph pools:\n{train_set.sampler.describe()}")
    print(f"Real eval crops: {len(real_set)}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_acc = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("val_acc", 0.0)
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            seen += labels.numel()
            if step % 50 == 0:
                print(
                    f"  epoch {epoch + 1} step {step + 1}/{len(train_loader)} "
                    f"loss {running_loss / seen:.4f}",
                    flush=True,
                )
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        real_top1, real_top3 = evaluate_real(model, real_set, device)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs} [{elapsed:.0f}s] "
            f"loss {running_loss / max(seen, 1):.4f} | synth val {val_acc:.2%} | "
            f"real top-1 {real_top1:.2%} top-3 {real_top3:.2%}",
            flush=True,
        )

        save_checkpoint(out_dir / "last.pt", model, epoch, val_acc, optimizer, classes=classes)
        if val_acc >= best_acc:
            best_acc = val_acc
            save_checkpoint(out_dir / "best.pt", model, epoch, val_acc, classes=classes)
            print(f"  new best ({val_acc:.2%}) -> {out_dir / 'best.pt'}")

    print(f"Done. Best synthetic val accuracy: {best_acc:.2%}")


if __name__ == "__main__":
    main()
