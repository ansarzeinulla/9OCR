"""Export the trained .pt checkpoints to ONNX for torch-free serving.

    python scripts/export_onnx.py

Writes, next to each source checkpoint:
    checkpoints/best.onnx          + checkpoints/best.classes.json
    checkpoints/kazan/best.onnx    + checkpoints/kazan/best.classes.json

ONNX carries no Python metadata, so the class list (which maps output index
-> move/kazan string) is saved as a JSON sidecar the serving code reads. The
graph has a dynamic batch axis so one session handles any number of cells.
"""

import argparse
import json
from pathlib import Path

import onnx
import torch

from togyz.model import build_model
from togyz.preprocess import TARGET_H, TARGET_W

OPSET = 17


def export_one(ckpt_path: Path) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    onnx_path = ckpt_path.with_suffix(".onnx")
    classes_path = ckpt_path.with_suffix(".classes.json")
    # Legacy TorchScript exporter (dynamo=False) embeds weights directly, so
    # the result is a single self-contained .onnx (no sidecar .data file) -
    # much cleaner to commit to the HuggingFace Space via Git LFS.
    dummy = torch.zeros(1, 1, TARGET_H, TARGET_W)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    # Belt and braces: if any external data slipped out, fold it back in so the
    # .onnx is guaranteed standalone, then drop the stray .data file.
    model_proto = onnx.load(str(onnx_path))
    onnx.save_model(model_proto, str(onnx_path), save_as_external_data=False)
    data_file = onnx_path.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()

    classes_path.write_text(json.dumps(classes))
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"{ckpt_path} -> {onnx_path} ({len(classes)} classes, {size_mb:.1f} MB)")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints",
        nargs="*",
        default=["checkpoints/best.pt", "checkpoints/kazan/best.pt"],
        help="checkpoint(s) to export (default: moves + kazan best.pt)",
    )
    args = parser.parse_args()
    for path in args.checkpoints:
        p = Path(path)
        if p.exists():
            export_one(p)
        else:
            print(f"skip: {p} not found")


if __name__ == "__main__":
    main()
