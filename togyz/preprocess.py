"""The single image -> array preprocessing used by training AND inference.

The legacy pipeline failed partly because train and inference normalized
differently. Every consumer (dataset, eval, predict, the ONNX pipeline) must
import from here.

This module is deliberately **torch-free** (numpy only) so the serving path
can import it without pulling in PyTorch. Torch consumers wrap the returned
array with ``torch.from_numpy(...)``.
"""

import numpy as np
from PIL import Image, ImageOps

TARGET_H = 64
TARGET_W = 128
MEAN = 0.5
STD = 0.5


def _estimate_background(gray: Image.Image) -> int:
    """Median of the image border, so padding blends with the paper."""
    arr = np.asarray(gray)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return int(np.median(border))


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """PIL image (any mode/size) -> float32 array [1, TARGET_H, TARGET_W]."""
    img = ImageOps.exif_transpose(img).convert("L")
    w, h = img.size
    target_ratio = TARGET_W / TARGET_H

    # Pad (never crop) to the target aspect ratio, centered on paper-colored canvas.
    if w / h < target_ratio:
        new_w, new_h = int(round(h * target_ratio)), h
    else:
        new_w, new_h = w, int(round(w / target_ratio))
    if (new_w, new_h) != (w, h):
        canvas = Image.new("L", (new_w, new_h), color=_estimate_background(img))
        canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
        img = canvas

    img = img.resize((TARGET_W, TARGET_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None, :, :]  # add channel axis -> [1, H, W]
    return (arr - MEAN) / STD


def preprocess_file(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return preprocess_pil(img)
