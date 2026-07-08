"""Synthesizes realistic scoresheet-cell photos for any of the 163 classes.

Design targets, derived from the real crops in data/real_crops/:
  * digits are large and fill most of the crop height (tight framing)
  * cursive right slant, overlapping kerning, variable stroke thickness
  * mid-gray paper with shading, blue-ink-turned-gray strokes (not black)
  * phone-camera artifacts: blur, sensor noise, JPEG compression
  * crop aspect ratio anywhere between ~1:1 and ~3.3:1

Everything is driven by an explicit `random.Random` so the validation set
can be fully deterministic while training data stays infinite.
"""

import argparse
import io
import math
import random

import cv2
import numpy as np
from PIL import Image

from .classes import CLASSES, EMPTY
from .glyphs import GlyphSampler

# working resolution: glyphs are rendered at this digit height, the finished
# cell is later downscaled to a random "photo" size (anti-aliased strokes)
RENDER_DIGIT_H = 96


def _transform_mask(mask: np.ndarray, rot_deg: float, shear: float) -> np.ndarray:
    """Rotate + italic-shear a glyph mask on an expanded canvas."""
    h, w = mask.shape
    pad = int(max(h, w) * 0.4) + 2
    mask = np.pad(mask, pad)
    ph, pw = mask.shape
    center = (pw / 2, ph / 2)
    m = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    # italic shear: top of the glyph shifts right relative to the bottom
    shear_m = np.array([[1.0, -shear, shear * ph / 2], [0.0, 1.0, 0.0]])
    full = np.vstack([shear_m, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]])
    out = cv2.warpAffine(mask, full[:2], (pw, ph), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(out > 0.1)
    if len(ys) == 0:
        return mask
    return out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _vary_thickness(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    op = rng.choice(["dilate", "none", "dilate", "erode", "none"])
    if op == "none":
        return mask
    kernel = np.ones((2, 2), np.uint8)
    fn = cv2.dilate if op == "dilate" else cv2.erode
    return fn(mask, kernel, iterations=1)


def _paper(rng: random.Random, h: int, w: int) -> np.ndarray:
    base = rng.uniform(160, 235)
    img = np.full((h, w), base, np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    # linear lighting gradient
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    plane = gx * xs / max(w, 1) + gy * ys / max(h, 1)
    ptp = plane.max() - plane.min()
    if ptp > 1e-6:
        img += rng.uniform(0, 22) * ((plane - plane.min()) / ptp - 0.5)
    # occasional soft shadow blob
    if rng.random() < 0.4:
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.5, 1.3) * max(w, h)
        d2 = ((xs - cx) ** 2 + (ys - cy) ** 2) / (r * r)
        img -= rng.uniform(5, 20) * np.exp(-d2)
    return img


def _compose_ink(paper: np.ndarray, alpha: np.ndarray, rng: random.Random) -> np.ndarray:
    """Blend an ink alpha mask onto paper with per-pixel ink tone variation."""
    ink_base = rng.uniform(30, 110)
    ink = ink_base + np.random.default_rng(rng.getrandbits(63)).normal(
        0, 8, paper.shape
    ).astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0) * rng.uniform(0.85, 1.0)
    return paper * (1 - alpha) + ink * alpha


def _add_border_lines(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Fragment of the printed cell border caught by an imperfect crop."""
    h, w = img.shape
    edge = rng.choice(["top", "bottom", "left", "right"])
    offset = rng.randint(0, max(1, int(0.06 * min(h, w))))
    darkness = rng.uniform(60, 140)
    thickness = rng.randint(1, 3)
    if edge in ("top", "bottom"):
        y = offset if edge == "top" else h - 1 - offset
        x0, x1 = sorted((rng.randint(0, w // 2), rng.randint(w // 2, w - 1)))
        cv2.line(img, (x0, y), (x1, y), darkness, thickness, cv2.LINE_AA)
    else:
        x = offset if edge == "left" else w - 1 - offset
        y0, y1 = sorted((rng.randint(0, h // 2), rng.randint(h // 2, h - 1)))
        cv2.line(img, (x, y0), (x, y1), darkness, thickness, cv2.LINE_AA)
    return img


def _camera_effects(img: np.ndarray, rng: random.Random) -> np.ndarray:
    # optical blur
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    # occasional slight motion blur
    if rng.random() < 0.15:
        k = rng.choice([3, 5])
        kernel = np.zeros((k, k), np.float32)
        angle = rng.uniform(0, math.pi)
        cv2.line(
            kernel,
            (0, int((k - 1) / 2 * (1 - math.sin(angle)))),
            (k - 1, int((k - 1) / 2 * (1 + math.sin(angle)))),
            1.0,
            1,
        )
        kernel /= max(kernel.sum(), 1e-6)
        img = cv2.filter2D(img, -1, kernel)
    # sensor noise
    noise_rng = np.random.default_rng(rng.getrandbits(63))
    img = img + noise_rng.normal(0, rng.uniform(1.5, 7.0), img.shape).astype(np.float32)
    # brightness / contrast jitter
    img = (img - 128.0) * rng.uniform(0.82, 1.15) + 128.0 + rng.uniform(-15, 15)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # JPEG artifacts
    if rng.random() < 0.6:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 85)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img


def synthesize_cell(
    class_name: str, sampler: GlyphSampler, rng: random.Random
) -> Image.Image:
    """Render one cell photo for `class_name` ('11'..'99x' or 'empty')."""
    if class_name == EMPTY:
        h = rng.randint(60, 180)
        w = int(h * rng.uniform(1.2, 3.3))
        img = _paper(rng, h, w)
        if rng.random() < 0.25:
            img = _add_border_lines(img, rng)
        return Image.fromarray(_camera_effects(img, rng))

    # --- prepare glyph masks ---
    slant = rng.uniform(-0.10, 0.40)  # shared cursive slant, biased rightward
    masks = []
    for char in class_name:
        mask = sampler.sample(char, rng)
        rel_h = rng.uniform(0.5, 0.8) if char == "x" else rng.uniform(0.85, 1.15)
        target_h = max(8, int(RENDER_DIGIT_H * rel_h))
        target_w = max(4, int(mask.shape[1] * target_h / mask.shape[0]))
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask = _transform_mask(mask, rng.uniform(-7, 7), slant + rng.uniform(-0.05, 0.05))
        mask = _vary_thickness(mask, rng)
        masks.append(mask)

    gaps = [
        int(rng.uniform(-0.18, 0.15) * masks[i].shape[1])
        for i in range(len(masks) - 1)
    ]
    total_w = sum(m.shape[1] for m in masks) + sum(gaps)
    max_h = max(m.shape[0] for m in masks)

    # --- cell geometry: digits fill `fill` of the crop height ---
    fill = rng.uniform(0.45, 0.92)
    cell_h = int(max_h / fill)
    aspect = rng.uniform(0.95, 3.3)
    cell_w = max(int(cell_h * aspect), int(total_w * rng.uniform(1.02, 1.2)))

    # horizontal placement: real crops are sometimes left-aligned with empty space
    slack = cell_w - total_w
    align = rng.random()
    if align < 0.35:  # left
        x = int(slack * rng.uniform(0.0, 0.15))
    elif align < 0.85:  # center-ish
        x = int(slack * rng.uniform(0.25, 0.6))
    else:  # right
        x = int(slack * rng.uniform(0.7, 0.95))

    # --- compose ink alpha on full-cell canvas ---
    alpha = np.zeros((cell_h, cell_w), np.float32)
    y_center = (cell_h - max_h) / 2
    for i, mask in enumerate(masks):
        mh, mw = mask.shape
        y = int(y_center + (max_h - mh) * rng.uniform(0.2, 0.8) + rng.uniform(-0.04, 0.04) * cell_h)
        y = min(max(y, 0), cell_h - mh)
        x = min(max(x, 0), cell_w - mw)
        region = alpha[y : y + mh, x : x + mw]
        np.maximum(region, mask, out=region)
        if i < len(gaps):
            x += mw + gaps[i]

    img = _compose_ink(_paper(rng, cell_h, cell_w), alpha, rng)
    if rng.random() < 0.12:
        img = _add_border_lines(img, rng)

    # downscale to a random photo resolution (real crops are 115-260 px tall)
    out_h = rng.randint(48, 190)
    out_w = max(16, int(cell_w * out_h / cell_h))
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return Image.fromarray(_camera_effects(img, rng))


def _preview(path: str, seed: int, count: int) -> None:
    """Render a comparison grid: the real-crop classes first, then random ones."""
    sampler = GlyphSampler()
    rng = random.Random(seed)
    fixed = ["96", "77", "13x", "37", "85", "42"] * 2
    names = fixed + [rng.choice(CLASSES) for _ in range(max(0, count - len(fixed)))]

    tile_w, tile_h, caption = 200, 100, 14
    cols = 6
    rows = math.ceil(len(names) / cols)
    sheet = Image.new("L", (cols * tile_w, rows * (tile_h + caption)), 255)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        cell = synthesize_cell(name, sampler, rng).resize((tile_w, tile_h))
        cx, cy = (i % cols) * tile_w, (i // cols) * (tile_h + caption)
        sheet.paste(cell, (cx, cy + caption))
        draw.text((cx + 4, cy + 1), name, fill=0)
    sheet.save(path)
    print(f"Saved {len(names)}-cell preview to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", default="preview.png", help="output image path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=48, help="number of cells")
    args = parser.parse_args()
    _preview(args.preview, args.seed, args.n)
