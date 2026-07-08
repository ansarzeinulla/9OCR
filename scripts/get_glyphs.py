"""Build the glyph pools used by the synthesizer (idempotent).

    python scripts/get_glyphs.py            # download + prepare ARDIS
    python scripts/get_glyphs.py --emnist   # also rebuild the EMNIST pool
                                            # (only needed if ingredients/ is absent,
                                            #  e.g. on Colab)

ARDIS (https://ardisdataset.github.io/ARDIS/) Dataset II contains ~10k digit
images cropped from 19th-century European church records - crossed 7s, serif
1s, cursive styles that match Kazakh/European handwriting far better than
EMNIST. Images are normalized here into white-on-black float masks under
glyph_data/ardis/<digit>/.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLYPH_DATA = PROJECT_ROOT / "glyph_data"
ARDIS_URL = (
    "https://github.com/ardisdataset/ARDIS/raw/Updates-Date-String/ARDIS_DATASET_II.rar"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    subprocess.run(["curl", "-L", "--fail", "-o", str(dest), url], check=True)


def extract_rar(archive: Path, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        print(f"Already extracted: {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("bsdtar"):
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(dest)], check=True)
    elif shutil.which("unar"):
        subprocess.run(["unar", "-quiet", "-o", str(dest), str(archive)], check=True)
    elif shutil.which("unrar"):
        subprocess.run(["unrar", "x", "-inul", str(archive), str(dest)], check=True)
    else:
        sys.exit(
            "No RAR extractor found (need bsdtar, unar, or unrar). "
            "On Ubuntu/Colab: apt-get install -y unar"
        )


def infer_digit_label(path: Path) -> str | None:
    if path.parent.name.isdigit() and len(path.parent.name) == 1:
        return path.parent.name
    stem = path.stem
    for sep in ("_", "-", " "):
        token = stem.split(sep)[0]
        if token.isdigit() and len(token) == 1:
            return token
    return None


def photo_to_mask(path: Path) -> np.ndarray | None:
    """Ink-on-paper photo -> white-on-black float mask, tight-cropped."""
    with Image.open(path) as img:
        gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.shape[0] < 10 or gray.shape[1] < 5:
        return None
    inverted = 255.0 - gray
    background = np.percentile(inverted, 50)
    signal = np.clip(inverted - background, 0, None)
    peak = np.percentile(signal, 99.5)
    if peak < 12:  # effectively blank
        return None
    mask = np.clip(signal / peak, 0.0, 1.0)
    mask[mask < 0.18] = 0.0  # suppress paper texture
    ys, xs = np.where(mask > 0.18)
    if len(ys) < 20:
        return None
    mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if mask.shape[0] < 8 or mask.shape[1] < 3:
        return None
    return mask


def prepare_ardis() -> None:
    archive = GLYPH_DATA / "_downloads" / "ARDIS_DATASET_II.rar"
    raw_dir = GLYPH_DATA / "_raw" / "ardis2"
    out_root = GLYPH_DATA / "ardis"
    if out_root.exists() and any(out_root.glob("*/*.png")):
        print(f"ARDIS pool already prepared at {out_root}")
        return

    download(ARDIS_URL, archive)
    extract_rar(archive, raw_dir)

    counts: dict[str, int] = {}
    skipped = 0
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = infer_digit_label(path)
        if label is None:  # 0 is kept: kazan checkpoint numbers use it
            skipped += 1
            continue
        mask = photo_to_mask(path)
        if mask is None:
            skipped += 1
            continue
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        index = counts.get(label, 0)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{label}_{index}.png")
        counts[label] = index + 1

    print(f"ARDIS pool: {sum(counts.values())} glyphs "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))}), "
          f"skipped {skipped}")
    if not counts:
        print("WARNING: no ARDIS glyphs extracted - the synthesizer will fall back "
              "to EMNIST + procedural styles.")


def prepare_emnist(per_class: int) -> None:
    from torchvision.datasets import EMNIST

    out_root = GLYPH_DATA / "emnist"
    if out_root.exists() and any(out_root.glob("*/*.png")):
        print(f"EMNIST pool already prepared at {out_root}")
        return

    print("Downloading EMNIST (byclass) via torchvision ...")
    dataset = EMNIST(str(GLYPH_DATA / "_downloads"), split="byclass", train=True, download=True)
    # byclass labels: 0-9 digits, 10-35 A-Z, 36-61 a-z
    wanted = {label: str(label) for label in range(0, 10)}
    wanted[33] = "x"  # 'X'
    wanted[59] = "x"  # 'x'

    counts: dict[str, int] = {}
    for img, label in zip(dataset.data, dataset.targets):
        char = wanted.get(int(label))
        if char is None:
            continue
        if counts.get(char, 0) >= per_class:
            if all(counts.get(c, 0) >= per_class for c in wanted.values()):
                break
            continue
        arr = img.numpy().T  # EMNIST images are stored transposed
        out_dir = out_root / char
        out_dir.mkdir(parents=True, exist_ok=True)
        index = counts.get(char, 0)
        Image.fromarray(arr).save(out_dir / f"{char}_{index}.png")
        counts[char] = index + 1
    print(f"EMNIST pool: {sum(counts.values())} glyphs "
          f"({', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emnist", action="store_true",
                        help="also build EMNIST pool (needed when ingredients/ is absent)")
    parser.add_argument("--per-class", type=int, default=2500,
                        help="max EMNIST glyphs per character")
    args = parser.parse_args()

    prepare_ardis()
    if args.emnist or not (PROJECT_ROOT / "ingredients").is_dir():
        prepare_emnist(args.per_class)

    # final summary through the sampler itself
    sys.path.insert(0, str(PROJECT_ROOT))
    from togyz.glyphs import GlyphSampler

    print("\nGlyph pools as seen by the synthesizer:")
    print(GlyphSampler(PROJECT_ROOT).describe())
