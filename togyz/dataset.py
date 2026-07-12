"""Datasets: on-the-fly synthetic cells and the labeled real crops."""

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .classes import CLASSES, CLASS_TO_IDX
from .glyphs import GlyphSampler
from .preprocess import preprocess_file, preprocess_pil
from .synth import synthesize_cell, synthesize_diagram_cell


class SyntheticCellDataset(Dataset):
    """Generates cells on the fly - no dataset folders on disk.

    With seed=None every access is fresh random data (infinite training
    variety; DataLoader workers are seeded independently by torch). With a
    seed, sample i is always the same image: a fixed validation set.
    """

    def __init__(self, num_samples: int, seed: int | None = None, project_root=".",
                 classes: list[str] | None = None, task: str = "moves"):
        self.sampler = GlyphSampler(project_root)
        self.num_samples = num_samples
        self.seed = seed
        self.classes = classes or CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.synth_fn = synthesize_diagram_cell if task == "diagram" else synthesize_cell

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        if self.seed is not None:
            rng = random.Random(f"{self.seed}:{index}")
        else:
            rng = random.Random(random.getrandbits(64))
        class_name = self.classes[rng.randrange(len(self.classes))]
        img = self.synth_fn(class_name, self.sampler, rng)
        return torch.from_numpy(preprocess_pil(img)), self.class_to_idx[class_name]


class RealCropDataset(Dataset):
    """The labeled real photos in data/real_crops (tiny; used for eval only)."""

    def __init__(self, root: str = "data/real_crops"):
        self.root = Path(root)
        self.items: list[tuple[str, str]] = []
        labels_file = self.root / "labels.csv"
        if labels_file.exists():
            with open(labels_file, newline="") as f:
                for row in csv.DictReader(f):
                    label = row["label"].strip()
                    if label in CLASS_TO_IDX:
                        self.items.append((row["filename"].strip(), label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        filename, label = self.items[index]
        tensor = torch.from_numpy(preprocess_file(str(self.root / filename)))
        return tensor, CLASS_TO_IDX[label]

    def batch(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """All crops as one batch plus their filenames (for reporting)."""
        tensors, labels = zip(*(self[i] for i in range(len(self))))
        names = [name for name, _ in self.items]
        return torch.stack(tensors), torch.tensor(labels), names
