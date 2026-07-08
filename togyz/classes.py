"""Canonical class list for Togyzkumalak move OCR.

162 moves (start hole 1-9, end hole 1-9, with or without a capture 'x')
plus 'empty'. The order matches the alphabetical order used by
torchvision ImageFolder in the legacy pipeline (legacy/class_mapping.json),
so old and new indices agree.
"""

MOVES = [
    f"{start}{end}{suffix}"
    for start in range(1, 10)
    for end in range(1, 10)
    for suffix in ("", "x")
]

EMPTY = "empty"
CLASSES = MOVES + [EMPTY]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)  # 163

# Kazan checkpoint numbers on the scoresheet summary strips: always exactly
# two digits (with leading zero), values 00-81, never an 'x'.
KAZAN_CLASSES = [f"{v:02d}" for v in range(82)] + [EMPTY]
