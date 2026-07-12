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

# Unified board-diagram classes for the summary strips: pit/kazan values
# 0-81 (rendered as 1 or 2 digits), 'x' = tuzdyk mark, '-' = zero mark.
# At inference the pipeline filters per context: kazan boxes allow only
# 10-81, pit cells allow '-', 'x' and 0-81.
DASH = "-"
TUZDYK = "x"
DIAGRAM_CLASSES = [str(v) for v in range(82)] + [TUZDYK, DASH, EMPTY]
