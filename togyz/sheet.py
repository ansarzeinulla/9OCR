"""Locate the printed blocks on a scoresheet photo and crop cells by FIXED
PROPORTIONS - no per-cell detection, no overlap between crops.

Federation scoresheet layout (strict):
  * 8 move blocks in two bands of 4; each block = header row + 10 move rows
    (header and every row have the SAME height: block_height / 11), columns
    [No | Bast. (White) | Kost. (Black)].
  * 8 board diagrams, one under each block: a 2x9 pit grid of equal cells,
    Black kazan box attached on TOP and White kazan attached BELOW, each
    exactly 2 pit-cells wide and one pit-row tall, over pits 5-6.

The 8 block borders are the ONLY thing detected. They are consolidated into
the page's invisible grid (shared band lines, shared column lines), the
sheet is deskewed (<= ~15 deg), and every cell is then cropped from the
block border using the LAYOUT PROPORTIONS below.

Tune the proportions by editing the constants and re-running:

    venv311/bin/python scripts/preview_split.py
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

# ========================================================================= #
#  LAYOUT PROPORTIONS - edit these, then: venv311/bin/python scripts/preview_split.py
# ========================================================================= #
ROWS_PER_TABLE = 10   # move rows per block (plus 1 header row of same height)

# --- move block columns, as fractions of the block width ------------------ #
COL_NO_END = 0.20     # right edge of the "No" column  (= left edge of Bast.)
COL_W_END = 0.60      # right edge of Bast. (White)    (= left edge of Kost.)

# --- board diagram, relative to the block above it ------------------------ #
# Measured defaults for the 2026-07-12 form. The 2026-07-08 form needs:
#   DIAG_TOP_GAP = 1.45, DIAG_ROW_H = 1.30, DIAG_LEFT = -0.02, DIAG_WIDTH = 0.99
DIAG_TOP_GAP = 2.49   # gap between block bottom and pit-grid top, in cell heights
DIAG_ROW_H = 1.10     # pit row height, in move-cell heights
DIAG_LEFT = -0.1    # pit-grid left edge offset from block left, x block width
DIAG_WIDTH = 1.05     # pit-grid width, x block width  (9 equal pit cells)

# --- kazan boxes ----------------------------------------------------------- #
KAZAN_FIRST_PIT = 4   # 0-based leftmost pit column under/over the kazan (4 = pit 5)
KAZAN_PITS = 2        # kazan width in pit cells

# --- crop margins (fractions of the cell size; 0 = crop exactly, no overlap) #
MOVE_MARGIN = 0.0
PIT_MARGIN = 0.0
KAZAN_MARGIN = 0.0
# ========================================================================= #

TABLES = 8
MAX_SKEW_DEG = 15.0


@dataclass
class Cell:
    move_no: int  # move cells: 1-80; diagram cells: checkpoint move (10..80)
    side: str  # "W" (Bast.) or "B" (Kost.); for pits this is the row owner
    bbox: tuple[int, int, int, int]  # x, y, w, h in deskewed-sheet coords
    image: Image.Image
    quad: np.ndarray | None = None  # optional 4x2 corners (unused when axis-aligned)
    kind: str = "move"  # "move" | "kazan" | "pit"
    pit_index: int = 0  # 1-9 for pit cells (scoresheet numbering), else 0


def _grid_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    horiz = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    )
    vert = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 40))
    )
    return cv2.dilate(cv2.add(horiz, vert), np.ones((3, 3), np.uint8))


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype(np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    return np.array(
        [points[sums.argmin()], points[diffs.argmin()],
         points[sums.argmax()], points[diffs.argmax()]],
        np.float32,
    )


def _find_table_quads(gray: np.ndarray) -> list[np.ndarray]:
    """Corner quads of the 8 move blocks, ordered top band then bottom band,
    each left to right."""
    h, w = gray.shape
    mask = _grid_mask(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    quads = []
    for i in range(1, n):
        if stats[i][2] <= w * 0.1 or stats[i][3] <= h * 0.1:
            continue
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad = _order_corners(approx)
        else:  # fall back to the min-area rectangle corners
            quad = _order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
        quads.append((cv2.contourArea(quad), quad))
    if len(quads) < TABLES:
        raise RuntimeError(
            f"Found only {len(quads)} move tables (need {TABLES}). "
            "Check photo quality/framing."
        )
    # the 8 move blocks all have near-identical area; drop outliers like the
    # footer/signature block, which can out-size a genuine block
    median_area = float(np.median([a for a, _ in quads]))
    consistent = [(a, q) for a, q in quads if 0.5 * median_area < a < 2.0 * median_area]
    if len(consistent) >= TABLES:
        quads = consistent
    quads = [q for _, q in sorted(quads, key=lambda t: -t[0])[:TABLES]]
    # split into top/bottom bands by y, order each band left to right
    quads.sort(key=lambda q: q[:, 1].mean())
    top = sorted(quads[:4], key=lambda q: q[:, 0].mean())
    bottom = sorted(quads[4:], key=lambda q: q[:, 0].mean())
    return top + bottom


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the sheet upright using the printed block borders (<= ~15 deg).

    After this every printed line is horizontal/vertical, so the strict
    layout can be applied with axis-aligned boxes.
    """
    quads = _find_table_quads(gray)
    angles = []
    for q in quads:
        for a, b in ((q[0], q[1]), (q[3], q[2])):  # top + bottom border edges
            angles.append(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
    angle = float(np.clip(np.median(angles), -MAX_SKEW_DEG, MAX_SKEW_DEG))
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.median(gray)),
    )


def _band_fit(xcs: list[float], ys: list[float]):
    """Fit y = a*x + b through the 4 blocks of one band (the blocks ARE the
    anchors: their borders lie on one shared line per band)."""
    a, b = np.polyfit(xcs, ys, 1)
    return lambda x: float(a * x + b)


def _page_grid(gray: np.ndarray):
    """The page's invisible grid, consolidated from all 8 block borders.

    Top/bottom edges of each band lie on one shared (fitted) line; each
    column's left/right edges are shared between the two bands; every cell
    (header included) has the same height H = block_height / 11.
    """
    quads = _find_table_quads(gray)
    boxes = [(float(q[:, 0].min()), float(q[:, 1].min()),
              float(q[:, 0].max()), float(q[:, 1].max())) for q in quads]
    top_band, bottom_band = boxes[:4], boxes[4:]

    col_x0 = [(t[0] + b[0]) / 2 for t, b in zip(top_band, bottom_band)]
    col_x1 = [(t[2] + b[2]) / 2 for t, b in zip(top_band, bottom_band)]

    top_fits, bot_fits = [], []
    for band in (top_band, bottom_band):
        xcs = [(b[0] + b[2]) / 2 for b in band]
        top_fits.append(_band_fit(xcs, [b[1] for b in band]))
        bot_fits.append(_band_fit(xcs, [b[3] for b in band]))

    H = float(np.mean([b[3] - b[1] for b in boxes])) / (ROWS_PER_TABLE + 1)

    return {
        "col_x0": col_x0, "col_x1": col_x1,
        "band_top": top_fits, "band_bot": bot_fits, "H": H,
    }


def _crop(gray: np.ndarray, x0: float, y0: float, x1: float, y1: float,
          mx: float, my: float):
    """Axis-aligned crop with an optional margin, clamped to the sheet."""
    sh, sw = gray.shape
    cx0 = int(max(0, x0 - mx))
    cy0 = int(max(0, y0 - my))
    cx1 = int(min(sw, x1 + mx))
    cy1 = int(min(sh, y1 + my))
    return (cx0, cy0, cx1 - cx0, cy1 - cy0), Image.fromarray(gray[cy0:cy1, cx0:cx1])


def extract_cells(image, with_gridlines: bool = False):
    """All 160 move cells (80 moves x W/B) plus the deskewed sheet image.

    Cells are cropped strictly by fixed proportions from the consolidated
    block grid: header + 10 rows of equal height H = block_height / 11,
    columns at COL_NO_END / COL_W_END. No overlap between neighbors.

    `image` is a path (str/Path) or a PIL.Image. Returns (cells, sheet) or,
    with `with_gridlines`, (cells, sheet, gridlines) where gridlines are the
    ((x0, y0), (x1, y1)) segments of the applied grid.
    """
    if isinstance(image, Image.Image):
        pil = ImageOps.exif_transpose(image).convert("L")
    else:
        with Image.open(image) as img:
            pil = ImageOps.exif_transpose(img).convert("L")
    gray = _deskew(np.asarray(pil))
    sheet = Image.fromarray(gray)
    grid = _page_grid(gray)
    H = grid["H"]

    cells: list[Cell] = []
    gridlines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for band in range(2):
        for col in range(4):
            bx0, bx1 = grid["col_x0"][col], grid["col_x1"][col]
            xc = (bx0 + bx1) / 2
            by0 = grid["band_top"][band](xc)
            bw = bx1 - bx0
            cols = [bx0, bx0 + COL_NO_END * bw, bx0 + COL_W_END * bw, bx1]

            if with_gridlines:
                for r in range(ROWS_PER_TABLE + 2):
                    y = by0 + r * H
                    gridlines.append(((int(bx0), int(y)), (int(bx1), int(y))))
                for x in cols:
                    gridlines.append(
                        ((int(x), int(by0)),
                         (int(x), int(by0 + (ROWS_PER_TABLE + 1) * H)))
                    )

            for r in range(ROWS_PER_TABLE):
                y0 = by0 + (r + 1) * H  # row 0 is the header (same height H)
                y1 = y0 + H
                for col_index, side in ((1, "W"), (2, "B")):
                    x0, x1 = cols[col_index], cols[col_index + 1]
                    bbox, crop = _crop(gray, x0, y0, x1, y1,
                                       (x1 - x0) * MOVE_MARGIN, H * MOVE_MARGIN)
                    cells.append(Cell(
                        (band * 4 + col) * ROWS_PER_TABLE + r + 1, side,
                        bbox, crop,
                    ))

    if with_gridlines:
        return cells, sheet, gridlines
    return cells, sheet


def extract_diagram_cells(sheet: Image.Image) -> list[Cell]:
    """All board-diagram cells, cropped by fixed proportions from the block
    above each diagram: pit-grid top at DIAG_TOP_GAP cell-heights below the
    block, rows of DIAG_ROW_H cell-heights, 9 equal pit columns spanning
    DIAG_WIDTH of the block width starting at DIAG_LEFT; kazans exactly over
    pits KAZAN_FIRST_PIT+1 .. +KAZAN_PITS (Black above, White below).

    Upper pit row = Black, indexes 9..1 left-to-right; lower row = White,
    1..9. `sheet` must be the deskewed image returned by `extract_cells`.
    """
    gray = np.asarray(sheet)
    grid = _page_grid(gray)
    H = grid["H"]
    row_h = H * DIAG_ROW_H

    cells: list[Cell] = []
    for band in range(2):
        for col in range(4):
            checkpoint = (band * 4 + col + 1) * 10
            bx0, bx1 = grid["col_x0"][col], grid["col_x1"][col]
            xc = (bx0 + bx1) / 2
            block_bottom = grid["band_bot"][band](xc)
            bw = bx1 - bx0

            top = block_bottom + DIAG_TOP_GAP * H
            mid = top + row_h
            bottom = mid + row_h
            gx0 = bx0 + DIAG_LEFT * bw
            pit_w = DIAG_WIDTH * bw / 9.0
            col_xs = [gx0 + j * pit_w for j in range(10)]

            for band_y0, band_y1, owner in ((top, mid, "B"), (mid, bottom, "W")):
                for j in range(9):
                    pit_index = 9 - j if owner == "B" else j + 1
                    bbox, crop = _crop(gray, col_xs[j], band_y0, col_xs[j + 1], band_y1,
                                       pit_w * PIT_MARGIN, row_h * PIT_MARGIN)
                    cells.append(Cell(
                        checkpoint, owner, bbox, crop,
                        kind="pit", pit_index=pit_index,
                    ))

            kx0 = col_xs[KAZAN_FIRST_PIT]
            kx1 = col_xs[KAZAN_FIRST_PIT + KAZAN_PITS]
            for ky0, ky1, side in ((top - row_h, top, "B"), (bottom, bottom + row_h, "W")):
                bbox, crop = _crop(gray, kx0, ky0, kx1, ky1,
                                   pit_w * KAZAN_MARGIN, row_h * KAZAN_MARGIN)
                cells.append(Cell(checkpoint, side, bbox, crop, kind="kazan"))
    return cells


def clean_cell(image: Image.Image) -> tuple[Image.Image, float]:
    """Remove printed grid-line fragments from a cell crop.

    Returns the cleaned crop and the fraction of remaining ink pixels
    (handwriting). Blank cells score near zero even when grid lines cross
    the crop, so this is a classifier-independent emptiness signal.
    """
    gray = np.asarray(image)
    h, w = gray.shape
    if h < 3 or w < 3:  # degenerate crop: bypass the OpenCV filters
        return image, 0.0

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    # printed lines: long straight runs spanning most of the crop
    horiz = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(w * 0.6)), 1))
    )
    vert = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(h * 0.6))))
    )
    lines = cv2.dilate(cv2.add(horiz, vert), np.ones((3, 3), np.uint8))

    background = int(np.median(gray[thr == 0])) if (thr == 0).any() else 255
    cleaned = gray.copy()
    cleaned[lines > 0] = background

    ink = (thr > 0) & (lines == 0)
    ink[: h // 8, :] = ink[-h // 8 :, :] = False  # ignore crop edges: neighbor
    ink[:, : w // 12] = ink[:, -w // 12 :] = False  # rows/cells bleeding in
    ink_ratio = float(ink.sum()) / (h * w)
    return Image.fromarray(cleaned), ink_ratio


def render_overlay(sheet: Image.Image, cells: list[Cell], labels: dict | None = None,
                   diagram_cells: list[Cell] | None = None,
                   diagram_labels: dict | None = None,
                   gridlines: list | None = None) -> Image.Image:
    """Debug image: move-cell boxes (green) with predicted labels (red),
    kazan boxes (blue), pit cells (orange) with their reads, and the
    segmentation gridlines (gray).

    `diagram_labels` is keyed by (move_no, kind, side, pit_index).
    """
    vis = cv2.cvtColor(np.asarray(sheet), cv2.COLOR_GRAY2BGR)
    for p0, p1 in gridlines or []:
        cv2.line(vis, p0, p1, (160, 160, 160), 1, cv2.LINE_AA)
    for cell in cells:
        x, y, w, h = cell.bbox
        if cell.quad is not None:
            cv2.polylines(vis, [cell.quad.astype(np.int32)], True, (0, 180, 0), 1)
        else:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 0), 1)
        if labels:
            text = labels.get((cell.move_no, cell.side))
            if text:
                cv2.putText(vis, text, (x + 2, y + h - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    for cell in diagram_cells or []:
        x, y, w, h = cell.bbox
        color = (200, 120, 0) if cell.kind == "kazan" else (0, 140, 255)  # BGR
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        if diagram_labels:
            text = diagram_labels.get((cell.move_no, cell.kind, cell.side, cell.pit_index))
            if text:
                cv2.putText(vis, text, (x + 1, y + h - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
