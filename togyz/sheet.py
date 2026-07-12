"""Locate the 8 printed move tables on a scoresheet photo and crop move cells.

Federation scoresheet layout: two bands of 4 tables (moves 1-40 top,
41-80 bottom), each table = header row + 10 move rows with columns
[No | Bast. (White) | Kost. (Black)]. Only these tables are read; the
header, summary strips, and footer are ignored.

Table boxes are found via printed grid lines (morphology + connected
components). Row/column separators inside a table are refined from detected
lines when they are complete, otherwise the known uniform structure is used
(photos are often too low-res for reliable thin-line detection).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

TABLES = 8
ROWS_PER_TABLE = 10  # plus one header row
# column boundaries as fractions of table width: No | Bast | Kost
COLUMN_FRACTIONS = (0.0, 0.20, 0.60, 1.0)
CELL_MARGIN = 0.15  # expand crops; handwriting overflows the printed cells


@dataclass
class Cell:
    move_no: int  # move cells: 1-80; diagram cells: checkpoint move (10..80)
    side: str  # "W" (Bast.) or "B" (Kost.); for pits this is the row owner
    bbox: tuple[int, int, int, int]  # x, y, w, h in original image coords
    image: Image.Image
    quad: np.ndarray | None = None  # 4x2 original-image corners (tilt-aware)
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
    """Corner quads of the 8 move tables (phone photos are tilted, so tables
    are general quadrilaterals, not axis-aligned rectangles)."""
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
    # the 8 move tables all have near-identical area; drop outliers like the
    # footer/signature block, which can out-size a genuine table
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


def _detect_lines(mask: np.ndarray, axis: int, min_frac: float) -> list[int]:
    """Positions of long line clusters along `axis` (0=rows, 1=cols)."""
    profile = mask.sum(axis=1 - axis)
    limit = 255 * mask.shape[1 - axis] * min_frac
    positions = np.where(profile > limit)[0]
    clusters: list[list[int]] = []
    for pos in positions:
        if clusters and pos - clusters[-1][-1] <= 2:
            clusters[-1].append(int(pos))
        else:
            clusters.append([int(pos)])
    return [int(np.mean(c)) for c in clusters]


UPSCALE = 2  # rectified tables are rendered at 2x for a little more pixel room


def _rectify_table(gray: np.ndarray, quad: np.ndarray):
    """Warp a (possibly tilted) table quad to a flat rectangle.

    Returns (warped gray image, inverse homography back to sheet coords).
    """
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    tw = int(round((top + bottom) / 2)) * UPSCALE
    th = int(round((left + right) / 2)) * UPSCALE
    target = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    warped = cv2.warpPerspective(gray, matrix, (tw, th), flags=cv2.INTER_CUBIC)
    return warped, np.linalg.inv(matrix)


def _row_lines(warped: np.ndarray) -> list[int]:
    """Row separators in a rectified table; uniform fallback."""
    h, w = warped.shape
    thr = cv2.adaptiveThreshold(
        warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )
    horiz = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 3, 3), 1))
    )
    rows = _detect_lines(horiz, axis=0, min_frac=0.4)
    if len(rows) != ROWS_PER_TABLE + 2:  # header + 10 rows needs 12 lines
        rows = [round(i * h / (ROWS_PER_TABLE + 1)) for i in range(ROWS_PER_TABLE + 2)]
    return rows


def _map_segment(inverse: np.ndarray, p0, p1) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map a rectified-table segment back onto the original sheet."""
    pts = np.array([p0, p1], np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, inverse).reshape(2, 2)
    return (
        (int(mapped[0][0]), int(mapped[0][1])),
        (int(mapped[1][0]), int(mapped[1][1])),
    )


def extract_cells(image, with_gridlines: bool = False):
    """All 160 move cells (80 moves x W/B) plus the upright sheet image.

    `image` is a path (str/Path) or a PIL.Image. Each table is
    perspective-rectified before being split, so cell crops stay aligned even
    on tilted phone photos.

    Returns (cells, sheet) or, with `with_gridlines`, (cells, sheet, gridlines)
    where gridlines is a list of ((x0, y0), (x1, y1)) sheet-coordinate segments
    for every row/column separator used during segmentation.
    """
    if isinstance(image, Image.Image):
        pil = ImageOps.exif_transpose(image).convert("L")
    else:
        with Image.open(image) as img:
            pil = ImageOps.exif_transpose(img).convert("L")
    gray = np.asarray(pil)

    cells: list[Cell] = []
    gridlines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for t, quad in enumerate(_find_table_quads(gray)):
        warped, inverse = _rectify_table(gray, quad)
        th, tw = warped.shape
        rows = _row_lines(warped)
        cols = [round(f * tw) for f in COLUMN_FRACTIONS]
        if with_gridlines:
            for y in rows:
                gridlines.append(_map_segment(inverse, (0, y), (tw, y)))
            for x in cols:
                gridlines.append(_map_segment(inverse, (x, rows[0]), (x, rows[-1])))
        for r in range(ROWS_PER_TABLE):
            y0, y1 = rows[r + 1], rows[r + 2]  # rows[0..1] is the header
            for col_index, side in ((1, "W"), (2, "B")):
                x0, x1 = cols[col_index], cols[col_index + 1]
                mx = round((x1 - x0) * CELL_MARGIN)
                my = round((y1 - y0) * CELL_MARGIN)
                cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
                cx1, cy1 = min(tw, x1 + mx), min(th, y1 + my)
                crop = Image.fromarray(warped[cy0:cy1, cx0:cx1])

                # map the cell corners back onto the original sheet
                corners = np.array(
                    [[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]], np.float32
                ).reshape(-1, 1, 2)
                sheet_quad = cv2.perspectiveTransform(corners, inverse).reshape(4, 2)
                x_min, y_min = sheet_quad.min(axis=0)
                x_max, y_max = sheet_quad.max(axis=0)
                bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
                cells.append(
                    Cell(t * ROWS_PER_TABLE + r + 1, side, bbox, crop, sheet_quad)
                )
    if with_gridlines:
        return cells, pil, gridlines
    return cells, pil


PIT_MARGIN_Y = 0.25  # pit digits regularly overflow the printed row height
PIT_MARGIN_X = 0.15  # and bleed a little into neighboring columns


def _strip_pit_grid(mask: np.ndarray, gray: np.ndarray,
                    strip_bbox: tuple[int, int, int, int], checkpoint: int) -> list[Cell]:
    """The 18 pit cells of the 2x9 board diagram inside one summary strip.

    Upper row = Kost. (Black), pit indexes 9..1 left-to-right; lower row =
    Bast. (White), indexes 1..9. Grid lines are detected inside the strip;
    when detection is incomplete the known uniform 2x9 structure is used.
    """
    x, y, bw, bh = strip_bbox
    h, w = gray.shape
    sub = mask[y : y + bh, x : x + bw]

    # horizontal separators: top / middle / bottom of the 2x9 grid
    hlines = _detect_lines(sub, axis=0, min_frac=0.55)
    if len(hlines) < 2:
        return []
    top, bottom = hlines[0], hlines[-1]
    if bottom - top < 6:
        return []
    mid_target = (top + bottom) / 2
    inner = [v for v in hlines if top + 2 < v < bottom - 2]
    middle = min(inner, key=lambda v: abs(v - mid_target)) if inner else int(round(mid_target))

    # vertical separators: 10 column lines across the grid band
    band = sub[top : bottom + 1]
    vlines = _detect_lines(band, axis=1, min_frac=0.5)
    if len(vlines) != 10:
        if len(vlines) >= 2:
            x0, x1 = vlines[0], vlines[-1]
        else:
            xs = np.where(band.sum(axis=0) > 0)[0]
            x0, x1 = (int(xs.min()), int(xs.max())) if len(xs) else (0, bw - 1)
        vlines = [round(x0 + j * (x1 - x0) / 9) for j in range(10)]

    cells: list[Cell] = []
    rows = ((top, middle, "B"), (middle, bottom, "W"))
    for ry0, ry1, owner in rows:
        my = round((ry1 - ry0) * PIT_MARGIN_Y)
        for j in range(9):
            cx0, cx1 = vlines[j], vlines[j + 1]
            mx = round((cx1 - cx0) * PIT_MARGIN_X)
            gx0 = max(0, x + cx0 - mx)
            gx1 = min(w, x + cx1 + mx)
            gy0 = max(0, y + ry0 - my)
            gy1 = min(h, y + ry1 + my)
            if gx1 - gx0 < 4 or gy1 - gy0 < 4:
                continue
            pit_index = 9 - j if owner == "B" else j + 1
            cells.append(Cell(
                checkpoint, owner, (gx0, gy0, gx1 - gx0, gy1 - gy0),
                Image.fromarray(gray[gy0:gy1, gx0:gx1]),
                kind="pit", pit_index=pit_index,
            ))
    return cells


def extract_diagram_cells(sheet: Image.Image) -> list[Cell]:
    """All board-diagram cells from the 8 summary strips between table bands.

    After every 10 moves the scorer fills a small board diagram: the box
    protruding ABOVE the strip is the Kost. (Black) kazan, the box BELOW is
    the Bast. (White) kazan, and the 2x9 grid between them holds the pit
    counts ('x' = tuzdyk, '-' = 0, or a 1-2 digit number). Returned as Cell
    objects with move_no = the checkpoint move (10, 20, ... 80), kind
    "kazan"/"pit", side "B"/"W" (row owner for pits) and pit_index 1-9.
    """
    gray = np.asarray(sheet)
    h, w = gray.shape
    mask = _grid_mask(gray)
    quads = _find_table_quads(gray)
    table_area = float(np.median([cv2.contourArea(q) for q in quads]))

    # Diagram strips are located by table position, not by enumeration order:
    # the strip for checkpoint 10*(4*band + col + 1) sits below table `col` of
    # band `band`, in the gap under that band. Handwriting sometimes bridges
    # neighboring strips into one connected component, so wide components are
    # kept and carved by each table's x-range.
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    components = [
        tuple(int(v) for v in stats[i][:4])
        for i in range(1, n)
        if stats[i][2] > w * 0.05
        and h * 0.02 < stats[i][3] < h * 0.12
        and stats[i][2] > 1.5 * stats[i][3]
        and stats[i][2] * stats[i][3] < 2.5 * table_area
    ]

    top_quads, bottom_quads = quads[:4], quads[4:]
    zones = (
        (top_quads, (max(q[:, 1].max() for q in top_quads),
                     min(q[:, 1].min() for q in bottom_quads))),
        (bottom_quads, (max(q[:, 1].max() for q in bottom_quads), h)),
    )

    strips: list[tuple[int, tuple[int, int, int, int]]] = []
    for band, (band_quads, (zy0, zy1)) in enumerate(zones):
        for col, quad in enumerate(band_quads):
            tx0, tx1 = float(quad[:, 0].min()), float(quad[:, 0].max())
            parts = []
            for cx, cy, cw, ch in components:
                yc = cy + ch / 2
                overlap = min(cx + cw, tx1) - max(cx, tx0)
                if zy0 < yc < zy1 and overlap > 0.3 * (tx1 - tx0):
                    parts.append((cx, cy, cw, ch))
            if not parts:
                continue
            pad = int(0.02 * w)
            sx0 = max(int(tx0) - pad, min(cx for cx, *_ in parts))
            sx1 = min(int(tx1) + pad, max(cx + cw for cx, _, cw, _ in parts))
            sy0 = min(cy for _, cy, *_ in parts)
            sy1 = max(cy + ch for _, cy, _, ch in parts)
            checkpoint = (band * 4 + col + 1) * 10
            strips.append((checkpoint, (sx0, sy0, sx1 - sx0, sy1 - sy0)))

    cells: list[Cell] = []
    for checkpoint, (x, y, bw, bh) in strips:
        cells.extend(_strip_pit_grid(mask, gray, (x, y, bw, bh), checkpoint))
        sub = mask[y : y + bh, x : x + bw]
        long_rows = np.where((sub > 0).sum(axis=1) > bw * 0.55)[0]
        if len(long_rows) == 0:
            continue
        # kazan boxes are ~1 row tall; a tight zone avoids swallowing the
        # neighboring table's header text below the strip
        zones = {
            "B": (max(0, y + int(long_rows.min()) - 30), y + int(long_rows.min()) - 1, False),
            "W": (y + int(long_rows.max()) + 3, min(h, y + int(long_rows.max()) + 31), True),
        }
        for side, (gy0, gy1, truncate_below) in zones.items():
            if gy1 - gy0 < 8:
                continue
            zone = mask[gy0:gy1, x : x + bw]
            # stop at any full-width line (a neighboring table's grid)
            full = np.where((zone > 0).sum(axis=1) > bw * 0.7)[0]
            if len(full):
                if truncate_below:
                    zone = zone[: full.min()]
                else:
                    zone = zone[full.max() + 1 :]
                    gy0 += int(full.max()) + 1
            ys, xs = np.where(zone > 0)
            if len(xs) < 15:
                continue
            bx0, bx1 = int(xs.min()), int(xs.max())
            by0, by1 = gy0 + int(ys.min()), gy0 + int(ys.max())
            if bx1 - bx0 < 10 or by1 - by0 < 8:
                continue
            cx0, cy0 = max(0, x + bx0 - 3), max(0, by0 - 2)
            cx1, cy1 = min(w, x + bx1 + 4), min(h, by1 + 3)
            cells.append(Cell(
                checkpoint, side, (cx0, cy0, cx1 - cx0, cy1 - cy0),
                Image.fromarray(gray[cy0:cy1, cx0:cx1]),
                kind="kazan",
            ))
    return cells


def clean_cell(image: Image.Image) -> tuple[Image.Image, float]:
    """Remove printed grid-line fragments from a cell crop.

    Returns the cleaned crop and the fraction of remaining ink pixels
    (handwriting). Blank cells score near zero even when grid lines cross
    the crop, so this is a classifier-independent emptiness signal.
    """
    gray = np.asarray(image)
    h, w = gray.shape
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
