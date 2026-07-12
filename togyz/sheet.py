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

def _get_page_geometry(gray: np.ndarray):
    """Phase 2 & 3: Find the outer constellation, warp the page to an orthogonal grid,
    and calculate the Master Ruler pixel dimensions."""
    quads = _find_table_quads(gray)
    
    # Phase 2: Lock the Invisible Frame using the 4 outermost corners of the 8 blocks
    src_pts = np.array([
        quads[0][0],   # Top-Left of Block 1
        quads[3][1],   # Top-Right of Block 4
        quads[7][2],   # Bottom-Right of Block 8
        quads[4][3]    # Bottom-Left of Block 5
    ], dtype=np.float32)
    
    # Measure dimensions to maintain native scale
    w_top = np.linalg.norm(src_pts[1] - src_pts[0])
    w_bot = np.linalg.norm(src_pts[2] - src_pts[3])
    h_left = np.linalg.norm(src_pts[3] - src_pts[0])
    h_right = np.linalg.norm(src_pts[2] - src_pts[1])
    
    tw = int(round(max(w_top, w_bot) * UPSCALE))
    th = int(round(max(h_left, h_right) * UPSCALE))
    
    dst_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    
    # Single global warp to make the entire page strictly 90-degrees orthogonal!
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inverse = np.linalg.inv(matrix)
    warped = cv2.warpPerspective(gray, matrix, (tw, th), flags=cv2.INTER_CUBIC)
    
    # Map the 8 blocks into our flat space to measure internal offsets
    mapped_quads = [cv2.perspectiveTransform(np.array([q], np.float32), matrix)[0] for q in quads]
    
    # Phase 3: The Master Ruler
    block_widths = [mq[:,0].max() - mq[:,0].min() for mq in mapped_quads]
    block_heights = [mq[:,1].max() - mq[:,1].min() for mq in mapped_quads]
    
    bw = float(np.mean(block_widths))
    bh = float(np.mean(block_heights))
    
    # Heights are uniform: 11 rows per block (header + 10 moves)
    cell_h = bh / 11.0
    
    # Widths: Move column is 40% of block. Pit width is 1/3 of move column.
    move_w = bw * 0.40
    pit_w = move_w / 3.0
    
    # Column X Anchors (Averaged from top/bottom bands for perfect vertical lines)
    col_xs = [float(np.mean([mapped_quads[i][:,0].min(), mapped_quads[i+4][:,0].min()])) for i in range(4)]
    
    # Row Y Anchors
    top_y = float(np.mean([mq[:,1].min() for mq in mapped_quads[:4]]))
    bot_y = float(np.mean([mq[:,1].min() for mq in mapped_quads[4:]]))
    
    return {
        "warped": warped, "inverse": inverse,
        "tw": tw, "th": th, "bw": bw, "bh": bh,
        "cell_h": cell_h, "pit_w": pit_w,
        "col_xs": col_xs, "top_y": top_y, "bot_y": bot_y
    }

def _create_cell_from_warped(geom, cx0, cy0, cx1, cy1, move_no, side, kind="move", pit_index=0) -> Cell:
    """Helper to crop blindly from the flat image and map coordinates back to the original photo."""
    warped, inverse = geom["warped"], geom["inverse"]
    th, tw = warped.shape
    cx0, cy0 = int(max(0, cx0)), int(max(0, cy0))
    cx1, cy1 = int(min(tw, cx1)), int(min(th, cy1))
    
    crop = Image.fromarray(warped[cy0:cy1, cx0:cx1])
    corners = np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]], np.float32).reshape(-1, 1, 2)
    sheet_quad = cv2.perspectiveTransform(corners, inverse).reshape(4, 2)
    
    x_min, y_min = sheet_quad.min(axis=0)
    x_max, y_max = sheet_quad.max(axis=0)
    bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    return Cell(move_no, side, bbox, crop, sheet_quad, kind=kind, pit_index=pit_index)

UPSCALE = 2  # rectified tables are rendered at 2x for a little more pixel room

def _map_segment(inverse: np.ndarray, p0, p1) -> tuple[tuple[int, int], tuple[int, int]]:
    """Map a rectified-table segment back onto the original sheet."""
    pts = np.array([p0, p1], np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, inverse).reshape(2, 2)
    return (
        (int(mapped[0][0]), int(mapped[0][1])),
        (int(mapped[1][0]), int(mapped[1][1])),
    )


def extract_cells(image, with_gridlines: bool = False):
    if isinstance(image, Image.Image):
        pil = ImageOps.exif_transpose(image).convert("L")
    else:
        with Image.open(image) as img:
            pil = ImageOps.exif_transpose(img).convert("L")
            
    gray = np.asarray(pil)
    geom = _get_page_geometry(gray)
    
    cells: list[Cell] = []
    gridlines: list[tuple[tuple[int, int], tuple[int, int]]] = []
    
    for band in range(2):
        base_y = geom["top_y"] if band == 0 else geom["bot_y"]
        for col in range(4):
            table_idx = band * 4 + col
            base_x = geom["col_xs"][col]
            
            if with_gridlines:
                for r in range(12):
                    y = base_y + r * geom["cell_h"]
                    gridlines.append(_map_segment(geom["inverse"], (base_x, y), (base_x + geom["bw"], y)))
                for frac in COLUMN_FRACTIONS:
                    x = base_x + geom["bw"] * frac
                    gridlines.append(_map_segment(geom["inverse"], (x, base_y), (x, base_y + geom["bh"])))
            
            # Slicing 10 move rows (skipping header at r=0)
            for r in range(10):
                cy0 = base_y + (r + 1) * geom["cell_h"]
                cy1 = cy0 + geom["cell_h"]
                my = (cy1 - cy0) * CELL_MARGIN
                
                for side in ("W", "B"):
                    # White starts at 20%, Black starts at 60%
                    offset_x = geom["bw"] * (0.2 if side == "W" else 0.6)
                    cx0 = base_x + offset_x
                    cx1 = cx0 + geom["bw"] * 0.4
                    mx = (cx1 - cx0) * CELL_MARGIN
                    
                    cells.append(_create_cell_from_warped(
                        geom, cx0 - mx, cy0 - my, cx1 + mx, cy1 + my,
                        table_idx * 10 + r + 1, side
                    ))
                    
    if with_gridlines:
        return cells, pil, gridlines
    return cells, pil

PIT_MARGIN_Y = 0.25  # pit digits regularly overflow the printed row height
PIT_MARGIN_X = 0.15  # and bleed a little into neighboring columns

def extract_diagram_cells(sheet: Image.Image) -> list[Cell]:
    gray = np.asarray(sheet)
    try:
        geom = _get_page_geometry(gray)
    except Exception:
        return []  # Defensive: If page is utterly broken, just skip diagrams
        
    cells: list[Cell] = []
    
    for band in range(2):
        for col in range(4):
            checkpoint = (band * 4 + col + 1) * 10
            base_x = geom["col_xs"][col]
            
            # Find the vertical center of the diagram gap
            if band == 0:
                diag_center_y = (geom["top_y"] + geom["bh"] + geom["bot_y"]) / 2.0
            else:
                offset = (geom["top_y"] + geom["bh"] + geom["bot_y"]) / 2.0 - geom["top_y"]
                diag_center_y = geom["bot_y"] + offset
            
            # 2x9 Grid Boundaries (exactly 2 cell heights total)
            grid_y0 = diag_center_y - geom["cell_h"]
            grid_y1 = diag_center_y
            grid_y2 = diag_center_y + geom["cell_h"]
            
            # Center the 9 pits horizontally under the move block
            total_grid_w = 9 * geom["pit_w"]
            grid_start_x = base_x + (geom["bw"] - total_grid_w) / 2.0
            
            # Extract 18 Pits
            for ry0, ry1, owner in ((grid_y0, grid_y1, "B"), (grid_y1, grid_y2, "W")):
                my = (ry1 - ry0) * PIT_MARGIN_Y
                for j in range(9):
                    px0 = grid_start_x + j * geom["pit_w"]
                    px1 = px0 + geom["pit_w"]
                    mx = (px1 - px0) * PIT_MARGIN_X
                    
                    pit_index = 9 - j if owner == "B" else j + 1
                    cells.append(_create_cell_from_warped(
                        geom, px0 - mx, ry0 - my, px1 + mx, ry1 + my,
                        checkpoint, owner, kind="pit", pit_index=pit_index
                    ))
            
            # Extract Kazans (Exactly above/below the 5th and 6th pit cells)
            # 5th and 6th pits = j=4 and j=5. So start offset is 4 * pit width, width is 2 * pit width.
            kx0 = grid_start_x + 4 * geom["pit_w"]
            kx1 = kx0 + 2 * geom["pit_w"]
            
            # Black Kazan (Top) - Exactly 1 height unit UP
            cells.append(_create_cell_from_warped(
                geom, kx0, grid_y0 - geom["cell_h"], kx1, grid_y0,
                checkpoint, "B", kind="kazan"
            ))
            
            # White Kazan (Bottom) - Exactly 1 height unit DOWN
            cells.append(_create_cell_from_warped(
                geom, kx0, grid_y2, kx1, grid_y2 + geom["cell_h"],
                checkpoint, "W", kind="kazan"
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
