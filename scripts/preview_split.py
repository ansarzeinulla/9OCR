"""Regenerate the split overlays for every photo in data/.

Edit the LAYOUT PROPORTIONS block at the top of togyz/sheet.py
(COL_NO_END, COL_W_END, DIAG_TOP_GAP, DIAG_ROW_H, DIAG_LEFT, DIAG_WIDTH,
KAZAN_FIRST_PIT, KAZAN_PITS, and the *_MARGIN values), then run:

    venv311/bin/python scripts/preview_split.py

For each data/<name>.jpg this writes data/<name>_split.jpg showing exactly
the boxes that would be fed to the classifiers: move cells (green), pit
cells (orange), kazans (blue), and the applied gridlines (gray).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from togyz.sheet import extract_cells, extract_diagram_cells, render_overlay


def main() -> None:
    data = Path(__file__).resolve().parent.parent / "data"
    photos = [p for p in sorted(data.glob("*.jpg")) if not p.stem.endswith("_split")]
    if not photos:
        print(f"No photos found in {data}/")
        return
    for photo in photos:
        try:
            cells, sheet, gridlines = extract_cells(photo, with_gridlines=True)
            diagrams = extract_diagram_cells(sheet)
        except Exception as exc:  # noqa: BLE001 - keep going over the batch
            print(f"{photo.name}: ERROR {exc}")
            continue
        kazans = sum(c.kind == "kazan" for c in diagrams)
        pits = sum(c.kind == "pit" for c in diagrams)
        out = photo.with_name(photo.stem + "_split.jpg")
        render_overlay(sheet, cells, diagram_cells=diagrams,
                       gridlines=gridlines).save(out, quality=90)
        print(f"{photo.name}: moves={len(cells)} kazans={kazans} pits={pits} "
              f"-> {out.name}")


if __name__ == "__main__":
    main()
