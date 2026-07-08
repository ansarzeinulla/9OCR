---
title: Togyzkumalak Scoresheet Reader
emoji: ♟️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Togyzkumalak Move OCR

Classifies a photo of a single scoresheet cell into one of **163 classes**:
162 moves (`11` … `99x` — start hole 1–9, end hole 1–9, optional capture `x`)
plus `empty`. A full-sheet reader turns a whole scoresheet photo into PGN game
records, and a **Gradio demo** (`app.py`) packages it for a HuggingFace Space.

## Live demo (HuggingFace Space)

`app.py` is a Gradio app: upload up to **5** scoresheet photos, optionally tag
each game's known result, and download the reconstructed PGNs (beam / raw /
legal) as a zip. Inference runs on the exported **ONNX** models (torch-free),
so the Space needs only the light serving deps in `requirements.txt`.

```bash
python scripts/export_onnx.py     # checkpoints/*.pt -> single-file .onnx (+ .classes.json)
cp checkpoints/best.onnx checkpoints/best.classes.json models/
cp checkpoints/kazan/best.onnx models/kazan.onnx
cp checkpoints/kazan/best.classes.json models/kazan.classes.json
python app.py                     # serve locally at http://127.0.0.1:7860
```

Deploy: create a Gradio Space and push this repo to its `origin` remote (the
`.onnx` files are tracked with Git LFS via `.gitattributes`). The Space reads
the YAML front-matter above and installs `requirements.txt`. No secrets or
database are required — it is a stateless demo capped at 5 images per run, with
requests serialized through Gradio's queue so a busy free Space degrades into a
short wait rather than 429 errors.

## Why the old pipeline failed (and what this one does differently)

| Problem | Old pipeline | This pipeline |
|---|---|---|
| Glyph style | EMNIST (American digits) — but players write **European/Kazakh** digits: crossed 7, serif 1, cursive 9 | ARDIS dataset (real European handwriting) + procedural crossbars/serifs on EMNIST glyphs |
| Preprocessing | Train and inference normalized **differently** (`0.5/0.5` vs ImageNet stats) | One shared module, `togyz/preprocess.py`, imported by both |
| Framing | Small digits with wide margins in fixed 80×40 cells | Random framing from tight crops to loose cells, matching real photos |
| Data | 48 900 fixed PNGs on disk | Infinite on-the-fly synthesis, deterministic validation set |

## Layout

```
togyz/            core library
  classes.py      canonical 163-class list (indices match legacy class_mapping.json)
  preprocess.py   THE single image->tensor preprocessing (train AND inference)
  glyphs.py       per-character glyph pools + procedural style edits
  synth.py        cell synthesizer;  python -m togyz.synth --preview preview.png
  dataset.py      on-the-fly synthetic dataset + real-crop eval dataset
  model.py        resnet18 (grayscale, 163 outputs), checkpoint helpers
train.py          training CLI (auto device: cuda/mps/cpu)
eval.py           synthetic val + per-file real-crop report
predict.py        classify images; --allowed restricts to legal moves
scripts/get_glyphs.py   downloads/prepares glyph pools (ARDIS, optional EMNIST)
data/real_crops/  10 labeled real crops (labels.csv) — evaluation only
legacy/           previous experiments, kept for reference
```

## Quickstart (local)

```bash
source venv311/bin/activate         # or: pip install -r requirements.txt
python scripts/get_glyphs.py        # one-time: download+prepare ARDIS glyphs
python -m togyz.synth --preview preview.png   # eyeball synthetic vs data/real_crops
python train.py --epochs 1 --samples-per-epoch 4000 --batch-size 64   # smoke test
python eval.py --ckpt checkpoints/best.pt
python predict.py "data/real_crops/*.jpg"
```

A real training run needs a GPU — see Colab below. Defaults
(`--epochs 20 --samples-per-epoch 50000`) are a sensible full run.

## Training on Google Colab (or any GPU machine)

Open `notebooks/colab_train.ipynb` in Colab, or manually:

```bash
git clone <this-repo> && cd 9OCR        # or upload a zip of the project
pip install -r requirements.txt
python scripts/get_glyphs.py --emnist   # ingredients/ is gitignored, so also
                                        # rebuild the EMNIST pool from torchvision
python train.py --epochs 30
# download checkpoints/best.pt when done
```

`predict.py`/`eval.py` run anywhere (CPU is fine) with the downloaded
checkpoint.

## Inference

```bash
python predict.py cell.jpg --topk 3
python predict.py cell.jpg --allowed "12,34x,56"
```

At any game state at most 9 moves are legal. If an upstream game tracker
passes them via `--allowed`, probabilities are renormalized over just those
moves — a large, free accuracy boost when reading whole games.

## Reading a whole scoresheet

```bash
python read_game.py "data/2026-07-06 00.00.20.jpg" --out out/sheet1 --result 0-1
```

Pass `--result` (1-0 / 0-1 / draw, from the sheet footer) when known: the
beam's final pool is re-ranked to prefer reconstructions whose end state is
consistent with how the game actually ended.

The summary strips between the tables record both kazan counts every 10
moves (Black's box above the strip, White's below, always two digits 00-81).
A second classifier reads them (`python train.py --task kazan`, saved to
`checkpoints/kazan/best.pt`) and the beam gains likelihood when a
hypothesis' computed kazans match these written checkpoints.

Finds the 8 printed move tables, classifies every cell, and writes:

- `game.json` — per ply: all 163 class probabilities, top-5, legality info
- `raw.pgn` — pure classifier argmax per ply, even if illegal
- `legal.pgn` — strict replay, stops at the first illegal argmax
- `beam.pgn` — **best reconstruction**: beam search over all legal
  continuations per ply; each legal move is scored by blending its exact
  class probability with the probability mass of its source digit
  (the landing digit is derived from the position, so a misread second
  digit doesn't discard the right source hole). Longest fully legal
  sequence wins, ties broken by joint probability.
- `annotated.jpg`, `cells/` — visual debugging

Move annotations: `+` capture, `x` tuzdyk creation. Strip `+` to feed the
moves to the 9Q engine. Rules implementation: `togyz/rules.py`, validated
against the 9Q C++ engine fixture (`python tests/test_rules.py`).

Accuracy depends heavily on photo resolution — the CLI warns when cells are
under 35 px tall; photograph sheets at full camera resolution.

## Improving accuracy further

1. **Label more real cells** — append rows to `data/real_crops/labels.csv`.
   Even ~50 real crops make the reported real-crop accuracy meaningful; a few
   hundred would allow fine-tuning on them.
2. Add glyph styles: put white-on-black PNG masks under
   `glyph_data/<source>/<char>/` and register the source in `togyz/glyphs.py`.
3. Full-game decoding with a Togyzkumalak rules engine (choose the most
   probable *legal* move per cell) — hook already exists via `--allowed`.

## Notes

- `train_data5/` (the old pre-generated dataset) is no longer used and can be
  deleted; synthesis now happens on the fly.
- Old models/scripts live in `legacy/` (`.pkl`/`.pth` files are gitignored).
