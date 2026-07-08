# Togyzkumalak Scoresheet Reader — Gradio demo on a HuggingFace Space

## Context

The OCR pipeline reads a scoresheet photo into PGN records (sheet 1 = 36/39
exact plies). We want a public, shareable **demo** — not production — where
anyone with the link uploads a few scoresheet photos and downloads the PGNs.

Hosting is a single **HuggingFace Space running Gradio**. A free Space has
ample RAM (≈16 GB), so no database and no external services are needed —
outputs are files downloaded in-session. Constraints: **≤5 images per run**
and **graceful handling of overload (429)** on shared free compute.

The blocker on smaller hosts was PyTorch (528 MiB peak RAM, 441 MB on disk).
The fix — kept here because it also makes the Space faster and lighter — is to
run inference under **onnxruntime + numpy**, dropping torch from the serving
path (torch stays only for training/export).

## Architecture

```
HuggingFace Space (Gradio SDK)
  app.py             UI + handler; loads 2 ONNX sessions once at import
  togyz/pipeline.py  run_pipeline() — torch-free core, shared with the CLI
  togyz/{sheet,rules,preprocess,classes}.py   already torch-free
  models/*.onnx      committed via Git LFS (+ .classes.json sidecars)
  requirements.txt   serving deps (torch-free); requirements-train.txt for training
```

## Key decisions

- **onnxruntime + numpy inference.** `togyz/preprocess.py` returns numpy;
  `togyz/pipeline.py` reimplements cell classification, legal-move scoring,
  kazan evidence, and beam search with numpy. Verified **byte-identical**
  `beam.pgn` vs the torch baseline on both sample sheets. Peak RAM 394 MiB
  (was 528), wall time 8.8 s (was 12.9).
- **Single-file ONNX.** `scripts/export_onnx.py` uses the legacy exporter
  (`dynamo=False`) + `onnx.save_model(save_as_external_data=False)` so each
  model is one self-contained ~45 MB `.onnx` (no `.data` sidecar) — clean for
  Git LFS. Class list saved as a `<stem>.classes.json` sidecar.
- **Fixed 5 slots** (image + result dropdown each) structurally enforce the
  5-image cap and give unambiguous per-game result tagging. Result improves
  accuracy via the beam's end-state re-ranking; `unknown` → no hint.
- **Graceful overload.** `demo.queue(default_concurrency_limit=1, max_size=16)`
  serializes CPU-heavy runs so callers wait rather than triggering 429s; a
  `_busy_wrapper` maps any 429/rate error to a friendly `gr.Error`.
- **Per-image isolation.** Each image runs in `try/except`; a failure reports
  an error row and the batch continues.

## Outputs

Per game: `beam` (best legal reconstruction), `raw` (pure OCR argmax), `legal`
(strict replay) PGNs, plus an annotated preview image. All PGNs are bundled
into a downloadable zip; filenames use the optional round label
(`R3_table1_beam.pgn`).

## Verification (all passed)

1. ONNX parity: `beam.pgn` identical to torch on both sheets.
2. RAM 394 MiB / 8.8 s per image via the ONNX CLI.
3. `app.convert()` headless on both samples → 39 + 77 plies, zip with 6 PGNs,
   beam matches the CLI.
4. Resilience: a table-less image reports an error row while the good image
   still produces output.

## Out of scope (demo)

Batches >5, persistence/history, auth, tournament round/table auto-detection.
Larger production hosting (Render/Supabase/Vercel) was considered and dropped
in favor of this single-Space demo.
