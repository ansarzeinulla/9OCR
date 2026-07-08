"""Gradio demo for the Togyzkumalak scoresheet reader (HuggingFace Space).

Upload up to 5 scoresheet photos, optionally tag each with its known result,
and download the reconstructed PGNs. Inference runs on the exported ONNX
models (torch-free) via `togyz.pipeline.run_pipeline`.

This is a demo, not production: state is per-session, work is capped at 5
images per run, and requests are serialized through Gradio's queue so a shared
free Space degrades into a wait rather than a flurry of 429s.
"""
import sys
print("!!! APP IS STARTING !!!", file=sys.stderr)
sys.stderr.flush()

# --- MONKEY PATCH GRADIO CLIENT BUG ---
try:
    import gradio_client.utils as client_utils
    orig_get_type = client_utils.get_type
    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return orig_get_type(schema)
    client_utils.get_type = patched_get_type
    print("[app] Applied monkey-patch to gradio_client.utils.get_type", flush=True)
except Exception as e:
    print(f"[app] Failed to apply monkey-patch: {e}", flush=True)
# --------------------------------------

import os
import tempfile
import zipfile
from pathlib import Path

# Unbuffered stdout so boot progress actually shows in the Space container logs
# (otherwise a slow import/model-load looks like a silent hang).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def _log(msg):
    print(f"[app] {msg}", flush=True)


_log("importing gradio ...")
import gradio as gr

_log("importing pipeline ...")
from togyz.pipeline import load_classifier, run_pipeline

MAX_IMAGES = 5
MODEL_DIR = Path(__file__).parent / "models"
RESULT_CHOICES = ["unknown", "1-0", "0-1", "draw"]

# Load the ONNX sessions once at import - warm for the whole process lifetime.
_log(f"loading move model from {MODEL_DIR / 'best.onnx'} ...")
_MOVES = load_classifier(MODEL_DIR / "best.onnx")
_KAZAN = None
_kazan_path = MODEL_DIR / "kazan.onnx"
if _kazan_path.exists():
    _log("loading kazan model ...")
    _KAZAN = load_classifier(_kazan_path)
_log("models loaded")


def _safe_slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "").strip())
    return keep.strip("_")


def _process_one(image_path, result_choice, base_name, out_dir: Path):
    """Run the pipeline on one image; return (row, gallery_item, pgn_files)."""
    result = None if result_choice in (None, "unknown") else result_choice
    out = run_pipeline(image_path, _MOVES, _KAZAN, result=result)

    stop = out["stopped"]
    stop_txt = stop.get("reason", "")
    if "winner" in stop:
        stop_txt += f" ({stop['winner']})"
    note = " ⚠ low-res" if out["low_resolution"] else ""

    files = []
    for kind in ("beam", "raw", "legal"):
        f = out_dir / f"{base_name}_{kind}.pgn"
        f.write_text(out[f"{kind}_pgn"])
        files.append(str(f))

    row = [base_name, out["beam_plies"], stop_txt + note, out["beam_pgn"].strip()]
    caption = f"{base_name}: {out['beam_plies']} plies"
    return row, (out["annotated_image"], caption), files


def convert(round_label, *slot_values):
    """slot_values = [img1, res1, img2, res2, ...] for the 5 fixed slots."""
    images = slot_values[0::2]
    results = slot_values[1::2]
    provided = [(img, res) for img, res in zip(images, results) if img]

    if not provided:
        raise gr.Error("Please upload at least one scoresheet image.")
    if len(provided) > MAX_IMAGES:  # defensive; the UI only exposes 5 slots
        raise gr.Error(f"This demo handles at most {MAX_IMAGES} images per run.")

    out_dir = Path(tempfile.mkdtemp(prefix="togyz_"))
    round_slug = _safe_slug(round_label)
    rows, gallery, all_files = [], [], []

    for i, (img, res) in enumerate(provided, start=1):
        prefix = f"{round_slug}_table{i}" if round_slug else f"table{i}"
        try:
            row, gal, files = _process_one(img, res, prefix, out_dir)
        except Exception as exc:  # one bad image must not kill the batch
            rows.append([prefix, 0, f"error: {exc}", ""])
            continue
        rows.append(row)
        gallery.append(gal)
        all_files.extend(files)

    if not all_files:
        # every image errored - still return the table so the user sees why
        return rows, gallery, None

    zip_path = out_dir / (f"{round_slug}_pgns.zip" if round_slug else "pgns.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in all_files:
            zf.write(f, arcname=Path(f).name)

    return rows, gallery, str(zip_path)


def _busy_wrapper(*args):
    """Turn infrastructure overload into a friendly message instead of a 500."""
    try:
        return convert(*args)
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001 - surface anything else gracefully
        msg = str(exc).lower()
        if "429" in msg or "too many" in msg or "rate" in msg:
            raise gr.Error("Server busy — please retry in a moment.")
        raise gr.Error(f"Something went wrong: {exc}")


with gr.Blocks(title="Togyzkumalak Scoresheet Reader", api_open=False) as demo:
    gr.Markdown(
        "# Togyzkumalak Scoresheet Reader\n"
        "Upload up to **5** scoresheet photos, optionally tag each game's known "
        "result (improves accuracy), then **Convert** to download the PGNs.\n\n"
        "Outputs per game: `beam` (best legal reconstruction), `raw` (pure OCR), "
        "`legal` (strict replay). Free demo — a first run may wake the Space, and "
        "images are processed one at a time."
    )
    round_label = gr.Textbox(label="Round (optional)", placeholder="e.g. 3",
                             scale=1, max_lines=1)

    slots = []
    for i in range(MAX_IMAGES):
        with gr.Row():
            img = gr.Image(label=f"Table {i + 1}", type="filepath", height=150, show_api=False)
            res = gr.Dropdown(RESULT_CHOICES, value="unknown",
                              label="Result (1-0 = White won)", scale=1, show_api=False)
        slots.extend([img, res])

    convert_btn = gr.Button("Convert", variant="primary")

    results_table = gr.Dataframe(
        headers=["game", "legal plies", "stopped", "beam PGN"],
        label="Results", wrap=True, interactive=False, show_api=False
    )
    gallery = gr.Gallery(label="Annotated reconstruction", columns=2, height="auto", show_api=False)
    zip_out = gr.File(label="Download all PGNs (zip)", show_api=False)

    convert_btn.click(
        _busy_wrapper,
        inputs=[round_label, *slots],
        outputs=[results_table, gallery, zip_out],
    )

# Serialize CPU-heavy runs: callers wait in a bounded queue instead of
# overloading the shared Space (which is what triggers 429s).
demo.queue(max_size=16, default_concurrency_limit=1)

if __name__ == "__main__":
    # Bind explicitly to 0.0.0.0 and the Space's port so HF can detect the
    # running app (the default 127.0.0.1 bind can leave a Space stuck "Starting").
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860)))
    _log(f"launching gradio on 0.0.0.0:{port} ...")
    demo.launch(server_name="0.0.0.0", server_port=port)