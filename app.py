"""Gradio demo for the Togyzkumalak scoresheet reader (HuggingFace Space).

Upload up to 5 scoresheet photos of one round as a single batch, describe the
tournament and the games in two CSV text fields, and download the
reconstructed PGNs (with proper PGN tags). Inference runs on the exported
ONNX models (torch-free) via `togyz.pipeline.run_pipeline`.

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

import csv
import io
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
BEAM_CHOICES = [str(2**k) for k in range(10, 31)]  # 1024 ... 2^30
# A full ~160-ply game costs roughly (width/1024) x 12s of beam search, so the
# default stays modest; the board-diagram evidence prunes well at this width.
DEFAULT_BEAM = "2048"

# accepted spellings of a game result -> pipeline result code
RESULT_MAP = {
    "": None, "*": None,
    "1": "1-0", "1-0": "1-0",
    "0": "0-1", "0-1": "0-1",
    "5": "draw", "0.5-0.5": "draw", "1/2-1/2": "draw", "draw": "draw",
}
RESULT_TAGS = {"1-0": "1-0", "0-1": "0-1", "draw": "1/2-1/2", None: "*"}

# Load the ONNX sessions once at import - warm for the whole process lifetime.
_log(f"loading move model from {MODEL_DIR / 'best.onnx'} ...")
_MOVES = load_classifier(MODEL_DIR / "best.onnx")
_DIAGRAM = None
_diagram_path = MODEL_DIR / "diagram.onnx"
if _diagram_path.exists():
    _log("loading diagram model ...")
    _DIAGRAM = load_classifier(_diagram_path)
else:
    # the old kazan.onnx has incompatible classes - do not fall back to it
    _log("no models/diagram.onnx - checkpoint evidence disabled")
_log("models loaded")


def _safe_slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "").strip())
    return keep.strip("_")


def _csv_fields(line: str) -> list[str]:
    """One CSV line -> stripped fields (handles quoted commas)."""
    rows = list(csv.reader(io.StringIO(line)))
    return [f.strip() for f in rows[0]] if rows else []


def _parse_meta(text: str) -> dict:
    """Shared metadata line: Tournament,Location,Date,Round (all optional)."""
    fields = _csv_fields((text or "").strip())
    if len(fields) > 4:
        raise gr.Error(
            "Metadata must be one CSV line: Tournament,Location,Date,Round "
            f"(got {len(fields)} fields). Quote fields that contain commas."
        )
    fields += [""] * (4 - len(fields))
    meta = {"event": fields[0], "site": fields[1], "date": fields[2],
            "round": fields[3]}
    if meta["round"]:
        try:
            rnd = int(meta["round"])
        except ValueError:
            raise gr.Error(f"Round must be a number 1-20, got {meta['round']!r}.")
        if not 1 <= rnd <= 20:
            raise gr.Error(f"Round must be between 1 and 20, got {rnd}.")
        meta["round"] = str(rnd)
    return meta


def _parse_games(text: str, n_images: int) -> list[dict]:
    """Per-game lines: WhiteName,BlackName,Result,WhiteTime,BlackTime.

    One line per uploaded image, in order. Fewer lines than images is fine
    (missing games get empty metadata); more lines is an error.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) > n_images:
        raise gr.Error(
            f"{len(lines)} game lines for {n_images} image(s). "
            "Provide at most one line per uploaded image, in order."
        )
    games = []
    for lineno, line in enumerate(lines, start=1):
        fields = _csv_fields(line)
        if len(fields) > 5:
            raise gr.Error(
                f"Game line {lineno}: expected at most 5 CSV fields "
                "(White,Black,Result,WhiteTime,BlackTime), got "
                f"{len(fields)}. Quote fields that contain commas."
            )
        fields += [""] * (5 - len(fields))
        raw_result = fields[2]
        if raw_result not in RESULT_MAP:
            raise gr.Error(
                f"Game line {lineno}: unknown result {raw_result!r}. Accepted: "
                "1 or 1-0 (White won), 0 or 0-1 (Black won), "
                "5 / 0.5-0.5 / 1/2-1/2 (draw), or empty."
            )
        games.append({"white": fields[0], "black": fields[1],
                      "result": RESULT_MAP[raw_result],
                      "white_time": fields[3], "black_time": fields[4]})
    games += [{"white": "", "black": "", "result": None,
               "white_time": "", "black_time": ""}] * (n_images - len(games))
    return games


def _pgn_tags(meta: dict, game: dict) -> str:
    """Standard PGN tag section from the shared + per-game metadata."""
    tags = [
        ("Event", meta["event"] or "?"),
        ("Site", meta["site"] or "?"),
        ("Date", meta["date"] or "?"),
        ("Round", meta["round"] or "?"),
        ("White", game["white"] or "?"),
        ("Black", game["black"] or "?"),
        ("Result", RESULT_TAGS[game["result"]]),
    ]
    if game["white_time"]:
        tags.append(("WhiteClock", game["white_time"]))
    if game["black_time"]:
        tags.append(("BlackClock", game["black_time"]))
    return "".join(f'[{k} "{v}"]\n' for k, v in tags) + "\n"


def _process_one(image_path, game: dict, meta: dict, beam_width: int,
                 base_name: str, out_dir: Path, progress_cb=None):
    """Run the pipeline on one image; return (row, gallery_item, files, warnings)."""
    out = run_pipeline(image_path, _MOVES, _DIAGRAM,
                       result=game["result"], beam_width=beam_width,
                       progress_cb=progress_cb)

    stop = out["stopped"]
    stop_txt = stop.get("reason", "")
    if "winner" in stop:
        stop_txt += f" ({stop['winner']})"
    note = " ⚠ low-res" if out["low_resolution"] else ""

    tags = _pgn_tags(meta, game)
    files = []
    for kind in ("beam", "raw", "legal"):
        f = out_dir / f"{base_name}_{kind}.pgn"
        f.write_text(tags + out[f"{kind}_pgn"])
        files.append(str(f))

    row = [base_name, out["beam_plies"], stop_txt + note, out["beam_pgn"].strip()]
    caption = f"{base_name}: {out['beam_plies']} plies"
    warnings = [f"{base_name}: {w}" for w in out["warnings"]]
    return row, (out["annotated_image"], caption), files, warnings


def _as_paths(files) -> list[str]:
    """Normalize the multi-file uploader value into a list of file paths."""
    if not files:
        return []
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    paths = []
    for f in files:
        # gr.File yields str paths (type="filepath") or objects with .name
        paths.append(f if isinstance(f, str) else getattr(f, "name", str(f)))
    return paths


def convert(meta_text, games_text, beam_choice, files, progress=gr.Progress()):
    """One batch: up to 5 images sharing tournament metadata.

    A generator: it yields (table, gallery, zip, warnings) after each image so
    results stream in one by one; the zip download is assembled only at the
    end and contains every game's PGNs appended together.
    """
    images = _as_paths(files)
    if not images:
        raise gr.Error("Please upload at least one scoresheet image.")
    if len(images) > MAX_IMAGES:
        raise gr.Error(f"This demo handles at most {MAX_IMAGES} images per run "
                       f"(got {len(images)}).")

    # parse everything up front - all input errors surface before any heavy work
    meta = _parse_meta(meta_text)
    games = _parse_games(games_text, len(images))
    try:
        beam_width = int(beam_choice)
    except (TypeError, ValueError):
        beam_width = int(DEFAULT_BEAM)

    out_dir = Path(tempfile.mkdtemp(prefix="togyz_"))
    round_slug = _safe_slug(meta["round"])
    n = len(images)
    rows, gallery, all_files, all_warnings = [], [], [], []

    def warn_md():
        return "\n".join(f"⚠ {w}" for w in dict.fromkeys(all_warnings))

    for i, (img, game) in enumerate(zip(images, games)):
        name = Path(img).name
        prefix = f"round{round_slug}_game{i + 1}" if round_slug else f"game{i + 1}"

        def cb(frac, desc, i=i, name=name):
            # blend per-image progress into an overall 0..1 bar
            progress((i + frac) / n, desc=f"Image {i + 1}/{n} ({name}): {desc}")

        cb(0.0, "starting")
        try:
            row, gal, files, warns = _process_one(
                img, game, meta, beam_width, prefix, out_dir, progress_cb=cb
            )
        except Exception as exc:  # one bad image must not kill the batch
            rows.append([f"{prefix} ({name})", 0, f"error: {exc}", ""])
            yield rows[:], gallery[:], None, warn_md()
            continue
        row[0] = f"{row[0]} ({name})"
        rows.append(row)
        gallery.append(gal)
        all_files.extend(files)
        all_warnings.extend(warns)
        # stream this image's result immediately; zip is built only at the end
        yield rows[:], gallery[:], None, warn_md()

    if not all_files:
        # every image errored - still return the table so the user sees why
        yield rows[:], gallery[:], None, warn_md()
        return

    zip_path = out_dir / (f"round{round_slug}_pgns.zip" if round_slug else "pgns.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in all_files:
            zf.write(f, arcname=Path(f).name)

    yield rows[:], gallery[:], str(zip_path), warn_md()


def _busy_wrapper(meta_text, games_text, beam_choice, files, progress=gr.Progress()):
    """Turn infrastructure overload into a friendly message instead of a 500."""
    try:
        yield from convert(meta_text, games_text, beam_choice, files, progress)
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001 - surface anything else gracefully
        msg = str(exc).lower()
        if "429" in msg or "too many" in msg or "rate" in msg:
            raise gr.Error("Server busy — please retry in a moment.")
        raise gr.Error(f"Something went wrong: {exc}")


with gr.Blocks(title="Togyzkumalak Scoresheet Reader") as demo:
    gr.Markdown(
        "# Togyzkumalak Scoresheet Reader\n"
        "Upload up to **5** scoresheet photos of one round in a single batch, "
        "describe the round and the games in the two text fields, then "
        "**Convert**. Results stream in image by image with a live progress "
        "bar; the combined PGN download appears once every image is done.\n\n"
        "Outputs per game: `beam` (best legal reconstruction), `raw` (pure OCR), "
        "`legal` (strict replay). Free demo — a first run may wake the Space, and "
        "images are processed one at a time."
    )
    meta_box = gr.Textbox(
        label="Tournament metadata (CSV): Tournament,Location,Date,Round — all optional",
        placeholder="World Championship among boys, Astana city, 2026 7 July, 7",
        max_lines=1,
    )
    games_box = gr.Textbox(
        label="Games (one CSV line per image, in order): "
              "WhiteName,BlackName,Result,WhiteTime,BlackTime — all optional; "
              "Result: 1 / 1-0, 0 / 0-1, 5 / 0.5-0.5 / 1/2-1/2",
        placeholder="Zhanabay Korkem, Kubzhasar Marhabat, 0-1, 0:07:55, 0:24:05",
        lines=MAX_IMAGES,
    )
    beam_dd = gr.Dropdown(
        BEAM_CHOICES, value=DEFAULT_BEAM,
        label="Beam width (game hypotheses kept; larger = slower but more "
              "thorough; very large values are trimmed to fit memory)",
    )

    image_files = gr.File(
        label=f"Scoresheet photos (up to {MAX_IMAGES}, in the same order as the "
              "game lines above)",
        file_count="multiple",
        file_types=["image"],
        type="filepath",
    )

    convert_btn = gr.Button("Convert", variant="primary")

    results_table = gr.Dataframe(
        headers=["game", "legal plies", "stopped", "beam PGN"],
        label="Results (stream in as each image finishes)",
        wrap=True, interactive=False,
    )
    warnings_md = gr.Markdown()
    gallery = gr.Gallery(label="Annotated reconstruction", columns=2, height="auto")
    zip_out = gr.File(label="Download all PGNs (zip)")

    convert_btn.click(
        _busy_wrapper,
        inputs=[meta_box, games_box, beam_dd, image_files],
        outputs=[results_table, gallery, zip_out, warnings_md],
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