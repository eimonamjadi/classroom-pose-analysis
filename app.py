#!/usr/bin/env python3
"""
Web app for run_classroom.py: upload a classroom video in the browser, optionally click 4 floor points to
calibrate, press Run, and get the annotated video, skeleton video, trajectory maps, CSVs and events JSON back
(also as one zip).

  python app.py                 # http://localhost:7860
  python app.py --share         # also prints a temporary public link (e.g. from Google Colab)
  python app.py --host 0.0.0.0  # on a cloud VM (a Hugging Face Space is detected automatically)

Gemini labels need GEMINI_API_KEY in the environment (a Colab / Space secret). Jobs run one at a time (one GPU);
each job's working folder is under jobs/ (CLASSROOM_JOBS_DIR; keep it on a fast local disk). Set
CLASSROOM_RESULTS_DIR (e.g. a Google Drive folder) to also copy each finished results zip there.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
JOBS = Path(os.environ.get("CLASSROOM_JOBS_DIR", ROOT / "jobs"))       # working folders (keep on a fast local disk)
RESULTS = Path(os.environ["CLASSROOM_RESULTS_DIR"]) if os.environ.get("CLASSROOM_RESULTS_DIR") else None  # zip copies
CORNERS = ["near-left", "near-right", "far-right", "far-left"]


# ------------------------------------------------------------------ calibration clicks
def first_frame(video: str | None):
    if not video:
        return None
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def draw_points(frame, pts):
    img = frame.copy()
    s = max(1.0, img.shape[0] / 720)
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(x), int(y)), int(7 * s), (255, 220, 0), -1)
        cv2.putText(img, CORNERS[i], (int(x + 10 * s), int(y - 10 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                    (255, 220, 0), max(1, int(2 * s)), cv2.LINE_AA)
    if len(pts) > 1:
        cv2.polylines(img, [np.int32(pts)], len(pts) == 4, (255, 220, 0), max(1, int(2 * s)))
    return img


def calib_status(pts):
    if len(pts) < 4:
        return f"Click the **{CORNERS[len(pts)]}** corner of a rectangle on the floor ({len(pts)}/4)."
    return "4 floor points set. Enter the rectangle's real width and depth below."


def on_video(video):
    frame = first_frame(video)
    if frame is None:
        return None, None, [], "Upload a video first."
    return frame, frame, [], calib_status([])


def on_click(frame, pts, evt: gr.SelectData):
    if frame is None:
        return None, pts, "Upload a video first."
    if len(pts) < 4:
        pts = pts + [[int(evt.index[0]), int(evt.index[1])]]
    return draw_points(frame, pts), pts, calib_status(pts)


def on_reset(frame):
    return frame, [], calib_status([])


# ------------------------------------------------------------------ running a job
def to_h264(src: Path, dst: Path) -> Path:
    """Browsers can't play OpenCV's mp4v; re-encode with ffmpeg when available."""
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "fast",
                            "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)])
        if r.returncode == 0:
            return dst
    return src


def parse_overrides(text: str) -> list[str]:
    args = []
    for item in re.split(r"[,\n;]+", text or ""):
        if "=" in item:
            args += ["--set", item.strip()]
    return args


JOB_LOCK = threading.Lock()   # one analysis at a time (one GPU); later jobs wait their turn
NOISE = ("Warning", "Loading weights", "Downloading", "warn(")


def _finish(job: Path) -> dict:
    """After the analysis: browser-playable videos, summary table, one zip with everything."""
    out = job / "outputs"
    annotated = next(out.glob("*_annotated.mp4"), None)
    skeleton = next(out.glob("*_skeletons.mp4"), None)
    result = dict(annotated=str(to_h264(annotated, job / "annotated_web.mp4")) if annotated else None,
                  skeleton=str(to_h264(skeleton, job / "skeletons_web.mp4")) if skeleton else None,
                  gallery=[str(p) for p in sorted(out.glob("*.png"))], table=None)
    summary_csv = next(out.glob("*_id_summary.csv"), None)
    if summary_csv:
        with open(summary_csv, newline="") as fh:
            rows = list(csv.reader(fh))
        keep = ["stable_id", "visible_sec", "total_distance_m", "mean_speed_mps", "moving_sec", "still_sec",
                "sec_at_home_seat", "hand_raises", "role", "remaps"]
        idx = [rows[0].index(k) for k in keep if k in rows[0]]
        result["table"] = [[rows[0][i] for i in idx]] + [[r[i] for i in idx] for r in rows[1:]]
    (job / "log.txt").write_text((job / "info.txt").read_text() + (job / "run.log").read_text())
    zip_path = job / f"{job.name}_results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out))
        z.write(job / "log.txt", "log.txt")
    result["zip"] = str(zip_path)
    if RESULTS is not None:  # e.g. Google Drive: copy only the finished zip (slow disk)
        RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, RESULTS / zip_path.name)
    (job / "result.json").write_text(json.dumps(result))
    return result


def _worker(job: Path, cmd: list[str]) -> None:
    """Runs in a background thread, so the job keeps going if the browser tab is closed."""
    status = job / "status.txt"
    with JOB_LOCK:
        status.write_text("running")
        with open(job / "run.log", "w") as logf:
            rc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, text=True).wait()
        if rc != 0:
            status.write_text(f"failed (exit code {rc})")
            return
        status.write_text("finishing")
        try:
            _finish(job)
            status.write_text("done")
        except Exception as e:  # keep the outputs folder even if packaging fails
            status.write_text(f"failed while packaging results: {e}")


def _log_text(job: Path, n_frames: int, tail: int = 40) -> str:
    info = (job / "info.txt").read_text()
    lines = []
    if (job / "run.log").exists():
        for line in (job / "run.log").read_text(errors="replace").splitlines():
            line = line.rstrip()
            if not line or any(k in line for k in NOISE):
                continue
            m = re.search(r"\[progress\] frame (\d+)", line)
            if m and n_frames:
                line += f"   ({100 * int(m.group(1)) / n_frames:.0f}%)"
            lines.append(line)
    return info + "\n".join(lines[-tail:])


def run_job(video, pts, width_m, depth_m, calib_file, max_seconds, tracker, use_gemini, overrides):
    empty = (None, None, [], None, None)
    if not video:
        yield ("Upload a video first.", *empty)
        return
    yield ("Starting... preparing the job (a large video can take a moment).", *empty)
    stem = Path(video).stem
    job = JOBS / f"{time.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^A-Za-z0-9_-]+', '_', stem)[:40]}"
    out = job / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    src = job / ("input" + Path(video).suffix.lower())
    try:
        os.link(video, src)        # keep the upload with the job (no copy when on the same disk)
    except OSError:
        shutil.copy2(video, src)

    calib = job / "floor_calib.json"
    pts = pts or []
    frame = first_frame(str(src))
    corner_like = False
    if frame is not None and len(pts) == 4:
        h, w = frame.shape[:2]
        corner_like = all(min(x, w - x) < 0.04 * w and min(y, h - y) < 0.04 * h for x, y in pts)
    if calib_file:
        shutil.copy(calib_file, calib)
        calib_note = "floor calibration: uploaded file"
    elif len(pts) == 4 and width_m and depth_m and width_m > 0 and depth_m > 0:
        calib.write_text(json.dumps({"image_points": pts,
                                     "floor_points": [[0, 0], [width_m, 0], [width_m, depth_m], [0, depth_m]],
                                     "source": "clicked in the web app"}, indent=2))
        calib_note = f"floor calibration: clicked, {width_m} m x {depth_m} m"
        if corner_like:
            calib_note += ("\nWARNING: the 4 points are the corners of the image, not of a rectangle on the floor; "
                           "floor distances will be wrong. Reset and click 4 floor points, or skip calibration.")
    elif len(pts) == 4:
        calib_note = ("floor calibration: NOT USED: 4 points were clicked but the rectangle's width / depth are empty "
                      "or 0. Using a rough default; distances are approximate.")
    elif pts:
        calib_note = f"floor calibration: NOT USED: only {len(pts)} of 4 points clicked. Using a rough default."
    else:
        calib_note = "floor calibration: none (rough default; distances will be approximate)"

    cmd = [sys.executable, str(ROOT / "run_classroom.py"), "--video", str(src), "--out", str(out),
           "--calib", str(calib), "--tracker", tracker] + parse_overrides(overrides)
    if max_seconds and max_seconds > 0:
        cmd += ["--max-seconds", str(max_seconds)]
    gemini_ready = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if not use_gemini:
        cmd.append("--no-vlm")

    cap = cv2.VideoCapture(str(src))
    fps, n_frames = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if max_seconds and max_seconds > 0:
        n_frames = min(n_frames, int(max_seconds * fps))
    gemini = ("on" if use_gemini and gemini_ready else "off" if not use_gemini
              else "requested but no GEMINI_API_KEY is set: skipped")
    (job / "info.txt").write_text(f"job: {job.name}\n{calib_note}\nGemini: {gemini}\nframes to process: {n_frames}\n"
                                  "You can close this tab: the job keeps running, and its zip appears under "
                                  "'Results from earlier runs' when it is done.\n\n")
    (job / "status.txt").write_text("queued")
    threading.Thread(target=_worker, args=(job, cmd), daemon=True).start()

    while True:
        status = (job / "status.txt").read_text().strip()
        log = _log_text(job, n_frames)
        if status == "queued":
            yield (log + "waiting for the previous job to finish...", *empty)
        elif status == "done":
            r = json.loads((job / "result.json").read_text())
            df = gr.Dataframe(value=r["table"][1:], headers=r["table"][0]) if r["table"] else None
            yield (_log_text(job, n_frames, tail=200) + "\n\nDONE. Download everything as one zip below.",
                   r["annotated"], r["skeleton"], r["gallery"], df, r["zip"])
            return
        elif status.startswith("failed"):
            yield (_log_text(job, n_frames, tail=200) + f"\n\nFAILED: {status}", *empty)
            return
        else:
            yield (log, *empty)
        time.sleep(1.0)


def past_results():
    zips = {p.name: p for p in JOBS.glob("*/*_results.zip")}
    if RESULTS is not None and RESULTS.exists():
        zips.update({p.name: p for p in RESULTS.glob("*_results.zip") if p.name not in zips})
    return [str(zips[k]) for k in sorted(zips, reverse=True)[:20]] or None


def jobs_status():
    rows = []
    for st in sorted(JOBS.glob("*/status.txt"), reverse=True)[:20]:
        s = st.read_text().strip()
        if s != "done":
            rows.append(f"- `{st.parent.name}`: {s}")
    return "Unfinished jobs:\n" + "\n".join(rows) if rows else "No jobs running."


# ------------------------------------------------------------------ UI
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Classroom pose analysis") as demo:
        gr.Markdown("# Classroom pose analysis\nUpload a fixed-camera classroom video, optionally calibrate the floor, "
                    "and run. Identities are anonymous numbers; no face recognition.")
        frame_state = gr.State(None)
        pts_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.File(label="1. Upload video", type="filepath",
                                file_types=[".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".MP4", ".MOV"])
                with gr.Accordion("Options", open=False):
                    max_seconds = gr.Number(label="Process only the first N seconds (0 = whole video)", value=0)
                    tracker = gr.Radio(["bytetrack", "botsort_reid"], value="bytetrack", label="Tracker")
                    use_gemini = gr.Checkbox(label="Gemini behavior labels (needs GEMINI_API_KEY)", value=True)
                    overrides = gr.Textbox(label="Advanced overrides (NAME=VALUE, comma separated)",
                                           placeholder="MIN_SIM=0.88, SEAT_MIN_SIM=0.95")
                run = gr.Button("3. Run analysis", variant="primary")
                log = gr.Textbox(label="Progress", lines=14, max_lines=14, autoscroll=True,
                                 placeholder="Progress appears here after you press Run.")
            with gr.Column(scale=1):
                gr.Markdown("### 2. Floor calibration (optional, once per camera)")
                calib_md = gr.Markdown("Upload a video first.")
                calib_img = gr.Image(label="Click 4 floor corners: near-left, near-right, far-right, far-left",
                                     type="numpy", interactive=False)
                with gr.Row():
                    width_m = gr.Number(label="Rectangle width (m)", value=None)
                    depth_m = gr.Number(label="Rectangle depth (m)", value=None)
                    reset = gr.Button("Reset points")
                calib_file = gr.File(label="...or upload an existing floor_calib.json", file_types=[".json"],
                                     type="filepath")
        with gr.Row():
            annotated = gr.Video(label="Annotated video (with floor radar)")
            skeleton = gr.Video(label="Skeleton video")
        gallery = gr.Gallery(label="Trajectory maps and heatmap", columns=3, height="auto")
        table = gr.Dataframe(label="Per-ID summary (full table in the zip)")
        results = gr.File(label="Download all results (zip)")
        with gr.Accordion("Results from earlier runs (also for jobs started before you closed the tab)", open=False):
            job_md = gr.Markdown()
            past = gr.File(label="Finished result zips", file_count="multiple")
            refresh = gr.Button("Refresh list")

        video.change(on_video, video, [calib_img, frame_state, pts_state, calib_md])
        calib_img.select(on_click, [frame_state, pts_state], [calib_img, pts_state, calib_md])
        reset.click(on_reset, frame_state, [calib_img, pts_state, calib_md])
        run.click(run_job, [video, pts_state, width_m, depth_m, calib_file, max_seconds, tracker, use_gemini,
                            overrides], [log, annotated, skeleton, gallery, table, results])
        refresh.click(past_results, None, past).then(jobs_status, None, job_md)
        demo.load(past_results, None, past).then(jobs_status, None, job_md)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link (Colab)")
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME",
                                                     "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    ap.add_argument("--password", default=os.environ.get("APP_PASSWORD"),
                    help="require login (user 'teacher' and this password); or set APP_PASSWORD")
    a = ap.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    build_ui().queue(default_concurrency_limit=1).launch(
        share=a.share, server_name=a.host, server_port=a.port, allowed_paths=[str(JOBS)],
        auth=("teacher", a.password) if a.password else None, show_error=True)
