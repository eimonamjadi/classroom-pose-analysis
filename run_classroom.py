#!/usr/bin/env python3
"""
run_classroom.py - offline classroom video analysis (single script).

Per frame:
  YOLO11 pose (conf=DET_CONF)
    -> tracker: ByteTrack (default) or Ultralytics BoT-SORT + ReID           -> raw tracker IDs
    -> IdentityManager: raw IDs -> stable anonymous IDs (SigLIP clothing embedding + floor position)
    -> homography floor projection + per-stable-ID EMA smoothing              -> 2D radar
    -> raised-hand detection from pose keypoints (must hold HAND_UP_SEC)
    -> every VLM_INTERVAL_SEC: Gemini keyframe labels (role, behavior, attention, appearance, id_mismatch)
  -> annotated video (frame on top, floor radar panel underneath), events JSON, per-frame tracks CSV.

All downstream code (labels, radar, VLM tags, state cache, JSON log) uses STABLE IDs only.

Privacy: no face recognition, names or biometric identification. Stable IDs are anonymous numbers that only
exist within one video. The appearance embedding is computed on a body crop that excludes the head (clothing
is the identity cue), and the VLM is instructed not to identify anyone.

Usage:
  python run_classroom.py --video input_videos/class.mp4 [--out output/class] [--max-seconds 30]
                          [--calib floor_calib.json] [--calibrate] [--tracker bytetrack|botsort_reid] [--no-vlm]
Gemini needs GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", message=".*template_mgs.*")  # supervision import-time deprecation noise

import cv2
import numpy as np
import supervision as sv
import torch
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

# =====================================================================================================
# TUNABLES
# =====================================================================================================
# --- detection / pose ---
POSE_MODEL = "yolo11m-pose.pt"     # Ultralytics pose weights (auto-downloaded on first run)
DET_CONF = 0.1                     # pose conf; low so the tracker's 2nd (low-score) stage receives weak boxes
DET_IMGSZ = 1280                   # inference size; >640 helps small back-row students (slower)
KP_CONF = 0.35                     # min keypoint confidence for a keypoint to count as visible

# --- tracking ---
TRACKER = "bytetrack"              # "bytetrack" (default) or "botsort_reid" (Ultralytics BoT-SORT with ReID)
TRACK_ACTIVATION_THRESHOLD = 0.6   # min box confidence to START a track; weaker boxes can only extend tracks
TRACK_HIGH_THRESH = 0.5            # split between 1st (high-score) and 2nd (low-score) association stages
TRACK_MIN_IOU = 0.2                # min IoU between a track's prediction and a box to associate them
LOST_TRACK_BUFFER_SEC = 2.0        # how long the tracker keeps a lost track before dropping it
TRACK_MIN_FRAMES = 3               # consecutive frames before a new track gets an ID (ByteTrack; filters flicker)

# --- stable identity layer ---
SIGLIP_MODEL = os.environ.get("SIGLIP_MODEL", "google/siglip-base-patch16-224")  # HF id or local dir
EMBED_CROP = "body"                # "body": person box below the head (clothing; non-facial) | "box": full box
EMBED_REFRESH_SEC = 1.0            # refresh each active ID's gallery embedding about this often
GALLERY_EMA = 0.2                  # weight of a fresh embedding in the gallery EMA (running mean until 1/EMA obs)
MIN_SIM = 0.90                     # min cosine sim (crop vs gallery) to reclaim a lost ID  (raise -> fewer wrong merges)
MAX_DIST = 1.5                     # max floor distance (m) to the lost ID's last position  (lower -> fewer wrong merges)
REMAP_WINDOW_SEC = 30.0            # lost IDs last seen within this window can be reclaimed (raise -> fewer new IDs)
OCCLUDED_FRAC = 0.3                # don't update a gallery from a crop whose box is >30% covered by another box
DRIFT_SIM = 0.80                   # refresh below this sim to its own gallery = "appearance drift": logged, not merged
MIN_CROP_PX = 16                   # don't embed crops smaller than this (either side)

# --- raised hand ---
HAND_UP_SEC = 0.5                  # wrist-above-nose AND elbow-above-shoulder must hold this long to be reported

# --- floor radar ---
RADAR_EMA_ALPHA = 0.3              # per-ID EMA on floor positions (anchor mode switches ankle/hip/box cause jumps)
HIP_TO_FLOOR = 1.0                 # hip-mode anchor: drop the hip midpoint by this x torso length (px)
RADAR_HEIGHT_FRAC = 0.5            # radar panel height as a fraction of the video frame height
RADAR_PAD_M = 0.5                  # radar shows the calibrated floor area plus this margin (m)

# --- VLM (Gemini) ---
VLM_ENABLED = True                 # also requires GEMINI_API_KEY / GOOGLE_API_KEY; --no-vlm disables
GEMINI_MODEL = "gemini-3.8-flash"
VLM_INTERVAL_SEC = 5.0             # seconds between labeled keyframes
VLM_FIRST_SEC = 1.0                # first keyframe time (lets tracks settle)
GEMINI_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"  # small ID tags need high input resolution
GEMINI_TIMEOUT_MS = 180_000        # per-request HTTP timeout
GEMINI_MAX_RETRIES = 4             # retries for retryable API errors
GEMINI_RETRY_CODES = (429, 500, 503)
GEMINI_BACKOFF_SEC = 2.0           # backoff = GEMINI_BACKOFF_SEC * 2**attempt (+ jitter)
VLM_JPEG_QUALITY = 90
SAVE_VLM_FRAMES = True             # keep the tagged JPEGs sent to Gemini (for auditing id_mismatch flags)
STATE_TTL_SEC = 3 * VLM_INTERVAL_SEC  # cached Gemini labels older than this are ignored

# =====================================================================================================
KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ank=15, r_ank=16)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def log(msg: str) -> None:
    print(msg, flush=True)


def id_color(sid: int) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR color per stable ID."""
    hue = int((sid * 0.61803398875 % 1.0) * 179)
    hsv = np.uint8([[[hue, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def put_label(img, text, org, color, scale, thickness=1, text_color=(255, 255, 255)):
    """Text with a filled background box. org = bottom-left of the text."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = int(org[0]), int(org[1])
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = max(th + 4, min(y, img.shape[0] - base))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


# =====================================================================================================
# Floor calibration (image -> floor homography, meters)
# =====================================================================================================
CALIB_HELP = ("Click 4 floor points forming a rectangle on the floor, in order: near-left, near-right, "
              "far-right, far-left (near = closest to the camera). Enter = accept, r = reset, Esc = cancel.")


def interactive_calibration(frame: np.ndarray) -> Optional[dict]:
    """Click 4 floor corners on the first frame, then type the rectangle's size in meters."""
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return None
    win = "floor calibration"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        return None  # headless OpenCV build
    pts: list[list[float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([float(x), float(y)])

    cv2.setMouseCallback(win, on_mouse)
    log(CALIB_HELP)
    names = ["near-left", "near-right", "far-right", "far-left"]
    while True:
        vis = frame.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
            put_label(vis, names[i], (p[0] + 8, p[1] - 8), (0, 0, 0), 0.6)
        if len(pts) > 1:
            cv2.polylines(vis, [np.int32(pts)], len(pts) == 4, (0, 255, 255), 2)
        put_label(vis, CALIB_HELP[:110], (10, 30), (0, 0, 0), 0.55)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(pts) == 4:
            break
        if key == ord("r"):
            pts.clear()
        if key == 27:
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    w = float(input("Width of the clicked rectangle (near-left -> near-right), meters: "))
    d = float(input("Depth of the clicked rectangle (near edge -> far edge), meters: "))
    return {"image_points": pts, "floor_points": [[0, 0], [w, 0], [w, d], [0, d]], "source": "interactive"}


def default_calibration(w: int, h: int) -> dict:
    """Rough guess used only when no calibration exists and no display is available."""
    return {
        "image_points": [[0.02 * w, 0.98 * h], [0.98 * w, 0.98 * h], [0.78 * w, 0.42 * h], [0.22 * w, 0.42 * h]],
        "floor_points": [[0, 0], [8.0, 0], [8.0, 8.0], [0, 8.0]],
        "source": "auto_default (UNCALIBRATED - edit this file or rerun with --calibrate)",
    }


def load_or_create_calibration(path: Path, first_frame: np.ndarray, force: bool) -> dict:
    if path.exists() and not force:
        calib = json.loads(path.read_text())
        log(f"[calib] loaded {path} ({calib.get('source', 'file')})")
        return calib
    calib = interactive_calibration(first_frame)
    if calib is None:
        calib = default_calibration(first_frame.shape[1], first_frame.shape[0])
        log(f"[calib] WARNING: no display for interactive calibration; using a rough default homography. "
            f"Distances (MAX_DIST) will be approximate. Edit {path} or rerun with --calibrate on a machine "
            f"with a display.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))
    log(f"[calib] saved {path}")
    return calib


def homography_from_calib(calib: dict) -> np.ndarray:
    img = np.float32(calib["image_points"])
    flr = np.float32(calib["floor_points"])
    if len(img) == 4:
        return cv2.getPerspectiveTransform(img, flr)
    H, _ = cv2.findHomography(img, flr, 0)
    return H


def radar_extent(calib: dict) -> tuple[float, float, float, float]:
    if "radar_extent_m" in calib:
        return tuple(float(v) for v in calib["radar_extent_m"])
    f = np.float32(calib["floor_points"])
    return (float(f[:, 0].min() - RADAR_PAD_M), float(f[:, 1].min() - RADAR_PAD_M),
            float(f[:, 0].max() + RADAR_PAD_M), float(f[:, 1].max() + RADAR_PAD_M))


# =====================================================================================================
# Ground anchors + floor projection
# =====================================================================================================
def estimate_ground_anchors(xyxy: np.ndarray, kp_xy: np.ndarray, kp_conf: np.ndarray):
    """Image point where each person touches the floor. Modes: ankle > hip (+torso offset) > box bottom.
    Mode switches cause jumps; positions are smoothed per stable ID downstream (FloorSmoother)."""
    anchors = np.zeros((len(xyxy), 2), dtype=np.float64)
    modes: list[str] = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        p, c = kp_xy[i], kp_conf[i]
        ank = [p[k] for k in (KP["l_ank"], KP["r_ank"]) if c[k] > KP_CONF]
        hips = [p[k] for k in (KP["l_hip"], KP["r_hip"]) if c[k] > KP_CONF]
        shs = [p[k] for k in (KP["l_sh"], KP["r_sh"]) if c[k] > KP_CONF]
        if ank:
            anchors[i] = [np.mean([a[0] for a in ank]), max(a[1] for a in ank)]
            modes.append("ankle")
        elif hips and shs:
            hip, sh = np.mean(hips, axis=0), np.mean(shs, axis=0)
            torso = max(hip[1] - sh[1], 1.0)
            anchors[i] = [hip[0], hip[1] + HIP_TO_FLOOR * torso]
            modes.append("hip")
        else:
            anchors[i] = [(x1 + x2) / 2.0, y2]
            modes.append("box")
    return anchors, modes


def to_floor(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


class FloorSmoother:
    """Per-stable-ID EMA on floor positions."""

    def __init__(self, alpha: float = RADAR_EMA_ALPHA):
        self.alpha = alpha
        self.pos: dict[int, np.ndarray] = {}

    def update(self, sid: int, xy: np.ndarray) -> np.ndarray:
        prev = self.pos.get(sid)
        cur = np.asarray(xy, dtype=np.float64) if prev is None else (1 - self.alpha) * prev + self.alpha * xy
        self.pos[sid] = cur
        return cur


# =====================================================================================================
# Radar panel
# =====================================================================================================
class FloorRadar:
    """Top-down floor map. Far side of the room (away from the camera) is at the top of the panel."""

    def __init__(self, extent: tuple[float, float, float, float], width: int, height: int):
        self.x0, self.y0, self.x1, self.y1 = extent
        self.w, self.h = width, height
        m = 28
        self.scale = min((width - 2 * m) / max(self.x1 - self.x0, 1e-3), (height - 2 * m) / max(self.y1 - self.y0, 1e-3))
        self.ox = (width - self.scale * (self.x1 - self.x0)) / 2
        self.oy = (height - self.scale * (self.y1 - self.y0)) / 2
        self.fs = max(0.4, height / 700)
        self.base = self._background()

    def to_px(self, xy) -> tuple[int, int]:
        x = min(max(float(xy[0]), self.x0), self.x1)
        y = min(max(float(xy[1]), self.y0), self.y1)
        return (int(round(self.ox + (x - self.x0) * self.scale)), int(round(self.oy + (self.y1 - y) * self.scale)))

    def _background(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 28, np.uint8)
        for gx in range(math.ceil(self.x0), math.floor(self.x1) + 1):
            cv2.line(img, self.to_px((gx, self.y0)), self.to_px((gx, self.y1)), (55, 55, 55), 1)
        for gy in range(math.ceil(self.y0), math.floor(self.y1) + 1):
            cv2.line(img, self.to_px((self.x0, gy)), self.to_px((self.x1, gy)), (55, 55, 55), 1)
        cv2.rectangle(img, self.to_px((self.x0, self.y1)), self.to_px((self.x1, self.y0)), (120, 120, 120), 1)
        put_label(img, "floor radar (1 m grid) - camera side at bottom", (8, 22), (28, 28, 28), self.fs * 0.8)
        return img

    def render(self, positions: dict[int, np.ndarray], hand_raised: dict[int, bool],
               states: dict[int, "PersonState"]) -> np.ndarray:
        """All inputs keyed by stable ID."""
        img = self.base.copy()
        r = max(5, int(self.h / 70))
        for sid, st in states.items():  # attention arrows first (underneath dots)
            if st is None or sid not in positions:
                continue
            tgt = st.attuned_to_id
            if tgt is not None and tgt != sid and tgt in positions:
                cv2.arrowedLine(img, self.to_px(positions[sid]), self.to_px(positions[tgt]), (200, 200, 90), 1,
                                cv2.LINE_AA, tipLength=0.08)
        for sid, xy in positions.items():
            c = self.to_px(xy)
            st = states.get(sid)
            color = id_color(sid)
            if st is not None and st.role == "teacher":
                cv2.rectangle(img, (c[0] - r - 2, c[1] - r - 2), (c[0] + r + 2, c[1] + r + 2), color, -1)
            else:
                cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            if hand_raised.get(sid, False):
                cv2.circle(img, c, r + 6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(sid), (c[0] + r + 3, c[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, self.fs * 0.6,
                        (235, 235, 235), 1, cv2.LINE_AA)
        legend = "o student  [] teacher  red ring = hand raised  arrow = attending to"
        put_label(img, legend, (8, self.h - 8), (28, 28, 28), self.fs * 0.7)
        return img


# =====================================================================================================
# Trackers
# =====================================================================================================
class ByteTrackAdapter:
    """Supervision-style ByteTrack. supervision>=0.31 moved ByteTrack into Roboflow `trackers`
    (trackers.ByteTrackTracker); older supervision has sv.ByteTrack. Both return a subset of the input
    sv.Detections built by indexing, so detections.data['kp_idx'] survives tracking."""

    def __init__(self, fps: float):
        buffer_30fps = int(round(LOST_TRACK_BUFFER_SEC * 30))  # both implementations count 30-FPS frames
        try:
            from trackers import ByteTrackTracker
            self.impl = ByteTrackTracker(
                lost_track_buffer=buffer_30fps, frame_rate=fps,
                track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
                high_conf_det_threshold=TRACK_HIGH_THRESH,
                minimum_iou_threshold=TRACK_MIN_IOU,
                minimum_consecutive_frames=TRACK_MIN_FRAMES)
            self.kind = "trackers.ByteTrackTracker"
        except ImportError:
            if not hasattr(sv, "ByteTrack"):
                raise ImportError("ByteTrack not found: `pip install trackers` (supervision>=0.31) "
                                  "or use supervision<=0.30")
            # sv.ByteTrack: track_activation_threshold is the high/low split; new tracks need +0.1 above it.
            self.impl = sv.ByteTrack(
                track_activation_threshold=TRACK_HIGH_THRESH, lost_track_buffer=buffer_30fps,
                minimum_matching_threshold=1.0 - TRACK_MIN_IOU, frame_rate=int(round(fps)),
                minimum_consecutive_frames=TRACK_MIN_FRAMES)
            self.kind = "supervision.ByteTrack"

    def update(self, dets: sv.Detections) -> sv.Detections:
        if self.kind.startswith("trackers"):
            out = self.impl.update(dets)  # returns every detection; untracked ones have tracker_id == -1
            return out if len(out) == 0 else out[out.tracker_id != -1]
        return self.impl.update_with_detections(dets)


def write_botsort_yaml(path: Path, fps: float) -> Path:
    """Ultralytics' installed botsort.yaml defaults with ReID enabled + our thresholds."""
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml
    cfg = dict(YAML.load(check_yaml("botsort.yaml")))
    cfg.update(
        tracker_type="botsort", with_reid=True, model="auto",  # "auto" = reuse YOLO features as ReID embedding
        gmc_method="none",                                     # fixed camera: no global motion compensation
        track_high_thresh=TRACK_HIGH_THRESH, track_low_thresh=DET_CONF,
        new_track_thresh=TRACK_ACTIVATION_THRESHOLD,
        track_buffer=int(round(LOST_TRACK_BUFFER_SEC * fps)),  # Ultralytics trackers assume 30 FPS -> use frames
        match_thresh=1.0 - TRACK_MIN_IOU,
    )
    import yaml
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def keypoints_of(res) -> tuple[np.ndarray, np.ndarray]:
    if res.keypoints is None or len(res.boxes) == 0:
        return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
    xy = res.keypoints.xy.cpu().numpy()
    conf = res.keypoints.conf
    conf = conf.cpu().numpy() if conf is not None else np.ones(xy.shape[:2], np.float32)
    return xy, conf


# =====================================================================================================
# Appearance embedding (SigLIP, as in roboflow/sports TeamClassifier)
# =====================================================================================================
class AppearanceEmbedder:
    def __init__(self, model_id: str = SIGLIP_MODEL):
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SiglipVisionModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size
        self.n_forward = 0
        self.n_crops = 0

    @torch.inference_mode()
    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """One forward pass for all crops -> (n, d) L2-normalized mean of last_hidden_state."""
        if not crops_bgr:
            return np.zeros((0, self.dim), np.float32)
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        feats = self.model(**inputs).last_hidden_state.mean(dim=1)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        self.n_forward += 1
        self.n_crops += len(crops_bgr)
        return feats.float().cpu().numpy()


def appearance_crop(frame: np.ndarray, box, kp_xy: np.ndarray, kp_conf: np.ndarray) -> Optional[np.ndarray]:
    """Person crop for the appearance embedding. EMBED_CROP='body' drops the head (starts at the shoulder line)
    so the embedding keys on clothing and contains no facial information."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if EMBED_CROP == "body":
        sh = [kp_xy[k][1] for k in (KP["l_sh"], KP["r_sh"]) if kp_conf[k] > KP_CONF]
        if sh:
            top = min(sh)
        elif kp_conf[KP["nose"]] > KP_CONF:
            top = kp_xy[KP["nose"]][1] + 0.15 * (y2 - y1)
        else:
            top = y1 + 0.3 * (y2 - y1)
        y1 = int(max(y1, min(top, y2 - MIN_CROP_PX)))
    if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def occlusion_fraction(xyxy: np.ndarray) -> np.ndarray:
    """For each box: max fraction of its area covered by any other box."""
    n = len(xyxy)
    if n < 2:
        return np.zeros(n)
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]), 1e-6, None)
    frac = inter / area[:, None]
    np.fill_diagonal(frac, 0)
    return frac.max(axis=1)


# =====================================================================================================
# Stable identity layer
# =====================================================================================================
@dataclass
class Identity:
    sid: int
    first_t: float
    last_t: float
    last_frame: int
    gallery: Optional[np.ndarray] = None
    n_gallery: int = 0
    last_embed_t: float = -1e9
    floor: Optional[np.ndarray] = None       # last smoothed floor position (m)
    raw_ids: list = field(default_factory=list)
    frames_seen: int = 0
    drifting: bool = False


class IdentityManager:
    """Maps raw tracker IDs to stable anonymous IDs.

    A new raw ID may reclaim a stable ID that is NOT active in this frame, was last seen within REMAP_WINDOW_SEC,
    has cosine similarity >= MIN_SIM between the new crop's embedding and its gallery, and whose last floor
    position is within MAX_DIST. Multiple new raw IDs in one frame are matched jointly (Hungarian, max similarity).
    Otherwise a new stable ID is created. One SigLIP forward pass per frame embeds all needed crops."""

    def __init__(self, embedder: AppearanceEmbedder):
        self.embedder = embedder
        self.raw_to_sid: dict[int, int] = {}
        self.ids: dict[int, Identity] = {}
        self.next_sid = 1
        self.all_raw_ids: set[int] = set()
        self.remaps: list[dict] = []
        self.events: list[dict] = []           # identity_new / identity_remap / appearance_drift

    def _gallery_update(self, ident: Identity, emb: np.ndarray) -> None:
        if ident.gallery is None:
            ident.gallery, ident.n_gallery = emb.copy(), 1
            return
        a = max(GALLERY_EMA, 1.0 / (ident.n_gallery + 1))  # running mean first, then EMA
        g = (1 - a) * ident.gallery + a * emb
        ident.gallery = g / (np.linalg.norm(g) + 1e-12)
        ident.n_gallery += 1

    def update(self, frame_idx: int, t: float, frame: np.ndarray, raw_ids: np.ndarray, xyxy: np.ndarray,
               kp_xy: np.ndarray, kp_conf: np.ndarray, floor_raw: np.ndarray) -> np.ndarray:
        n = len(raw_ids)
        sids = np.full(n, -1, dtype=int)
        new_idx: list[int] = []
        for i, r in enumerate(raw_ids):
            r = int(r)
            self.all_raw_ids.add(r)
            sid = self.raw_to_sid.get(r)
            if sid is None:
                new_idx.append(i)
            else:
                sids[i] = sid
        active = set(int(s) for s in sids if s >= 0)
        occl = occlusion_fraction(xyxy) if n else np.zeros(0)

        # ---- one batched SigLIP pass: new tracks immediately + ~1 Hz gallery refresh for active IDs ----
        refresh_idx = [i for i in range(n) if sids[i] >= 0 and occl[i] <= OCCLUDED_FRAC
                       and t - self.ids[sids[i]].last_embed_t >= EMBED_REFRESH_SEC]
        want = new_idx + refresh_idx
        crops, owners = [], []
        for i in want:
            c = appearance_crop(frame, xyxy[i], kp_xy[i], kp_conf[i])
            if c is not None:
                crops.append(c)
                owners.append(i)
        embs = self.embedder.embed(crops)
        emb_of = {i: embs[k] for k, i in enumerate(owners)}

        # ---- gallery refresh (skip if the crop no longer looks like this ID: possible swap) ----
        for i in refresh_idx:
            ident = self.ids[sids[i]]
            ident.last_embed_t = t
            e = emb_of.get(i)
            if e is None:
                continue
            if ident.gallery is not None and ident.n_gallery >= 3:
                sim = float(ident.gallery @ e)
                if sim < DRIFT_SIM:
                    if not ident.drifting:
                        ev = dict(type="appearance_drift", frame=frame_idx, t=round(t, 3), stable_id=ident.sid,
                                  raw_id=int(raw_ids[i]), similarity=round(sim, 4))
                        self.events.append(ev)
                        log(f"[drift] frame {frame_idx} t={t:.2f}s stable {ident.sid} (raw {int(raw_ids[i])}) "
                            f"crop sim {sim:.3f} < {DRIFT_SIM}: possible swap, gallery frozen")
                    ident.drifting = True
                    continue
            ident.drifting = False
            self._gallery_update(ident, e)

        # ---- new raw IDs: reclaim a lost stable ID or create a new one ----
        if new_idx:
            cands = [d for d in self.ids.values() if d.sid not in active and d.gallery is not None
                     and d.floor is not None and t - d.last_t <= REMAP_WINDOW_SEC]
            S = np.full((len(new_idx), len(cands)), -1.0)
            D = np.full((len(new_idx), len(cands)), np.inf)
            for a, i in enumerate(new_idx):
                e = emb_of.get(i)
                for b, d in enumerate(cands):
                    D[a, b] = float(np.linalg.norm(floor_raw[i] - d.floor))
                    if e is not None:
                        S[a, b] = float(d.gallery @ e)
            feasible = (S >= MIN_SIM) & (D <= MAX_DIST)
            matched: dict[int, int] = {}
            if feasible.any():
                rows, cols = linear_sum_assignment(np.where(feasible, 1.0 - S, 1e6))
                matched = {int(r): int(c) for r, c in zip(rows, cols) if feasible[r, c]}
            for a, i in enumerate(new_idx):
                raw = int(raw_ids[i])
                e = emb_of.get(i)
                if a in matched:
                    ident = cands[matched[a]]
                    for rr in [rr for rr, s in self.raw_to_sid.items() if s == ident.sid]:
                        del self.raw_to_sid[rr]  # retire the old raw ID; if it comes back it is re-matched
                    gap = t - ident.last_t
                    ev = dict(type="identity_remap", frame=frame_idx, t=round(t, 3), raw_id=raw, stable_id=ident.sid,
                              similarity=round(float(S[a, matched[a]]), 4),
                              distance_m=round(float(D[a, matched[a]]), 3), gap_sec=round(gap, 3))
                    self.remaps.append(ev)
                    self.events.append(ev)
                    log(f"[remap] frame {frame_idx} t={t:.2f}s raw {raw} -> stable {ident.sid} "
                        f"(sim {ev['similarity']:.3f}, dist {ev['distance_m']:.2f} m, gap {gap:.1f}s)")
                    if e is not None and occl[i] <= OCCLUDED_FRAC:
                        self._gallery_update(ident, e)
                    ident.drifting = False
                else:
                    ident = Identity(sid=self.next_sid, first_t=t, last_t=t, last_frame=frame_idx)
                    self.next_sid += 1
                    self.ids[ident.sid] = ident
                    if e is not None:
                        self._gallery_update(ident, e)
                    ev = dict(type="identity_new", frame=frame_idx, t=round(t, 3), raw_id=raw, stable_id=ident.sid)
                    if len(cands):  # best rejected candidate, for tuning MIN_SIM / MAX_DIST
                        b = int(np.argmax(np.where(D[a] <= MAX_DIST, S[a], S[a] - 10)))
                        ev.update(best_candidate=cands[b].sid, best_sim=round(float(S[a, b]), 4),
                                  best_dist_m=round(float(D[a, b]), 3))
                    self.events.append(ev)
                ident.last_embed_t = t
                ident.raw_ids.append(raw)
                self.raw_to_sid[raw] = ident.sid
                sids[i] = ident.sid
                active.add(ident.sid)

        for i in range(n):
            ident = self.ids[sids[i]]
            ident.last_t, ident.last_frame = t, frame_idx
            ident.frames_seen += 1
        return sids

    def set_floor(self, sid: int, xy: np.ndarray) -> None:
        self.ids[sid].floor = np.asarray(xy, dtype=np.float64)


# =====================================================================================================
# Raised hand
# =====================================================================================================
def hand_up_pose(xy: np.ndarray, conf: np.ndarray) -> bool:
    """(wrist above nose) AND (same-side elbow above same-side shoulder); all keypoints confident.
    Image y grows downward, so 'above' = smaller y."""
    if conf[KP["nose"]] <= KP_CONF:
        return False
    nose_y = xy[KP["nose"]][1]
    for wr, el, sh in ((KP["l_wr"], KP["l_el"], KP["l_sh"]), (KP["r_wr"], KP["r_el"], KP["r_sh"])):
        if min(conf[wr], conf[el], conf[sh]) > KP_CONF and xy[wr][1] < nose_y and xy[el][1] < xy[sh][1]:
            return True
    return False


class HandRaiseTracker:
    """Per stable ID: report a raised hand only after the pose held for HAND_UP_SEC."""

    def __init__(self):
        self.since: dict[int, float] = {}
        self.reported: dict[int, float] = {}   # sid -> time the raise started
        self.events: list[dict] = []
        self.last_seen: dict[int, float] = {}

    def update(self, sid: int, pose_up: bool, t: float, frame_idx: int) -> bool:
        self.last_seen[sid] = t
        if pose_up:
            start = self.since.setdefault(sid, t)
            if sid not in self.reported and t - start >= HAND_UP_SEC:
                self.reported[sid] = start
                self.events.append(dict(type="hand_raise_start", frame=frame_idx, t=round(t, 3), stable_id=sid,
                                        raised_since=round(start, 3)))
        else:
            self.since.pop(sid, None)
            self._end(sid, t, frame_idx)
        return sid in self.reported

    def _end(self, sid: int, t: float, frame_idx: int) -> None:
        if sid in self.reported:
            start = self.reported.pop(sid)
            self.events.append(dict(type="hand_raise_end", frame=frame_idx, t=round(t, 3), stable_id=sid,
                                    duration_sec=round(t - start, 3)))

    def close_all(self, t: float, frame_idx: int) -> None:
        for sid in list(self.reported):
            self._end(sid, t, frame_idx)
        self.since.clear()

    def expire_missing(self, present: set[int], t: float, frame_idx: int) -> None:
        for sid in list(set(self.since) | set(self.reported)):
            if sid not in present and t - self.last_seen.get(sid, t) > HAND_UP_SEC:
                self.since.pop(sid, None)
                self._end(sid, t, frame_idx)


# =====================================================================================================
# Gemini keyframe labeling
# =====================================================================================================
class Role(str, Enum):
    teacher = "teacher"
    student = "student"
    other = "other"


class Behavior(str, Enum):
    listening = "listening"            # attending to teacher/board/speaker
    writing = "writing"
    reading = "reading"
    using_device = "using_device"      # laptop/phone/tablet
    talking = "talking"                # talking with a peer
    hand_raised = "hand_raised"
    presenting = "presenting"          # lecturing / presenting / explaining at the board
    walking = "walking"
    off_task = "off_task"              # e.g. head down, looking away for long, sleeping
    other = "other"


class PersonLabel(BaseModel):
    id: int
    role: Role
    behavior: Behavior
    attuned_to_id: Optional[int] = None
    appearance: str                    # clothing color, hair, glasses, seat area (no identity)
    id_mismatch: bool


class KeyframeLabels(BaseModel):
    people: list[PersonLabel]


@dataclass
class PersonState:
    role: str
    behavior: str
    attuned_to_id: Optional[int]
    appearance: str
    id_mismatch: bool
    frame: int
    t: float


class StateCache:
    """Last Gemini label per stable ID, with timestamp; entries older than STATE_TTL_SEC are ignored."""

    def __init__(self, ttl: float = STATE_TTL_SEC):
        self.ttl = ttl
        self._d: dict[int, tuple[PersonState, float]] = {}

    def put(self, sid: int, state: PersonState, t: float) -> None:
        self._d[sid] = (state, t)

    def get(self, sid: int, t: float) -> Optional[PersonState]:
        entry = self._d.get(sid)
        if entry is None or t - entry[1] > self.ttl:
            return None
        return entry[0]

    def fresh(self, sids, t: float) -> dict[int, PersonState]:
        out = {}
        for sid in sids:
            st = self.get(sid, t)
            if st is not None:
                out[sid] = st
        return out


def head_anchor(box, kp_xy: np.ndarray, kp_conf: np.ndarray) -> tuple[float, float]:
    """Point just above the head: from the nose keypoint if confident, else top-center of the box."""
    x1, y1, x2, y2 = box
    if kp_conf[KP["nose"]] > KP_CONF:
        nx, ny = kp_xy[KP["nose"]]
        shs = [kp_xy[k][1] for k in (KP["l_sh"], KP["r_sh"]) if kp_conf[k] > KP_CONF]
        head = 1.25 * (np.mean(shs) - ny) if shs else 0.15 * (y2 - y1)  # nose -> crown, from neck length
        head = float(np.clip(head, 6, 0.4 * (y2 - y1)))
        return float(nx), float(max(y1 - 2, ny - head))
    return float((x1 + x2) / 2), float(y1)


def draw_vlm_tags(frame: np.ndarray, people: list[dict]) -> np.ndarray:
    """ID tags only (no boxes/skeletons), each just above the person's head with a short pointer."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.6, h / 1100)
    th = max(2, int(round(scale * 2)))
    placed: list[tuple[int, int, int, int]] = []

    def overlaps(r):
        return any(not (r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3]) for q in placed)

    for p in sorted(people, key=lambda q: head_anchor(q["box"], q["kp_xy"], q["kp_conf"])[1]):
        ax, ay = head_anchor(p["box"], p["kp_xy"], p["kp_conf"])
        text = str(p["sid"])
        (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        bw, bh = tw + 8, tht + base + 6
        gap = int(6 * scale)
        def rect_at(dx, dy):  # tag box whose bottom-center sits at (ax+dx, ay-gap+dy), kept inside the frame
            x0 = int(max(0, min(ax + dx - bw / 2, w - bw)))
            y0 = int(max(0, min(ay - gap + dy - bh, h - bh)))
            return (x0, y0, x0 + bw, y0 + bh)

        candidates = [rect_at(dx, dy) for dx, dy in ((0, 0), (0, -bh), (-bw, 0), (bw, 0), (0, -2 * bh))]
        rect = next((r for r in candidates if not overlaps(r)), candidates[0])
        placed.append(rect)
        cv2.line(img, ((rect[0] + rect[2]) // 2, rect[3]), (int(ax), int(ay)), (255, 255, 255), max(1, th - 1),
                 cv2.LINE_AA)
        cv2.rectangle(img, rect[:2], rect[2:], (0, 0, 0), -1)
        cv2.rectangle(img, rect[:2], rect[2:], (255, 255, 255), 1)
        cv2.putText(img, text, (rect[0] + 4, rect[3] - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), th, cv2.LINE_AA)
    return img


def build_prompt(active_ids: list[int], memory: dict[int, str]) -> str:
    notes = "\n".join(f"  #{sid}: {memory[sid]}" if sid in memory else f"  #{sid}: (no notes yet)"
                      for sid in active_ids)
    roles = ", ".join(r.value for r in Role)
    behaviors = ", ".join(b.value for b in Behavior)
    return f"""You are labeling one frame from a fixed classroom camera for an anonymous classroom-activity study.
Each tracked person has a white-on-black number tag drawn just above their head, with a short line pointing down
to the head. The number is an anonymous tracking ID. Do NOT identify anyone, guess names, age, ethnicity, or any
facial features other than whether glasses are worn.

Tagged IDs in this frame: {", ".join(map(str, active_ids))}

Stored appearance notes for these IDs from earlier frames:
{notes}

Return one entry per tagged ID that you can see (only IDs from the list above):
- id: the tag number.
- role: one of [{roles}].
- behavior: one of [{behaviors}] - what the person is doing right now.
- attuned_to_id: tag number of the person this person is visibly attending to (looking at, listening to,
  talking with), or null if none, if it is the board/screen, or if that person has no tag.
- appearance: short non-identifying description: clothing colors/type, hair color/length, glasses yes/no,
  and seat area (e.g. "row 2, left side", "standing at the front"). Max ~20 words.
- id_mismatch: true ONLY if a stored note exists for this ID AND the person under the tag now clearly differs
  from it (e.g. clearly different clothing color or type). Lighting, pose, partial occlusion or a changed
  seat/position alone are NOT mismatches. Use false when unsure or when there is no stored note."""


class GeminiLabeler:
    def __init__(self):
        from google import genai
        from google.genai import errors, types
        self.types, self.errors = types, errors
        self.client = genai.Client(http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
        self.memory: dict[int, str] = {}     # stable ID -> latest accepted appearance description
        self.calls = 0
        self.failed = 0
        self.mismatch_flags = 0
        self.ignored_ids = 0

    def _generate(self, jpeg: bytes, prompt: str) -> KeyframeLabels:
        """Retry APIError 429/500/503 with exponential backoff; raise anything else immediately."""
        types = self.types
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=KeyframeLabels,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            media_resolution=GEMINI_MEDIA_RESOLUTION,
        )
        contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt]
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
                break
            except self.errors.APIError as e:
                if e.code in GEMINI_RETRY_CODES and attempt < GEMINI_MAX_RETRIES:
                    delay = GEMINI_BACKOFF_SEC * (2 ** attempt) * (1 + 0.25 * random.random())
                    log(f"[gemini] {e.code} {e.status}; retry {attempt + 1}/{GEMINI_MAX_RETRIES} in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise
        parsed = resp.parsed
        if isinstance(parsed, KeyframeLabels):
            return parsed
        if not resp.text:
            raise ValueError("Gemini returned no text (blocked or empty response)")
        return KeyframeLabels.model_validate_json(resp.text)

    def label(self, frame: np.ndarray, people: list[dict], frame_idx: int, t: float,
              save_dir: Optional[Path]) -> tuple[list[PersonLabel], list[dict]]:
        """Returns (valid labels for active stable IDs, possible-swap events). Raises on failure."""
        active_ids = sorted(p["sid"] for p in people)
        tagged = draw_vlm_tags(frame, people)
        ok, jpg = cv2.imencode(".jpg", tagged, [cv2.IMWRITE_JPEG_QUALITY, VLM_JPEG_QUALITY])
        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"keyframe_{frame_idx:06d}.jpg"), tagged)
        self.calls += 1
        result = self._generate(jpg.tobytes(), build_prompt(active_ids, self.memory))

        active, seen = set(active_ids), set()
        labels, swaps = [], []
        for lab in result.people:
            if lab.id not in active or lab.id in seen:
                self.ignored_ids += 1
                continue
            seen.add(lab.id)
            if lab.attuned_to_id is not None and (lab.attuned_to_id not in active or lab.attuned_to_id == lab.id):
                lab.attuned_to_id = None
            prior = self.memory.get(lab.id)
            if lab.id_mismatch and prior is not None:
                self.mismatch_flags += 1
                ev = dict(type="possible_swap", source="gemini_id_mismatch", frame=frame_idx, t=round(t, 3),
                          stable_id=lab.id, stored_appearance=prior, observed_appearance=lab.appearance)
                swaps.append(ev)
                log(f"[gemini] id_mismatch on stable {lab.id} at t={t:.2f}s: stored '{prior}' vs "
                    f"observed '{lab.appearance}' (stored appearance kept)")
            else:
                lab.id_mismatch = False  # a flag without a stored note to compare against is meaningless
                self.memory[lab.id] = lab.appearance
            labels.append(lab)
        return labels, swaps


# =====================================================================================================
# Drawing
# =====================================================================================================
def draw_annotations(frame: np.ndarray, people: list[dict], states: dict[int, PersonState],
                     hand_up: dict[int, bool], header: str) -> np.ndarray:
    out = frame.copy()
    h = frame.shape[0]
    lw = max(1, round(h / 540))
    fs = max(0.4, h / 1800)
    for p in people:
        sid, (x1, y1, x2, y2) = p["sid"], (int(v) for v in p["box"])
        color = id_color(sid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)
        xy, c = p["kp_xy"], p["kp_conf"]
        for a, b in SKELETON:
            if c[a] > KP_CONF and c[b] > KP_CONF:
                cv2.line(out, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[b]), color, max(1, lw - 1),
                         cv2.LINE_AA)
        for k in range(17):
            if c[k] > KP_CONF:
                cv2.circle(out, tuple(int(v) for v in xy[k]), lw + 1, color, -1)
        label = f"#{sid}"
        st = states.get(sid)
        if st is not None:
            label += f" {st.role[:1].upper()}:{st.behavior}"
        put_label(out, label, (x1, y1 - 2), color, fs, 1, (0, 0, 0))
        if hand_up.get(sid, False):
            put_label(out, "HAND UP", (x1, y1 + int(30 * fs) + 4), (0, 0, 255), fs, 1)
    put_label(out, header, (8, int(34 * fs) + 8), (0, 0, 0), fs * 1.1, 1)
    return out


# =====================================================================================================
# Main
# =====================================================================================================
def config_snapshot() -> dict:
    return {k: v for k, v in globals().items()
            if k.isupper() and isinstance(v, (int, float, str, bool, tuple)) and k not in ("CALIB_HELP",)}


def main() -> int:
    global TRACKER, VLM_ENABLED, SIGLIP_MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="output directory (default: output/<video stem>)")
    ap.add_argument("--calib", default=None, help="floor calibration JSON (default: <out>/floor_calib.json)")
    ap.add_argument("--calibrate", action="store_true", help="force interactive floor calibration")
    ap.add_argument("--max-seconds", type=float, default=None, help="process only the first N seconds")
    ap.add_argument("--tracker", choices=["bytetrack", "botsort_reid"], default=None, help=f"default: {TRACKER}")
    ap.add_argument("--no-vlm", action="store_true", help="skip Gemini labeling")
    ap.add_argument("--siglip", default=None, help=f"SigLIP model id or local dir (default: {SIGLIP_MODEL})")
    args = ap.parse_args()
    TRACKER = args.tracker or TRACKER
    SIGLIP_MODEL = args.siglip or SIGLIP_MODEL
    if args.no_vlm:
        VLM_ENABLED = False

    video = Path(args.video)
    out_dir = Path(args.out) if args.out else Path("output") / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log(f"cannot open {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(math.ceil(args.max_seconds * fps)) if args.max_seconds else None
    log(f"[video] {video} {W}x{H} @ {fps:.2f} fps, {total} frames"
        + (f"; processing first {args.max_seconds:g}s ({max_frames} frames)" if max_frames else ""))

    ok, first = cap.read()
    if not ok:
        log("empty video")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    calib_path = Path(args.calib) if args.calib else out_dir / "floor_calib.json"
    calib = load_or_create_calibration(calib_path, first, args.calibrate)
    Hm = homography_from_calib(calib)

    from ultralytics import YOLO
    model = YOLO(POSE_MODEL)
    if TRACKER == "bytetrack":
        tracker = ByteTrackAdapter(fps)
        tracker_desc = f"bytetrack ({tracker.kind})"
    elif TRACKER == "botsort_reid":
        botsort_yaml = write_botsort_yaml(out_dir / "botsort_reid.yaml", fps)
        tracker_desc = f"botsort_reid (ultralytics, {botsort_yaml})"
    else:
        raise ValueError(f"unknown TRACKER {TRACKER!r}")
    log(f"[tracker] {tracker_desc}")

    embedder = AppearanceEmbedder(SIGLIP_MODEL)
    log(f"[identity] SigLIP {SIGLIP_MODEL} on {embedder.device}")
    idm = IdentityManager(embedder)
    smoother = FloorSmoother()
    hands = HandRaiseTracker()
    cache = StateCache()

    labeler, vlm_status = None, "disabled (--no-vlm)" if args.no_vlm else ""
    if VLM_ENABLED:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            labeler = GeminiLabeler()
            vlm_status = f"enabled ({GEMINI_MODEL})"
        else:
            vlm_status = "disabled (no GEMINI_API_KEY / GOOGLE_API_KEY in environment)"
    log(f"[vlm] {vlm_status}")
    vlm_dir = None
    if labeler is not None and SAVE_VLM_FRAMES:
        vlm_dir = out_dir / "vlm_keyframes"
        vlm_dir.mkdir(exist_ok=True)

    radar_h = int(round(H * RADAR_HEIGHT_FRAC)) // 2 * 2
    radar = FloorRadar(radar_extent(calib), W, radar_h)
    out_video = out_dir / f"{video.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + radar_h))
    tracks_csv = open(out_dir / f"{video.stem}_tracks.csv", "w", newline="")
    tw = csv.writer(tracks_csv)
    tw.writerow(["frame", "t", "stable_id", "raw_id", "x1", "y1", "x2", "y2", "conf",
                 "floor_x", "floor_y", "anchor_mode", "hand_raised"])

    keyframes: list[dict] = []
    vlm_events: list[dict] = []
    possible_swaps: list[dict] = []
    next_vlm_t = VLM_FIRST_SEC
    frame_idx, t, t0 = 0, 0.0, time.time()
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps

            # ---------------- detection + tracking (keypoints aligned 1:1 with tracks) ----------------
            if TRACKER == "bytetrack":
                res = model(frame, conf=DET_CONF, imgsz=DET_IMGSZ, verbose=False)[0]
                dets = sv.Detections.from_ultralytics(res)
                kxy_all, kconf_all = keypoints_of(res)
                dets.data["kp_idx"] = np.arange(len(dets))
                dets = tracker.update(dets)
                kp_idx = np.asarray(dets.data.get("kp_idx", np.zeros(0, int)), dtype=int)  # empty -> no data
                kp_xy, kp_conf = kxy_all[kp_idx], kconf_all[kp_idx]
            else:
                res = model.track(frame, persist=True, tracker=str(botsort_yaml), conf=DET_CONF, imgsz=DET_IMGSZ,
                                  verbose=False)[0]
                dets = sv.Detections.from_ultralytics(res)  # IDs + keypoints from the same results object
                kp_xy, kp_conf = keypoints_of(res)
                if dets.tracker_id is None:
                    keep = np.zeros(len(dets), dtype=bool)
                    dets.tracker_id = np.full(len(dets), -1, dtype=int)
                else:
                    keep = dets.tracker_id >= 0
                dets, kp_xy, kp_conf = dets[keep], kp_xy[keep], kp_conf[keep]
            raw_ids = dets.tracker_id.astype(int) if len(dets) else np.zeros(0, int)
            xyxy = dets.xyxy.astype(np.float64)

            # ---------------- floor + stable identities ----------------
            anchors, modes = estimate_ground_anchors(xyxy, kp_xy, kp_conf)
            floor_raw = to_floor(Hm, anchors)
            sids = idm.update(frame_idx, t, frame, raw_ids, xyxy, kp_xy, kp_conf, floor_raw)
            positions: dict[int, np.ndarray] = {}
            for i, sid in enumerate(sids):
                positions[int(sid)] = smoother.update(int(sid), floor_raw[i])
                idm.set_floor(int(sid), positions[int(sid)])

            # ---------------- raised hands ----------------
            hand_up = {int(sid): hands.update(int(sid), hand_up_pose(kp_xy[i], kp_conf[i]), t, frame_idx)
                       for i, sid in enumerate(sids)}
            hands.expire_missing(set(hand_up), t, frame_idx)

            people = [dict(sid=int(sids[i]), raw=int(raw_ids[i]), box=xyxy[i], kp_xy=kp_xy[i], kp_conf=kp_conf[i])
                      for i in range(len(sids))]

            # ---------------- Gemini keyframe ----------------
            if labeler is not None and t >= next_vlm_t and people:
                next_vlm_t = t + VLM_INTERVAL_SEC
                try:
                    labels, swaps = labeler.label(frame, people, frame_idx, t, vlm_dir)
                    possible_swaps.extend(swaps)
                    for lab in labels:
                        cache.put(lab.id, PersonState(lab.role.value, lab.behavior.value, lab.attuned_to_id,
                                                      lab.appearance, lab.id_mismatch, frame_idx, t), t)
                    keyframes.append(dict(frame=frame_idx, t=round(t, 3), active_ids=sorted(positions), people=[
                        dict(id=lab.id, role=lab.role.value, behavior=lab.behavior.value,
                             attuned_to_id=lab.attuned_to_id, appearance=lab.appearance,
                             id_mismatch=lab.id_mismatch, hand_raised=hand_up.get(lab.id, False),
                             floor_xy=[round(float(v), 3) for v in positions[lab.id]]) for lab in labels]))
                    log(f"[gemini] keyframe {frame_idx} t={t:.2f}s: {len(labels)}/{len(people)} labeled")
                except Exception as e:  # caller logs and continues
                    labeler.failed += 1
                    vlm_events.append(dict(type="vlm_error", frame=frame_idx, t=round(t, 3),
                                           error=f"{type(e).__name__}: {e}"[:500]))
                    log(f"[gemini] keyframe {frame_idx} failed: {type(e).__name__}: {str(e)[:200]}")

            # ---------------- render ----------------
            states = cache.fresh(positions.keys(), t)
            header = (f"t={t:6.2f}s  people={len(people)}  hands={sum(hand_up.values())}  "
                      f"stable IDs={len(idm.ids)}  raw IDs={len(idm.all_raw_ids)}  remaps={len(idm.remaps)}")
            top = draw_annotations(frame, people, states, hand_up, header)
            panel = radar.render(positions, hand_up, states)
            writer.write(np.vstack([top, panel]))

            for i, p in enumerate(people):
                fx, fy = positions[p["sid"]]
                tw.writerow([frame_idx, f"{t:.3f}", p["sid"], p["raw"], *(f"{v:.1f}" for v in p["box"]),
                             f"{float(dets.confidence[i]):.3f}" if dets.confidence is not None else "",
                             f"{fx:.3f}", f"{fy:.3f}", modes[i], int(hand_up[p["sid"]])])

            frame_idx += 1
            if frame_idx % 50 == 0:
                el = time.time() - t0
                log(f"[progress] frame {frame_idx} t={t:.1f}s  {frame_idx / el:.2f} fps  people={len(people)} "
                    f"stable={len(idm.ids)} raw={len(idm.all_raw_ids)} remaps={len(idm.remaps)}")
    finally:
        cap.release()
        writer.release()
        tracks_csv.close()
        hands.close_all(frame_idx / fps, frame_idx)

        events = sorted(idm.events + hands.events + possible_swaps + vlm_events, key=lambda e: (e["frame"], e["type"]))
        summary = dict(
            frames_processed=frame_idx, seconds_processed=round(frame_idx / fps, 3),
            raw_tracker_ids=len(idm.all_raw_ids), stable_ids=len(idm.ids), remaps=len(idm.remaps),
            appearance_drift_events=sum(e["type"] == "appearance_drift" for e in idm.events),
            hand_raise_events=sum(e["type"] == "hand_raise_start" for e in hands.events),
            gemini_status=vlm_status,
            gemini_calls=labeler.calls if labeler else 0,
            gemini_failed=labeler.failed if labeler else 0,
            gemini_id_mismatch_flags=labeler.mismatch_flags if labeler else 0,
            gemini_ignored_ids=labeler.ignored_ids if labeler else 0,
            siglip_forward_passes=embedder.n_forward, siglip_crops=embedder.n_crops,
            wall_time_sec=round(time.time() - t0, 1),
        )
        doc = dict(
            schema_version=2, video=str(video), fps=fps, frame_size=[W, H], max_seconds=args.max_seconds,
            tracker=tracker_desc, config=config_snapshot(), calibration=calib,
            keyframes=keyframes, events=events, remaps=idm.remaps, possible_swaps=possible_swaps,
            identities={str(sid): dict(first_t=round(d.first_t, 3), last_t=round(d.last_t, 3),
                                       frames_seen=d.frames_seen, raw_ids=d.raw_ids,
                                       appearance=labeler.memory.get(sid) if labeler else None)
                        for sid, d in sorted(idm.ids.items())},
            summary=summary,
        )
        events_path = out_dir / f"{video.stem}_events.json"
        events_path.write_text(json.dumps(doc, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))

        log("")
        log("=" * 64)
        log(f"SUMMARY  ({summary['seconds_processed']:.1f}s, {frame_idx} frames, {tracker_desc})")
        log(f"  total raw tracker IDs      : {summary['raw_tracker_ids']}")
        log(f"  total stable IDs           : {summary['stable_ids']}")
        log(f"  identity remaps            : {summary['remaps']}")
        log(f"  Gemini id_mismatch flags   : {summary['gemini_id_mismatch_flags']}")
        log(f"  failed Gemini calls        : {summary['gemini_failed']} / {summary['gemini_calls']} attempted"
            f"  [{vlm_status}]")
        log(f"  outputs: {out_video}, {events_path}")
        log("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
