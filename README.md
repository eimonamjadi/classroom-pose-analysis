# classroom-pose-analysis

Offline analysis of fixed-camera classroom video, in a single script (`run_classroom.py`).

```
YOLO11 pose (conf=0.1)
  -> tracker: ByteTrack (default) or Ultralytics BoT-SORT + ReID         -> raw tracker IDs
  -> IdentityManager: raw IDs -> stable anonymous IDs
       (SigLIP clothing embedding + floor-position gating)
  -> homography floor projection + per-ID EMA smoothing                  -> 2D floor radar
  -> raised-hand detection from pose (must hold HAND_UP_SEC)
  -> every VLM_INTERVAL_SEC: Gemini keyframe labels
       (role, behavior, attention target, appearance, id_mismatch)
  -> annotated video (frame on top, radar underneath) + events JSON + per-frame tracks CSV
```

Everything downstream of the tracker (labels, radar, Gemini tags, state cache, JSON log) uses **stable IDs only**.

**Privacy.** No face recognition, names or biometric identification. Stable IDs are anonymous numbers that
exist only within one video. The appearance embedding uses a body crop that starts at the shoulder line (the head
is excluded), and Gemini is told not to identify anyone or describe facial features beyond "glasses yes/no".

## Setup

```bash
pip install -r requirements.txt
```

```bash
export GEMINI_API_KEY=...   # optional; without it the Gemini step is skipped and reported as disabled
```

The YOLO weights (`yolo11m-pose.pt`) and SigLIP (`google/siglip-base-patch16-224`) download on first run. CUDA is used
automatically when available. To use a local SigLIP copy, set `SIGLIP_MODEL=/path/to/dir` or pass `--siglip`.

## Usage

```bash
python run_classroom.py --video input_videos/class.mp4 --out output/class
```

```bash
python run_classroom.py --video input_videos/class.mp4 --max-seconds 30 --tracker botsort_reid
```

| flag | meaning |
|---|---|
| `--max-seconds N` | process only the first N seconds |
| `--tracker bytetrack\|botsort_reid` | override `TRACKER` |
| `--calib FILE` | floor calibration JSON (default `<out>/floor_calib.json`) |
| `--calibrate` | force interactive floor calibration |
| `--no-vlm` | skip Gemini |
| `--siglip ID_OR_DIR` | SigLIP model id or local directory |

At the end the script prints: total raw tracker IDs, total stable IDs, identity remaps, Gemini `id_mismatch`
flags, and failed Gemini calls.

### Floor calibration

On first run (or with `--calibrate`) a window opens on the first frame: click 4 floor points that form a rectangle
on the floor (near-left, near-right, far-right, far-left; "near" = closest to the camera), press Enter, then type
the rectangle's width and depth in meters. The result is saved as JSON and reused:

```json
{"image_points": [[x, y], ...4], "floor_points": [[0, 0], [W, 0], [W, D], [0, D]],
 "radar_extent_m": [xmin, ymin, xmax, ymax]}
```

`radar_extent_m` is optional (defaults to the floor rectangle plus 0.5 m). Without a display, a rough default
homography is written instead and a warning is printed. Edit it, because `MAX_DIST` is in meters.

## Outputs (in `--out`)

- `<video>_annotated.mp4`: boxes, skeletons, `#ID role:behavior` labels, `HAND UP` badges; floor radar underneath
  (dot = student, square = teacher, red ring = hand raised, arrow = attending to).
- `<video>_events.json`:
  - `summary`
  - `keyframes`: per-keyframe Gemini labels with floor position and hand state.
  - `events`: chronological `identity_new`, `identity_remap`, `appearance_drift`, `possible_swap`,
    `hand_raise_start` / `hand_raise_end`, `vlm_error`.
  - `remaps`, `possible_swaps`
  - `identities`: per stable ID, which raw IDs it absorbed, time span and last appearance note.
  - `config`: every tunable, as used for the run.
- `<video>_tracks.csv`: one row per person per frame, with columns
  `frame, t, stable_id, raw_id, bbox, conf, floor_x, floor_y, anchor_mode, hand_raised`.
- `vlm_keyframes/`: the exact tagged JPEGs sent to Gemini, for auditing `id_mismatch` flags.

## How stable IDs work

- **Gallery.** Each stable ID keeps one gallery embedding: SigLIP `last_hidden_state` mean, L2-normalized, as in
  roboflow/sports `TeamClassifier`.
  - Updated about once per second with an EMA. It uses a running mean for the first few observations.
  - Not updated from crops that are more than `OCCLUDED_FRAC` covered by another box.
  - Not updated from crops whose similarity to the ID's own gallery drops below `DRIFT_SIM`. That is logged as
    `appearance_drift` (a possible swap), and the gallery stays frozen so it can't absorb the wrong person.
- **New raw tracker IDs** are embedded immediately. All crops in a frame go through one batched forward pass.
  Each new raw ID is compared with stable IDs that are **not active** in this frame and were last seen within
  `REMAP_WINDOW_SEC`. It reclaims the most similar one that passes both gates:
  - cosine similarity `>= MIN_SIM`
  - floor distance to that ID's last position `<= MAX_DIST`

  Otherwise it gets a new stable ID. Several new raw IDs in one frame are matched jointly (Hungarian).
  Every remap is logged with frame, raw ID, stable ID, similarity, distance and gap.
- **Gemini** keeps a short appearance note per stable ID and sees the stored notes in each prompt.
  - When it reports `id_mismatch`, the event is logged as a possible swap and the stored note is **not**
    overwritten.
  - Returned IDs that are not currently active stable IDs are ignored.
  - API errors 429/500/503 are retried with exponential backoff. Other errors fail that keyframe; the run
    continues and counts it as a failed call.

## Tuning guide

All tunables are constants at the top of `run_classroom.py`. Start from the events JSON:

- `identity_new` events carry the **best rejected candidate** (`best_sim`, `best_dist_m`), so you can see
  near-misses.
- `identity_remap`, `appearance_drift` and `possible_swap` events show merges and suspected swaps.

### Too many new IDs (the same person comes back with a new number)

| symptom in `identity_new` events | change |
|---|---|
| `best_sim` just below `MIN_SIM` | lower `MIN_SIM` in steps of 0.02 (not much below ~0.86 for SigLIP-base; different people commonly score 0.75-0.88) |
| `best_dist_m` just above `MAX_DIST` | fix the floor calibration first, then raise `MAX_DIST` (1.5 -> 2.0 m) |
| gap longer than `REMAP_WINDOW_SEC` | raise `REMAP_WINDOW_SEC` (seated students are safe to keep for 60 s+) |
| no `best_*` fields (no candidate) | the lost ID was still "active" or had no gallery: raise `LOST_TRACK_BUFFER_SEC` (2 -> 3-4 s) so the tracker bridges the gap itself |

Also worth trying:
- `TRACKER = "botsort_reid"`.
- A lower `TRACK_MIN_IOU` (0.2 -> 0.1) for fast movers.
- `EMBED_CROP = "box"` if most students are hidden behind desks below the shoulders, so body crops get tiny.

### Wrong merges or swaps (two people share one ID; labels land on the wrong person)

Signs: Gemini `possible_swap` events, `appearance_drift` events, or a remap with low similarity or a large distance.

- Raise `MIN_SIM` in steps of 0.02.
- Lower `MAX_DIST` (1.0 m works when students stay seated).
- Lower `REMAP_WINDOW_SEC`.
- Raise `DRIFT_SIM` (freeze galleries sooner) and/or lower `GALLERY_EMA` (galleries adapt more slowly).
- Lower `OCCLUDED_FRAC` so crops overlapping a neighbor never update a gallery.
- For swaps between two tracks that never break (no new raw ID, so no remap can fix them):
  - raise `TRACK_ACTIVATION_THRESHOLD` and `TRACK_MIN_IOU`
  - shorten `LOST_TRACK_BUFFER_SEC`
  - try `botsort_reid`

### Recalibrating `MIN_SIM` for your room

Thresholds depend on the room, the clothing and the camera. To recalibrate:
1. Run 30-60 s with `--no-vlm`.
2. From `<video>_tracks.csv`, take crops of the same raw track at least 2 s apart, and crops of different
   tracks in the same frame.
3. Embed them and compare the two similarity distributions.
4. Put `MIN_SIM` just above the high tail of the different-person scores, and `DRIFT_SIM` just below the low
   tail of the same-person scores.

## Verification run

Results from the first 30 s of a public test clip are added below once the run completes.
