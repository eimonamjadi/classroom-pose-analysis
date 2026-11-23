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
| `--set NAME=VALUE` | override any tunable constant for this run, e.g. `--set MIN_SIM=0.88 --set SKELETON_BG=faded` (repeatable; constants derived from others, like `STATE_TTL_SEC`, are not recomputed) |

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
  (dot = student, square = teacher, red ring = hand raised, arrow = attending to, thin line = last
  `TRAIL_SEC` of movement).
- `<video>_skeletons.mp4`: skeletons only, with ID labels and fading movement trails, on black (no faces or
  background; `SKELETON_BG = "faded"` shows a dimmed frame behind them).
- `<video>_events.json`:
  - `summary`
  - `keyframes`: per-keyframe Gemini labels with floor position and hand state.
  - `events`: chronological `identity_new`, `identity_remap`, `appearance_drift`, `possible_swap`,
    `hand_raise_start` / `hand_raise_end`, `vlm_error`.
  - `remaps`, `possible_swaps`
  - `identities`: per stable ID, which raw IDs it absorbed, time span, home seat and last appearance note.
  - `config`: every tunable, as used for the run.
- `<video>_tracks.csv`: one row per person per frame (final stable IDs; late merges already applied):

  | columns | meaning |
  |---|---|
  | `frame, t, stable_id, raw_id` | time and identity |
  | `x1, y1, x2, y2, conf` | person box (pixels) and detector confidence |
  | `floor_x, floor_y` | smoothed floor position (m); `floor_x_raw, floor_y_raw` unsmoothed |
  | `anchor_x, anchor_y, anchor_mode` | image point used as the ground contact, and how it was estimated (`ankle` / `hip` / `box`) |
  | `speed_mps, heading_deg, dist_cum_m` | speed over the last second, direction of travel (degrees, 0 = +x), distance travelled so far |
  | `still, dist_home_seat_m` | standing/sitting still now (0/1); distance to this ID's home seat |
  | `hand_raised` | raised hand (after the `HAND_UP_SEC` hold) |
  | `occluded_frac, truncated, provisional` | box covered by another detection; cut off by the frame edge; ID still provisional |
  | `role, behavior, attuned_to_id, appearance` | latest Gemini labels if fresher than `STATE_TTL_SEC` |
  | `nose_x, nose_y, nose_conf, ... r_ankle_conf` | all 17 COCO keypoints (pixels + confidence) |

  Distances and speeds are in floor meters, so they are only as good as the floor calibration.
- `<video>_id_summary.csv`: one row per stable ID:
  - time on screen: `first_t`, `last_t`, `visible_sec`, `frames`;
  - identity: absorbed `raw_tracker_ids`, `remaps`;
  - movement: `total_distance_m`, `mean_speed_mps`, `p95_speed_mps`, `moving_sec`, `still_sec`, mean position;
  - seat: home seat and `sec_at_home_seat`;
  - hands: `hand_raises`, `hand_raised_sec`;
  - Gemini: most common `role`, `top_attention_id`, `appearance`, and seconds spent in each behavior
    (`behavior_sec_*`).
- `<video>_trajectories_floor.png`: every ID's path on the floor plan, with start (o), end (filled) and home seat
  ([]).
- `<video>_trajectories_camera.png`: the same paths drawn over the first frame, in camera view.
- `<video>_heatmap_floor.png`: where people spent their time on the floor.
- `vlm_keyframes/`: the exact tagged JPEGs sent to Gemini, for auditing `id_mismatch` flags.

## Running in the cloud

The pipeline runs anywhere Python and (ideally) an NVIDIA GPU are available. On a GPU it is many times faster
than on CPU.

- **Google Colab (easiest):** open `cloud/run_on_colab.ipynb` in Colab and pick a GPU runtime. Add the secrets
  `GEMINI_API_KEY` (and `GITHUB_TOKEN` while this repo is private), point it at a Google Drive folder of videos,
  and run all cells. Outputs go back to Drive. Colab can't open the click-to-calibrate window, so the notebook
  has a cell to enter the 4 floor points instead.
- **Any cloud GPU VM** (Google Cloud, AWS, Azure, Lambda, RunPod, ...) with a recent NVIDIA driver:

  ```bash
  git clone https://github.com/eimonamjadi/classroom-pose-analysis && cd classroom-pose-analysis
  pip install -r requirements.txt
  export GEMINI_API_KEY=...
  python run_classroom.py --video class.mp4 --out output/class --calib floor_calib.json
  ```

  Calibrate once on a machine with a display (`--calibrate`), or write `floor_calib.json` by hand, then copy it to
  the VM.

The script uses CUDA automatically when available, for both YOLO and SigLIP.

## How stable IDs work

- **Gallery.** Each stable ID keeps one gallery embedding: SigLIP `last_hidden_state` mean, L2-normalized, as in
  roboflow/sports `TeamClassifier`.
  - Updated about once per second with an EMA. It uses a running mean for the first few observations.
  - Not updated from crops that are more than `OCCLUDED_FRAC` covered by another box.
  - Not updated from crops whose similarity to the ID's own gallery drops below `DRIFT_SIM`. That is logged as
    `appearance_drift` (a possible swap), and the gallery stays frozen so it can't absorb the wrong person.
- **New raw tracker IDs** are embedded immediately. All crops in a frame go through one batched forward pass.
  Each new raw ID is compared with stable IDs that are **not visible now** and **vanished before this person
  appeared**, so two people who were ever on screen together are never merged. It reclaims the most similar one
  that passes one of these rules:

  | rule | missing for | similarity | position |
  |---|---|---|---|
  | `short` | `<= REMAP_WINDOW_SEC` (30 s) | `>= MIN_SIM` (0.90) | within `MAX_DIST` of where it was last seen |
  | `long` | `<= LONG_ABSENCE_SEC` (10 min) | `>= LONG_MIN_SIM` (0.93) | appears within `MAX_DIST` of where it vanished (same door or frame edge) |

  Otherwise it gets a new stable ID. Several new raw IDs in one frame are matched jointly (Hungarian).
  Every remap is logged with its rule, frame, raw ID, stable ID, similarity, distance and how long it was missing.
- **Home seats.** Each ID records where it stands or sits still for at least `SEAT_DWELL_SEC` (within
  `SEAT_RADIUS`). The place held longest is its **home seat**.
  - `seat` rule: a newcomer who settles within `SEAT_MATCH_DIST` of a missing ID's home seat, within
    `SEAT_CLAIM_WINDOW_SEC` of appearing, is merged into that ID if their clothing galleries match with similarity
    `>= LONG_MIN_SIM`.
  - This covers a student who leaves through one door, comes back minutes later through another, and sits down
    in their usual seat.
- **Provisional IDs.** A person often comes back half-hidden behind whoever was in front of them, and that first
  crop is too contaminated to match. When the crop is occluded or cut off by the frame edge, the new ID is
  **provisional** for `PENDING_SEC`:
  - It is re-tested every `PENDING_RETEST_SEC` on clean crops, using the same `MIN_SIM` + `MAX_DIST` rule.
  - If it passes, it is merged into the lost ID. That counts as a remap with `deferred: true`.
- **Late merges** (provisional or `seat` rule) are applied to the tracks CSV and keyframes at the end of the run,
  and listed under `merged_ids` in the events JSON. The annotated video is written live, so it shows the temporary
  number until the merge happens.
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
| person came back partly hidden, crop is a mix of two people | raise `PENDING_SEC` (3 -> 5 s) so the provisional ID gets more time to show a clean crop |
| person left the room and came back much later as a new ID | check `best_missing_sec` / `best_sim`: raise `LONG_ABSENCE_SEC`, or lower `LONG_MIN_SIM` a little (not below `MIN_SIM`); for seated classes check that `home seat` positions are stable (calibration) and raise `SEAT_MATCH_DIST` slightly |
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
- Lower `PENDING_SEC` (or set it to 0 to disable deferred re-checks) if provisional IDs merge into the wrong person.
- If people who left are later merged with someone else (`rule: long` or `rule: seat` remaps that look wrong in
  `tools/identity_audit.py`): raise `LONG_MIN_SIM`, lower `SEAT_MATCH_DIST`, or shorten `LONG_ABSENCE_SEC` /
  `SEAT_CLAIM_WINDOW_SEC`. Setting `LONG_ABSENCE_SEC = 0` and `SEAT_CLAIM_WINDOW_SEC = 0` turns the
  re-entry rules off.
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

## Verification

All runs used CPU only. Gemini was disabled in the test sandbox (no API key, no network route to the API), so the
Gemini numbers below are 0 / 0. The Gemini code path was tested separately with a mocked client:
- retries on 429/500/503 with backoff;
- immediate raise on other errors;
- returned IDs that are not active stable IDs are ignored;
- an `id_mismatch` flag is logged and the stored appearance note is kept.

The test clips come from [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos)
(CC BY 4.0). None of them has ground-truth IDs, so swaps and wrong merges were checked visually with
`tools/id_filmstrips.py` (one row of crops per stable ID) and `tools/identity_audit.py` (each remap and each new
ID next to its best candidate).

| clip | what it tests | people | raw tracker IDs | stable IDs | remaps | swaps | wrong merges |
|---|---|---|---|---|---|---|---|
| `classroom.mp4`, first 30 s (`--max-seconds 30`) | the reference run | 4 | 4 | 4 | 0 | 0 | 0 |
| same, 3 seated people hidden for 3.5 s each (longer than the tracker buffer) | forced re-acquisition | 4 | 7 | 4 | 3 | 0 | 0 |
| `store-aisle-detection.mp4` 14-64 s, ByteTrack | real overlaps and crossings | 6 | 7 | 6 | 1 (deferred) | 0 | 0 |
| same, `--tracker botsort_reid` | same | 6 | 7 | 6 | 1 (deferred) | 0 | 0 |
| `worker-zone-detection.mp4` 14-50 s | two people in identical vests and hard hats side by side; re-entries | 4 appearances | 4 | 4 | 0 | 0 | 0 |

**Leaving the frame and coming back** (long-absence and seat rules):

| clip | what it tests | people | raw tracker IDs | stable IDs | remaps | wrong merges |
|---|---|---|---|---|---|---|
| P1: classroom, one student out of view for 50 s, then back in his seat (frames after the return were never seen before) | return after more than the 30 s window | 4 | 5 | 4 | 1 (`long`, sim 0.939, 0.16 m) | 0 |
| P2: `one-by-one-person-detection.mp4`, a woman walks out of frame, the room is empty for 60 s, she walks back in through the same edge | walk-out / walk-in | 1 | 3 | 1 (+1 one-frame blip) | 1 (`long`, sim 0.974, 0.09 m) | 0 |
| P3: classroom, a student stands up, steps into the aisle and sits back down in his own seat; hidden for 43 s while out of his seat | `seat` rule on real footage | 4 | 5 | 4 | 1 (`seat`, gallery sim 0.968, 0.06 m from his home seat) | 0 |
| N1: `one-by-one-person-detection.mp4` (139 s, unmodified) | 6 **different** people take turns at the same table, 3-54 s apart, two look-alike pairs | 6 | 6 | 6 | 0 | 0 |
| N2: `people-detection.mp4` (50 s, unmodified) | 7 different people using the same doors, similar outfits | 7 | 7 | 7 | 0 | 0 |

In P3 the long-absence rule also checked him when he reappeared, but rejected him (0.85 < `LONG_MIN_SIM`; he was
mid-sit). He got a temporary ID, and 5 s after he settled into his home seat the seat rule merged it back. A second
run with the reappearance rules switched off (`--set MAX_DIST=0.01`) gave the same result. His walk, his return
and the frames after it are real footage; only the 43 s absence (his 5.4 s away from the seat, slowed 8x behind a
grey panel) and a looped seated tail are edited.

P1 and P2 are built from real footage; there was no public clip of the same person leaving and returning. P2's
return is the original walk played in reverse with brighter lighting, so its appearance match is easier than in
real life. In N1/N2 the closest different-person similarity was 0.80 (well below `LONG_MIN_SIM` = 0.93).

Reference run summary (classroom, first 30 s): raw tracker IDs 4, stable IDs 4, remaps 0, Gemini `id_mismatch`
flags 0, failed Gemini calls 0 / 0 (disabled).

Notes:
- **Aisle.** A person crouched behind someone, the tracker dropped him, and he came back half-hidden. The right
  lost ID was 0.15-0.5 m away, but his contaminated first crop scored only 0.76 similarity. Before provisional IDs
  existed, that produced a duplicate ID with both trackers. Now the provisional ID is re-tested on a clean crop and
  merged 1.5 s later (similarity 0.91-0.92). BoT-SORT's own ReID did not bridge this gap either; the
  stable-ID layer did.
- **Workers.** The identical-vest pair never physically crossed, so this shows no swaps and no wrong merges with
  identical clothing, not a true crossing. Their SigLIP similarity to the earlier vest-wearing worker was 0.41-0.72,
  well below `MIN_SIM`, and the floor-distance gate (5.9-11 m) also rejected them.
- **Thresholds.** `MIN_SIM = 0.90` and `DRIFT_SIM = 0.80` come from gallery-vs-crop similarities on the classroom
  clip:
  - same person: 1st percentile 0.82, median 0.98;
  - different people: median 0.79, max 0.899.
- **Raised hands.** The pose rule found the one real raised hand in the classroom clip (17.4-19.8 s, after the
  0.5 s hold). In the aisle clip it also fired on someone reaching up to a shelf. The pose rule cannot tell those
  apart; Gemini's behavior labels can.
- **Floor calibration.** The calibrations for these clips were estimated by eye, so radar distances are
  approximate.
