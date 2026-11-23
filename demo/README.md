# Demo outputs

- `demo_tracking_30s.mp4`: `run_classroom.py` output (default ByteTrack + stable-ID layer, Gemini disabled) on the
  first 30 s of the test clip. Video frame on top, floor radar underneath. Watch #1 raise a hand at ~17-20 s.
- `gemini_tagged_frame.jpg`: the ID-tagged frame format sent to Gemini (tags only, just above each head).
- `demo_occlusion_stress_30s.mp4`: the same 30 s with three seated people hidden behind grey panels for 3.5 s each,
  longer than the tracker's 2 s buffer, so ByteTrack issues new raw IDs. The stable-ID layer reclaims all three:
  7 raw tracker IDs -> 4 stable IDs, 3 remaps (similarity 0.97-0.98, floor distance 0.01 m).
- `audit_occlusion_stress.jpg`: `tools/identity_audit.py` sheet for that run. For each remap, the left crop is the
  stable ID when last seen and the right crop is the track that reclaimed it (all three correct).

The floor calibration for this clip was estimated by eye, so radar depth is compressed toward the camera side.

- `demo_overlap_aisle_50s.mp4` and `filmstrips_overlap_aisle.jpg`: overlap test on a shop aisle (6 people, constant
  crossings). 7 raw tracker IDs -> 6 stable IDs, no swaps. At ~10 s the hat man (#2) crouches behind another
  shopper, briefly shows provisional #4, and is merged back into #2 at 13.3 s.
- `demo_identical_vests_36s.mp4` and `filmstrips_identical_vests.jpg`: two workers in identical vests and hard
  hats walking side by side keep separate IDs; newcomers in the same outfit are not merged into earlier workers.

- `demo_reentry_classroom_away50s.mp4` and `filmstrips_reentry_classroom.jpg`: the left-back student is out of view
  from 10 s to 60 s (grey panel) and gets his original ID (#2) back when he reappears, 50 s later, through the
  long-absence rule.
- `demo_reentry_walkout_back60s.mp4`: a woman walks out of frame, the room is empty for 60 s, and she walks back in
  through the same edge. Her temporary ID is merged back into #1 1.2 s after she reappears.
- `demo_seat_rule_return.mp4` and `filmstrips_seat_rule.jpg`: seat rule on real footage. The hoodie student (#3)
  leaves his seat at 21 s and is hidden for 43 s. He reappears sitting down, gets a temporary #5, and is merged back
  into #3 at 69.4 s, 5 s after settling into his home seat.
- `filmstrips_negative_same_spot.jpg` and `filmstrips_negative_same_doors.jpg`: six and seven **different** people
  using the same table / doors keep separate IDs. Nobody inherits the ID of someone who left.

New output types, from the classroom (first 30 s) and the shop aisle (50 s):
- `skeletons_classroom.mp4`, `skeletons_aisle.mp4`: skeleton-only video with ID labels and fading 10 s trails.
- `trajectories_floor_*.png`: each ID's path on the floor plan (ring = start, dot = end, square = home seat).
- `trajectories_camera_*.png`: the same paths drawn over the first frame.
- `heatmap_floor_*.png`: where people spent their time.
- `id_summary_*.csv`: one row per ID (time on screen, distance, speeds, moving / still time, home seat, hand
  raises, Gemini fields when available).
- `tracks_sample_*.csv`: the first 300 rows of the per-frame CSV (80 columns, including all 17 keypoints).

The floor calibrations for these clips were estimated by eye, so distances and speeds are approximate.

All demos were produced with the current code (default ByteTrack + stable-ID layer).

**Attribution:** source footage is `classroom.mp4`, `store-aisle-detection.mp4`, `worker-zone-detection.mp4`,
`one-by-one-person-detection.mp4` and `people-detection.mp4` from
[intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: trimmed or re-edited (re-entry clips: segments masked, slowed, reversed or brightened; aisle/worker
clips resampled to 30 fps and resized), annotated with tracking
overlays and a radar panel, downscaled and re-encoded.
