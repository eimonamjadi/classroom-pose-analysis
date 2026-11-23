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

All demos were produced with the current code (default ByteTrack + stable-ID layer).

**Attribution:** source footage is `classroom.mp4`, `store-aisle-detection.mp4` and `worker-zone-detection.mp4` from
[intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: trimmed (and for the aisle/worker clips resampled to 30 fps and resized), annotated with tracking
overlays and a radar panel, downscaled and re-encoded.
