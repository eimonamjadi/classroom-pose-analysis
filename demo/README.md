# Demo outputs

- `demo_tracking_30s.mp4`: `run_classroom.py` output (default ByteTrack + stable-ID layer, Gemini disabled) on the
  first 30 s of the test clip. Video frame on top, floor radar underneath. Watch #1 raise a hand at ~17-20 s.
- `gemini_tagged_frame.jpg`: the ID-tagged frame format sent to Gemini (tags only, just above each head).

The floor calibration for this clip was estimated by eye, so radar depth is compressed toward the camera side.

**Attribution:** source footage is `classroom.mp4` from
[intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: trimmed to 30 s, annotated with tracking
overlays and a radar panel, downscaled and re-encoded.
