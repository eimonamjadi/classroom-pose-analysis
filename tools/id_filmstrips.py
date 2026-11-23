#!/usr/bin/env python3
"""One row per stable ID: crops sampled every STEP seconds over its lifetime. A row that changes person = a swap.
Usage: python tools/id_filmstrips.py VIDEO OUT_DIR [STEP_SEC]   -> OUT_DIR/id_filmstrips.jpg"""
import csv, sys
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

video, out_dir = sys.argv[1], Path(sys.argv[2])
step = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
rows = defaultdict(dict)
for r in csv.DictReader(open(out_dir / f"{Path(video).stem}_tracks.csv")):
    rows[int(r["stable_id"])][int(r["frame"])] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = max(f for d in rows.values() for f in d) + 1
samples = list(range(0, n_frames, max(1, int(step * fps))))
cache = {}
TW, TH = 70, 130
strips = []
for sid in sorted(rows):
    tiles = []
    for f in samples:
        near = [g for g in rows[sid] if abs(g - f) <= fps * step / 2]
        if not near:
            tiles.append(np.full((TH, TW, 3), 40, np.uint8)); continue
        g = min(near, key=lambda g: abs(g - f))
        if g not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, g); cache[g] = cap.read()[1]
        x1, y1, x2, y2 = (int(round(v)) for v in rows[sid][g])
        c = cache[g][max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        tiles.append(cv2.resize(c, (TW, TH)))
    label = np.full((TH, 60, 3), 20, np.uint8)
    cv2.putText(label, f"#{sid}", (4, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    strips.append(np.hstack([label] + tiles))
axis = np.full((22, strips[0].shape[1], 3), 20, np.uint8)
for k, f in enumerate(samples):
    cv2.putText(axis, f"{f / fps:.0f}s", (60 + k * TW + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
cv2.imwrite(str(out_dir / "id_filmstrips.jpg"), np.vstack([axis] + strips))
print(out_dir / "id_filmstrips.jpg")
