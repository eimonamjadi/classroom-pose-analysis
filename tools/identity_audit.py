#!/usr/bin/env python3
"""Visual audit of the stable-ID layer.

For every identity_remap event (short / long / seat rule): crop of the stable ID when it was last seen  |  crop of
the person who reclaimed it (similarity, floor distance, how long it was missing). Wrong merges are obvious at a glance.
For every identity_new event that had a candidate: crop of the new person  |  crop of the best rejected candidate
(best_sim, best_dist_m). Same person on both sides = a missed reclaim (too many new IDs).

Usage: python tools/identity_audit.py VIDEO OUT_DIR   (OUT_DIR = the run's --out directory)
Writes OUT_DIR/identity_audit.jpg
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

video, out_dir = sys.argv[1], Path(sys.argv[2])
stem = Path(video).stem
events = json.loads((out_dir / f"{stem}_events.json").read_text())
merged = events.get("merged_ids", events.get("merged_provisional_ids", {}))
rows = defaultdict(dict)  # stable_id -> frame -> bbox
for r in csv.DictReader(open(out_dir / f"{stem}_tracks.csv")):
    rows[int(r["stable_id"])][int(r["frame"])] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
H = 220


def crop(frame_idx, box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    if not ok or box is None:
        return np.zeros((H, H // 2, 3), np.uint8)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    pad = int(0.08 * (y2 - y1))
    c = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    return cv2.resize(c, (max(1, int(c.shape[1] * H / max(1, c.shape[0]))), H))


def last_before(sid, frame):
    fs = [f for f in rows.get(sid, {}) if f < frame]
    return (max(fs), rows[sid][max(fs)]) if fs else (None, None)


def tile(left, right, title, color):
    gap = np.full((H, 8, 3), 255, np.uint8)
    body = np.hstack([left, gap, right])
    head = np.full((44, max(body.shape[1], 420), 3), 30, np.uint8)
    for k, line in enumerate(title.split("\n")):
        cv2.putText(head, line, (6, 18 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    body = np.pad(body, ((0, 0), (0, head.shape[1] - body.shape[1]), (0, 0)))
    return np.vstack([head, body])


tiles = []
for e in events["events"]:
    if e["type"] == "identity_remap":
        sid, f = e["stable_id"], e["frame"]
        # for a late merge the newcomer's earlier rows already carry the final ID: look before it appeared
        cutoff = f - int(round(e.get("merged_after_sec", 0) * fps)) if e.get("deferred") else f
        lf, lbox = last_before(sid, cutoff)
        left = crop(lf, lbox) if lf is not None else np.zeros((H, H // 2, 3), np.uint8)
        right = crop(f, rows[sid].get(f))
        t = (f"REMAP ({e.get('rule', 'short')}) raw {e['raw_id']} -> #{sid}  t={e['t']:.1f}s\n"
             f"sim {e['similarity']:.3f}  dist {e['distance_m']:.2f}m  missing {e['gap_sec']:.1f}s")
        tiles.append(tile(left, right, t, (120, 255, 120)))
    elif e["type"] == "identity_new" and "best_candidate" in e and e["frame"] > 0:
        sid, f, cand = e["stable_id"], e["frame"], e["best_candidate"]
        final = merged.get(str(sid), sid)  # the CSV already carries the final ID of merged newcomers
        lf, lbox = last_before(cand, f)
        left = crop(f, rows[final].get(f))
        right = crop(lf, lbox) if lf is not None else np.zeros((H, H // 2, 3), np.uint8)
        tag = f" (later merged into #{final})" if final != sid else ""
        t = (f"NEW #{sid}{tag} t={e['t']:.1f}s | best rejected #{cand}\n"
             f"sim {e['best_sim']:.3f}  dist {e['best_dist_m']:.2f}m")
        tiles.append(tile(left, right, t, (120, 200, 255)))

if not tiles:
    print("no remap / new-identity events with candidates")
    sys.exit(0)
w = max(t.shape[1] for t in tiles)
tiles = [np.pad(t, ((0, 6), (0, w - t.shape[1]), (0, 0))) for t in tiles]
cols = 4
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(str(out_dir / "identity_audit.jpg"), sheet)
print(f"{len(tiles)} tiles -> {out_dir / 'identity_audit.jpg'}")
