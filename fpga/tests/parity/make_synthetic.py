#!/usr/bin/env python3
"""Synthetic Anti-UAV-like IR sequence for the C++ <-> Python parity test.

Writes <out>/<seq_name>/{000001.png ...  IR_label.json} and <out>/detections_<seq_name>.json
(detector replay, letterbox coords == image coords because 512x640 -> 512x640).

Scene: smooth cloud-like background panning/rotating slightly (camera motion,
exercises the global-motion IRLS), a small hot target moving across it, absent
for a stretch of frames, plus a hot distractor. Detections: noisy GT boxes with
misses and false positives (exercises all FSM tiers + tentative tracks).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def blob(canvas, cx, cy, sigma, amp):
    h, w = canvas.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    canvas += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="synthetic_seq")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    H, W = 512, 640
    seq_dir = Path(args.out) / args.name
    seq_dir.mkdir(parents=True, exist_ok=True)

    big = rng.normal(0, 1, (H + 200, W + 200)).astype(np.float32)
    big = cv2.GaussianBlur(big, (0, 0), 12)
    big = (big - big.min()) / (big.max() - big.min()) * 90 + 60  # 60..150 grey level background

    exist, gt_rect, det_frames = [], [], []
    for t in range(args.frames):
        ang = 0.05 * np.sin(t / 15.0)  # degrees
        M = cv2.getRotationMatrix2D(((W + 200) / 2, (H + 200) / 2), ang, 1.0)
        M[0, 2] -= 100 + 1.5 * t  # pan right->left
        M[1, 2] -= 100 + 0.6 * np.sin(t / 7.0) * 4
        frame = cv2.warpAffine(big, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frame += rng.normal(0, 1.5, frame.shape).astype(np.float32)

        # target
        cx, cy = 120 + 3.2 * t, 300 - 1.1 * t + 10 * np.sin(t / 9.0)
        present = not (60 <= t < 75)
        size = 10.0 + 4.0 * np.sin(t / 20.0)
        if present:
            blob(frame, cx, cy, size / 3.5, 70)
        # distractor (hot, moving the other way)
        blob(frame, 520 - 2.0 * t, 120 + 0.5 * t, 3.0, 55)

        img = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.imwrite(str(seq_dir / f"{t + 1:06d}.png"), img)

        box = [float(cx - size / 2), float(cy - size / 2), float(size), float(size)]
        exist.append(1 if present else 0)
        gt_rect.append([round(v, 2) for v in box] if present else [])

        dets = []
        if present and rng.random() < 0.7:
            j = rng.normal(0, 1.2, 4)
            x1, y1 = box[0] + j[0], box[1] + j[1]
            x2, y2 = box[0] + box[2] + j[2], box[1] + box[3] + j[3]
            dets.append({"box": [x1, y1, x2, y2], "score": float(rng.uniform(0.3, 0.95))})
        if rng.random() < 0.12:
            fx, fy = rng.uniform(20, W - 40), rng.uniform(20, H - 40)
            fs = rng.uniform(6, 14)
            dets.append({"box": [fx, fy, fx + fs, fy + fs], "score": float(rng.uniform(0.3, 0.7))})
        dets.sort(key=lambda d: -d["score"])
        # float32-representable values so both sides start from identical inputs
        for d in dets:
            d["box"] = [float(np.float32(v)) for v in d["box"]]
            d["score"] = float(np.float32(d["score"]))
        det_frames.append(dets)

    with open(seq_dir / "IR_label.json", "w") as f:
        json.dump({"exist": exist, "gt_rect": gt_rect}, f)
    with open(Path(args.out) / f"detections_{args.name}.json", "w") as f:
        json.dump({"frames": det_frames}, f)
    manifest = {"root_dir": str(Path(args.out).resolve()), "image_glob": ["*.png"],
                "train_sequences": [], "val_sequences": [args.name]}
    with open(Path(args.out) / "manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"wrote {args.frames} frames to {seq_dir}")


if __name__ == "__main__":
    main()
