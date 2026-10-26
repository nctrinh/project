#!/usr/bin/env python3
"""Random int8 YOLOv8 head tensors (NHWC, 512x640 input) for the decode parity test.
A few clusters of confident cells (overlapping boxes -> NMS work, equal int8
logits -> score ties) on top of a low-confidence background."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distinct", action="store_true",
                    help="every confident cell gets a different int8 logit (no score ties -> NMS order is defined)")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--max-det", type=int, default=12)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg_max, nc = 16, 1
    levels = []
    distinct_q = iter(rng.permutation(np.arange(-12, 127)))
    for li, (h, w, s) in enumerate(((64, 80, 8), (32, 40, 16), (16, 20, 32))):
        box_fp = int(rng.integers(3, 6))
        cls_fp = 4 if args.distinct else int(rng.integers(3, 6))  # one scale -> distinct q == distinct score
        box = rng.integers(-60, 60, (h, w, 4 * reg_max), dtype=np.int16).astype(np.int8)
        cls = rng.integers(-128, -40, (h, w, nc), dtype=np.int16).astype(np.int8)
        for _ in range(4 if args.distinct else 6):  # clusters of confident cells
            cy, cx = rng.integers(2, h - 2), rng.integers(2, w - 2)
            q = int(rng.integers(-12, 40))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if args.distinct:
                        cls[cy + dy, cx + dx, 0] = np.int8(next(distinct_q))
                    else:
                        cls[cy + dy, cx + dx, 0] = np.int8(q if rng.random() < 0.5 else q - int(rng.integers(0, 4)))
        box.tofile(out / f"box{li}.bin")
        cls.tofile(out / f"cls{li}.bin")
        levels.append({"h": h, "w": w, "stride": s, "box_fp": box_fp, "cls_fp": cls_fp,
                       "box_file": f"box{li}.bin", "cls_file": f"cls{li}.bin"})
    meta = {"reg_max": reg_max, "nc": nc, "conf": 0.3, "iou": args.iou, "max_det": args.max_det, "levels": levels}
    json.dump(meta, open(out / "meta.json", "w"))


if __name__ == "__main__":
    main()
