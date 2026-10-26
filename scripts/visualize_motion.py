#!/usr/bin/env python3
"""
Chay RIENG nhanh Motion (FrameBuffer + gradient_motion_candidates - xem
src1/antiuav_sot/motion/motion_eval.py) tren MOT sequence duy nhat, xuat
video overlay cac motion box SAU KHI DA HOP NHAT (gop cum fragment, loc
kich thuoc/ty le canh, loc min_strength), kem strength cua tung box.

Khac voi visualize_sequence.py (chay TOAN BO pipeline: detector + motion +
fusion + tracker + Kalman), script nay CHI chay nhanh Motion mot minh -
muc dich la xem va chon nguong (min_box_size, max_box_size, merge_gap_ratio,
min_strength, ...) truoc khi dua vao pipeline day du.

Sau khi luu video, in ra ty le overlap voi ground-truth: moi frame CO GT,
chi tinh IoU LON NHAT trong so cac motion box hop le (KHONG trung binh nhieu
box - xem docstring motion_eval.py).

Cach dung:
    python scripts/visualize_motion.py --config configs1/default.yaml \
        --seq-name 20190925_101846_1_2

    # Thu nguong khac (khong can sua yaml):
    python scripts/visualize_motion.py --seq-name 20190925_101846_1_2 \
        --min-box-size 6 --max-box-size 48 --min-strength 50 \
        --strength-metric mean
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src1"))

from antiuav_sot.data.dataset import AntiUAVDataset
from antiuav_sot.motion.motion_eval import (
    DEFAULT_IOU_THRESHOLDS,
    MotionSearchParams,
    run_motion_on_sequence,
    summarize_motion_results,
)
from antiuav_sot.utils.config import load_config
from antiuav_sot.utils.viz import SequenceVideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs1/default.yaml")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Neu truyen, lay root_dir/image_glob tu manifest thay vi config")
    parser.add_argument("--seq-name", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Duong dan video output (mac dinh: experiments_motion/<seq_name>_motion.mp4)")
    parser.add_argument("--fps", type=int, default=15)

    # Moi tham so motion duoi day: khong truyen (None) -> lay tu khoi motion cua
    # fpga_constraints.yaml, CUNG gia tri SotPipeline/eval_tracker dang dung.
    # ---- So luong box motion co dinh (K) + loc kich thuoc/hinh dang/hop nhat ----
    parser.add_argument("--max-motion-candidates", type=int, default=None)
    parser.add_argument("--min-box-size", type=int, default=None)
    parser.add_argument("--max-box-size", type=int, default=None)
    parser.add_argument("--merge-gap-ratio", type=float, default=None)
    parser.add_argument("--max-aspect-ratio", type=float, default=None)
    parser.add_argument("--merge-min-fill-ratio", type=float, default=None)

    # ---- Strength cua box SAU KHI hop nhat + nguong loc ----
    parser.add_argument("--strength-metric", type=str, choices=["sum", "mean", "contrast"], default=None,
                        help="sum = tong residual trong box (thien vi box lon); "
                             "mean = residual trung binh/pixel (mat do, khong thien vi dien tich); "
                             "contrast = mean trong box tru mean vanh xung quanh (phat vung clutter)")
    parser.add_argument("--min-strength", type=float, default=None,
                        help="Box hop nhat co strength < gia tri nay bi loai bo truoc khi hien thi/tinh overlap")

    # ---- Tham so thuat toan gradient global-motion (hiem khi can doi) ----
    parser.add_argument("--model", type=str, choices=["affine", "translation"], default=None)
    parser.add_argument("--irls-iters", type=int, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--pyramid-levels", type=int, default=None)
    parser.add_argument("--thresh-factor", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=None)
    parser.add_argument("--frame-lag", type=int, default=None,
                        help="So frame T-lag dung de tinh motion (xem motion/frame_buffer.py)")

    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=list(DEFAULT_IOU_THRESHOLDS))
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.manifest is not None:
        import json
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        root_dir = manifest["root_dir"]
        image_glob = manifest["image_glob"]
    else:
        root_dir = cfg.get("data", "root_dir")
        image_glob = cfg.get("data", "image_glob", default=["*.jpg", "*.jpeg", "*.png"])

    dataset = AntiUAVDataset(root_dir, image_glob=image_glob)
    seq = dataset.load_sequence(args.seq_name)
    print(f"[visualize_motion] Sequence '{args.seq_name}': {len(seq)} frame")

    params = MotionSearchParams.from_constraints(
        cfg.fpga, **{f.name: getattr(args, f.name) for f in dataclasses.fields(MotionSearchParams)}
    )
    print("[visualize_motion] " + " ".join(f"{k}={v}" for k, v in dataclasses.asdict(params).items()))

    output_path = args.output or f"experiments_motion/{args.seq_name}_motion.mp4"
    video_writer = SequenceVideoWriter(output_path, fps=args.fps)

    result = run_motion_on_sequence(seq, params, video_writer=video_writer)
    print(f"[visualize_motion] Da luu video: {output_path}")

    summary = summarize_motion_results([result], iou_thresholds=tuple(args.iou_thresholds))
    print("\n[visualize_motion] ==== OVERLAP VOI GROUND-TRUTH (sau khi luu video) ====")
    print(
        f"[visualize_motion] So frame: {summary['n_frames_total']} "
        f"(warm-up bo qua: {summary['n_frames_warmup_skipped']}, "
        f"co GT: {summary['n_present_frames']}, vang mat GT: {summary['n_absent_frames']})"
    )
    print(f"[visualize_motion] Mean best IoU (frame co GT): {summary['mean_best_iou_present']:.4f}")
    for t in args.iou_thresholds:
        key = f"coverage_at_iou_{t:g}"
        print(f"[visualize_motion]   Coverage @ IoU>={t:g}: {summary[key]:.4f}")
    print(
        f"[visualize_motion] So box hop le trung binh/frame - co GT: "
        f"{summary['avg_valid_motion_boxes_present']:.2f}, vang mat GT: "
        f"{summary['avg_valid_motion_boxes_absent']:.2f}"
    )


if __name__ == "__main__":
    main()