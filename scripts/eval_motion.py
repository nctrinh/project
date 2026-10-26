#!/usr/bin/env python3
"""
Danh gia RIENG nhanh Motion (FrameBuffer + gradient_motion_candidates - xem
src1/antiuav_sot/motion/motion_eval.py) tren tap sequence val (hoac chi
dinh), KHONG chay detector/fusion/tracker/Kalman - muc dich la tra loi cau
hoi "nguong nao cho nhanh Motion (min_box_size, max_box_size, merge_gap_ratio,
min_strength, ...) lam no bat trung UAV tot nhat" TRUOC KHI dua vao
SotPipeline day du, tach biet hoan toan khoi anh huong cua detector/tracker.

Nhan vao cac tham so split/manifest/sequence GIONG eval_tracker_motion.py de
de doi chieu cung mot tap du lieu, nhung chi so bao cao la overlap giua GT
va motion box: moi frame CO GT, CHI tinh IoU LON NHAT trong so cac motion box
hop le (sau loc min_strength) voi GT - KHONG trung binh nhieu box (xem
docstring motion_eval.py ve ly do).

Cach dung:
    python scripts/eval_motion.py --config configs1/default.yaml \
        --manifest experiments/data_manifest.json --split val

    # Quet thu mot nguong min_strength khac, khong can sua yaml:
    python scripts/eval_motion.py --manifest experiments/data_manifest.json \
        --split val --min-strength 80 --strength-metric mean

    # Chi 1 sequence + luu video debug:
    python scripts/eval_motion.py --manifest experiments/data_manifest.json \
        --split val --sequence 20190925_101846_1_2 --save-viz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src1"))

from antiuav_sot.data.dataset import AntiUAVDataset
from antiuav_sot.motion.motion_eval import (
    DEFAULT_IOU_THRESHOLDS,
    MotionSearchParams,
    SequenceMotionEvalResult,
    run_motion_on_sequence,
    summarize_motion_results,
)
from antiuav_sot.utils.config import load_config
from antiuav_sot.utils.viz import SequenceVideoWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs1/default.yaml")
    parser.add_argument("--manifest", type=str, default="experiments/data_manifest.json")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Chi danh gia MOT sequence trong split da chon")
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Gioi han so sequence de danh gia nhanh khi debug",
    )
    parser.add_argument("--save-viz", action="store_true",
                        help="Luu video debug (motion box + GT) cho tung sequence duoc danh gia")
    parser.add_argument("--output-dir", type=str, default="experiments_motion_eval_mean5",
                        help="Thu muc goc luu ket qua danh gia va video (neu --save-viz)")

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
                        help="Box hop nhat co strength < gia tri nay bi loai bo truoc khi tinh overlap")

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
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if args.split == "train":
        seq_names = manifest["train_sequences"]
    elif args.split == "val":
        seq_names = manifest["val_sequences"]
    else:
        seq_names = manifest["train_sequences"] + manifest["val_sequences"]

    if args.max_sequences is not None:
        seq_names = seq_names[: args.max_sequences]
    if args.sequence is not None:
        if args.sequence in seq_names:
            seq_names = [args.sequence]
        else:
            print(f"Loi: Sequence '{args.sequence}' khong ton tai trong split '{args.split}'")
            return

    dataset = AntiUAVDataset(manifest["root_dir"], image_glob=manifest["image_glob"])
    params = MotionSearchParams.from_constraints(
        cfg.fpga, **{f.name: getattr(args, f.name) for f in dataclasses.fields(MotionSearchParams)}
    )

    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "eval_viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_motion] Danh gia {len(seq_names)} sequence (split={args.split})")
    print("[eval_motion] " + " ".join(f"{k}={v}" for k, v in dataclasses.asdict(params).items()))
    all_results: list[SequenceMotionEvalResult] = []
    for seq_name in tqdm(seq_names, desc="eval motion sequences"):
        seq = dataset.load_sequence(seq_name)
        video_writer = SequenceVideoWriter(viz_dir / f"{seq_name}_motion.mp4") if args.save_viz else None
        result = run_motion_on_sequence(seq, params, video_writer=video_writer)
        all_results.append(result)

    iou_thresholds = tuple(args.iou_thresholds)
    summary = summarize_motion_results(all_results, iou_thresholds=iou_thresholds)

    print("\n[eval_motion] ==== KET QUA TONG HOP ====")
    print(
        f"[eval_motion] So sequence: {summary['n_sequences']}  So frame: {summary['n_frames_total']} "
        f"(warm-up bo qua: {summary['n_frames_warmup_skipped']})"
    )
    print(
        f"[eval_motion] Frame co GT: {summary['n_present_frames']}  "
        f"Frame vang mat GT: {summary['n_absent_frames']}"
    )
    print(f"[eval_motion] Mean best IoU (frame co GT, 1 box tot nhat/frame): "
          f"{summary['mean_best_iou_present']:.4f}")
    for t in iou_thresholds:
        key = f"coverage_at_iou_{t:g}"
        print(f"[eval_motion]   Coverage @ IoU>={t:g}: {summary[key]:.4f}")
    print(
        f"[eval_motion] So box hop le trung binh/frame - co GT: "
        f"{summary['avg_valid_motion_boxes_present']:.2f}, vang mat GT: "
        f"{summary['avg_valid_motion_boxes_absent']:.2f} "
        "(tham khao de danh gia muc do nhieu khi chon nguong)"
    )

    per_seq_results = []
    for r in all_results:
        s = summarize_motion_results([r], iou_thresholds=iou_thresholds)
        per_seq_results.append({"seq_name": r.seq_name, **s})

    results_path = out_dir / f"eval_motion_results_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,

                "params": {
                    k: (None if isinstance(v, float) and v == float("-inf") else v)
                    for k, v in dataclasses.asdict(params).items()
                },
                "summary": summary,
                "per_sequence": per_seq_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[eval_motion] Da luu ket qua chi tiet: {results_path}")


if __name__ == "__main__":
    main()