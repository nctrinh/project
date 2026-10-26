#!/usr/bin/env python3
"""
Chay toan bo pipeline (detector + motion + fusion + Simple Tracker FSM +
Kalman) tren tap sequence val (hoac chi dinh), tinh Precision/Success theo
chuan OPE - day la con so THUC TE tra loi cau hoi "phuong an nay hieu qua
cao hay khong" thay vi chi phan tich kien truc tren giay.

Cach dung:
    python scripts/eval_tracker.py --config configs/default.yaml \
        --manifest experiments/data_manifest.json --split val

    python scripts/eval_tracker.py --config configs/default.yaml \
        --manifest experiments/data_manifest.json --split val --save-viz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src1"))

from antiuav_sot.data.dataset import AntiUAVDataset
from antiuav_sot.detector.model import LightweightYoloDetector
from antiuav_sot.pipeline.sot_pipeline import SotPipeline
from antiuav_sot.utils.box_ops import INVALID_BOX
from antiuav_sot.utils.config import load_config
from antiuav_sot.utils.box_ops import xyxy_to_xywh
from antiuav_sot.utils.metrics import (
    SequenceEvalResult,
    evaluate_sequence,
    norm_precision_plot_values,
    success_plot_values,
    summarize,
)
from antiuav_sot.utils.viz import SequenceVideoWriter, render_frame


def run_pipeline_on_sequence(
    pipeline: SotPipeline, seq, save_viz_path: str | None = None
) -> tuple[SequenceEvalResult, list[float]]:
    """Tra ve (ket qua danh gia, danh sach latency tung frame tinh bang giay).
    Latency chi do pipeline.step() (khong gom doc anh va ve visualization)."""
    pipeline.reset()

    predicted_boxes: list[np.ndarray | None] = []
    predicted_active: list[bool] = []
    gt_boxes_xywh: list[np.ndarray] = []
    gt_exist: list[bool] = []
    latencies: list[float] = []

    video_writer = SequenceVideoWriter(save_viz_path) if save_viz_path else None

    for sample in seq:
        raw_frame = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            print(f"[eval_tracker] CANH BAO: khong doc duoc {sample.image_path}, bo qua frame.")
            continue

        t0 = time.perf_counter()
        output = pipeline.step(raw_frame)
        latencies.append(time.perf_counter() - t0)

        # Voi N=1 (SOT), chi quan tam slot 0.
        slot_active = bool(output.track_active[0])
        slot_box = (
            output.track_boxes_smoothed[0]
            if slot_active and output.track_boxes_smoothed[0, 0] != INVALID_BOX
            else None
        )

        predicted_boxes.append(slot_box)
        predicted_active.append(slot_active and slot_box is not None)
        gt_boxes_xywh.append(sample.gt_box_xywh)
        gt_exist.append(bool(sample.exist))

        if video_writer is not None:
            frame_bgr = render_frame(
                raw_frame,
                gt_box_xywh=sample.gt_box_xywh if sample.exist else None,
                track_boxes_xyxy=output.track_boxes_smoothed,
                track_active=output.track_active,
                track_ids=output.track_ids,
                detection_result=output.detection_result,
                letterbox_scale=output.letterbox_scale,
                letterbox_pad=output.letterbox_pad,
                motion_boxes_xyxy=output.motion_boxes,
                motion_valid_mask=output.motion_valid_mask,
            )
            video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.close()

    result = evaluate_sequence(
        seq_name=seq.seq_name,
        predicted_boxes_xyxy=predicted_boxes,
        predicted_active=predicted_active,
        gt_boxes_xywh=gt_boxes_xywh,
        gt_exist=gt_exist,
    )
    return result, latencies


def run_pipeline_on_test_sequence(
    pipeline: SotPipeline, seq, save_viz_path: str | None = None
) -> list[list[float]]:
    """Chay pipeline tren sequence KHONG co ground-truth (tap test).

    Tra ve list do dai n_frames: moi phan tu la [x, y, w, h] (float) neu pipeline
    co track, hoac [] neu khong. Dung save_submission_json() de ghi ra file.
    Khong goi evaluate_sequence() vi khong co GT de so sanh."""
    pipeline.reset()

    predictions_xywh: list[list[float]] = []

    video_writer = SequenceVideoWriter(save_viz_path) if save_viz_path else None

    for sample in seq:
        raw_frame = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            # Giu nguyen so frame de file nop khong bi lech chi so.
            print(f"[eval_tracker] CANH BAO: khong doc duoc {sample.image_path}, ghi [] cho frame nay.")
            predictions_xywh.append([])
            continue

        output = pipeline.step(raw_frame)

        # Voi N=1 (SOT), chi quan tam slot 0.
        slot_active = bool(output.track_active[0])
        if slot_active and output.track_boxes_smoothed[0, 0] != INVALID_BOX:
            box_xywh = xyxy_to_xywh(output.track_boxes_smoothed[0])
            predictions_xywh.append([float(v) for v in box_xywh])
        else:
            predictions_xywh.append([])

        if video_writer is not None:
            frame_bgr = render_frame(
                raw_frame,
                gt_box_xywh=None,
                track_boxes_xyxy=output.track_boxes_smoothed,
                track_active=output.track_active,
                track_ids=output.track_ids,
                detection_result=output.detection_result,
                letterbox_scale=output.letterbox_scale,
                letterbox_pad=output.letterbox_pad,
                motion_boxes_xyxy=output.motion_boxes,
                motion_valid_mask=output.motion_valid_mask,
            )
            video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.close()

    return predictions_xywh


def save_submission_json(predictions_xywh: list[list[float]], path: str | Path) -> None:
    """Ghi du doan cua mot sequence ra JSON dang {"res": [[x,y,w,h] | [], ...]}.
    Hay doi chieu dinh dang voi huong dan nop bai chinh thuc tai
    https://anti-uav.github.io/ truoc khi nop."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"res": predictions_xywh}, f)


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    """Latency pipeline.step() tren may chay eval (GPU/CPU), KHONG phai latency tren FPGA."""
    if not latencies:
        return {"n_frames_timed": 0}
    arr = np.asarray(latencies, dtype=np.float64) * 1000.0  # ms
    mean_ms = float(arr.mean())
    return {
        "n_frames_timed": int(arr.shape[0]),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps_from_mean": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs1/default.yaml")
    parser.add_argument("--manifest", type=str, default="experiments/data_manifest.json")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument(
        "--detector-checkpoint", type=str, default=None,
        help="Ghi de detector_checkpoint trong config neu can",
    )
    parser.add_argument("--save-viz", action="store_true")
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Gioi han so sequence de eval nhanh khi debug",
    )
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments_motion_mean5_again",
                    help="Thư mục gốc lưu kết quả eval và visualization")
    parser.add_argument(
        "--detector-activation", type=str, default=None,
        choices=["silu", "hardswish", "relu", "leakyrelu"],
        help="Ghi de tracker_eval.detector_activation. Bo trong = giu activation trong checkpoint",
    )
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
                print(f"Lỗi: Sequence '{args.sequence}' không tồn tại trong split '{args.split}'")
                return

    dataset = AntiUAVDataset(manifest["root_dir"], image_glob=manifest["image_glob"])

    checkpoint = args.detector_checkpoint or cfg.get(
        "tracker_eval", "detector_checkpoint"
    )
    activation = args.detector_activation or cfg.get(
        "tracker_eval", "detector_activation", default=None
    )
    detector = LightweightYoloDetector(
        weights_path=checkpoint,
        max_det_boxes=cfg.fpga.max_det_boxes,
        conf_threshold=cfg.fpga.det_conf_threshold,
        nms_iou_threshold=cfg.fpga.det_nms_iou_threshold,
        device=cfg.get("tracker_eval", "device", default="cuda:0"),
        activation=activation,
    )
    print(f"[eval_tracker] Detector: {checkpoint} | activation yeu cau={activation} "
          f"| activation thuc te={detector.activation}")

    pipeline = SotPipeline(cfg.fpga, detector)
    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "eval_viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Toan bo rang buoc (motion + tracker) cua lan chay nay - in ra va ghi vao
    # JSON de doi chieu ket qua giua cac cau hinh yaml khac nhau. -inf
    # (motion.min_strength = khong loc) ghi thanh null de JSON hop le.
    fpga_snapshot = {
        k: (None if isinstance(v, float) and v == float("-inf") else v)
        for k, v in dataclasses.asdict(cfg.fpga).items()
    }
    print(f"[eval_tracker] Danh gia {len(seq_names)} sequence (split={args.split})")
    print("[eval_tracker] motion: " + " ".join(
        f"{k[len('motion_'):]}={v}" for k, v in fpga_snapshot.items() if k.startswith("motion_")
    ))

    all_results: list[SequenceEvalResult] = []
    all_latencies: list[float] = []
    per_seq_latencies: dict[str, list[float]] = {}
    for seq_name in tqdm(seq_names, desc="eval sequences"):
        seq = dataset.load_sequence(seq_name)
        save_viz_path = str(viz_dir / f"{seq_name}.mp4") if args.save_viz else None
        result, latencies = run_pipeline_on_sequence(
            pipeline, seq, save_viz_path=save_viz_path
        )
        all_results.append(result)
        all_latencies.extend(latencies)
        per_seq_latencies[seq_name] = latencies

    summary = summarize(all_results)
    latency = _latency_stats(all_latencies)
    allf = summary["all_frames"]
    pres = summary["present_only"]

    print("\n[eval_tracker] ==== KET QUA TONG HOP ====")
    print(f"[eval_tracker] So sequence: {summary['n_sequences']}  So frame: {summary['n_frames']}")
    print(f"[eval_tracker] mSA (trung binh SA tren sequence): {summary['mSA']:.4f}")
    print(f"[eval_tracker] SA gop theo frame:                  {summary['SA_pooled']:.4f}")
    print("[eval_tracker] -- Tinh tren MOI frame (frame vang mat dung = 1 diem) --")
    print(f"[eval_tracker]   Precision@20px:   {allf['precision_at_20px']:.4f}")
    print(f"[eval_tracker]   P_Norm (AUC):     {allf['norm_precision_auc']:.4f}")
    print(f"[eval_tracker]   Success (AUC):    {allf['success_auc']:.4f}")
    print("[eval_tracker] -- Chi tinh frame CO UAV (kieu OTB) --")
    print(f"[eval_tracker]   Precision@20px:   {pres['precision_at_20px']:.4f}")
    print(f"[eval_tracker]   P_Norm (AUC):     {pres['norm_precision_auc']:.4f}")
    print(f"[eval_tracker]   Success (AUC):    {pres['success_auc']:.4f}")
    print(
        f"[eval_tracker] FN rate (frame co UAV, khong bao track): "
        f"{summary['fn_rate_on_present_frames']:.4f} "
        f"({summary['n_false_negative_frames']}/{summary['n_present_frames']})"
    )
    print(
        f"[eval_tracker] FP rate (frame vang mat, van bao track): "
        f"{summary['fp_rate_on_absent_frames']:.4f} "
        f"({summary['n_false_positive_frames']}/{summary['n_absent_frames']})"
    )
    if latency.get("n_frames_timed"):
        print(
            f"[eval_tracker] Latency pipeline.step: mean {latency['mean_ms']:.2f} ms, "
            f"median {latency['median_ms']:.2f} ms, p95 {latency['p95_ms']:.2f} ms "
            f"(~{latency['fps_from_mean']:.1f} FPS) - tren may chay eval, khong phai FPGA"
        )

    # Ve va luu Success plot + Normalized Precision plot (ca hai che do)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for present_only, label in ((False, "all frames"), (True, "present only")):
            th, sr = success_plot_values(all_results, present_only=present_only)
            axes[0].plot(th, sr, linewidth=2, label=label)
            nth, nr = norm_precision_plot_values(all_results, present_only=present_only)
            axes[1].plot(nth, nr, linewidth=2, label=label)

        axes[0].set_xlabel("IoU threshold")
        axes[0].set_ylabel("Success rate")
        axes[0].set_title(
            f"Success plot (AUC all={allf['success_auc']:.3f}, present={pres['success_auc']:.3f})"
        )
        axes[0].set_xlim(0, 1)
        axes[1].set_xlabel("Normalized center distance")
        axes[1].set_ylabel("Precision")
        axes[1].set_title(
            f"Normalized precision (AUC all={allf['norm_precision_auc']:.3f}, "
            f"present={pres['norm_precision_auc']:.3f})"
        )
        axes[1].set_xlim(0, 0.5)
        for ax in axes:
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plot_path = viz_dir / "success_plot.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[eval_tracker] Da luu Success / P_Norm plot: {plot_path}")
    except ImportError:
        print("[eval_tracker] matplotlib khong co san, bo qua ve bieu do.")

    # Luu ket qua chi tiet tung sequence ra JSON
    per_seq_results = []
    for r in all_results:
        s = summarize([r])
        entry = {
            "seq_name": r.seq_name,
            "n_frames": r.n_frames,
            "SA": s["mSA"],
            "mean_iou": float(np.mean(r.ious)),  # giu lai ten cu; bang SA
            "all_frames": s["all_frames"],
            "present_only": s["present_only"],
            "fp_rate_on_absent_frames": s["fp_rate_on_absent_frames"],
            "fn_rate_on_present_frames": s["fn_rate_on_present_frames"],
        }
        entry["precision_at_20px"] = s["all_frames"]["precision_at_20px"]
        seq_lat = _latency_stats(per_seq_latencies.get(r.seq_name, []))
        if seq_lat.get("n_frames_timed"):
            entry["mean_latency_ms"] = seq_lat["mean_ms"]
        per_seq_results.append(entry)

    results_path = out_dir / f"eval_results_{args.split}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "config": args.config,
                "detector_checkpoint": checkpoint,
                "detector_activation": detector.activation,
                "fpga_constraints": fpga_snapshot,
                "overall_precision_at_20px": allf["precision_at_20px"],
                "overall_success_auc": allf["success_auc"],
                "summary": summary,
                "latency": latency,
                "per_sequence": per_seq_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[eval_tracker] Da luu ket qua chi tiet: {results_path}")


if __name__ == "__main__":
    main()