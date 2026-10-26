#!/usr/bin/env python3
"""
Chan doan pipeline THEO TUNG GIAI DOAN: chay full pipeline (detector +
motion + fusion + FSM + Kalman) nhung, khac voi eval_tracker.py, GHI LAI
IoU voi GT tai TUNG giai doan trung gian cho moi frame - detector tho,
sau FSM, sau Kalman - de xac dinh CHINH XAC giai doan nao lam track bi
lech, thay vi chi biet "ket qua cuoi cung sai".

Dung khi quan sat hien tuong: detector bat trung o mot so frame nhung
track hien thi (qua thoi gian) lai khong bat trung, du vat the van to/ro
rang - hien tuong nay CO THE do:
    - Detector tu no da mat muc tieu roi (khong lien quan tracker) ->
      chay them scripts/eval_detector.py de xac nhan/loai truong hop nay.
    - FSM khong khop duoc detection dung voi track (vd track da bi Kalman
      keo lech truoc do qua nhieu, IoU tut duoi ca nguong tier4).
    - Kalman lam muot (smooth) qua muc, keo vi tri lech khoi quan sat that.
    - Track dang "troi" o che do predict-only qua nhieu frame lien tiep.

Cach dung:
    python scripts/eval_pipeline_stages.py --config configs/default.yaml \
        --seq-name 20190926_095902_1_3 \
        --detector-checkpoint experiments/detector/weights/best.pt

    # Chay tren toan bo tap val va tong hop:
    python scripts/eval_pipeline_stages.py --config configs/default.yaml \
        --manifest experiments/data_manifest.json --split val \
        --detector-checkpoint experiments/detector/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from antiuav_sot.data.dataset import AntiUAVDataset
from antiuav_sot.detector.model import LightweightYoloDetector
from antiuav_sot.pipeline.sot_pipeline import SotPipeline
from antiuav_sot.utils.config import load_config
from antiuav_sot.utils.stage_metrics import (
    FrameStageDiagnostic,
    diagnose_frame,
    summarize_diagnostics,
)

# Slot 0 - N=1 (Single-Object Tracking), dung quy uoc giong toan repo.
TRACK_SLOT = 0


def run_stage_diagnosis_on_sequence(
    pipeline: SotPipeline,
    seq,
    detector_missed_threshold: float,
    fsm_lost_threshold_gap: float,
    kalman_diverge_threshold_gap: float,
    long_predict_only_threshold: int,
) -> list[FrameStageDiagnostic]:
    pipeline.reset()
    diagnostics: list[FrameStageDiagnostic] = []

    for sample in seq:
        raw_frame = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            print(f"[eval_pipeline_stages] CANH BAO: khong doc duoc {sample.image_path}, bo qua.")
            continue

        # QUAN TRONG: lay track box TRUOC KHI goi pipeline.step() - day la
        # vi tri track dang o TU FRAME TRUOC, chinh la gia tri Simple Tracker
        # FSM se dung de tinh IoU voi detection cua frame NAY. Phai lay truoc
        # step(), vi step() se ghi de track_state.boxes ngay trong noi bo FSM.
        track_active_before = bool(pipeline.track_state.active[TRACK_SLOT])
        track_box_before = (
            pipeline.track_state.boxes[TRACK_SLOT].copy() if track_active_before else None
        )

        output = pipeline.step(raw_frame)

        diag = diagnose_frame(
            frame_index=sample.frame_index,
            gt_exists=sample.exist,
            gt_box_xywh=sample.gt_box_xywh,
            detector_boxes_xyxy=_rescale_detector_boxes(output, pipeline),
            detector_scores=output.detection_result.scores,
            detector_valid_mask=output.detection_result.valid_mask,
            track_box_before_this_frame_xyxy=track_box_before,
            post_fsm_box_xyxy=output.track_boxes[TRACK_SLOT],
            post_fsm_active=bool(output.track_active[TRACK_SLOT]),
            post_kalman_box_xyxy=output.track_boxes_smoothed[TRACK_SLOT],
            track_age=int(output.track_ages[TRACK_SLOT]),
            predict_only_streak=int(output.predict_only_streak[TRACK_SLOT]),
            matched_tier=int(output.fsm_result.tier_for_candidate[
                _find_tier_index_for_slot(output, TRACK_SLOT)
            ]) if _find_tier_index_for_slot(output, TRACK_SLOT) is not None else 0,
            detector_missed_threshold=detector_missed_threshold,
            fsm_lost_threshold_gap=fsm_lost_threshold_gap,
            kalman_diverge_threshold_gap=kalman_diverge_threshold_gap,
            long_predict_only_threshold=long_predict_only_threshold,
        )
        diagnostics.append(diag)

    return diagnostics


def _rescale_detector_boxes(output, pipeline: SotPipeline) -> np.ndarray:
    """Detector boxes trong PipelineFrameOutput.detection_result van o he
    toa do SAU letterbox - quy doi ve he toa do anh goc de so sanh voi GT,
    dung cach lam giong het sot_pipeline._rescale_boxes_from_letterbox
    (khong import truc tiep ham private, tu tinh lai o day cho ro rang)."""
    from antiuav_sot.utils.box_ops import INVALID_BOX

    boxes = output.detection_result.boxes.copy()
    scale = output.letterbox_scale
    pad_left, pad_top = output.letterbox_pad
    valid = boxes[:, 0] != INVALID_BOX
    boxes[valid, 0] = (boxes[valid, 0] - pad_left) / scale
    boxes[valid, 1] = (boxes[valid, 1] - pad_top) / scale
    boxes[valid, 2] = (boxes[valid, 2] - pad_left) / scale
    boxes[valid, 3] = (boxes[valid, 3] - pad_top) / scale
    return boxes


def _find_tier_index_for_slot(output, slot_idx: int) -> int | None:
    """Tim candidate index nao (neu co) da duoc khop vao slot_idx frame nay,
    de biet FSM da dung TIER nao. Tra ve None neu khong co candidate nao
    khop slot nay frame nay (vd track dang predict-only, hoac khong active)."""
    matches = np.nonzero(output.fsm_result.matched_slot_for_candidate == slot_idx)[0]
    if matches.shape[0] == 0:
        return None
    return int(matches[0])


def print_sequence_report(seq_name: str, diagnostics: list[FrameStageDiagnostic]) -> None:
    summary = summarize_diagnostics(diagnostics)

    print(f"\n[eval_pipeline_stages] ==== Sequence: {seq_name} ====")
    print(f"  Tong so frame: {summary['n_frames_total']} "
          f"(co object: {summary['n_frames_with_gt']})")

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"  Mean IoU - Detector (tho):    {_fmt(summary['mean_detector_iou'])}")
    print(f"  Mean IoU - Sau FSM:           {_fmt(summary['mean_post_fsm_iou'])}")
    print(f"  Mean IoU - Sau Kalman:        {_fmt(summary['mean_post_kalman_iou'])}")
    print(f"  So frame detector bo sot muc tieu:      {summary['n_detector_missed']}")
    print(f"  So frame FSM danh mat detection tot:    {summary['n_fsm_lost_good_detection']}")
    print(f"    trong do khong the khop VE HINH HOC "
          f"(track qua xa detection moi): {summary['n_fsm_lost_structurally_unmatchable']}")
    print(f"    trong do nguyen nhan KHAC (dang can xem lai):  "
          f"{summary['n_fsm_lost_other_cause']}")
    print(f"  So frame Kalman lam lech so voi FSM:    {summary['n_kalman_diverged']}")
    print(f"  So frame dang o 'predict-only' keo dai: {summary['n_long_predict_only_streak']}")
    print(f"  Predict-only streak dai nhat:           {summary['max_predict_only_streak']}")

    # In chi tiet cac frame co van de "FSM lost" hoac "Kalman diverged" -
    # day thuong la cac frame nguoi dung can xem lai truc tiep tren video.
    problem_frames = [
        d for d in diagnostics
        if d.flag_fsm_lost_good_detection or d.flag_kalman_diverged_from_fsm
    ]
    if problem_frames:
        print(f"\n  --- Chi tiet {len(problem_frames)} frame nghi van ---")
        for d in problem_frames[:20]:  # gioi han in ra man hinh, JSON day du van luu het
            reasons = []
            if d.flag_fsm_lost_good_detection:
                track_vs_det = (
                    f"{d.track_vs_best_detection_iou:.2f}"
                    if d.track_vs_best_detection_iou is not None else "N/A"
                )
                reasons.append(
                    f"FSM danh mat (det_iou={d.detector_best_iou:.2f} -> "
                    f"fsm_iou={d.post_fsm_iou:.2f}, track_vs_detection_iou={track_vs_det})"
                )
            if d.flag_kalman_diverged_from_fsm:
                reasons.append(
                    f"Kalman lech (fsm_iou={d.post_fsm_iou:.2f} -> "
                    f"kalman_iou={d.post_kalman_iou:.2f})"
                )
            print(f"    Frame {d.frame_index}: {'; '.join(reasons)}, "
                  f"tier={d.matched_tier}, age={d.track_age}, "
                  f"predict_streak={d.predict_only_streak}")
        if len(problem_frames) > 20:
            print(f"    ... va {len(problem_frames) - 20} frame khac (xem file JSON)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--detector-checkpoint", type=str, required=True)

    # Che do 1: mot sequence duy nhat
    parser.add_argument("--seq-name", type=str, default=None)

    # Che do 2: toan bo mot split (can manifest)
    parser.add_argument("--manifest", type=str, default="experiments/data_manifest.json")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default=None)
    parser.add_argument("--max-sequences", type=int, default=None)

    parser.add_argument("--detector-missed-threshold", type=float, default=0.3)
    parser.add_argument("--fsm-lost-threshold-gap", type=float, default=0.3)
    parser.add_argument("--kalman-diverge-threshold-gap", type=float, default=0.2)
    parser.add_argument("--long-predict-only-threshold", type=int, default=2)
    args = parser.parse_args()

    if args.seq_name is None and args.split is None:
        parser.error("Can chi dinh --seq-name (mot sequence) hoac --split (toan bo mot tap)")

    cfg = load_config(args.config)

    if args.seq_name is not None:
        root_dir = cfg.get("data", "root_dir")
        image_glob = cfg.get("data", "image_glob", default=["*.jpg", "*.jpeg", "*.png"])
        dataset = AntiUAVDataset(root_dir, image_glob=image_glob)
        seq_names = [args.seq_name]
    else:
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
        dataset = AntiUAVDataset(manifest["root_dir"], image_glob=manifest["image_glob"])

    detector = LightweightYoloDetector(
        weights_path=args.detector_checkpoint,
        max_det_boxes=cfg.fpga.max_det_boxes,
        conf_threshold=cfg.fpga.det_conf_threshold,
        nms_iou_threshold=cfg.fpga.det_nms_iou_threshold,
        device=cfg.get("tracker_eval", "device", default="cuda:0"),
    )
    pipeline = SotPipeline(cfg.fpga, detector)

    all_diagnostics_by_seq: dict[str, list[FrameStageDiagnostic]] = {}

    iterator = seq_names if len(seq_names) == 1 else tqdm(seq_names, desc="stage diagnosis")
    for seq_name in iterator:
        seq = dataset.load_sequence(seq_name)
        diagnostics = run_stage_diagnosis_on_sequence(
            pipeline, seq,
            detector_missed_threshold=args.detector_missed_threshold,
            fsm_lost_threshold_gap=args.fsm_lost_threshold_gap,
            kalman_diverge_threshold_gap=args.kalman_diverge_threshold_gap,
            long_predict_only_threshold=args.long_predict_only_threshold,
        )
        all_diagnostics_by_seq[seq_name] = diagnostics
        print_sequence_report(seq_name, diagnostics)

    # ---- Tong hop toan bo (huu ich khi chay --split) ----
    if len(seq_names) > 1:
        flat_all = [d for diags in all_diagnostics_by_seq.values() for d in diags]
        overall_summary = summarize_diagnostics(flat_all)
        print("\n[eval_pipeline_stages] ==== TONG HOP TOAN BO SPLIT ====")
        for key, value in overall_summary.items():
            print(f"  {key}: {value}")

    # ---- Luu JSON chi tiet ----
    output_name = args.seq_name if args.seq_name else f"split_{args.split}"
    output_path = Path("experiments") / f"stage_diagnosis_{output_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        seq_name: [
            {
                "frame_index": d.frame_index,
                "gt_exists": d.gt_exists,
                "detector_best_iou": d.detector_best_iou,
                "detector_best_conf": d.detector_best_conf,
                "track_vs_best_detection_iou": d.track_vs_best_detection_iou,
                "post_fsm_iou": d.post_fsm_iou,
                "post_kalman_iou": d.post_kalman_iou,
                "track_active": d.track_active,
                "track_age": d.track_age,
                "predict_only_streak": d.predict_only_streak,
                "matched_tier": d.matched_tier,
                "flag_detector_missed_but_object_visible": d.flag_detector_missed_but_object_visible,
                "flag_fsm_lost_good_detection": d.flag_fsm_lost_good_detection,
                "flag_kalman_diverged_from_fsm": d.flag_kalman_diverged_from_fsm,
                "flag_long_predict_only_streak": d.flag_long_predict_only_streak,
            }
            for d in diags
        ]
        for seq_name, diags in all_diagnostics_by_seq.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n[eval_pipeline_stages] Da luu chan doan chi tiet: {output_path}")


if __name__ == "__main__":
    main()
