#!/usr/bin/env python3
"""
Danh gia RIENG detector - hoan toan doc lap voi motion cue, fusion, Simple
Tracker FSM, Kalman. Tra loi cau hoi "detector TU NO co bat trung muc tieu
hay khong", tach bach voi "pipeline SAU detector co giu duoc muc tieu hay
khong" (cau hoi do thuoc ve scripts/eval_pipeline_stages.py).

Khac voi scripts/eval_tracker.py (chay full pipeline, chi bao cao 1 box
cuoi cung/frame do la SOT N=1), script nay:
  - Chay detector.predict() TRUC TIEP tren tung frame, lay TOAN BO box sau
    NMS (khong gioi han 1 track).
  - Tinh Precision/Recall/AP theo chuan detection (IoU >= threshold la
    true positive), khong phai Precision/Success kieu OPE cua SOT.
  - Bao cao rieng cho frame CO object (exist=1) va frame KHONG co object
    (exist=0) - vi false positive tren frame rong la mot loai loi khac
    han voi bo sot muc tieu tren frame co object.

Cach dung:
    python scripts/eval_detector.py --config configs/default.yaml \
        --manifest experiments/data_manifest.json --split val \
        --checkpoint experiments/detector/weights/best.pt
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
from antiuav_sot.data.transforms import letterbox_grayscale, normalize_grayscale
from antiuav_sot.detector.model import LightweightYoloDetector
from antiuav_sot.utils.box_ops import xywh_to_xyxy
from antiuav_sot.utils.config import load_config


def rescale_boxes_from_letterbox(boxes_xyxy: np.ndarray, letterbox) -> np.ndarray:
    """Tuong tu antiuav_sot.pipeline.sot_pipeline._rescale_boxes_from_letterbox,
    lap lai o day de script nay khong phu thuoc vao pipeline (giu tinh doc lap
    hoan toan voi tracker)."""
    from antiuav_sot.utils.box_ops import INVALID_BOX

    boxes = boxes_xyxy.copy()
    valid = boxes[:, 0] != INVALID_BOX
    boxes[valid, 0] = (boxes[valid, 0] - letterbox.pad_left) / letterbox.scale
    boxes[valid, 1] = (boxes[valid, 1] - letterbox.pad_top) / letterbox.scale
    boxes[valid, 2] = (boxes[valid, 2] - letterbox.pad_left) / letterbox.scale
    boxes[valid, 3] = (boxes[valid, 3] - letterbox.pad_top) / letterbox.scale
    return boxes


def iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def evaluate_detector_on_sequence(
    detector: LightweightYoloDetector,
    seq,
    det_input_height: int,
    det_input_width: int,
    iou_threshold: float,
) -> dict:
    """
    Vi Anti-UAV chi co 1 object/frame (Single-Object), "true positive" duoc
    dinh nghia don gian: trong so cac detector box, box co IoU CAO NHAT voi
    GT co vuot iou_threshold hay khong. Cac box con lai (neu co) duoc tinh
    la false positive - dung de phat hien detector co "nham lam nhieu vat"
    (vd nhieu block nhiet nen bi nham UAV) hay khong.
    """
    n_tp = 0            # frame co object, detector bat trung (IoU >= threshold)
    n_fn = 0            # frame co object, detector KHONG bat trung (miss hoan toan hoac IoU thap)
    n_fp_on_object_frame = 0   # frame co object: SO LUONG box thua (khong phai box tot nhat)
    n_fp_on_empty_frame = 0    # frame KHONG co object nhung detector van ra box (false alarm)
    n_true_negative = 0        # frame khong co object, detector cung khong ra box nao - dung

    iou_values_on_object_frames: list[float] = []
    conf_values_at_best_iou: list[float] = []

    for sample in seq:
        raw_frame = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            continue

        letterbox = letterbox_grayscale(raw_frame, det_input_height, det_input_width)
        det_input = normalize_grayscale(letterbox.image)
        result = detector.predict(det_input)

        boxes_orig = rescale_boxes_from_letterbox(result.boxes, letterbox)
        valid_indices = np.nonzero(result.valid_mask)[0]

        if not sample.exist:
            if valid_indices.shape[0] == 0:
                n_true_negative += 1
            else:
                n_fp_on_empty_frame += valid_indices.shape[0]
            continue

        gt_xyxy = xywh_to_xyxy(sample.gt_box_xywh[None, :])[0]

        if valid_indices.shape[0] == 0:
            n_fn += 1
            continue

        ious = np.array([iou_single(gt_xyxy, boxes_orig[i]) for i in valid_indices])
        best_local_idx = int(np.argmax(ious))
        best_iou = float(ious[best_local_idx])

        iou_values_on_object_frames.append(best_iou)
        conf_values_at_best_iou.append(float(result.scores[valid_indices[best_local_idx]]))

        if best_iou >= iou_threshold:
            n_tp += 1
        else:
            n_fn += 1

        # Moi box KHAC box tot nhat tren mot frame co object duoc tinh la FP
        # (gia dinh N=1 object that su trong frame).
        n_fp_on_object_frame += valid_indices.shape[0] - 1

    n_object_frames = n_tp + n_fn

    return {
        "seq_name": seq.seq_name,
        "n_object_frames": n_object_frames,
        "n_empty_frames_true_negative": n_true_negative,
        "n_tp": n_tp,
        "n_fn": n_fn,
        "n_fp_on_object_frame": n_fp_on_object_frame,
        "n_fp_on_empty_frame": n_fp_on_empty_frame,
        "recall": n_tp / n_object_frames if n_object_frames > 0 else None,
        "mean_iou_on_object_frames": (
            float(np.mean(iou_values_on_object_frames)) if iou_values_on_object_frames else None
        ),
        "mean_conf_at_best_iou": (
            float(np.mean(conf_values_at_best_iou)) if conf_values_at_best_iou else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--manifest", type=str, default="experiments/data_manifest.json")
    parser.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence", type=str, default=None)
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

    # FIX: FpgaConstraints (config.py) luu cac truong nay voi tien to "det_"
    # (det_conf_threshold, det_nms_iou_threshold) - KHONG phai conf_threshold/
    # nms_iou_threshold nhu ban goc cua script nay dang goi. Ban goc se raise
    # AttributeError ngay khi tao Config vi FpgaConstraints la frozen dataclass
    # khong co field ten do. Sua lai dung ten field.
    detector = LightweightYoloDetector(
        weights_path=args.checkpoint,
        max_det_boxes=cfg.fpga.max_det_boxes,
        conf_threshold=cfg.fpga.det_conf_threshold,
        nms_iou_threshold=cfg.fpga.det_nms_iou_threshold,
        device=cfg.get("tracker_eval", "device", default="cuda:0"),
    )

    print(f"[eval_detector] Danh gia detector tren {len(seq_names)} sequence "
          f"(split={args.split}), iou_threshold={args.iou_threshold}")
    print(f"[eval_detector] conf_threshold={cfg.fpga.det_conf_threshold}, "
          f"nms_iou_threshold={cfg.fpga.det_nms_iou_threshold}, "
          f"max_det_boxes={cfg.fpga.max_det_boxes}")

    per_seq_results = []
    for seq_name in tqdm(seq_names, desc="eval detector"):
        seq = dataset.load_sequence(seq_name)
        result = evaluate_detector_on_sequence(
            detector, seq,
            det_input_height=cfg.fpga.det_input_height,
            det_input_width=cfg.fpga.det_input_width,
            iou_threshold=args.iou_threshold,
        )
        per_seq_results.append(result)

    total_object_frames = sum(r["n_object_frames"] for r in per_seq_results)
    total_tp = sum(r["n_tp"] for r in per_seq_results)
    total_fn = sum(r["n_fn"] for r in per_seq_results)
    total_fp_object = sum(r["n_fp_on_object_frame"] for r in per_seq_results)
    total_fp_empty = sum(r["n_fp_on_empty_frame"] for r in per_seq_results)
    total_true_negative = sum(r["n_empty_frames_true_negative"] for r in per_seq_results)

    all_ious = [
        r["mean_iou_on_object_frames"] for r in per_seq_results
        if r["mean_iou_on_object_frames"] is not None
    ]

    overall_recall = total_tp / total_object_frames if total_object_frames > 0 else 0.0
    overall_precision = (
        total_tp / (total_tp + total_fp_object)
        if (total_tp + total_fp_object) > 0 else 0.0
    )

    print("\n[eval_detector] ==== KET QUA DETECTOR (DOC LAP VOI TRACKER) ====")
    print(f"[eval_detector] So frame co object: {total_object_frames}")
    print(f"[eval_detector]   True Positive (IoU >= {args.iou_threshold}): {total_tp}")
    print(f"[eval_detector]   False Negative (bo sot/IoU thap): {total_fn}")
    print(f"[eval_detector]   False Positive thua (nhieu box hon 1 vat): {total_fp_object}")
    print(f"[eval_detector] Recall tren frame co object: {overall_recall:.4f}")
    print(f"[eval_detector] Precision (tren so box da xuat): {overall_precision:.4f}")
    if all_ious:
        print(f"[eval_detector] Mean IoU (best box/frame, chi frame co object): "
              f"{np.mean(all_ious):.4f}")
    else:
        print("[eval_detector] Mean IoU: N/A (khong co frame nao tinh duoc)")
    print(f"[eval_detector] So frame KHONG co object: "
          f"{total_true_negative + total_fp_empty}")
    print(f"[eval_detector]   True Negative (dung, khong ra box): {total_true_negative}")
    print(f"[eval_detector]   False Positive tren frame rong (bao dong gia): {total_fp_empty}")

    output_path = Path("experiments") / f"eval_detector_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "iou_threshold": args.iou_threshold,
                "overall_recall": overall_recall,
                "overall_precision": overall_precision,
                "mean_iou_on_object_frames": float(np.mean(all_ious)) if all_ious else None,
                "total_true_negative": total_true_negative,
                "total_fp_on_empty_frame": total_fp_empty,
                "per_sequence": per_seq_results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"[eval_detector] Da luu ket qua chi tiet: {output_path}")


if __name__ == "__main__":
    main()