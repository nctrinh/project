"""
motion_eval.py
================
Chay RIENG nhanh Motion (FrameBuffer + gradient_motion_candidates, xem
motion/gradient_candidates.py) tren mot sequence, KHONG dung den detector/
Simple Tracker FSM/Kalman - muc dich la co the chon nguong cho nhanh Motion
(min_box_size, max_box_size, merge_gap_ratio, min_strength...) MOT CACH DOC
LAP voi phan con lai cua pipeline, truoc khi dua vao SotPipeline day du.

Dung chung boi hai script:
    scripts/visualize_motion.py - mot sequence, xuat video debug + in overlap
    scripts/eval_motion.py      - toan bo split, tong hop so lieu + luu JSON

Quy uoc do "overlap voi GT" (khac eval_tracker_motion.py):
    Moi frame CO GT, lay IoU LON NHAT trong so CAC motion box HOP LE (sau loc
    min_strength) voi GT - CHI mot box "tot nhat", KHONG trung binh IoU cua
    nhieu box cung luc. Ly do: nhanh Motion co the tra ve nhieu candidate/
    frame (fragment, nhieu), nhung chi CAN mot trong so do dung vi tri de
    Simple Tracker FSM khop duoc (xem tracker/simple_tracker_fsm.py) - trung
    binh se bi cac candidate nhieu "keo" xuong sai ban chat cau hoi dang hoi.

    Frame VANG MAT GT (exist=False) khong tinh vao overlap (khong co GT de
    so sanh) - nhanh Motion khong co khai niem "false positive" ro rang nhu
    tracker (vat chuyen dong nen/camera rung sinh candidate la binh thuong,
    khong phai loi), nen chi bao cao so box hop le trung binh tren frame
    vang mat de danh gia muc do nhieu (dung de tham khao khi chon nguong).
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from antiuav_sot.motion.frame_buffer import FrameBuffer
from antiuav_sot.motion.gradient_candidates import gradient_motion_candidates
from antiuav_sot.utils.box_ops import iou_vector_xyxy, make_empty_boxes, xywh_to_xyxy
from antiuav_sot.utils.viz import COLOR_GT, COLOR_MOTION, COLOR_TRACK, draw_box, grayscale_to_bgr

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = (0.1, 0.3, 0.5)


@dataclasses.dataclass
class MotionSearchParams:
    """Tat ca tham so dieu khien nhanh Motion, gom lai mot cho de
    visualize_motion.py / eval_motion.py cung xay dung tu argparse + fallback
    tu FpgaConstraints roi truyen nguyen vao run_motion_on_sequence() - tranh
    hai script phai khai bao lai cung mot danh sach tham so dai."""

    max_motion_candidates: int
    min_box_size: int
    max_box_size: int
    merge_gap_ratio: float
    max_aspect_ratio: float
    merge_min_fill_ratio: float
    model: str
    irls_iters: int
    huber_delta: float
    pyramid_levels: int
    thresh_factor: float
    min_area: int
    strength_metric: str   # "sum" / "mean" / "contrast" - xem gradient_candidates.py
    min_strength: float    # -inf = khong loc
    frame_lag: int         # FrameBuffer lag - xem motion/frame_buffer.py

    @classmethod
    def from_constraints(cls, fpga, **overrides) -> "MotionSearchParams":
        """Gia tri mac dinh lay tu khoi motion cua fpga_constraints.yaml - CUNG
        nguon voi SotPipeline, nen nguong chon o day ap vao pipeline chi bang
        cach ghi lai vao yaml. overrides co gia tri None bi bo qua."""
        params = cls(
            max_motion_candidates=fpga.max_motion_candidates,
            min_box_size=fpga.motion_min_box_size,
            max_box_size=fpga.motion_max_box_size,
            merge_gap_ratio=fpga.motion_merge_gap_ratio,
            max_aspect_ratio=fpga.motion_max_aspect_ratio,
            merge_min_fill_ratio=fpga.motion_merge_min_fill_ratio,
            model=fpga.motion_model,
            irls_iters=fpga.motion_irls_iters,
            huber_delta=fpga.motion_huber_delta,
            pyramid_levels=fpga.motion_pyramid_levels,
            thresh_factor=fpga.motion_thresh_factor,
            min_area=fpga.motion_min_area,
            strength_metric=fpga.motion_strength_metric,
            min_strength=fpga.motion_min_strength,
            frame_lag=fpga.motion_frame_lag,
        )
        return dataclasses.replace(
            params, **{k: v for k, v in overrides.items() if v is not None}
        )


@dataclasses.dataclass
class MotionFrameResult:
    frame_index: int
    gt_exists: bool
    n_valid_boxes: int
    best_iou: float                    # 0.0 neu gt_exists=False hoac khong co box hop le nao
    best_strength: float | None        # strength cua box dat best_iou, None neu khong co


@dataclasses.dataclass
class SequenceMotionEvalResult:
    seq_name: str
    frame_results: list[MotionFrameResult]
    n_frames: int
    n_warmup_skipped: int   # so frame dau sequence chua du lich su (FrameBuffer.has_previous()=False)


def render_motion_debug_frame(
    raw_frame_gray: np.ndarray,
    gt_box_xywh: np.ndarray | None,
    motion_boxes_xyxy: np.ndarray,
    motion_valid_mask: np.ndarray,
    motion_strengths: np.ndarray,
    best_candidate_idx: int | None,
) -> np.ndarray:
    """Ve GT (xanh la) + tat ca motion box hop le (vang, kem strength), rieng
    box dang khop GT tot nhat frame nay (neu co) ve mau do de de nhan ra."""
    canvas = grayscale_to_bgr(raw_frame_gray)

    if gt_box_xywh is not None and np.any(gt_box_xywh > 0):
        gt_xyxy = xywh_to_xyxy(gt_box_xywh[None, :])[0]
        draw_box(canvas, gt_xyxy, COLOR_GT, label="GT")

    for idx in np.nonzero(motion_valid_mask)[0]:
        is_best = idx == best_candidate_idx
        color = COLOR_TRACK if is_best else COLOR_MOTION
        label = f"s={motion_strengths[idx]:.1f}"
        draw_box(canvas, motion_boxes_xyxy[idx], color, label=label)

    return canvas


def run_motion_on_sequence(
    seq,
    params: MotionSearchParams,
    video_writer=None,
) -> SequenceMotionEvalResult:
    """Chay nhanh Motion tren tung frame cua mot sequence (giong thu tu goi
    trong SotPipeline.step(), xem pipeline/sot_pipeline.py), tach khoi phan
    con lai cua pipeline. video_writer (tuy chon) la mot
    antiuav_sot.utils.viz.SequenceVideoWriter - neu truyen vao, moi frame
    (ke ca frame warm-up chua co motion candidate) deu duoc ghi ra video de
    giu dong bo thoi gian voi sequence goc."""
    frame_buffer = FrameBuffer(lag=params.frame_lag)
    frame_results: list[MotionFrameResult] = []
    n_warmup_skipped = 0

    for sample in seq:
        raw_frame = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if raw_frame is None:
            print(f"[motion_eval] CANH BAO: khong doc duoc {sample.image_path}, bo qua frame.")
            continue

        has_motion_stage = frame_buffer.has_previous()
        if has_motion_stage:
            prev_frame = frame_buffer.get_previous()
            boxes, valid_mask, strengths = gradient_motion_candidates(
                frame_t=raw_frame,
                frame_t_minus_1=prev_frame,
                max_motion_candidates=params.max_motion_candidates,
                min_box_size=params.min_box_size,
                max_box_size=params.max_box_size,
                merge_gap_ratio=params.merge_gap_ratio,
                max_aspect_ratio=params.max_aspect_ratio,
                merge_min_fill_ratio=params.merge_min_fill_ratio,
                priority_boxes=None,
                model=params.model,
                irls_iters=params.irls_iters,
                huber_delta=params.huber_delta,
                pyramid_levels=params.pyramid_levels,
                thresh_factor=params.thresh_factor,
                min_area=params.min_area,
                strength_metric=params.strength_metric,
                min_strength=params.min_strength,
            )
        else:
            boxes = make_empty_boxes(params.max_motion_candidates)
            valid_mask = np.zeros((params.max_motion_candidates,), dtype=bool)
            strengths = np.zeros((params.max_motion_candidates,), dtype=np.float32)

        frame_buffer.update(raw_frame)

        gt_exists = bool(sample.exist)
        best_iou = 0.0
        best_strength: float | None = None
        best_idx: int | None = None

        if has_motion_stage and gt_exists:
            valid_idx = np.nonzero(valid_mask)[0]
            if valid_idx.shape[0] > 0:
                gt_xyxy = xywh_to_xyxy(
                    np.asarray(sample.gt_box_xywh, dtype=np.float32)[None, :]
                )[0]
                ious = iou_vector_xyxy(gt_xyxy, boxes[valid_idx])
                local_best = int(np.argmax(ious))
                candidate_best_iou = float(ious[local_best])
                # np.argmax tra ve mot index NGAY CA KHI moi IoU = 0 (khong box
                # nao thuc su cham GT) - chi coi la "best match" thuc su (va to
                # do khi ve) neu IoU > 0, tranh highlight nham mot box khong
                # overlap GT chi vi no la "lon nhat trong cac so 0".
                if candidate_best_iou > 0.0:
                    best_idx = int(valid_idx[local_best])
                    best_iou = candidate_best_iou
                    best_strength = float(strengths[best_idx])

        if has_motion_stage:
            frame_results.append(
                MotionFrameResult(
                    frame_index=sample.frame_index,
                    gt_exists=gt_exists,
                    n_valid_boxes=int(valid_mask.sum()),
                    best_iou=best_iou,
                    best_strength=best_strength,
                )
            )
        else:
            n_warmup_skipped += 1

        if video_writer is not None:
            frame_bgr = render_motion_debug_frame(
                raw_frame,
                gt_box_xywh=sample.gt_box_xywh if gt_exists else None,
                motion_boxes_xyxy=boxes,
                motion_valid_mask=valid_mask,
                motion_strengths=strengths,
                best_candidate_idx=best_idx,
            )
            video_writer.write(frame_bgr)

    if video_writer is not None:
        video_writer.close()

    return SequenceMotionEvalResult(
        seq_name=seq.seq_name,
        frame_results=frame_results,
        n_frames=len(seq),
        n_warmup_skipped=n_warmup_skipped,
    )


def summarize_motion_results(
    results: list[SequenceMotionEvalResult],
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
) -> dict:
    """Tong hop nhieu SequenceMotionEvalResult thanh mot dict (in console/luu
    JSON). coverage_at_iou_<t>: ty le frame CO GT ma best_iou >= t (giong
    tinh than Success plot cua eval_tracker_motion.py, nhung chi tren nhanh
    Motion doc lap, va dung best_iou - mot box tot nhat - khong phai IoU cua
    track cuoi cung)."""
    present = [fr for r in results for fr in r.frame_results if fr.gt_exists]
    absent = [fr for r in results for fr in r.frame_results if not fr.gt_exists]

    best_ious = np.array([fr.best_iou for fr in present], dtype=np.float32)
    coverage = {
        f"coverage_at_iou_{t:g}": float((best_ious >= t).mean()) if best_ious.size else 0.0
        for t in iou_thresholds
    }

    return {
        "n_sequences": len(results),
        "n_frames_total": int(sum(r.n_frames for r in results)),
        "n_frames_warmup_skipped": int(sum(r.n_warmup_skipped for r in results)),
        "n_present_frames": len(present),
        "n_absent_frames": len(absent),
        "mean_best_iou_present": float(best_ious.mean()) if best_ious.size else 0.0,
        **coverage,
        "avg_valid_motion_boxes_present": (
            float(np.mean([fr.n_valid_boxes for fr in present])) if present else 0.0
        ),
        "avg_valid_motion_boxes_absent": (
            float(np.mean([fr.n_valid_boxes for fr in absent])) if absent else 0.0
        ),
    }