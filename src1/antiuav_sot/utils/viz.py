"""
Ve box + Track ID len frame de debug truc quan, va xuat video overlay cho
mot sequence. Module nay CHI dung khi chay tren GPU/CPU de kiem tra bang
mat thuong - khong lien quan gi den thiet ke FPGA (khong can port).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

COLOR_GT = (0, 255, 0)          # xanh la - ground truth
COLOR_DETECTOR = (255, 128, 0)   # cam - box tu detector (truoc tracker)
COLOR_TRACK = (0, 0, 255)        # do - track output cuoi cung
COLOR_MOTION = (255, 255, 0)     # cyan (BGR) - candidate tu motion cue
COLOR_DETECTOR_BEST = (0, 255, 255)  # vang (BGR) - box detector diem cao nhat


def draw_box(
    image_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    color: tuple[int, int, int],
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            image_bgr, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return image_bgr


def grayscale_to_bgr(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.dtype != np.uint8:
        gray_image = (np.clip(gray_image, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)


def render_frame(
    raw_frame_gray: np.ndarray,
    gt_box_xywh: np.ndarray | None = None,
    track_boxes_xyxy: np.ndarray | None = None,
    track_active: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    detection_result=None,
    letterbox_scale: float | None = None,
    letterbox_pad: tuple[int, int] | None = None,
    motion_boxes_xyxy: np.ndarray | None = None,
    motion_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Ve mot frame debug day du: motion candidate dua vao FSM (cyan, vien
    manh), ground-truth (xanh la) + cac track active (do, kem Track ID). Tra
    ve anh BGR san sang ghi ra video/anh."""
    from antiuav_sot.utils.box_ops import xywh_to_xyxy

    canvas = grayscale_to_bgr(raw_frame_gray)

    if motion_boxes_xyxy is not None and motion_valid_mask is not None:
        for idx in np.nonzero(motion_valid_mask)[0]:
            draw_box(canvas, motion_boxes_xyxy[idx], COLOR_MOTION, thickness=1)

    if gt_box_xywh is not None and np.any(gt_box_xywh > 0):
        gt_xyxy = xywh_to_xyxy(gt_box_xywh[None, :])[0]
        draw_box(canvas, gt_xyxy, COLOR_GT, label="GT")

    if detection_result is not None and detection_result.valid_mask.any():
        best_idx = int(np.argmax(np.where(detection_result.valid_mask, detection_result.scores, -1)))
        best_box = detection_result.boxes[best_idx].astype(np.float32).copy()
        pad_x, pad_y = letterbox_pad
        best_box[[0, 2]] = (best_box[[0, 2]] - pad_x) / letterbox_scale
        best_box[[1, 3]] = (best_box[[1, 3]] - pad_y) / letterbox_scale
        draw_box(canvas, best_box, COLOR_DETECTOR_BEST, label="DET")
        conf = float(detection_result.scores[best_idx])
        x2, y1 = int(round(best_box[2])), int(round(best_box[1]))
        cv2.putText(
            canvas, f"{conf:.2f}", (x2 + 5, max(0, y1) + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DETECTOR_BEST, 1, cv2.LINE_AA,
        )

    if track_boxes_xyxy is not None and track_active is not None:
        for slot_idx in range(track_boxes_xyxy.shape[0]):
            if not track_active[slot_idx]:
                continue
            tid = int(track_ids[slot_idx]) if track_ids is not None else slot_idx
            draw_box(
                canvas, track_boxes_xyxy[slot_idx], COLOR_TRACK, label=f"ID:{tid}"
            )

    return canvas


class SequenceVideoWriter:
    """Ghi cac frame da ve overlay thanh mot file video .mp4, dung de xem
    lai toan bo sequence sau khi chay eval_tracker.py hoac visualize_sequence.py."""

    def __init__(self, output_path: str | Path, fps: int = 15):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (w, h))
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "SequenceVideoWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()