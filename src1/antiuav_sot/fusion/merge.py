"""
Merge: ghep candidate tu detector va tu motion cue thanh mot danh sach thong
nhat, kich thuoc co dinh = max_total_candidates (muc 3.4 bao cao FPGA).

Vi ca hai nguon dau vao DA duoc thiet ke voi kich thuoc toi da co dinh
(detector.max_det_boxes va motion.max_motion_candidates), thao tac merge chi
la NOI hai mang co dinh - khong can logic dong, khong can cap phat bo nho
linh hoat tai runtime. Day chinh la ly do nguyen tac "kich thuoc du lieu co
dinh, biet truoc" duoc dat lam nguyen tac thiet ke hang dau.

Moi box sau merge duoc gan mot truong "source" (0 = detector, 1 = motion) vi
Simple Tracker FSM xu ly hai loai nguon nay khac nhau: box tu motion thieu
do tin cay ngu nghia hon nen can nguong IoU khi khop cao hon (tier2 > tier1
trong config).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.utils.box_ops import make_empty_boxes

SOURCE_DETECTOR = 0
SOURCE_MOTION = 1


@dataclasses.dataclass
class CandidateSet:
    """Danh sach candidate box hop nhat, kich thuoc co dinh.

    boxes:       (max_total_candidates, 4) xyxy, INVALID_BOX cho slot rong
    scores:      (max_total_candidates,) float32, confidence (0.0 cho motion box
                 khong co "confidence" ngu nghia that su - xem ghi chu duoi)
    source:      (max_total_candidates,) int32, SOURCE_DETECTOR hoac SOURCE_MOTION
    valid_mask:  (max_total_candidates,) bool, True neu slot la candidate that
    """

    boxes: np.ndarray
    scores: np.ndarray
    source: np.ndarray
    valid_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.valid_mask.sum())


def merge_candidates(
    detector_boxes: np.ndarray,
    detector_scores: np.ndarray,
    detector_valid_mask: np.ndarray,
    motion_boxes: np.ndarray,
    motion_valid_mask: np.ndarray,
    max_total_candidates: int,
) -> CandidateSet:
    """
    Ghep hai nguon candidate thanh mot CandidateSet duy nhat, fixed-size.

    detector_boxes:       (max_det_boxes, 4) xyxy
    detector_scores:      (max_det_boxes,) float32 confidence tu detector
    detector_valid_mask:  (max_det_boxes,) bool
    motion_boxes:         (max_motion_candidates, 4) xyxy
    motion_valid_mask:    (max_motion_candidates,) bool

    Motion box khong co confidence tu mang neural, nen gan score = 0.0 va
    de Simple Tracker FSM phan biet qua truong 'source' thay vi qua score
    (dung nhu bai goc: Motion Boxes Linking la mot TANG rieng, khong tron
    lan voi nguong score cua detector).
    """
    n_det = detector_boxes.shape[0]
    n_mot = motion_boxes.shape[0]
    expected_total = n_det + n_mot

    if expected_total != max_total_candidates:
        raise ValueError(
            f"n_det ({n_det}) + n_mot ({n_mot}) = {expected_total} phai bang "
            f"max_total_candidates ({max_total_candidates}). Kiem tra lai "
            "fpga_constraints.yaml (fusion.max_total_candidates)."
        )

    boxes = make_empty_boxes(max_total_candidates)
    scores = np.zeros((max_total_candidates,), dtype=np.float32)
    source = np.full((max_total_candidates,), SOURCE_DETECTOR, dtype=np.int32)
    valid_mask = np.zeros((max_total_candidates,), dtype=bool)

    boxes[:n_det] = detector_boxes
    scores[:n_det] = detector_scores
    source[:n_det] = SOURCE_DETECTOR
    valid_mask[:n_det] = detector_valid_mask

    boxes[n_det:] = motion_boxes
    scores[n_det:] = 0.0
    source[n_det:] = SOURCE_MOTION
    valid_mask[n_det:] = motion_valid_mask

    return CandidateSet(boxes=boxes, scores=scores, source=source, valid_mask=valid_mask)
