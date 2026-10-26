"""
Wrapper mong tren antiuav_sot.utils.box_ops, chuyen biet cho ngu canh tracker:
tinh IoU giua cac TRACK SLOT dang active va cac CANDIDATE BOX dang valid,
dong thoi ap dung mask de loai bo cap lien quan slot/candidate rong (INVALID_BOX)
ma khong can if/else phan tan khap noi trong FSM.

Voi n_track_slots = 1 (Single-Object Tracking, muc 4.2), ham nay tra ve
mot ma tran (1, M) - ve ban chat chinh la vector 1xM ma bao cao mo ta, chi
khac o cho khong can nhanh rieng cho truong hop N=1: cung mot code path
dung cho ca N=1 va N>1, dung tinh than "tham so hoa duoc" cua muc 4.5.
"""

from __future__ import annotations

import numpy as np

from antiuav_sot.utils.box_ops import containment_matrix_xyxy, iou_matrix_xyxy


def compute_masked_iou_matrix(
    track_boxes: np.ndarray,
    track_active_mask: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_valid_mask: np.ndarray,
) -> np.ndarray:
    """
    track_boxes:           (N, 4) xyxy - toan bo N khe (ca active lan trong)
    track_active_mask:     (N,) bool
    candidate_boxes:       (M, 4) xyxy - toan bo M candidate (ca valid lan padding)
    candidate_valid_mask:  (M,) bool

    return: (N, M) IoU matrix, voi gia tri = 0.0 tai moi cap (i, j) ma
            track i khong active HOAC candidate j khong valid - dam bao FSM
            khong bao gio "khop nham" vao mot slot/candidate rong.
    """
    iou = iou_matrix_xyxy(track_boxes, candidate_boxes)  # (N, M)

    # Zero-out hang ung voi track khong active
    iou = iou * track_active_mask[:, None].astype(np.float32)
    # Zero-out cot ung voi candidate khong valid
    iou = iou * candidate_valid_mask[None, :].astype(np.float32)

    return iou


def compute_masked_containment_matrix(
    track_boxes: np.ndarray,
    track_active_mask: np.ndarray,
    candidate_boxes: np.ndarray,
    candidate_valid_mask: np.ndarray,
) -> np.ndarray:
    """Ban sao cua compute_masked_iou_matrix nhung dung containment_matrix_xyxy
    (Intersection-over-Smaller-Area) thay vi IoU chuan - xem box_ops.py ve ly
    do can metric nay (sua Tier-2 IoU Failure: box motion nho lot hoan toan
    trong track/detector box lon hon van phai duoc coi la cung mot vat)."""
    containment = containment_matrix_xyxy(track_boxes, candidate_boxes)

    containment = containment * track_active_mask[:, None].astype(np.float32)
    containment = containment * candidate_valid_mask[None, :].astype(np.float32)

    return containment
