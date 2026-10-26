"""
Chuan hoa dau ra detector (sau NMS, so luong box THAY DOI tuy anh) ve dung
kich thuoc co dinh max_det_boxes - buoc bat buoc de tuong thich voi
fusion/merge.py (muc 3.3, 3.4 bao cao FPGA: "Dau ra: danh sach box da qua
NMS, kich thuoc toi da co dinh").

NMS ban than no (loai bo box trung lap) van chay VOI SO LUONG BOX DAU VAO
BIEN DOI - day la buoc dung Ultralytics/OpenCV tren GPU trong giai doan
train/eval. Khi port sang FPGA, NMS on-chip se duoc trien khai bang cau
truc ma tran so sanh tuong tu Simple Tracker (muc 3.3 ghi ro dieu nay: "NMS
on-chip co the trien khai bang cau truc ma tran so sanh tuong tu nhu o buoc
Simple Tracker"). Ham pad_to_fixed_size() o day CHI xu ly buoc chuan hoa
shape SAU KHI NMS da chay xong, khong tu trien khai lai NMS.
"""

from __future__ import annotations

import numpy as np

from antiuav_sot.utils.box_ops import make_empty_boxes


def pad_to_fixed_size(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    max_det_boxes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    boxes_xyxy: (K, 4) - K box that su sau NMS, K co the tu 0 den bat ky gia tri nao
    scores:     (K,)

    Neu K > max_det_boxes: giu lai max_det_boxes box co score CAO NHAT (NMS
        that su hiem khi de lai qua nhieu box trung lap nen truong hop nay
        it xay ra trong thuc te, nhung van can xu ly an toan).
    Neu K < max_det_boxes: pad them (max_det_boxes - K) slot INVALID_BOX,
        score = 0.0, valid_mask = False.

    return: (boxes (max_det_boxes,4), scores (max_det_boxes,), valid_mask (max_det_boxes,))
    """
    k = boxes_xyxy.shape[0]

    if k > max_det_boxes:
        top_k_indices = np.argsort(-scores)[:max_det_boxes]
        boxes_xyxy = boxes_xyxy[top_k_indices]
        scores = scores[top_k_indices]
        k = max_det_boxes

    out_boxes = make_empty_boxes(max_det_boxes)
    out_scores = np.zeros((max_det_boxes,), dtype=np.float32)
    valid_mask = np.zeros((max_det_boxes,), dtype=bool)

    out_boxes[:k] = boxes_xyxy
    out_scores[:k] = scores
    valid_mask[:k] = True

    return out_boxes, out_scores, valid_mask
