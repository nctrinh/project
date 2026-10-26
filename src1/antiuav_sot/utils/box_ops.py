"""
Cac phep toan hinh hoc co ban tren bounding box: chuyen doi format, tinh IoU.

Nguyen tac thiet ke (bam sat muc 3.5 bao cao FPGA):
    IoU chi can cong/tru/nhan/chia don gian tren du lieu KICH THUOC CO DINH.
    Moi ham o day deu vector hoa bang numpy, KHONG dung vong lap Python tren
    tung cap box - vi ly do song song hoa nay chinh la ly do khien Simple
    Tracker duoc chon la khoi "phu hop FPGA nhat" trong toan bo pipeline.
    Giu tinh vector-hoa nay cung giup logic anh xa 1-1 sang mot N x M mang
    PE (processing element) khi viet HLS/RTL sau nay.

Box format quy uoc trong toan repo:
    - "xywh": (x_top_left, y_top_left, width, height)  <- format cua IR_label.json
    - "xyxy": (x1, y1, x2, y2)  <- format thuan tien hon de tinh IoU
"""

from __future__ import annotations

import numpy as np

# Gia tri dung de danh dau mot slot box "rong" (khong co object), thay vi
# dung None hay list co do dai thay doi - giu dung nguyen tac fixed-size.
INVALID_BOX = -1.0


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """boxes: (..., 4) dang [x, y, w, h] -> [x1, y1, x2, y2]."""
    boxes = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """boxes: (..., 4) dang [x1, y1, x2, y2] -> [x, y, w, h]."""
    boxes = np.asarray(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    w = x2 - x1
    h = y2 - y1
    return np.stack([x1, y1, w, h], axis=-1)


def box_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Dien tich box dang xyxy. boxes: (..., 4) -> (...,)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    w = np.clip(boxes[..., 2] - boxes[..., 0], a_min=0, a_max=None)
    h = np.clip(boxes[..., 3] - boxes[..., 1], a_min=0, a_max=None)
    return w * h


def iou_matrix_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Tinh ma tran IoU day du giua N box va M box, hoan toan vector hoa.

    boxes_a: (N, 4) dang xyxy
    boxes_b: (M, 4) dang xyxy
    return:  (N, M) ma tran IoU, moi phan tu trong [0, 1]

    Day chinh la "Compare matrix" trong muc 3.5 bao cao: moi cap (track, box)
    duoc tinh doc lap, tuong duong N x M don vi tinh toan song song tren FPGA.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    if boxes_a.ndim != 2 or boxes_a.shape[-1] != 4:
        raise ValueError(f"boxes_a phai co shape (N, 4), nhan duoc {boxes_a.shape}")
    if boxes_b.ndim != 2 or boxes_b.shape[-1] != 4:
        raise ValueError(f"boxes_b phai co shape (M, 4), nhan duoc {boxes_b.shape}")

    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    # Broadcast: (N, 1, 4) vs (1, M, 4) -> (N, M, 4)
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy(boxes_a)[:, None]
    area_b = box_area_xyxy(boxes_b)[None, :]
    union_area = area_a + area_b - inter_area

    # Tranh chia cho 0 khi ca hai box deu suy bien (dien tich = 0)
    iou = np.where(union_area > 0, inter_area / np.maximum(union_area, 1e-9), 0.0)
    return iou.astype(np.float32)


def containment_matrix_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    "Enclosure" / Intersection-over-Smaller-Area (IoS) matrix giua N box va M
    box, vector hoa giong het iou_matrix_xyxy.

    return[i, j] = inter_area(a_i, b_j) / min(area(a_i), area(b_j))

    Khac IoU (chia cho UNION), metric nay chia cho dien tich box NHO HON
    trong cap - nen bang 1.0 khi mot box nam TRON VEN trong box kia, bat ke
    chenh lech kich thuoc lon den dau. Dung de sua Tier-2 IoU Failure: khi
    motion box nho (chi bat vung pixel dang chuyen dong) nam LOT HOAN TOAN
    trong mot detector/track box lon hon (vd vat bi mo/blur khien detector
    tra ve box lon hon vung chuyen dong that), IoU chuan se rat thap (dien
    tich vung giao nho so voi UNION rat lon) va khong khop duoc, trong khi
    containment o day van cho gia tri gan 1.0 - phan anh dung thuc te "box
    nho la MOT PHAN cua cung vat the", khong phai mot vat khac.

    KHONG thay the IoU hoan toan - chi dung ket hop (vd max(iou, containment))
    o nhung noi ro rang mot trong hai box co the la fragment cua box kia
    (motion vs track/detector), tranh lam long nguong IoU chuan cho cac cap
    box thuc su khong lien quan (containment cao gia tao voi box rat nho o
    goc anh, v.v. van bi loai vi cac buoc loc kich thuoc/aspect ratio khac).
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    n = boxes_a.shape[0]
    m = boxes_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy(boxes_a)[:, None]
    area_b = box_area_xyxy(boxes_b)[None, :]
    min_area = np.minimum(area_a, area_b)

    containment = np.where(min_area > 0, inter_area / np.maximum(min_area, 1e-9), 0.0)
    return containment.astype(np.float32)


def iou_vector_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Truong hop dac biet N=1 (Single-Object Tracking, muc 4.2): so mot track
    duy nhat voi M box ung vien -> vector IoU (M,) thay vi ma tran.

    box:   (4,) dang xyxy - track hien tai
    boxes: (M, 4) dang xyxy - cac box ung vien
    return: (M,)
    """
    box = np.asarray(box, dtype=np.float32).reshape(1, 4)
    return iou_matrix_xyxy(box, boxes)[0]


def clip_boxes_xyxy(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Gioi han toa do box trong bien anh [0, width] x [0, height]."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0, width)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, height)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, width)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, height)
    return boxes


def is_valid_box(boxes: np.ndarray) -> np.ndarray:
    """Danh dau slot nao trong mang fixed-size la box that (khong phai padding).

    Quy uoc: mot box "rong"/padding co toa do dau tien = INVALID_BOX (-1).
    boxes: (..., 4) -> (...,) bool
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    return boxes[..., 0] != INVALID_BOX


def make_empty_boxes(n: int) -> np.ndarray:
    """Tao mang box (n, 4) toan bo la INVALID_BOX - dung de khoi tao cac
    slot fixed-size con trong (vi du track slot chua co track, hoac padding
    candidate list cho du max_total_candidates)."""
    return np.full((n, 4), INVALID_BOX, dtype=np.float32)
