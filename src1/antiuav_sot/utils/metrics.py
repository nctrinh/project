# """
# Precision va Success plot theo chuan One-Pass Evaluation (OPE) dung trong
# cac benchmark SOT (OTB, VOT, va Anti-UAV Challenge goc ma bao cao FPGA tham
# chieu). Hai chi so nay danh gia TOAN BO pipeline (detector + motion + FSM +
# Kalman) tren tung sequence, la con so thuc te de tra loi cau hoi "hieu qua
# cao hay khong" thay vi chi phan tich kien truc tren giay.

# Dinh nghia:
#     Precision(threshold):
#         Ty le frame ma khoang cach tam (center distance) giua box du doan
#         va box ground-truth <= threshold (pixel). Precision score chuan =
#         gia tri tai threshold = 20 pixel.

#     Success(threshold):
#         Ty le frame ma IoU giua box du doan va ground-truth >= threshold.
#         Success score chuan (AUC) = dien tich duoi duong cong Success khi
#         threshold quet tu 0 den 1.

# Frame co exist=False trong ground-truth (UAV khong xuat hien) duoc XU LY
# RIENG: neu pipeline cung khong du doan track nao (track_active=False) thi
# tinh la "dung" (implicit true negative theo quy uoc Anti-UAV Challenge);
# neu pipeline van du doan track (false positive) thi tinh la "sai" (khoang
# cach/IoU = gia tri xau nhat).
# """

# from __future__ import annotations

# import dataclasses

# import numpy as np

# from antiuav_sot.utils.box_ops import box_area_xyxy, xywh_to_xyxy


# @dataclasses.dataclass
# class SequenceEvalResult:
#     seq_name: str
#     center_distances: np.ndarray   # (n_frames,) float, np.inf neu khong the so sanh duoc
#     ious: np.ndarray                # (n_frames,) float, 0.0 neu khong the so sanh duoc
#     n_frames: int


# def _center_distance(box_a_xyxy: np.ndarray, box_b_xyxy: np.ndarray) -> float:
#     cx_a = (box_a_xyxy[0] + box_a_xyxy[2]) / 2.0
#     cy_a = (box_a_xyxy[1] + box_a_xyxy[3]) / 2.0
#     cx_b = (box_b_xyxy[0] + box_b_xyxy[2]) / 2.0
#     cy_b = (box_b_xyxy[1] + box_b_xyxy[3]) / 2.0
#     return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


# def _iou_single(box_a_xyxy: np.ndarray, box_b_xyxy: np.ndarray) -> float:
#     x1 = max(box_a_xyxy[0], box_b_xyxy[0])
#     y1 = max(box_a_xyxy[1], box_b_xyxy[1])
#     x2 = min(box_a_xyxy[2], box_b_xyxy[2])
#     y2 = min(box_a_xyxy[3], box_b_xyxy[3])

#     inter_w = max(0.0, x2 - x1)
#     inter_h = max(0.0, y2 - y1)
#     inter_area = inter_w * inter_h

#     area_a = box_area_xyxy(box_a_xyxy[None, :])[0]
#     area_b = box_area_xyxy(box_b_xyxy[None, :])[0]
#     union = area_a + area_b - inter_area

#     if union <= 0:
#         return 0.0
#     return float(inter_area / union)


# def evaluate_sequence(
#     seq_name: str,
#     predicted_boxes_xyxy: list[np.ndarray | None],
#     predicted_active: list[bool],
#     gt_boxes_xywh: list[np.ndarray],
#     gt_exist: list[bool],
# ) -> SequenceEvalResult:
#     """
#     predicted_boxes_xyxy: list do dai n_frames, moi phan tu la (4,) xyxy
#         hoac None neu pipeline khong co track active frame do.
#     predicted_active:     list[bool] do dai n_frames - co track active hay khong.
#     gt_boxes_xywh:        list do dai n_frames, tu IR_label.json (gt_rect).
#     gt_exist:             list[bool] do dai n_frames, tu IR_label.json (exist).
#     """
#     n_frames = len(gt_exist)
#     center_distances = np.full((n_frames,), np.inf, dtype=np.float32)
#     ious = np.zeros((n_frames,), dtype=np.float32)

#     for i in range(n_frames):
#         exists = gt_exist[i]
#         pred_active = predicted_active[i]

#         if not exists and not pred_active:
#             # Ca hai deu "khong co object" - dung, quy uoc distance=0, iou=1
#             center_distances[i] = 0.0
#             ious[i] = 1.0
#             continue

#         if not exists and pred_active:
#             # False positive: object khong ton tai nhung pipeline bao co track.
#             # Giu center_distances = inf, ious = 0.0 (gia tri xau nhat, da khoi tao san).
#             continue

#         if exists and not pred_active:
#             # False negative: object ton tai nhung pipeline khong track duoc.
#             continue

#         # Ca hai deu co object -> so sanh binh thuong.
#         gt_xyxy = xywh_to_xyxy(gt_boxes_xywh[i][None, :])[0]
#         pred_xyxy = predicted_boxes_xyxy[i]
#         center_distances[i] = _center_distance(pred_xyxy, gt_xyxy)
#         ious[i] = _iou_single(pred_xyxy, gt_xyxy)

#     return SequenceEvalResult(
#         seq_name=seq_name,
#         center_distances=center_distances,
#         ious=ious,
#         n_frames=n_frames,
#     )


# def precision_score(
#     results: list[SequenceEvalResult], threshold_px: float = 20.0
# ) -> float:
#     """Ty le frame co center_distance <= threshold_px, gop tren TOAN BO
#     frame cua TOAN BO sequence dua vao (khong trung binh theo tung sequence
#     roi trung binh lan hai - dung chuan OPE: gop frame-level truoc)."""
#     all_distances = np.concatenate([r.center_distances for r in results])
#     valid = np.isfinite(all_distances)
#     if valid.sum() == 0:
#         return 0.0
#     return float((all_distances[valid] <= threshold_px).mean())


# def success_auc(results: list[SequenceEvalResult], n_thresholds: int = 21) -> float:
#     """Area-under-curve cua Success plot, quet threshold IoU tu 0 den 1."""
#     all_ious = np.concatenate([r.ious for r in results])
#     if all_ious.shape[0] == 0:
#         return 0.0

#     thresholds = np.linspace(0, 1, n_thresholds)
#     success_rates = np.array(
#         [(all_ious >= t).mean() for t in thresholds], dtype=np.float32
#     )
#     return float(np.trapz(success_rates, thresholds))


# def success_plot_values(
#     results: list[SequenceEvalResult], n_thresholds: int = 21
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Tra ve (thresholds, success_rates) de ve bieu do Success plot."""
#     all_ious = np.concatenate([r.ious for r in results])
#     thresholds = np.linspace(0, 1, n_thresholds)
#     if all_ious.shape[0] == 0:
#         return thresholds, np.zeros_like(thresholds)
#     success_rates = np.array(
#         [(all_ious >= t).mean() for t in thresholds], dtype=np.float32
#     )
#     return thresholds, success_rates

"""
Cac chi so danh gia theo chuan One-Pass Evaluation (OPE) cho Anti-UAV:
Precision, Normalized Precision, Success (AUC), State Accuracy (SA / mSA),
cung FP rate / FN rate. Danh gia TOAN BO pipeline (detector + motion + FSM +
Kalman) tren tung sequence.

Dinh nghia:
    Precision(threshold):
        Ty le frame ma khoang cach tam giua box du doan va box ground-truth
        <= threshold (pixel). Precision score chuan = gia tri tai 20 pixel.

    Normalized Precision (P_Norm):
        Khoang cach tam duoc chuan hoa theo kich thuoc box GT
        (dx / gt_w, dy / gt_h), roi tinh ty le frame co khoang cach chuan hoa
        <= t. P_Norm AUC = dien tich duoi duong cong khi t quet tu 0 den 0.5
        (chia cho 0.5 de dua ve [0, 1]).

    Success(threshold):
        Ty le frame ma IoU giua box du doan va ground-truth >= threshold.
        Success score chuan (AUC) = dien tich duoi duong cong khi threshold
        quet tu 0 den 1.

    State Accuracy (SA) cua mot sequence (chi so chinh cua Anti-UAV Challenge):
        trung binh tren moi frame cua: IoU neu UAV co mat (exist=1); 1 neu UAV
        vang mat (exist=0) va pipeline cung khong bao track; 0 neu bao nham.
        mSA = trung binh SA tren cac sequence.

Quy uoc frame vang mat (exist=False):
    - pipeline khong bao track -> dung (distance=0, iou=1)
    - pipeline van bao track   -> sai (FP, distance=inf, iou=0)
Vi vay cac chi so "all frames" bi anh huong boi ty le frame vang mat. Cac ham
Precision / Success / P_Norm deu co tuy chon present_only=True de chi tinh tren
frame co UAV (kieu OTB). Frame co UAV ma pipeline bo lo (FN) van nam trong
mau so o ca hai che do.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.utils.box_ops import box_area_xyxy, xywh_to_xyxy

# np.trapz bi doi ten thanh np.trapezoid tu NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclasses.dataclass
class SequenceEvalResult:
    seq_name: str
    center_distances: np.ndarray   # (n_frames,) float, np.inf neu khong the so sanh duoc
    ious: np.ndarray                # (n_frames,) float, 0.0 neu khong the so sanh duoc
    n_frames: int
    # Cac truong moi (mac dinh None de code cu tao SequenceEvalResult truc tiep van chay):
    norm_distances: np.ndarray | None = None   # (n_frames,) khoang cach tam / kich thuoc GT, inf neu khong so sanh duoc
    gt_exist: np.ndarray | None = None          # (n_frames,) bool
    pred_active: np.ndarray | None = None       # (n_frames,) bool

    def __post_init__(self) -> None:
        if self.norm_distances is None:
            self.norm_distances = np.full((self.n_frames,), np.inf, dtype=np.float32)


def _center_distance(box_a_xyxy: np.ndarray, box_b_xyxy: np.ndarray) -> float:
    cx_a = (box_a_xyxy[0] + box_a_xyxy[2]) / 2.0
    cy_a = (box_a_xyxy[1] + box_a_xyxy[3]) / 2.0
    cx_b = (box_b_xyxy[0] + box_b_xyxy[2]) / 2.0
    cy_b = (box_b_xyxy[1] + box_b_xyxy[3]) / 2.0
    return float(np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2))


def _normalized_center_distance(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> float:
    """Khoang cach tam, moi truc duoc chia cho kich thuoc box GT tuong ung."""
    gt_w = max(float(gt_xyxy[2] - gt_xyxy[0]), 1e-6)
    gt_h = max(float(gt_xyxy[3] - gt_xyxy[1]), 1e-6)
    dx = ((pred_xyxy[0] + pred_xyxy[2]) - (gt_xyxy[0] + gt_xyxy[2])) / 2.0
    dy = ((pred_xyxy[1] + pred_xyxy[3]) - (gt_xyxy[1] + gt_xyxy[3])) / 2.0
    return float(np.hypot(dx / gt_w, dy / gt_h))


def _iou_single(box_a_xyxy: np.ndarray, box_b_xyxy: np.ndarray) -> float:
    x1 = max(box_a_xyxy[0], box_b_xyxy[0])
    y1 = max(box_a_xyxy[1], box_b_xyxy[1])
    x2 = min(box_a_xyxy[2], box_b_xyxy[2])
    y2 = min(box_a_xyxy[3], box_b_xyxy[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = box_area_xyxy(box_a_xyxy[None, :])[0]
    area_b = box_area_xyxy(box_b_xyxy[None, :])[0]
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return float(inter_area / union)


def evaluate_sequence(
    seq_name: str,
    predicted_boxes_xyxy: list[np.ndarray | None],
    predicted_active: list[bool],
    gt_boxes_xywh: list[np.ndarray],
    gt_exist: list[bool],
) -> SequenceEvalResult:
    """
    predicted_boxes_xyxy: list do dai n_frames, moi phan tu la (4,) xyxy
        hoac None neu pipeline khong co track active frame do.
    predicted_active:     list[bool] do dai n_frames - co track active hay khong.
    gt_boxes_xywh:        list do dai n_frames, tu IR_label.json (gt_rect).
    gt_exist:             list[bool] do dai n_frames, tu IR_label.json (exist).
    """
    n_frames = len(gt_exist)
    center_distances = np.full((n_frames,), np.inf, dtype=np.float32)
    norm_distances = np.full((n_frames,), np.inf, dtype=np.float32)
    ious = np.zeros((n_frames,), dtype=np.float32)

    for i in range(n_frames):
        exists = bool(gt_exist[i])
        pred_active = bool(predicted_active[i])

        if not exists and not pred_active:
            # Ca hai deu "khong co object" - dung, quy uoc distance=0, iou=1
            center_distances[i] = 0.0
            norm_distances[i] = 0.0
            ious[i] = 1.0
            continue

        if not exists and pred_active:
            # False positive: object khong ton tai nhung pipeline bao co track.
            # Giu distance = inf, iou = 0.0 (gia tri xau nhat, da khoi tao san).
            continue

        if exists and not pred_active:
            # False negative: object ton tai nhung pipeline khong track duoc.
            continue

        # Ca hai deu co object -> so sanh binh thuong.
        gt_xyxy = xywh_to_xyxy(np.asarray(gt_boxes_xywh[i], dtype=np.float32)[None, :])[0]
        pred_xyxy = predicted_boxes_xyxy[i]
        center_distances[i] = _center_distance(pred_xyxy, gt_xyxy)
        norm_distances[i] = _normalized_center_distance(pred_xyxy, gt_xyxy)
        ious[i] = _iou_single(pred_xyxy, gt_xyxy)

    return SequenceEvalResult(
        seq_name=seq_name,
        center_distances=center_distances,
        ious=ious,
        n_frames=n_frames,
        norm_distances=norm_distances,
        gt_exist=np.asarray(gt_exist, dtype=bool),
        pred_active=np.asarray(predicted_active, dtype=bool),
    )


# --------------------------------------------------------------------------
# Gop du lieu o muc frame
# --------------------------------------------------------------------------

def _pool(results: list[SequenceEvalResult], field: str, present_only: bool) -> np.ndarray:
    """Gop mot truong theo frame tren moi sequence. present_only=True chi giu
    frame co UAV (exist=1). Ket qua cu khong co gt_exist coi nhu moi frame deu co UAV."""
    parts = []
    for r in results:
        values = getattr(r, field)
        if present_only and r.gt_exist is not None:
            values = values[r.gt_exist]
        parts.append(values)
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts)


# --------------------------------------------------------------------------
# Precision / Normalized Precision / Success
# --------------------------------------------------------------------------

def precision_score(
    results: list[SequenceEvalResult],
    threshold_px: float = 20.0,
    present_only: bool = False,
) -> float:
    """Ty le frame co center_distance <= threshold_px, gop tren TOAN BO frame
    (khong trung binh theo sequence). Frame FN / FP co distance = inf nen tinh
    la sai va VAN nam trong mau so (inf <= threshold la False)."""
    d = _pool(results, "center_distances", present_only)
    if d.shape[0] == 0:
        return 0.0
    return float((d <= threshold_px).mean())


def norm_precision_plot_values(
    results: list[SequenceEvalResult],
    max_threshold: float = 0.5,
    n_thresholds: int = 51,
    present_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0, max_threshold, n_thresholds)
    d = _pool(results, "norm_distances", present_only)
    if d.shape[0] == 0:
        return thresholds, np.zeros_like(thresholds)
    rates = np.array([(d <= t).mean() for t in thresholds], dtype=np.float32)
    return thresholds, rates


def norm_precision_auc(
    results: list[SequenceEvalResult],
    max_threshold: float = 0.5,
    n_thresholds: int = 51,
    present_only: bool = False,
) -> float:
    """P_Norm AUC: dien tich duoi duong cong Normalized Precision, t in [0, 0.5],
    chia cho max_threshold de dua ve [0, 1]."""
    thresholds, rates = norm_precision_plot_values(
        results, max_threshold, n_thresholds, present_only
    )
    return float(_trapezoid(rates, thresholds) / max_threshold)


def success_plot_values(
    results: list[SequenceEvalResult],
    n_thresholds: int = 21,
    present_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (thresholds, success_rates) de ve bieu do Success plot."""
    thresholds = np.linspace(0, 1, n_thresholds)
    all_ious = _pool(results, "ious", present_only)
    if all_ious.shape[0] == 0:
        return thresholds, np.zeros_like(thresholds)
    success_rates = np.array(
        [(all_ious >= t).mean() for t in thresholds], dtype=np.float32
    )
    return thresholds, success_rates


def success_auc(
    results: list[SequenceEvalResult],
    n_thresholds: int = 21,
    present_only: bool = False,
) -> float:
    """Area-under-curve cua Success plot, quet threshold IoU tu 0 den 1."""
    thresholds, success_rates = success_plot_values(results, n_thresholds, present_only)
    return float(_trapezoid(success_rates, thresholds))


# --------------------------------------------------------------------------
# State Accuracy
# --------------------------------------------------------------------------

def sequence_state_accuracy(result: SequenceEvalResult) -> float:
    """SA cua mot sequence: ious da chua diem cho ca frame vang mat (1 neu dung,
    0 neu bao nham) nen SA chinh la trung binh ious."""
    if result.ious.shape[0] == 0:
        return 0.0
    return float(result.ious.mean())


def state_accuracy(results: list[SequenceEvalResult]) -> float:
    """mSA: trung binh SA tren cac sequence (moi sequence mot trong so)."""
    if not results:
        return 0.0
    return float(np.mean([sequence_state_accuracy(r) for r in results]))


def state_accuracy_pooled(results: list[SequenceEvalResult]) -> float:
    """SA gop theo frame (sequence dai co trong so lon hon). De doi chieu voi mSA."""
    ious = _pool(results, "ious", present_only=False)
    return float(ious.mean()) if ious.shape[0] else 0.0


# --------------------------------------------------------------------------
# FP rate / FN rate
# --------------------------------------------------------------------------

def detection_error_rates(results: list[SequenceEvalResult]) -> dict[str, float | int]:
    """FP rate: ty le frame vang mat (exist=0) ma pipeline van bao track.
    FN rate: ty le frame co UAV (exist=1) ma pipeline khong bao track.
    (FN o day chi la 'khong bao track'; track bao sai cho van tinh vao IoU thap.)
    Mau so bang 0 -> ty le = 0.0 (kem so frame de nguoi doc biet)."""
    n_present = n_absent = n_fn = n_fp = 0
    for r in results:
        if r.gt_exist is None or r.pred_active is None:
            raise ValueError(
                "detection_error_rates can gt_exist va pred_active; "
                "hay tao SequenceEvalResult bang evaluate_sequence()."
            )
        n_present += int(r.gt_exist.sum())
        n_absent += int((~r.gt_exist).sum())
        n_fn += int((r.gt_exist & ~r.pred_active).sum())
        n_fp += int((~r.gt_exist & r.pred_active).sum())
    return {
        "fp_rate_on_absent_frames": (n_fp / n_absent) if n_absent else 0.0,
        "fn_rate_on_present_frames": (n_fn / n_present) if n_present else 0.0,
        "n_present_frames": n_present,
        "n_absent_frames": n_absent,
        "n_false_positive_frames": n_fp,
        "n_false_negative_frames": n_fn,
    }


# --------------------------------------------------------------------------
# Tong hop
# --------------------------------------------------------------------------

def summarize(results: list[SequenceEvalResult]) -> dict:
    """Tat ca chi so tong hop trong mot dict (dung cho in console va JSON)."""
    summary = {
        "n_sequences": len(results),
        "n_frames": int(sum(r.n_frames for r in results)),
        "mSA": state_accuracy(results),
        "SA_pooled": state_accuracy_pooled(results),
        "all_frames": {
            "precision_at_20px": precision_score(results, 20.0),
            "norm_precision_auc": norm_precision_auc(results),
            "success_auc": success_auc(results),
        },
        "present_only": {
            "precision_at_20px": precision_score(results, 20.0, present_only=True),
            "norm_precision_auc": norm_precision_auc(results, present_only=True),
            "success_auc": success_auc(results, present_only=True),
        },
    }
    if all(r.gt_exist is not None and r.pred_active is not None for r in results):
        summary.update(detection_error_rates(results))
    return summary