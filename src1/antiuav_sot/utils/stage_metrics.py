"""
Metric CHAN DOAN theo TUNG GIAI DOAN cua pipeline, khac voi utils/metrics.py
(chi danh gia dau ra CUOI CUNG cua toan bo pipeline).

Muc dich: khi quan sat thay "detector bat trung o frame dau nhung track
sau do bi lech dan" (hien tuong nguoi dung bao cao), can biet CHINH XAC
loi phat sinh o buoc nao:

    (a) Detector tu no da mat/lech muc tieu (khong lien quan FSM/Kalman)?
    (b) Detector van dung, nhung FSM khong khop duoc detection voi track
        (vd track da bi Kalman keo lech truoc do, IoU voi detection that
        rot xuong duoi ca nguong tier4)?
    (c) FSM khop dung, nhung Kalman lam muot (smooth) lam lech vi tri so
        voi quan sat that (kalman qua "cham" dieu chinh)?
    (d) Track dang chay o che do "predict-only" (Kalman du bao thuan, khong
        co quan sat that) qua nhieu frame lien tiep - day thuong la NGUYEN
        NHAN GOC neu (b) xay ra ngay sau do.

Module nay so sanh GT voi TUNG giai doan MOT CACH DOC LAP tren cung
mot frame, roi de scripts/eval_pipeline_stages.py tong hop thanh bao cao.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.utils.box_ops import INVALID_BOX, box_area_xyxy, xywh_to_xyxy


@dataclasses.dataclass
class FrameStageDiagnostic:
    """Chan doan cho MOT frame: IoU/khoang cach GT so voi tung giai doan,
    cong them cac co (flag) giai thich TAI SAO neu co bat thuong.

    Gia tri IoU/distance = None neu khong the tinh (vd GT khong ton tai,
    hoac giai doan do khong co box hop le) - PHAN BIET RO voi 0.0 (co box
    nhung sai hoan toan), tranh nham lan "khong co du lieu" voi "sai".
    """

    frame_index: int
    gt_exists: bool

    detector_best_iou: float | None       # IoU cao nhat giua GT va TOAN BO detector box (truoc FSM)
    detector_best_conf: float | None       # confidence cua detector box co IoU cao nhat do
    track_vs_best_detection_iou: float | None  # IoU giua TRACK BOX HIEN TAI (truoc khi cap nhat
                                                 # frame nay) va detector box gan GT nhat - day CHINH
                                                 # LA gia tri FSM thuc su dung de quyet dinh khop/khong
                                                 # khop (khac voi detector_best_iou, von chi so voi GT).
                                                 # Neu gia tri nay THAP trong khi detector_best_iou CAO,
                                                 # nghia la detector dung nhung track (dang o vi tri cu)
                                                 # da qua xa detection moi de bat ky tier nao khop duoc -
                                                 # KHONG phai loi cua FSM/Kalman, ma la gioi han von co
                                                 # cua "chi dung IoU, khong dung appearance" khi vat the
                                                 # di chuyen nhanh giua 2 frame.
    post_fsm_iou: float | None             # IoU giua GT va track box SAU FSM (truoc Kalman), slot 0
    post_kalman_iou: float | None          # IoU giua GT va track box SAU Kalman, slot 0

    track_active: bool
    track_age: int                          # so frame lien tiep KHONG khop (tu TrackStateTable)
    predict_only_streak: int                # so frame lien tiep dang chi dua Kalman du bao

    matched_tier: int                       # tier FSM da khop track nay frame nay (0 = khong khop)

    # Co canh bao suy ra tu cac gia tri tren, tinh san de scripts khong
    # phai tu suy luan lai logic nguong.
    flag_detector_missed_but_object_visible: bool  # GT ton tai, detector_best_iou thap (object van la
                                                     # nhung detector khong bat duoc GAN vi tri GT)
    flag_fsm_lost_good_detection: bool             # detector_best_iou cao NHUNG post_fsm_iou thap
                                                     # (FSM khong khop duoc detection tot vao track)
    flag_kalman_diverged_from_fsm: bool            # post_fsm_iou cao NHUNG post_kalman_iou thap
                                                     # (Kalman lam lech so voi vi tri FSM vua khop)
    flag_long_predict_only_streak: bool            # predict_only_streak vuot nguong canh bao


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


def diagnose_frame(
    frame_index: int,
    gt_exists: bool,
    gt_box_xywh: np.ndarray,
    detector_boxes_xyxy: np.ndarray,
    detector_scores: np.ndarray,
    detector_valid_mask: np.ndarray,
    track_box_before_this_frame_xyxy: np.ndarray | None,
    post_fsm_box_xyxy: np.ndarray,
    post_fsm_active: bool,
    post_kalman_box_xyxy: np.ndarray,
    track_age: int,
    predict_only_streak: int,
    matched_tier: int,
    detector_missed_threshold: float = 0.3,
    fsm_lost_threshold_gap: float = 0.3,
    kalman_diverge_threshold_gap: float = 0.2,
    long_predict_only_threshold: int = 2,
) -> FrameStageDiagnostic:
    """
    Chan doan MOT frame. Cac tham so *_boxes*/*_box*/*_valid_mask* deu la
    dau ra da co san tu antiuav_sot.pipeline.sot_pipeline.PipelineFrameOutput
    + antiuav_sot.data.dataset.SequenceSample cho frame do - ham nay KHONG
    tu chay lai pipeline, chi so sanh/suy luan tu ket qua da co.

    detector_boxes_xyxy/scores/valid_mask: TOAN BO candidate tu detector
        (chua qua FSM) - dung de biet detector TU NO co thay object hay
        khong, doc lap voi viec FSM co khop duoc hay khong.
    """
    if not gt_exists:
        return FrameStageDiagnostic(
            frame_index=frame_index,
            gt_exists=False,
            detector_best_iou=None,
            detector_best_conf=None,
            track_vs_best_detection_iou=None,
            post_fsm_iou=None,
            post_kalman_iou=None,
            track_active=post_fsm_active,
            track_age=track_age,
            predict_only_streak=predict_only_streak,
            matched_tier=matched_tier,
            flag_detector_missed_but_object_visible=False,
            flag_fsm_lost_good_detection=False,
            flag_kalman_diverged_from_fsm=False,
            flag_long_predict_only_streak=predict_only_streak >= long_predict_only_threshold,
        )

    gt_xyxy = xywh_to_xyxy(gt_box_xywh[None, :])[0]

    # ---- Giai doan 1: Detector tho (truoc FSM) ----
    valid_indices = np.nonzero(detector_valid_mask)[0]
    detector_best_iou: float | None = None
    detector_best_conf: float | None = None
    best_detection_box: np.ndarray | None = None
    if valid_indices.shape[0] > 0:
        ious = np.array(
            [_iou_single(gt_xyxy, detector_boxes_xyxy[i]) for i in valid_indices]
        )
        best_local_idx = int(np.argmax(ious))
        detector_best_iou = float(ious[best_local_idx])
        detector_best_conf = float(detector_scores[valid_indices[best_local_idx]])
        best_detection_box = detector_boxes_xyxy[valid_indices[best_local_idx]]

    # ---- IoU giua TRACK (truoc khi cap nhat frame nay) va detection tot nhat ----
    # Day la gia tri Simple Tracker FSM THUC SU dung de quyet dinh khop/khong
    # khop o frame nay - khac han detector_best_iou (chi so voi GT). Neu
    # track_vs_best_detection_iou THAP trong khi detector_best_iou CAO, nghia
    # la detector dung nhung track (dang o vi tri CU, tu frame truoc) da qua
    # xa detection MOI (vd vat the di chuyen nhanh) de bat ky tier IoU nao
    # khop duoc - day la gioi han von co cua "chi dung IoU thuan tuy, khong
    # dung appearance/template matching nhu SOT", KHONG phai loi cua Kalman.
    track_vs_best_detection_iou: float | None = None
    if (
        best_detection_box is not None
        and track_box_before_this_frame_xyxy is not None
        and track_box_before_this_frame_xyxy[0] != INVALID_BOX
    ):
        track_vs_best_detection_iou = _iou_single(
            track_box_before_this_frame_xyxy, best_detection_box
        )

    # ---- Giai doan 2: Sau FSM (truoc Kalman) ----
    post_fsm_iou: float | None = None
    if post_fsm_active and post_fsm_box_xyxy[0] != INVALID_BOX:
        post_fsm_iou = _iou_single(gt_xyxy, post_fsm_box_xyxy)

    # ---- Giai doan 3: Sau Kalman ----
    post_kalman_iou: float | None = None
    if post_fsm_active and post_kalman_box_xyxy[0] != INVALID_BOX:
        post_kalman_iou = _iou_single(gt_xyxy, post_kalman_box_xyxy)

    # ---- Suy luan co canh bao ----
    flag_detector_missed = (
        detector_best_iou is not None and detector_best_iou < detector_missed_threshold
    ) or detector_best_iou is None

    flag_fsm_lost = (
        detector_best_iou is not None
        and post_fsm_iou is not None
        and (detector_best_iou - post_fsm_iou) >= fsm_lost_threshold_gap
    )

    flag_kalman_diverged = (
        post_fsm_iou is not None
        and post_kalman_iou is not None
        and (post_fsm_iou - post_kalman_iou) >= kalman_diverge_threshold_gap
    )

    flag_long_streak = predict_only_streak >= long_predict_only_threshold

    return FrameStageDiagnostic(
        frame_index=frame_index,
        gt_exists=True,
        detector_best_iou=detector_best_iou,
        detector_best_conf=detector_best_conf,
        track_vs_best_detection_iou=track_vs_best_detection_iou,
        post_fsm_iou=post_fsm_iou,
        post_kalman_iou=post_kalman_iou,
        track_active=post_fsm_active,
        track_age=track_age,
        predict_only_streak=predict_only_streak,
        matched_tier=matched_tier,
        flag_detector_missed_but_object_visible=flag_detector_missed,
        flag_fsm_lost_good_detection=flag_fsm_lost,
        flag_kalman_diverged_from_fsm=flag_kalman_diverged,
        flag_long_predict_only_streak=flag_long_streak,
    )


def summarize_diagnostics(diagnostics: list[FrameStageDiagnostic]) -> dict:
    """Tong hop danh sach chan doan tung frame thanh thong ke toan sequence -
    dung de in bao cao gon trong scripts/eval_pipeline_stages.py."""
    frames_with_gt = [d for d in diagnostics if d.gt_exists]
    n_frames_with_gt = len(frames_with_gt)

    def _mean_ignore_none(values: list[float | None]) -> float | None:
        filtered = [v for v in values if v is not None]
        return float(np.mean(filtered)) if filtered else None

    fsm_lost_frames = [d for d in frames_with_gt if d.flag_fsm_lost_good_detection]
    # Trong so cac frame "FSM danh mat", tach rieng truong hop track_vs_best_detection_iou
    # THAP (duoi 0.1, tuc duoi ca tier4_iou_threshold mac dinh) - day la truong
    # hop "khong the khop duoc VE MAT HINH HOC" (track qua xa detection moi),
    # KHAC voi truong hop track_vs_best_detection_iou con kha nhung van khong
    # khop (co the do logic FSM/nguong sai, dang can xem lai).
    n_fsm_lost_structurally_unmatchable = sum(
        1 for d in fsm_lost_frames
        if d.track_vs_best_detection_iou is not None and d.track_vs_best_detection_iou < 0.1
    )
    n_fsm_lost_other = len(fsm_lost_frames) - n_fsm_lost_structurally_unmatchable

    return {
        "n_frames_total": len(diagnostics),
        "n_frames_with_gt": n_frames_with_gt,
        "mean_detector_iou": _mean_ignore_none([d.detector_best_iou for d in frames_with_gt]),
        "mean_post_fsm_iou": _mean_ignore_none([d.post_fsm_iou for d in frames_with_gt]),
        "mean_post_kalman_iou": _mean_ignore_none([d.post_kalman_iou for d in frames_with_gt]),
        "n_detector_missed": sum(
            1 for d in frames_with_gt if d.flag_detector_missed_but_object_visible
        ),
        "n_fsm_lost_good_detection": len(fsm_lost_frames),
        "n_fsm_lost_structurally_unmatchable": n_fsm_lost_structurally_unmatchable,
        "n_fsm_lost_other_cause": n_fsm_lost_other,
        "n_kalman_diverged": sum(
            1 for d in frames_with_gt if d.flag_kalman_diverged_from_fsm
        ),
        "n_long_predict_only_streak": sum(
            1 for d in diagnostics if d.flag_long_predict_only_streak
        ),
        "max_predict_only_streak": max(
            (d.predict_only_streak for d in diagnostics), default=0
        ),
    }
