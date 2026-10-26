import numpy as np

from antiuav_sot.utils.box_ops import INVALID_BOX, make_empty_boxes
from antiuav_sot.utils.stage_metrics import diagnose_frame, summarize_diagnostics


def test_diagnose_frame_no_gt_returns_none_ious():
    diag = diagnose_frame(
        frame_index=0,
        gt_exists=False,
        gt_box_xywh=np.zeros((4,), dtype=np.float32),
        detector_boxes_xyxy=make_empty_boxes(4),
        detector_scores=np.zeros((4,), dtype=np.float32),
        detector_valid_mask=np.zeros((4,), dtype=bool),
        track_box_before_this_frame_xyxy=None,
        post_fsm_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
        post_fsm_active=False,
        post_kalman_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
        track_age=0,
        predict_only_streak=0,
        matched_tier=0,
    )
    assert diag.gt_exists is False
    assert diag.detector_best_iou is None
    assert diag.track_vs_best_detection_iou is None
    assert diag.post_fsm_iou is None
    assert diag.post_kalman_iou is None
    assert diag.flag_detector_missed_but_object_visible is False


def test_diagnose_frame_perfect_match_all_stages():
    """GT va tat ca 3 giai doan deu khop hoan hao -> khong co canh bao nao."""
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)  # xywh
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [0, 0, 10, 10]  # xyxy, khop hoan hao
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    diag = diagnose_frame(
        frame_index=1,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes,
        detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        track_age=0,
        predict_only_streak=0,
        matched_tier=1,
    )

    assert diag.detector_best_iou == 1.0
    assert diag.track_vs_best_detection_iou == 1.0
    assert diag.post_fsm_iou == 1.0
    assert diag.post_kalman_iou == 1.0
    assert diag.flag_detector_missed_but_object_visible is False
    assert diag.flag_fsm_lost_good_detection is False
    assert diag.flag_kalman_diverged_from_fsm is False


def test_diagnose_frame_flags_fsm_lost_structurally_unmatchable():
    """Detector dung (IoU cao voi GT) nhung track (vi tri CU, tu frame
    truoc) da qua xa detection MOI de bat ky tier nao khop duoc -
    track_vs_best_detection_iou PHAI THAP, giai thich DUNG nguyen nhan
    (khac voi loi FSM/Kalman: day la gioi han hinh hoc thuan tuy khi vat
    the di chuyen nhanh giua 2 frame)."""
    gt_box = np.array([50, 50, 10, 10], dtype=np.float32)  # xywh (x,y,w,h) = xyxy[50,50,60,60], GT o vi tri MOI
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [50, 50, 60, 60]  # detector dung, khop hoan hao voi GT
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    diag = diagnose_frame(
        frame_index=5,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes,
        detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        # Track van o vi tri CU (frame truoc), rat xa detection moi
        track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),  # khong khop, giu nguyen
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        track_age=3,
        predict_only_streak=3,
        matched_tier=0,
    )

    assert diag.detector_best_iou == 1.0
    assert diag.track_vs_best_detection_iou == 0.0  # track qua xa detection - khong the khop
    assert diag.flag_fsm_lost_good_detection is True
    assert diag.flag_detector_missed_but_object_visible is False  # detector VAN dung


def test_diagnose_frame_flags_kalman_diverged():
    """FSM khop tot (post_fsm_iou cao) nhung Kalman lam lech (post_kalman_iou thap)."""
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [0, 0, 10, 10]
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    diag = diagnose_frame(
        frame_index=2,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes,
        detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),  # FSM khop dung
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([20, 20, 30, 30], dtype=np.float32),  # Kalman lech xa
        track_age=0,
        predict_only_streak=0,
        matched_tier=1,
    )

    assert diag.post_fsm_iou == 1.0
    assert diag.post_kalman_iou == 0.0
    assert diag.flag_kalman_diverged_from_fsm is True
    assert diag.flag_fsm_lost_good_detection is False  # FSM khong co loi


def test_diagnose_frame_flags_detector_missed():
    """Detector khong ra box nao dat IoU du cao - object van ton tai
    nhung detector tu no da mat muc tieu."""
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [100, 100, 110, 110]  # rat xa GT
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    diag = diagnose_frame(
        frame_index=10,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes,
        detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        track_box_before_this_frame_xyxy=None,
        post_fsm_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
        post_fsm_active=False,
        post_kalman_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
        track_age=1,
        predict_only_streak=0,
        matched_tier=0,
    )

    assert diag.detector_best_iou == 0.0
    assert diag.flag_detector_missed_but_object_visible is True


def test_diagnose_frame_track_vs_detection_none_when_no_track():
    """Khi khong co track truoc do (vd frame dau sequence, hoac track vua
    bi expire), track_vs_best_detection_iou phai la None (khong the tinh),
    KHAC voi 0.0 (co tinh nhung IoU bang 0)."""
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [0, 0, 10, 10]
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    diag = diagnose_frame(
        frame_index=0,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes,
        detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        track_box_before_this_frame_xyxy=None,  # khong co track truoc do
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        track_age=0,
        predict_only_streak=0,
        matched_tier=3,  # High Score Initiation - track moi tao
    )

    assert diag.track_vs_best_detection_iou is None


def test_diagnose_frame_long_predict_only_streak_flag():
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)
    diag = diagnose_frame(
        frame_index=7,
        gt_exists=True,
        gt_box_xywh=gt_box,
        detector_boxes_xyxy=make_empty_boxes(4),
        detector_scores=np.zeros((4,), dtype=np.float32),
        detector_valid_mask=np.zeros((4,), dtype=bool),
        track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        track_age=3,
        predict_only_streak=5,
        matched_tier=0,
        long_predict_only_threshold=2,
    )
    assert diag.flag_long_predict_only_streak is True


def test_summarize_diagnostics_aggregates_correctly():
    gt_box = np.array([0, 0, 10, 10], dtype=np.float32)
    empty_boxes = make_empty_boxes(4)
    empty_valid = np.zeros((4,), dtype=bool)
    empty_scores = np.zeros((4,), dtype=np.float32)

    good_boxes = make_empty_boxes(4)
    good_boxes[0] = [0, 0, 10, 10]
    good_valid = np.array([True, False, False, False])
    good_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    diagnostics = [
        diagnose_frame(
            frame_index=0, gt_exists=True, gt_box_xywh=gt_box,
            detector_boxes_xyxy=good_boxes, detector_scores=good_scores,
            detector_valid_mask=good_valid,
            track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            post_fsm_active=True,
            post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
            track_age=0, predict_only_streak=0, matched_tier=1,
        ),
        diagnose_frame(
            frame_index=1, gt_exists=True, gt_box_xywh=gt_box,
            detector_boxes_xyxy=empty_boxes, detector_scores=empty_scores,
            detector_valid_mask=empty_valid,
            track_box_before_this_frame_xyxy=None,
            post_fsm_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
            post_fsm_active=False,
            post_kalman_box_xyxy=np.full((4,), INVALID_BOX, dtype=np.float32),
            track_age=1, predict_only_streak=0, matched_tier=0,
        ),
    ]

    summary = summarize_diagnostics(diagnostics)
    assert summary["n_frames_total"] == 2
    assert summary["n_frames_with_gt"] == 2
    # Frame 0: detector_best_iou=1.0. Frame 1: detector khong co box hop le
    # nao -> detector_best_iou=None (loai khoi mean, KHAC voi 0.0 - xem
    # docstring FrameStageDiagnostic ve ly do phan biet "khong co du lieu"
    # voi "co du lieu nhung sai"). Mean chi tinh tren frame 0 -> 1.0.
    assert summary["mean_detector_iou"] == 1.0
    assert summary["n_detector_missed"] == 1


def test_summarize_diagnostics_breaks_down_fsm_lost_by_cause():
    """n_fsm_lost_structurally_unmatchable phai dem dung cac frame co
    track_vs_best_detection_iou thap (< 0.1), tach biet khoi n_fsm_lost_other_cause."""
    gt_box = np.array([50, 50, 10, 10], dtype=np.float32)  # xywh -> xyxy[50,50,60,60]
    detector_boxes = make_empty_boxes(4)
    detector_boxes[0] = [50, 50, 60, 60]
    detector_valid = np.array([True, False, False, False])
    detector_scores = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)

    # Frame A: structurally unmatchable (track qua xa detection moi)
    diag_a = diagnose_frame(
        frame_index=0, gt_exists=True, gt_box_xywh=gt_box,
        detector_boxes_xyxy=detector_boxes, detector_scores=detector_scores,
        detector_valid_mask=detector_valid,
        track_box_before_this_frame_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        post_fsm_active=True,
        post_kalman_box_xyxy=np.array([0, 0, 10, 10], dtype=np.float32),
        track_age=1, predict_only_streak=1, matched_tier=0,
    )

    summary = summarize_diagnostics([diag_a])
    assert summary["n_fsm_lost_good_detection"] == 1
    assert summary["n_fsm_lost_structurally_unmatchable"] == 1
    assert summary["n_fsm_lost_other_cause"] == 0
