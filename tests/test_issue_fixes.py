"""
tests/test_issue_fixes.py
============================
Kiem tra CO CHU DICH cho 8 van de fusion detector/motion duoc sua trong lan
cap nhat nay (xem update_log.md o root repo de biet day du ly do/thiet ke).
KHONG can dataset that hay GPU - toan bo dung du lieu tong hop (synthetic),
chi can numpy + pyyaml (+ opencv-python/numba NEU muon import trang thai
day du antiuav_sot.motion.gradient_candidates, vi module do import
gradient_motion_core.py o cap module).

Chay: python3 tests/test_issue_fixes.py
(chay tu thu muc goc repo, hoac them src/ vao PYTHONPATH truoc - xem lenh
vi du o cuoi file nay/trong update_log.md)

Khong dung pytest de giam phu thuoc toi thieu - chi assert + runner don gian,
in PASS/FAIL ro rang tung muc.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# Cho phep chay truc tiep "python3 tests/test_issue_fixes.py" tu root repo
# ma khong can cai dat package (pip install -e .) truoc.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from antiuav_sot.utils.box_ops import (  # noqa: E402
    containment_matrix_xyxy,
    iou_matrix_xyxy,
)
from antiuav_sot.tracker.simple_tracker_fsm import (  # noqa: E402
    SOURCE_DETECTOR,
    SOURCE_MOTION,
    TIER_CONFIDENCE_OVERRIDE,
    TIER_MIDDLE_SCORE,
    TIER_MOTION_LINK,
    TIER_UNMATCHED,
    SimpleTrackerFSM,
    TrackerFsmConfig,
)
from antiuav_sot.tracker.track_state import TrackStateTable  # noqa: E402
from antiuav_sot.fusion.merge import CandidateSet  # noqa: E402
from antiuav_sot.motion.adaptive_size import AdaptiveBoxSizeEstimator  # noqa: E402
from antiuav_sot.utils.box_ops import make_empty_boxes  # noqa: E402


def _make_candidates(
    n_total: int,
    det_boxes: list[tuple[float, float, float, float]] | None = None,
    det_scores: list[float] | None = None,
    motion_boxes: list[tuple[float, float, float, float]] | None = None,
) -> CandidateSet:
    """Helper dung rieng cho file test nay - dung SOURCE_DETECTOR truoc,
    SOURCE_MOTION sau, giong dung quy uoc cua fusion/merge.merge_candidates."""
    det_boxes = det_boxes or []
    det_scores = det_scores or []
    motion_boxes = motion_boxes or []
    assert len(det_boxes) == len(det_scores)

    boxes = make_empty_boxes(n_total)
    scores = np.zeros((n_total,), dtype=np.float32)
    source = np.full((n_total,), SOURCE_DETECTOR, dtype=np.int32)
    valid = np.zeros((n_total,), dtype=bool)

    i = 0
    for b, s in zip(det_boxes, det_scores):
        boxes[i] = b
        scores[i] = s
        source[i] = SOURCE_DETECTOR
        valid[i] = True
        i += 1
    for b in motion_boxes:
        boxes[i] = b
        scores[i] = 0.0
        source[i] = SOURCE_MOTION
        valid[i] = True
        i += 1

    return CandidateSet(boxes=boxes, scores=scores, source=source, valid_mask=valid)


def test_issue1_tier2_containment_fix():
    """Box motion nho (10x10) nam LOT trong track box lon (100x100, cung tam)
    -> IoU chuan RAT THAP (~0.01) nhung containment PHAI = 1.0. Metric
    max(IoU, containment) dung o Tier 2 phai vuot nguong dien hinh (0.5) du
    IoU chuan khong the."""
    track_box = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
    small_motion_box = np.array([[45.0, 45.0, 55.0, 55.0]], dtype=np.float32)

    iou = iou_matrix_xyxy(track_box, small_motion_box)[0, 0]
    containment = containment_matrix_xyxy(track_box, small_motion_box)[0, 0]

    assert iou < 0.02, f"IoU ky vong rat thap, nhan duoc {iou}"
    assert abs(containment - 1.0) < 1e-5, f"Containment ky vong = 1.0, nhan duoc {containment}"
    assert max(iou, containment) >= 0.5, "max(IoU, containment) phai vuot nguong tier2 dien hinh"


def _default_fsm_config(**overrides) -> TrackerFsmConfig:
    base = dict(
        tier1_iou_threshold=0.3,
        tier1_conf_min=0.6,
        tier2_iou_threshold=0.5,
        tier3_conf_min=0.7,
        tier4_iou_threshold=0.1,
        max_track_age=5,
        max_consecutive_motion_matches=3,
        override_conf_min=0.85,
        override_iou_threshold=0.05,
    )
    base.update(overrides)
    return TrackerFsmConfig(**base)


def test_issue2_and_4_motion_streak_lockout():
    """Mot track duoc detector khoi tao, sau do CHI duoc motion khop lien
    tuc (khong detector nao dat tier1_conf_min nua). Sau
    max_consecutive_motion_matches khung hinh, track phai bi "khoa" khoi
    Tang 2 - candidate motion tiep theo (du IoU/containment cao) KHONG duoc
    phep khop no nua (Motion Priority Conflict / vai tro fail-safe cua motion
    phai bi gioi han)."""
    config = _default_fsm_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    fsm = SimpleTrackerFSM(config, track_state)

    # Frame 0: detector khoi tao track qua Tang 3.
    cand0 = _make_candidates(2, det_boxes=[(10, 10, 30, 30)], det_scores=[0.9])
    result0 = fsm.step(cand0)
    assert len(result0.newly_initiated_track_ids) == 1
    assert track_state.active[0]
    assert track_state.motion_streak[0] == 0

    # 3 frame lien tiep CHI co motion khop (IoU du cao voi track hien tai,
    # khong co detector nao). max_consecutive_motion_matches=3 nen frame 1,2,3
    # van duoc phep khop (streak tang 1,2,3), frame thu 4 PHAI bi khoa.
    box = (10, 10, 30, 30)  # trung khop hoan hao voi track hien tai
    tiers_seen = []
    for _ in range(4):
        # det_boxes rong -> _make_candidates dat candidate MOTION duy nhat
        # ngay tai index 0 (xem cai dat _make_candidates: vong lap detector
        # chay truoc nhung khong co phan tu nao, khong "danh cho" slot 0).
        cand = _make_candidates(2, motion_boxes=[box])
        result = fsm.step(cand)
        motion_cand_idx = 0
        tiers_seen.append(int(result.tier_for_candidate[motion_cand_idx]))

    assert tiers_seen[0] == TIER_MOTION_LINK, tiers_seen
    assert tiers_seen[1] == TIER_MOTION_LINK, tiers_seen
    assert tiers_seen[2] == TIER_MOTION_LINK, tiers_seen
    # Sau 3 lan khop motion lien tiep, track.motion_streak == 3 == cap ->
    # bi khoa TRUOC khi xu ly candidate cua lan goi step() thu 4 (thu tu
    # kiem tra dung LUC BAT DAU step, nen chinh candidate thu 4 nay bi tu choi).
    assert tiers_seen[3] == TIER_UNMATCHED, (
        f"Track phai bi khoa motion sau {config.max_consecutive_motion_matches} lan khop "
        f"motion lien tiep, nhung van khop o tang {tiers_seen[3]}"
    )


def test_issue5_confidence_override():
    """Tiep tuc kich ban tren: sau khi track bi khoa motion (motion-dragged),
    mot detector box CONF RAT CAO (>= override_conf_min) nhung IoU voi vi tri
    hien tai cua track chi o muc trung binh-thap (duoi tier1_iou_threshold,
    qua do khong the khop o Tang 1 thong thuong) PHAI van chiem lai duoc
    track do qua Tang Confidence Override."""
    config = _default_fsm_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    fsm = SimpleTrackerFSM(config, track_state)

    fsm.step(_make_candidates(1, det_boxes=[(10, 10, 30, 30)], det_scores=[0.9]))
    for _ in range(config.max_consecutive_motion_matches):
        fsm.step(_make_candidates(1, motion_boxes=[(10, 10, 30, 30)]))

    assert track_state.is_motion_dragged(0, config.max_consecutive_motion_matches)

    # Box detector conf rat cao, lech mot phan so voi track (IoU ~0.087 - duoi
    # tier1_iou_threshold=0.3 nhung tren override_iou_threshold=0.05).
    shifted_box = (22, 22, 42, 42)
    iou_with_track = float(
        iou_matrix_xyxy(
            np.array([track_state.boxes[0]], dtype=np.float32),
            np.array([shifted_box], dtype=np.float32),
        )[0, 0]
    )
    assert config.override_iou_threshold < iou_with_track < config.tier1_iou_threshold, (
        f"Kich ban test can IoU nam giua hai nguong, nhan duoc {iou_with_track}"
    )

    result = fsm.step(_make_candidates(1, det_boxes=[shifted_box], det_scores=[0.95]))
    assert int(result.tier_for_candidate[0]) == TIER_CONFIDENCE_OVERRIDE, result.tier_for_candidate
    assert track_state.motion_streak[0] == 0, "Override phai reset motion_streak ve 0"
    np.testing.assert_allclose(track_state.boxes[0], np.array(shifted_box, dtype=np.float32))


def test_issue3_adaptive_size_estimator():
    """Uoc luong dong phai CO PHAN UNG voi kich thuoc detector: cho box nho
    lien tuc -> khoang [min,max] phai THU HEP lai gan phia duoi cua khoang
    (floor, ceiling) tinh cu."""
    estimator = AdaptiveBoxSizeEstimator(window=30, floor=4, ceiling=64, min_samples=5)

    # Chua du mau -> fallback dung floor/ceiling tinh.
    lo, hi = estimator.estimate()
    assert (lo, hi) == (4, 64)

    # Nap 20 box nho, dong deu quanh 8x8 (UAV o xa).
    small_boxes = np.array([[0, 0, 8, 8]] * 20, dtype=np.float32)
    estimator.update(small_boxes, np.ones((20,), dtype=bool))

    lo, hi = estimator.estimate()
    assert lo >= 4 and hi <= 64, (lo, hi)
    assert hi < 64, f"max_box_size dong phai thu hep lai tu tran 64, nhan duoc {hi}"


def test_issue7_aspect_ratio_filter():
    from antiuav_sot.motion.gradient_candidates import _aspect_ratio

    assert _aspect_ratio(10.0, 10.0) == 1.0
    assert abs(_aspect_ratio(10.0, 50.0) - 5.0) < 1e-6
    assert _aspect_ratio(0.0, 10.0) == float("inf")


def test_issue8_clustering_paradox():
    from antiuav_sot.motion.gradient_candidates import _cluster_and_merge_boxes

    # Ca A: hai fragment CUNG mot UAV - gan nhau, kich thuoc tuong duong,
    # gop lai se cho fill ratio cao -> PHAI duoc gop thanh 1 box.
    uav_fragments = [(10.0, 10.0, 18.0, 18.0), (19.0, 10.0, 27.0, 18.0)]
    merged_uav = _cluster_and_merge_boxes(
        uav_fragments, merge_gap_ratio=0.5, min_fill_ratio=0.3, max_aspect_ratio=5.0
    )
    assert len(merged_uav) == 1, f"Fragment cung UAV phai duoc gop, nhan duoc {merged_uav}"

    # Ca B: dung merge_gap_ratio RAT LON de "vo hieu hoa" gan nhu hoan toan
    # dieu kien khoang cach (moi cap deu duoi nguong, tuong duong tinh huong
    # xau nhat cua "chuyen sang gop THUAN TUY theo khoang cach" ma Issue 8 mo
    # ta) - CHI con lai dieu kien fill ratio quyet dinh. UAV (8x8) va 3 blob
    # nhieu nho (2x2) nam RAI RAC xa nhau: neu khong co fill-ratio guard,
    # thuat toan Kruskal se gop het thanh 1 box khong lo bao trum toan bo
    # khu vuc; VOI guard, moi lan gop deu bi tu choi vi box hop nhat qua
    # "rong" so voi tong dien tich thuc - ket qua PHAI la 4 box rieng biet.
    uav_like = (10.0, 10.0, 18.0, 18.0)
    scattered_noise = [
        (100.0, 100.0, 102.0, 102.0),
        (200.0, 50.0, 202.0, 52.0),
        (50.0, 200.0, 52.0, 202.0),
    ]
    boxes_b = [uav_like] + scattered_noise
    merged_b = _cluster_and_merge_boxes(
        boxes_b, merge_gap_ratio=100.0, min_fill_ratio=0.3, max_aspect_ratio=5.0
    )
    assert len(merged_b) == 4, (
        "Fill-ratio guard phai tu choi MOI lan gop giua cac box rai rac du dieu kien "
        f"khoang cach da bi vo hieu hoa gan het (merge_gap_ratio=100) - nhan duoc {merged_b}"
    )


def test_issue6_priority_anchoring():
    from antiuav_sot.motion.gradient_candidates import _overlaps_any_priority_box

    weak_but_tracked = (10.0, 10.0, 20.0, 20.0)
    priority_boxes = np.array([[9.0, 9.0, 21.0, 21.0]], dtype=np.float32)  # track gan trung khop

    assert _overlaps_any_priority_box(weak_but_tracked, priority_boxes, threshold=0.5)

    far_away_noise = (200.0, 200.0, 210.0, 210.0)
    assert not _overlaps_any_priority_box(far_away_noise, priority_boxes, threshold=0.5)


ALL_TESTS = [
    test_issue1_tier2_containment_fix,
    test_issue2_and_4_motion_streak_lockout,
    test_issue5_confidence_override,
    test_issue3_adaptive_size_estimator,
    test_issue7_aspect_ratio_filter,
    test_issue8_clustering_paradox,
    test_issue6_priority_anchoring,
]


def main() -> int:
    failures = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {name}: {e}")
        except Exception:
            failures += 1
            print(f"ERROR {name}:")
            traceback.print_exc()

    print()
    if failures == 0:
        print(f"Tat ca {len(ALL_TESTS)} test PASS.")
    else:
        print(f"{failures}/{len(ALL_TESTS)} test FAIL/ERROR.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
