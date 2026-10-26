import numpy as np
import pytest

from antiuav_sot.fusion.merge import SOURCE_DETECTOR, SOURCE_MOTION, CandidateSet
from antiuav_sot.tracker.simple_tracker_fsm import (
    TIER_HIGH_SCORE_INIT,
    TIER_LOW_SCORE,
    TIER_MIDDLE_SCORE,
    TIER_MOTION_LINK,
    TIER_UNMATCHED,
    SimpleTrackerFSM,
    TrackerFsmConfig,
)
from antiuav_sot.tracker.track_state import EMPTY_TRACK_ID, TrackStateTable
from antiuav_sot.utils.box_ops import INVALID_BOX, make_empty_boxes


def make_config(**overrides) -> TrackerFsmConfig:
    defaults = dict(
        tier1_iou_threshold=0.3,
        tier1_conf_min=0.4,
        tier2_iou_threshold=0.5,
        tier3_conf_min=0.7,
        tier4_iou_threshold=0.1,
        max_track_age=5,
    )
    defaults.update(overrides)
    return TrackerFsmConfig(**defaults)


def make_candidate_set(
    boxes: list[list[float]],
    scores: list[float],
    sources: list[int],
    max_total: int,
) -> CandidateSet:
    n = len(boxes)
    all_boxes = make_empty_boxes(max_total)
    all_scores = np.zeros((max_total,), dtype=np.float32)
    all_source = np.full((max_total,), SOURCE_DETECTOR, dtype=np.int32)
    all_valid = np.zeros((max_total,), dtype=bool)

    if n > 0:
        all_boxes[:n] = np.array(boxes, dtype=np.float32)
        all_scores[:n] = scores
        all_source[:n] = sources
        all_valid[:n] = True

    return CandidateSet(boxes=all_boxes, scores=all_scores, source=all_source, valid_mask=all_valid)


# ---------------------------------------------------------------------------
# Tang 3: High Score Initiation - track moi duoc tao khi chua co track nao
# ---------------------------------------------------------------------------

def test_tier3_initiates_new_track_from_empty_state():
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[10, 10, 20, 20]],
        scores=[0.9],  # >= tier3_conf_min=0.7
        sources=[SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert track_state.active[0]
    assert track_state.track_ids[0] != EMPTY_TRACK_ID
    np.testing.assert_allclose(track_state.boxes[0], [10, 10, 20, 20])
    assert result.tier_for_candidate[0] == TIER_HIGH_SCORE_INIT
    assert len(result.newly_initiated_track_ids) == 1


def test_tier3_does_not_initiate_when_confidence_too_low():
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[10, 10, 20, 20]],
        scores=[0.5],  # < tier3_conf_min=0.7, cung < tier1_conf_min neu khong co track de khop
        sources=[SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert not track_state.active[0]
    assert result.tier_for_candidate[0] == TIER_UNMATCHED


def test_tier3_respects_free_slot_limit():
    """Voi N=1 slot, 2 box confidence cao chi 1 box duoc khoi tao track."""
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[10, 10, 20, 20], [50, 50, 60, 60]],
        scores=[0.9, 0.95],
        sources=[SOURCE_DETECTOR, SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert track_state.num_active() == 1
    n_initiated = sum(1 for t in result.tier_for_candidate if t == TIER_HIGH_SCORE_INIT)
    assert n_initiated == 1


# ---------------------------------------------------------------------------
# Tang 1: Middle Score Linking - khop track da co voi box detector IoU cao
# ---------------------------------------------------------------------------

def test_tier1_matches_existing_track_to_detector_box():
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([10, 10, 20, 20], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[11, 11, 21, 21]],  # dich nhe, IoU cao voi track hien tai
        scores=[0.5],  # >= tier1_conf_min=0.4, < tier3_conf_min=0.7 (khong bi Tang 3 tranh)
        sources=[SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert result.tier_for_candidate[0] == TIER_MIDDLE_SCORE
    np.testing.assert_allclose(track_state.boxes[0], [11, 11, 21, 21])
    assert track_state.ages[0] == 0  # vua khop, age phai reset ve 0


def test_tier1_does_not_match_low_iou():
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([10, 10, 20, 20], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[100, 100, 110, 110]],  # qua xa, IoU = 0
        scores=[0.5],
        sources=[SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    # Khong khop o tang nao (tier4 threshold=0.1 cung khong dat vi IoU=0)
    assert result.tier_for_candidate[0] == TIER_UNMATCHED
    assert track_state.ages[0] == 1  # track khong khop -> age tang len


# ---------------------------------------------------------------------------
# Tang 2: Motion Boxes Linking - box tu motion can nguong IoU cao hon
# ---------------------------------------------------------------------------

def test_tier2_matches_motion_box_with_higher_threshold():
    # tier1_iou_threshold=0.3, tier2_iou_threshold=0.6: mot box co IoU=0.4
    # se KHONG du cho tier2 (0.4 < 0.6) nhung neu la detector se du cho tier1.
    config = make_config(tier1_iou_threshold=0.3, tier2_iou_threshold=0.6)
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    # Box dich chuyen it, IoU ~0.68 (du cho ca tier1 va tier2) tu MOTION source
    candidates = make_candidate_set(
        boxes=[[1, 1, 11, 11]],
        scores=[0.0],  # motion box khong co confidence ngu nghia
        sources=[SOURCE_MOTION],
        max_total=4,
    )

    result = fsm.step(candidates)

    # Vi la SOURCE_MOTION, khong du dieu kien tier1 (chi xet is_detector),
    # phai roi xuong tier2.
    assert result.tier_for_candidate[0] == TIER_MOTION_LINK


def test_tier2_not_reached_if_tier1_already_matched():
    """Neu track da khop o Tang 1 (box detector), Tang 2 khong duoc xet lai
    track do nua trong cung khung hinh."""
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[
            [1, 1, 11, 11],   # detector, se khop Tang 1
            [1, 1, 11, 11],   # motion, IoU cao nhung track da bi Tang 1 chiem
        ],
        scores=[0.5, 0.0],
        sources=[SOURCE_DETECTOR, SOURCE_MOTION],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert result.tier_for_candidate[0] == TIER_MIDDLE_SCORE
    assert result.tier_for_candidate[1] == TIER_UNMATCHED  # khong con track de khop


# ---------------------------------------------------------------------------
# Tang 4: Low Score Linking - du phong cuoi, nguong noi long nhat
# ---------------------------------------------------------------------------

def test_tier4_catches_low_iou_leftover():
    config = make_config(tier1_conf_min=0.9, tier3_conf_min=0.99, tier4_iou_threshold=0.05)
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    # confidence=0.3 qua thap cho tier1 (can 0.9) va tier3 (can 0.99).
    # track=[0,0,10,10], box=[6,6,16,16] -> overlap=[6,6,10,10]=16,
    # union=100+100-16=184, IoU=16/184~=0.087, du cho tier4 (0.05) nhung
    # khong du cho tier1 hay bat ky nguong cao hon nao khac trong config.
    candidates = make_candidate_set(
        boxes=[[6, 6, 16, 16]],
        scores=[0.3],
        sources=[SOURCE_DETECTOR],
        max_total=4,
    )

    result = fsm.step(candidates)

    assert result.tier_for_candidate[0] == TIER_LOW_SCORE


# ---------------------------------------------------------------------------
# Track expiration
# ---------------------------------------------------------------------------

def test_track_expires_after_max_age():
    config = make_config(max_track_age=2)
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    empty_candidates = make_candidate_set(boxes=[], scores=[], sources=[], max_total=4)

    fsm.step(empty_candidates)  # age -> 1
    assert track_state.active[0]
    fsm.step(empty_candidates)  # age -> 2
    assert track_state.active[0]
    fsm.step(empty_candidates)  # age -> 3 > max_track_age=2 -> expire
    assert not track_state.active[0]
    assert track_state.track_ids[0] == EMPTY_TRACK_ID
    assert np.all(track_state.boxes[0] == INVALID_BOX)


def test_expired_slot_can_be_reused_for_new_track():
    config = make_config(max_track_age=1)
    track_state = TrackStateTable.create_empty(n_slots=1)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    fsm = SimpleTrackerFSM(config, track_state)

    empty_candidates = make_candidate_set(boxes=[], scores=[], sources=[], max_total=4)
    fsm.step(empty_candidates)
    fsm.step(empty_candidates)
    assert not track_state.active[0]

    new_candidates = make_candidate_set(
        boxes=[[50, 50, 60, 60]], scores=[0.9], sources=[SOURCE_DETECTOR], max_total=4
    )
    result = fsm.step(new_candidates)

    assert track_state.active[0]
    assert result.tier_for_candidate[0] == TIER_HIGH_SCORE_INIT


# ---------------------------------------------------------------------------
# Multi-object (N > 1) sanity check - dam bao logic khong hard-code N=1
# ---------------------------------------------------------------------------

def test_multi_object_two_tracks_two_boxes_no_id_switch():
    config = make_config()
    track_state = TrackStateTable.create_empty(n_slots=2)
    track_state.init_track(0, np.array([0, 0, 10, 10], dtype=np.float32), confidence=0.9)
    track_state.init_track(1, np.array([100, 100, 110, 110], dtype=np.float32), confidence=0.9)
    id0, id1 = track_state.track_ids[0], track_state.track_ids[1]
    fsm = SimpleTrackerFSM(config, track_state)

    candidates = make_candidate_set(
        boxes=[[1, 1, 11, 11], [101, 101, 111, 111]],
        scores=[0.5, 0.5],
        sources=[SOURCE_DETECTOR, SOURCE_DETECTOR],
        max_total=4,
    )

    fsm.step(candidates)

    # Track ID phai giu nguyen, khong bi hoan doi (identity switch)
    assert track_state.track_ids[0] == id0
    assert track_state.track_ids[1] == id1
    np.testing.assert_allclose(track_state.boxes[0], [1, 1, 11, 11])
    np.testing.assert_allclose(track_state.boxes[1], [101, 101, 111, 111])
