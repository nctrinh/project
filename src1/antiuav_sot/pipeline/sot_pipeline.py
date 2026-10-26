"""
Rap toan bo pipeline theo dung so do khoi trong bao cao FPGA (muc 1.2),
CAP NHAT (2 lan): (1) nhanh Motion dung Gradient Global-Motion Compensation
+ Residual Connected-Components thay cho Frame Diff + Block Threshold cu;
(2) them Kalman R thich ung theo nguon va co che track moi tu motion
(tentative tracks) - xem docstring cac khoi lien quan de biet chi tiet:

    Frame T (IR, grayscale)
      |
      +--> Frame Buffer (giu T-1) --> Gradient global-motion + residual CCL --> Motion candidates
      |                                                                                |
      +--> Detector nhe (YOLOv8-n/s, INT8) --> NMS on-chip --> Detector boxes          |
                                                                                |       |
                                            Merge (concat, kich thuoc co dinh) <--------+
                                                        |
                                                        v
                                    Simple Tracker FSM (IoU matrix, 4 tang uu tien,
                                    Tier 2/4 GIU NGUYEN SIZE track cu khi khop voi
                                    motion - xem tracker/simple_tracker_fsm.py)
                                                        |
                            +---------------------------+---------------------------+
                            |                                                       |
                            v                                                       v
              Tentative Tracks (motion con TIER_UNMATCHED,          Bo loc Kalman (tuy chon) - R lon
              tich luy streak, promote thanh track that             hon cho quan sat tu motion (xem
              sau motion_track_confirm_frames - xem                 kalman_motion_measurement_noise)
              tracker/tentative_tracks.py)
                                                        |
                                                        v
                                Dau ra: hop gioi han + ma track (Track ID)

Class SotPipeline duoi day GIU NGUYEN thu tu cac khoi nay, moi khoi la mot
loi goi ro rang toi module tuong ung - de code o day doc duoc SONG SONG voi
so do khoi trong bao cao, thuan tien khi doi chieu lai voi thiet ke RTL sau nay.

CAU HINH CAN CO trong FpgaConstraints / fpga_constraints.yaml (xem
utils/config.py va README_integration.md):
    motion.min_box_size, motion.max_box_size, motion.merge_gap_ratio (BAT BUOC)
    kalman.motion_measurement_noise (TUY CHON, fallback = measurement_noise)
    khoi tentative_motion_tracks (TUY CHON CA KHOI, mac dinh tat)

CAP NHAT (3): sua 8 van de fusion detector/motion - xem update_log.md o root
repo de biet day du boi canh/ly do. Tom tat cac khoi/co che THEM vao so do
tren (KHONG doi thu tu tong the, chi chen them nhanh xu ly o giua):
    - AdaptiveBoxSizeEstimator (motion/adaptive_size.py): [min_box_size,
      max_box_size] cua nhanh Motion co the la GIA TRI DONG suy tu lich su
      box Detector, thay vi hang so tinh (Issue 3, tat theo
      motion.size_adapt_enabled).
    - Priority anchoring: track active + tentative candidate duoc truyen
      NGUOC vao gradient_motion_candidates() de "neo" candidate trung vi tri
      dang theo doi khoi bi loai boi buoc chon top-K (Issue 6).
    - Simple Tracker FSM them Tang Confidence Override + co che khoa
      motion_streak (Issue 1/2/4/5, xem tracker/simple_tracker_fsm.py).
    - Tentative Tracks loai truoc cac candidate motion chong lan track dang
      active (Issue 2), va track duoc promote tu day duoc gan motion_streak
      = max_consecutive_motion_matches ngay tu dau (Issue 5).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.data.transforms import letterbox_grayscale, normalize_grayscale
from antiuav_sot.detector.model import DetectionResult, LightweightYoloDetector
from antiuav_sot.fusion.merge import SOURCE_MOTION, CandidateSet, merge_candidates
from antiuav_sot.motion.adaptive_size import AdaptiveBoxSizeEstimator
from antiuav_sot.motion.frame_buffer import FrameBuffer
from antiuav_sot.motion.gradient_candidates import gradient_motion_candidates
from antiuav_sot.tracker.kalman import ConstantVelocityKalmanBox
from antiuav_sot.tracker.simple_tracker_fsm import (
    TIER_UNMATCHED,
    FsmStepResult,
    SimpleTrackerFSM,
    TrackerFsmConfig,
)
from antiuav_sot.tracker.tentative_tracks import TentativeTrackTable
from antiuav_sot.tracker.track_state import TrackStateTable
from antiuav_sot.utils.box_ops import (
    INVALID_BOX,
    containment_matrix_xyxy,
    iou_matrix_xyxy,
    make_empty_boxes,
)
from antiuav_sot.utils.config import FpgaConstraints


@dataclasses.dataclass
class PipelineFrameOutput:
    """Ket qua pipeline cho MOT khung hinh - du lieu can de eval/visualize."""

    track_boxes: np.ndarray          # (n_track_slots, 4) xyxy, vi tri sau FSM (truoc Kalman)
    track_boxes_smoothed: np.ndarray  # (n_track_slots, 4) xyxy, sau Kalman (neu bat)
    track_active: np.ndarray          # (n_track_slots,) bool
    track_ids: np.ndarray             # (n_track_slots,) int32
    track_ages: np.ndarray            # (n_track_slots,) int32 - so frame lien tiep KHONG khop (tu TrackStateTable)
    predict_only_streak: np.ndarray   # (n_track_slots,) int32 - so frame lien tiep dang chi dua vao Kalman du bao,
                                       # KHONG co quan sat that tu FSM - chi so chan doan quan trong: streak cao
                                       # dong nghia track dang "troi" theo mo hinh chuyen dong tuyen tinh, khong
                                       # con duoc detector/motion xac nhan lai.
    fsm_result: FsmStepResult
    detection_result: DetectionResult
    letterbox_scale: float
    letterbox_pad: tuple[int, int]
    motion_boxes: np.ndarray          # (max_motion_candidates, 4) xyxy anh goc - candidate motion dua vao FSM
    motion_valid_mask: np.ndarray     # (max_motion_candidates,) bool
    motion_strengths: np.ndarray      # (max_motion_candidates,) float32, theo motion.strength_metric


class SotPipeline:
    """
    Pipeline SOT day du, chay tren mot sequence tung frame mot (giong dung
    cach du lieu se den tren FPGA - streaming, khong batch).

    Cach dung:
        pipeline = SotPipeline(fpga_constraints, detector)
        pipeline.reset()  # bat dau sequence moi
        for frame in sequence:
            output = pipeline.step(frame)
    """

    def __init__(
        self,
        constraints: FpgaConstraints,
        detector: LightweightYoloDetector,
    ):
        self.constraints = constraints
        self.detector = detector

        self.frame_buffer = FrameBuffer(lag=constraints.motion_frame_lag)
        self.track_state = TrackStateTable.create_empty(constraints.n_track_slots)
        self.fsm = SimpleTrackerFSM(
            config=TrackerFsmConfig(
                tier1_iou_threshold=constraints.tier1_iou_threshold,
                tier1_conf_min=constraints.tier1_conf_min,
                tier2_iou_threshold=constraints.tier2_iou_threshold,
                tier3_conf_min=constraints.tier3_conf_min,
                tier4_iou_threshold=constraints.tier4_iou_threshold,
                max_track_age=constraints.max_track_age,
                max_consecutive_motion_matches=constraints.max_consecutive_motion_matches,
                override_conf_min=constraints.override_conf_min,
                override_iou_threshold=constraints.override_iou_threshold,
                override_requires_overlap=constraints.override_requires_overlap,
                override_motion_dragged_only=constraints.override_motion_dragged_only,
                override_max_center_distance_ratio=(
                    constraints.override_max_center_distance_ratio
                ),
                reinit_near_lost_enabled=constraints.reinit_near_lost_enabled,
                reinit_max_center_distance_ratio=(
                    constraints.reinit_max_center_distance_ratio
                ),
                reinit_lost_memory_frames=constraints.reinit_lost_memory_frames,
                reinit_trust_conf_min=constraints.reinit_trust_conf_min,
                image_diagonal=float(
                    np.hypot(constraints.image_width, constraints.image_height)
                ),
            ),
            track_state=self.track_state,
        )

        # Issue 3 (Static Size Limits): uoc luong [min_box_size, max_box_size]
        # dong theo lich su kich thuoc box DETECTOR gan nhat, thay vi dung
        # thang hai hang so tinh moi frame. Tat theo config
        # (motion.size_adapt_enabled) - None nghia la GIU HANH VI CU y het,
        # luon dung constraints.motion_min_box_size/max_box_size tinh.
        self.size_estimator: AdaptiveBoxSizeEstimator | None = (
            AdaptiveBoxSizeEstimator(
                window=constraints.motion_size_adapt_window,
                floor=constraints.motion_min_box_size,
                ceiling=constraints.motion_max_box_size,
                shrink_factor=constraints.motion_size_adapt_shrink_factor,
                grow_factor=constraints.motion_size_adapt_grow_factor,
                min_samples=constraints.motion_size_adapt_min_samples,
            )
            if constraints.motion_size_adapt_enabled
            else None
        )

        self.kalman_filters: list[ConstantVelocityKalmanBox] = [
            ConstantVelocityKalmanBox(
                process_noise=constraints.kalman_process_noise,
                measurement_noise=constraints.kalman_measurement_noise,
            )
            for _ in range(constraints.n_track_slots)
        ]
        # So khung hinh LIEN TIEP moi slot dang o che do "predict-only"
        # (khong co quan sat that, chi Kalman du bao) - dung de biet khi nao
        # ngung tin tuong du bao Kalman (kalman_max_predict_only_frames).
        self._predict_only_streak = np.zeros((constraints.n_track_slots,), dtype=np.int32)

        # Tentative tracks tu motion (Tier E) - TUY CHON, None neu tat trong config.
        self.tentative_tracks: TentativeTrackTable | None = (
            TentativeTrackTable.create_empty(constraints.max_tentative_motion_tracks)
            if constraints.tentative_motion_tracks_enabled
            else None
        )

    def reset(self) -> None:
        """Goi khi bat dau mot sequence MOI - tranh frame cuoi sequence truoc
        bi dung nham lam T-1, hoac track cu sequence truoc con ton tai."""
        self.frame_buffer.reset()
        self.track_state = TrackStateTable.create_empty(self.constraints.n_track_slots)
        self.fsm.track_state = self.track_state
        for kf in self.kalman_filters:
            kf.reset()
        self._predict_only_streak[:] = 0
        if self.tentative_tracks is not None:
            self.tentative_tracks.reset()
        if self.size_estimator is not None:
            self.size_estimator.reset()

    def step(self, raw_frame: np.ndarray) -> PipelineFrameOutput:
        """
        raw_frame: (H, W) uint8 grayscale, kich thuoc anh GOC (truoc letterbox),
                   PHAI dung dung constraints.image_height/width da khai bao
                   trong fpga_constraints.yaml.

        Xu ly dung thu tu so do khoi trong docstring module.
        """
        c = self.constraints

        # Kiem tra som: raw_frame phai dung kich thuoc da khai bao trong
        # config (image.height/width). Nhanh Detector tu resize qua
        # letterbox nen se KHONG bao loi neu raw_frame lech kich thuoc,
        # nhung nhanh Motion doi hoi H/W nhat quan giua cac frame lien tiep
        # (gradient global-motion can hai frame CUNG kich thuoc) va se bao
        # loi kho hieu neu raw_frame thuc te khac voi kich thuoc da dung de
        # validate config luc load. Bao loi ro rang ngay tai day thay vi de
        # loi mo ho xuat hien sau, sau khi da chay qua ca nhanh Detector.
        if raw_frame.shape[:2] != (c.image_height, c.image_width):
            raise ValueError(
                f"raw_frame co kich thuoc {raw_frame.shape[:2]}, khong khop voi "
                f"image.height/width da khai bao trong config "
                f"({c.image_height}, {c.image_width}). Sua lai fpga_constraints.yaml "
                "cho dung kich thuoc anh IR thuc te, hoac kiem tra lai nguon du lieu dau vao."
            )

        # ---- Nhanh Detector (stateless) ----
        letterbox = letterbox_grayscale(raw_frame, c.det_input_height, c.det_input_width)
        det_input = normalize_grayscale(letterbox.image)
        detection_result = self.detector.predict(det_input)

        # Quy doi detector box (he toa do sau letterbox) ve he toa do anh
        # goc SOM (truoc ca nhanh Motion) - can vi (a) merge sau nay doi hoi
        # ca hai nguon cung he quy chieu, VA (b) AdaptiveBoxSizeEstimator +
        # priority anchoring cho nhanh Motion (duoi day) can box detector va
        # track o CUNG he toa do voi motion candidate (anh goc), khong phai
        # he toa do sau letterbox.
        detector_boxes_orig = _rescale_boxes_from_letterbox(
            detection_result.boxes, letterbox
        )

        # ---- Issue 3 (Static Size Limits): cap nhat lich su kich thuoc
        # detector va suy ra [min_box_size, max_box_size] dong cho nhanh
        # Motion frame nay - fallback ve cap gia tri TINH cu neu tat trong
        # config (motion.size_adapt_enabled=false, mac dinh). ----
        if self.size_estimator is not None:
            self.size_estimator.update(detector_boxes_orig, detection_result.valid_mask)
            motion_min_box_size, motion_max_box_size = self.size_estimator.estimate()
        else:
            motion_min_box_size, motion_max_box_size = c.motion_min_box_size, c.motion_max_box_size

        # ---- Issue 6 (Candidate List Overflow): vi tri dang duoc theo doi
        # (track active + candidate tentative dang tich luy streak) duoc
        # "neo" trong buoc chon top-K candidate cua nhanh Motion, de UAV
        # dung yen/lo lung (residual yeu) khong bi nhieu nen manh hon day
        # ra khoi danh sach K candidate co dinh. ----
        priority_boxes_parts = [self.track_state.boxes[self.track_state.active]]
        if self.tentative_tracks is not None:
            priority_boxes_parts.append(
                self.tentative_tracks.boxes[self.tentative_tracks.active]
            )
        priority_boxes = (
            np.concatenate(priority_boxes_parts, axis=0)
            if any(p.shape[0] > 0 for p in priority_boxes_parts)
            else None
        )

        # ---- Nhanh Motion (stateful - can Frame Buffer) ----
        # Gradient global-motion compensation + residual CCL: tu bu duoc
        # pan/rung camera (frame_diff/block_threshold cu KHONG lam duoc dieu
        # nay, chi diff tho hai frame), va tra ve bounding box sat vat hon
        # vi la box that cua tung connected component thay vi luoi block
        # co dinh. Xem gradient_candidates.py ve buoc gop cum chong phan manh,
        # loc ty le canh, va loc theo min/max_box_size (dong hoac tinh).
        if self.frame_buffer.has_previous():
            prev_frame = self.frame_buffer.get_previous()
            motion_boxes, motion_valid_mask, motion_strengths = gradient_motion_candidates(
                frame_t=raw_frame,
                frame_t_minus_1=prev_frame,
                max_motion_candidates=c.max_motion_candidates,
                min_box_size=motion_min_box_size,
                max_box_size=motion_max_box_size,
                merge_gap_ratio=c.motion_merge_gap_ratio,
                max_aspect_ratio=c.motion_max_aspect_ratio,
                merge_min_fill_ratio=c.motion_merge_min_fill_ratio,
                priority_boxes=priority_boxes,
                priority_iou_threshold=c.motion_candidate_anchor_iou,
                model=c.motion_model,
                irls_iters=c.motion_irls_iters,
                huber_delta=c.motion_huber_delta,
                pyramid_levels=c.motion_pyramid_levels,
                thresh_factor=c.motion_thresh_factor,
                min_area=c.motion_min_area,
                strength_metric=c.motion_strength_metric,
                min_strength=c.motion_min_strength,
            )
        else:
            # Frame dau tien cua sequence: chua co T-1, khong co motion candidate.
            motion_boxes = make_empty_boxes(c.max_motion_candidates)
            motion_valid_mask = np.zeros((c.max_motion_candidates,), dtype=bool)
            motion_strengths = np.zeros((c.max_motion_candidates,), dtype=np.float32)

        self.frame_buffer.update(raw_frame)

        # ---- Merge ----
        candidates = merge_candidates(
            detector_boxes=detector_boxes_orig,
            detector_scores=detection_result.scores,
            detector_valid_mask=detection_result.valid_mask,
            motion_boxes=motion_boxes,
            motion_valid_mask=motion_valid_mask,
            max_total_candidates=c.max_total_candidates,
        )

        # ---- Simple Tracker FSM ----
        fsm_result = self.fsm.step(candidates)

        # ---- Tentative tracks tu motion (Tier E, tuy chon) ----
        # Danh cho truong hop vat qua nho/mo de detector TUNG dat
        # tier3_conf_min du chi mot lan - Tang 3 se khong bao gio khoi tao
        # track cho no. O day, cac candidate MOTION con TIER_UNMATCHED sau
        # het 4 tang duoc theo doi rieng; neu mot vi tri duoc motion xac
        # nhan LIEN TUC du so khung hinh (motion_track_confirm_frames), no
        # duoc "thang chuc" thanh mot track that trong track_state (neu con
        # slot trong) - xem tracker/tentative_tracks.py.
        if self.tentative_tracks is not None:
            motion_unmatched_mask = (
                candidates.valid_mask
                & (candidates.source == SOURCE_MOTION)
                & (fsm_result.tier_for_candidate == TIER_UNMATCHED)
            )
            # ---- Issue 2 (Motion Priority Conflict): mot candidate motion
            # con TIER_UNMATCHED co the van chong lan mot track DANG active
            # (vd track vua duoc khop boi mot candidate KHAC frame nay, hoac
            # overlap chua du de vuot ca nguong long nhat tier4_iou_threshold).
            # Cho no vao Tentative Tracks trong truong hop nay se tao ra mot
            # ung vien tu-khoi-tao "gia", canh tranh khong can thiet voi
            # chinh track that no dang chong len - loai truoc khi dua vao
            # tentative_tracks.step(). Dung max(IoU, containment) - giong
            # tinh than Tier 2 (Issue 1) - vi mot fragment nho co the nam LOT
            # trong track lon ma IoU chuan van thap. ----
            active_track_boxes = self.track_state.boxes[self.track_state.active]
            if motion_unmatched_mask.any() and active_track_boxes.shape[0] > 0:
                overlap = np.maximum(
                    iou_matrix_xyxy(candidates.boxes, active_track_boxes),
                    containment_matrix_xyxy(candidates.boxes, active_track_boxes),
                )  # (n_candidates, n_active_tracks)
                overlaps_active_track = (
                    overlap.max(axis=1) >= c.tentative_active_overlap_max_iou
                )
                motion_unmatched_mask = motion_unmatched_mask & ~overlaps_active_track

            ready_slots = self.tentative_tracks.step(
                motion_boxes=candidates.boxes,
                motion_valid_mask=motion_unmatched_mask,
                iou_threshold=c.tentative_iou_threshold,
                max_miss=c.tentative_max_miss,
                confirm_frames=c.motion_track_confirm_frames,
            )
            reinit_max_distance = (
                c.reinit_max_center_distance_ratio
                * float(np.hypot(c.image_width, c.image_height))
            )
            for tentative_slot_idx in ready_slots:
                free_track_slot = self.track_state.find_free_slot()
                if free_track_slot is None:
                    # TrackStateTable da day - thu lai o khung hinh sau
                    # (streak cua slot tentative van giu nguyen >= confirm_frames).
                    continue
                # ---- Gate "track moi phai o gan track vua mat" ----
                # Track promote tu day la track TU KHOI TAO BOI MOTION, khong co
                # confidence ngu nghia nao de tin. Ngay sau khi mat mot track,
                # nhieu nen o goc khac cua anh rat de tich luy du
                # motion_track_confirm_frames khung hinh lien tiep va sinh ra
                # mot track moi o vi tri hoan toan vo ly (vat that khong the
                # dich chuyen tuc thoi den do). Chan truoc khi promote; slot
                # tentative KHONG bi xoa nen no se duoc thu lai o cac khung hinh
                # sau - va se duoc chap nhan khi ghost het han
                # (reinit_lost_memory_frames), luc khong con co so nao de chan.
                pending_box = self.tentative_tracks.boxes[tentative_slot_idx]
                if c.reinit_near_lost_enabled and not self.track_state.is_near_recent_lost(
                    pending_box, reinit_max_distance
                ):
                    continue
                promoted_box = self.tentative_tracks.promote_and_clear(tentative_slot_idx)
                # confidence=0.0: track nay khong tu detector, khong co
                # confidence ngu nghia that su - giong quy uoc motion box
                # trong fusion/merge.py.
                #
                # initial_motion_streak = max_consecutive_motion_matches (Issue 2):
                # track nay TU KHOI TAO BOI MOTION, khong phai detector, nen
                # phai bi coi la DA O trang thai "motion-dragged" tu khung
                # hinh dau tien - danh dau ngay lap tuc san sang cho Tang
                # Confidence Override (Issue 5) chiem lai boi bat ky detector
                # box tin cay cao nao xuat hien sau do, thay vi duoc mot
                # "thoi gian an" max_consecutive_motion_matches khung hinh
                # nhu mot track binh thuong moi bi motion dan dat.
                self.track_state.init_track(
                    free_track_slot,
                    promoted_box,
                    confidence=0.0,
                    initial_motion_streak=c.max_consecutive_motion_matches,
                )
                fsm_result.slot_matched_source[free_track_slot] = SOURCE_MOTION

        # ---- Kalman (tuy chon) ----
        track_boxes_smoothed = self.track_state.boxes.copy()
        if c.kalman_enabled:
            track_boxes_smoothed = self._apply_kalman(fsm_result)

        return PipelineFrameOutput(
            track_boxes=self.track_state.boxes.copy(),
            track_boxes_smoothed=track_boxes_smoothed,
            track_active=self.track_state.active.copy(),
            track_ids=self.track_state.track_ids.copy(),
            track_ages=self.track_state.ages.copy(),
            predict_only_streak=self._predict_only_streak.copy(),
            fsm_result=fsm_result,
            detection_result=detection_result,
            letterbox_scale=letterbox.scale,
            letterbox_pad=(letterbox.pad_left, letterbox.pad_top),
            motion_boxes=motion_boxes,
            motion_valid_mask=motion_valid_mask,
            motion_strengths=motion_strengths,
        )

    def _apply_kalman(self, fsm_result: FsmStepResult) -> np.ndarray:
        """Cap nhat/du bao Kalman cho tung slot, dua theo viec slot do co
        duoc FSM khop khung hinh nay hay khong (age == 0 nghia la vua khop).

        Khi quan sat vua khop den tu nguon MOTION (fsm_result.slot_matched_source
        == SOURCE_MOTION), truyen measurement_noise lon hon
        (kalman_motion_measurement_noise) cho Kalman - vi tam cua motion box
        co the lech so voi tam that cua vat neu no chi bat duoc mot phan
        (fragment), nen it tin quan sat nay hon binh thuong. Luu y: kich
        thuoc (w/h) KHONG di qua Kalman (xem kalman.py) - viec giu size cu
        cho motion da duoc xu ly truoc do o simple_tracker_fsm.py, khong
        lien quan gi den R o day."""
        c = self.constraints
        output_boxes = self.track_state.boxes.copy()

        for slot_idx in range(c.n_track_slots):
            kf = self.kalman_filters[slot_idx]

            if not self.track_state.active[slot_idx]:
                kf.reset()
                self._predict_only_streak[slot_idx] = 0
                continue

            just_matched = self.track_state.ages[slot_idx] == 0
            if just_matched:
                measurement_noise = None
                if fsm_result.slot_matched_source[slot_idx] == SOURCE_MOTION:
                    measurement_noise = c.kalman_motion_measurement_noise
                smoothed = kf.update(
                    self.track_state.boxes[slot_idx], measurement_noise=measurement_noise
                )
                output_boxes[slot_idx] = smoothed
                self._predict_only_streak[slot_idx] = 0
            else:
                if kf.is_initialized() and (
                    self._predict_only_streak[slot_idx] < c.kalman_max_predict_only_frames
                ):
                    predicted = kf.predict()
                    output_boxes[slot_idx] = predicted
                    self._predict_only_streak[slot_idx] += 1
                else:
                    output_boxes[slot_idx] = INVALID_BOX

        return output_boxes


def _rescale_boxes_from_letterbox(boxes_xyxy: np.ndarray, letterbox) -> np.ndarray:
    """Quy doi mang box (K,4) xyxy tu he toa do sau letterbox ve anh goc,
    vector hoa (khong loop tung box) - giu dung nguyen tac vector hoa toan repo."""
    boxes = boxes_xyxy.copy()
    valid = boxes[:, 0] != INVALID_BOX

    boxes[valid, 0] = (boxes[valid, 0] - letterbox.pad_left) / letterbox.scale
    boxes[valid, 1] = (boxes[valid, 1] - letterbox.pad_top) / letterbox.scale
    boxes[valid, 2] = (boxes[valid, 2] - letterbox.pad_left) / letterbox.scale
    boxes[valid, 3] = (boxes[valid, 3] - letterbox.pad_top) / letterbox.scale

    return boxes

