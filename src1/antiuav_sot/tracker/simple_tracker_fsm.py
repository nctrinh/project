"""
Simple Tracker FSM: co che lien ket nhieu tang uu tien, dung tinh than bai
goc Tang et al. "Strong Detector with Simple Tracker" (muc 3.5 bao cao FPGA),
MO RONG voi hai co che kiem soat vai tro cua motion (xem update_log.md,
muc "Issue 2/4/5" de biet day du boi canh/ly do thiet ke):

    Tang 1 (Middle Score Linking):
        Cap (track, box) dat nguong IoU cao, box tu DETECTOR voi confidence
        o muc trung binh -> khop, cap nhat track (CA vi tri lan kich thuoc,
        detector duoc tin tuong hoan toan). Khop o tang nay LUON reset
        motion_streak cua track ve 0 (track.note_detector_match()).

    Tang "Confidence Override" (MOI - chay NGAY SAU Tang 1, TRUOC Tang 2):
        Moi box DETECTOR con chua khop co confidence >= override_conf_min
        duoc quyen "chiem lai" mot track con chua khop trong khung hinh nay,
        CUONG CHE track quay ve dung vi tri/kich thuoc cua detector (khong
        ap dung _apply_motion_size_preservation vi nguon la detector), thay
        vi tiep tuc de motion/Kalman dan dat sai. Day la co che
        "Confidence-Based Override" giai quyet truong hop detector cuc ky
        tin cay nhung khong the thang duoc mot track da bi keo lech qua
        Tang 1 thong thuong.

        Hai tham so quyet dinh pham vi cua tang nay:

        override_requires_overlap (mac dinh False): KHONG bat buoc box
            detector phai chong lan track cu. Khi track da troi qua xa, IoU
            voi vi tri hien tai cua track bang dung 0 va moi nguong IoU du
            noi long cung vo nghia - nen tieu chi xep hang cua tang nay la
            CONFIDENCE, khong phai IoU (xem _build_override_affinity): box
            conf cao hon luon thang, du khong chong lan. IoU va khoang cach
            tam chi la term PHA HOA khi confidence gan nhu bang nhau. Dieu
            kien duy nhat con lai voi cap khong chong lan la gioi han
            override_max_center_distance_ratio (ti le duong cheo anh). Dat
            True de tro lai hanh vi cu: chi khop khi
            IoU >= override_iou_threshold.

        override_motion_dragged_only (mac dinh False): tang nay nham vao MOI
            track active. Dat True de gioi han lai hanh vi cu - chi nham vao
            track da bi danh dau "motion-dragged"
            (track_state.is_motion_dragged() == True, tuc motion_streak da
            vuot max_consecutive_motion_matches: track dang troi theo motion
            lien tuc qua nhieu khung hinh ma khong duoc detector xac nhan).

    Tang 2 (Motion Boxes Linking):
        Voi track/box con chua khop o cac tang truoc, xet tiep box tu MOTION
        cue. Metric so khop KHONG con la IoU chuan don thuan ma la
        max(IoU, containment) voi containment = Intersection-over-Smaller-
        Area (xem utils/box_ops.containment_matrix_xyxy) - sua "Tier-2 IoU
        Failure": mot box motion nho (chi bat vung pixel dang chuyen dong)
        nam LOT HOAN TOAN trong track box lon hon (vd vat bi mo/blur) van
        duoc coi la cung mot vat du IoU chuan rat thap. Nguong so sanh van
        la tier2_iou_threshold (ap dung cho gia tri max(IoU, containment)).

        Track da bi danh dau motion-dragged (is_motion_dragged()) bi LOAI
        KHOI dieu kien duoc tang nay khop (motion "khoa" khong duoc tiep tuc
        dan dat track do nua cho den khi mot box DETECTOR xac nhan lai qua
        Tang 1/Confidence Override/Tang 4, reset motion_streak) - day la co
        che "kiem soat chat" Motion Priority Conflict, gioi han so khung
        hinh LIEN TIEP toi da ma motion duoc phep tu quyet dinh vi tri track
        ma khong co detector xac nhan.

        Khi khop o tang nay, CHI lay VI TRI (tam) tu box motion, GIU NGUYEN
        kich thuoc cua track cu (_apply_motion_size_preservation) - vi
        motion detector khong dam bao bao phu toan bo vat the, co the chi
        bat duoc mot phan (fragment) cua no. Khop o tang nay lam TANG
        motion_streak them 1 (track.note_motion_match()).

    Tang 3 (High Score Initiation):
        Box co confidence rat cao (tu detector) nhung chua khop track nao
        -> khoi tao track moi tai khe trong dau tien. Chi tu detector nen
        kich thuoc track moi luon dang tin cay ngay tu dau, motion_streak
        khoi tao = 0.

        Gate "gan track vua mat" (reinit_near_lost_enabled, mac dinh True):
        khi mot track bi xoa, vi tri cuoi cung cua no duoc giu lai lam
        "ghost" trong reinit_lost_memory_frames khung hinh (xem
        TrackStateTable.expire_track/age_lost_memory). Trong thoi gian do,
        candidate co conf < reinit_trust_conf_min chi duoc khoi tao track moi
        neu tam cua no nam trong ban kinh reinit_max_center_distance_ratio *
        duong cheo anh quanh mot ghost con han. Ly do: mot UAV khong dich
        chuyen tuc thoi, nen ngay sau khi mat track, mot "vat" xuat hien o
        goc khac cua anh gan nhu chac chan la nhieu nen. Candidate conf >=
        reinit_trust_conf_min KHONG bi gate - detector du tin cay thi tin
        ngay, o bat ky dau. Khi khong con ghost nao con han, gate mo hoan
        toan (khong co co so nao de chan).

        CUNG gate nay duoc ap dung cho buoc promote Tentative Tracks (track
        tu-khoi-tao boi MOTION) trong pipeline/sot_pipeline.py - do moi la
        nhanh de sinh ra track rac nhat, vi no khong co confidence ngu nghia
        nao de loc.

    Tang 4 (Low Score Linking):
        Khop not phan con lai voi nguong noi long hon, buoc du phong cuoi
        cung. Candidate o tang nay co the la detector HOAC motion - ap dung
        CUNG co che giu-nguyen-kich-thuoc, containment-fallback, motion-lock
        va cap nhat motion_streak nhu Tang 2 khi nguon la motion; reset
        motion_streak ve 0 khi nguon la detector.

FsmStepResult.slot_matched_source ghi lai NGUON (SOURCE_DETECTOR/
SOURCE_MOTION) cua candidate vua khop moi track trong khung hinh nay (hoac
-1 neu khong khop) - de pipeline.py biet chon measurement_noise phu hop khi
goi ConstantVelocityKalmanBox.update() (quan sat tu motion co the kem tin
cay ve vi tri hon detector).

Day la khoi PHU HOP NHAT voi kien truc FPGA trong toan bo pipeline: toan bo
chi la cac phep so sanh so hoc (IoU/containment, deu combinational thuan
tuy tren du lieu kich thuoc co dinh biet truoc) va mot bo dem nho
(motion_streak, 1 thanh ghi/slot) so sanh voi mot hang so cau hinh - khong
co logic dieu khien phu thuoc du lieu dong voi so vong lap khong bien truoc.
Phep tinh box-tu-tam-va-kich-thuoc-cu o Tang 2/4 cung la phep toan hang so -
dinh dang IEEE754/fixed-point don gian, khong them dieu khien dong.

Ghi chu ve cai dat: moi tang duyet TUAN TU qua cac track/candidate con
"chua khop" (unmatched) - day la phan CO tinh dieu khien tuan tu (sequential
logic), khac voi phep tinh IoU/containment matrix o buoc truoc (thuan tuy
combinational/parallel). Trong RTL, phan nay se la mot may trang thai huu
han co so buoc CO DINH (boi vi N va M deu co dinh), khong phai vong lap
khong gioi han.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.fusion.merge import SOURCE_DETECTOR, SOURCE_MOTION, CandidateSet
from antiuav_sot.tracker.iou_matrix import (
    compute_masked_containment_matrix,
    compute_masked_iou_matrix,
)
from antiuav_sot.tracker.track_state import TrackStateTable


@dataclasses.dataclass(frozen=True)
class TrackerFsmConfig:
    tier1_iou_threshold: float
    tier1_conf_min: float
    tier2_iou_threshold: float
    tier3_conf_min: float
    tier4_iou_threshold: float
    max_track_age: int
    # ---- Kiem soat vai tro motion (Motion Priority Conflict / Confidence
    # Override, xem docstring module) ----
    max_consecutive_motion_matches: int
    override_conf_min: float
    override_iou_threshold: float
    # Tang Confidence Override co BAT BUOC detector phai chong lan (IoU > 0 va
    # >= override_iou_threshold) voi track cu khong. False = tin detector ngay
    # ca khi IoU = 0 (box detector va track hoan toan khong giao nhau) - khi do
    # xep hang bang KHOANG CACH TAM thay vi IoU (xem _build_override_affinity).
    override_requires_overlap: bool
    # Chi cho Tang Confidence Override nham vao track da bi danh dau
    # motion-dragged (hanh vi cu), hay MOI track active (detector conf cao
    # luon duoc quyen chiem lai track).
    override_motion_dragged_only: bool
    # Gioi han khoang cach tam cho phep khi khop KHONG CAN chong lan, tinh
    # theo ti le duong cheo anh. 1.0 = khong gioi han thuc te.
    override_max_center_distance_ratio: float
    # ---- Gate "track moi phai o gan track vua mat" (xem docstring module) ----
    # Bat/tat ca co che.
    reinit_near_lost_enabled: bool
    # Ban kinh cho phep quanh vi tri track vua mat, theo ti le duong cheo anh.
    reinit_max_center_distance_ratio: float
    # So khung hinh ghost con hieu luc sau khi mat track. Qua han -> het chan.
    reinit_lost_memory_frames: int
    # Detector conf >= gia tri nay duoc khoi tao track o BAT KY dau, khong bi
    # gate chan. Dat = tier3_conf_min de detector khong bao gio bi gate.
    reinit_trust_conf_min: float
    # Duong cheo anh (pixel) - dung de chuan hoa khoang cach tam.
    image_diagonal: float


@dataclasses.dataclass
class FsmStepResult:
    """Ket qua cua mot buoc FSM (xu ly mot khung hinh).

    matched_slot_for_candidate: (M,) int32, gia tri = slot_idx neu candidate
        do khop voi mot track, hoac -1 neu khong khop voi track nao.
    tier_for_candidate: (M,) int32, tang nao da khop candidate do (1-4),
        hoac 0 neu khong khop / khong duoc xu ly (bi bo qua vi het slot).
    slot_matched_source: (n_slots,) int32, SOURCE_DETECTOR/SOURCE_MOTION cua
        candidate vua khop track do KHUNG HINH NAY, hoac -1 neu track do
        khong duoc khop khung hinh nay (dang predict-only hoac khe trong).
    newly_initiated_track_ids: danh sach Track ID moi duoc tao o Tang 3.
    expired_track_ids: danh sach Track ID bi xoa do vuot max_track_age.
    """

    matched_slot_for_candidate: np.ndarray
    tier_for_candidate: np.ndarray
    slot_matched_source: np.ndarray
    newly_initiated_track_ids: list[int]
    expired_track_ids: list[int]


TIER_UNMATCHED = 0
TIER_MIDDLE_SCORE = 1
TIER_MOTION_LINK = 2
TIER_HIGH_SCORE_INIT = 3
TIER_LOW_SCORE = 4
TIER_CONFIDENCE_OVERRIDE = 5

# Nguong noi bo dung khi goi _match_tier() cho Tang Confidence Override.
# Ma tran dua vao tang do KHONG con la IoU tho ma la "affinity" lay
# CONFIDENCE lam tieu chi chinh (xem _build_override_affinity). Moi cap HOP LE
# deu >= _OVERRIDE_AFFINITY_BASE = 1.0 va moi cap KHONG hop le bang dung 0.0,
# nen nguong so sanh chi can la mot gia tri duong rat nho.
_OVERRIDE_AFFINITY_EPS = 1e-6

# Hang so cong them de moi cap HOP LE luon > _OVERRIDE_AFFINITY_EPS, ke ca khi
# confidence = 0.0 - giu 0.0 lam gia tri "cap khong hop le" duy nhat.
_OVERRIDE_AFFINITY_BASE = 1.0
# Trong so cua hai tieu chi PHA HOA, theo dung thu tu uu tien:
#   confidence (he so 1.0)  >>  IoU (1e-3)  >>  gan tam (1e-6)
# Chenh lech confidence du nho (>= 1e-3) van thang moi muc chong lan; IoU chi
# quyet dinh khi confidence gan nhu bang nhau; khoang cach tam chi quyet dinh
# khi CA confidence va IoU bang nhau (vi du moi cap deu IoU = 0 - khong co
# term nay thi argmax se chon cap dau tien mot cach tuy y).
_OVERRIDE_IOU_TIE_BREAK = 1e-3
_OVERRIDE_DISTANCE_TIE_BREAK = 1e-6


def _apply_motion_size_preservation(
    candidate_box: np.ndarray, track_box: np.ndarray
) -> np.ndarray:
    """Khi candidate khop la tu MOTION: lay TAM (center) cua candidate_box
    lam vi tri moi, nhung GIU NGUYEN chieu rong/cao cua track_box hien tai
    (box track truoc khi cap nhat khung hinh nay).

    Ly do: motion detector (frame-diff/gradient-residual-CCL) khong dam bao
    bao phu toan bo vat the - co the chi bat duoc mot phan (fragment), nen
    kich thuoc box cua no khong dang tin cay bang vi tri (vi tri dua tren
    thay doi anh THAT). Gia dinh kich thuoc vat thay doi cham giua cac frame
    lien tiep (hop ly cho bai toan air-to-air) nen giu size cu la lua chon
    an toan hon la tin thang size tu motion.

    QUAN TRONG: ket qua ham nay se duoc dua thang vao
    ConstantVelocityKalmanBox.update() (kalman.py) - filter do KHONG loc
    w/h, chi ghi de _last_wh truc tiep tu box duoc truyen vao, nen buoc
    giu-size-cu nay BAT BUOC phai lam TRUOC khi goi Kalman, khong co co
    che nao khac sua duoc sau do.

    Ca hai box deu xyxy (x0, y0, x1, y1)."""
    cx = (candidate_box[0] + candidate_box[2]) / 2.0
    cy = (candidate_box[1] + candidate_box[3]) / 2.0
    w = track_box[2] - track_box[0]
    h = track_box[3] - track_box[1]
    return np.array(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        dtype=candidate_box.dtype,
    )


class SimpleTrackerFSM:
    """Trien khai co che lien ket 4 tang tren mot TrackStateTable co dinh N slot."""

    def __init__(self, config: TrackerFsmConfig, track_state: TrackStateTable):
        self.config = config
        self.track_state = track_state

    def step(self, candidates: CandidateSet) -> FsmStepResult:
        """Xu ly mot khung hinh: nhan CandidateSet (da merge tu detector +
        motion), cap nhat self.track_state tai cho (in-place), tra ve thong
        tin chi tiet ve viec khop de phuc vu debug/visualize/eval.
        """
        n_slots = self.track_state.n_slots
        n_candidates = candidates.boxes.shape[0]

        # Ghost cua cac track da mat gia di mot khung hinh (va het han neu qua
        # reinit_lost_memory_frames) - xem TrackStateTable.age_lost_memory().
        self.track_state.age_lost_memory(self.config.reinit_lost_memory_frames)

        matched_slot_for_candidate = np.full((n_candidates,), -1, dtype=np.int32)
        tier_for_candidate = np.full((n_candidates,), TIER_UNMATCHED, dtype=np.int32)
        slot_matched_source = np.full((n_slots,), -1, dtype=np.int32)
        slot_matched_this_frame = np.zeros((n_slots,), dtype=bool)

        iou_matrix = compute_masked_iou_matrix(
            self.track_state.boxes,
            self.track_state.active,
            candidates.boxes,
            candidates.valid_mask,
        )  # (n_slots, n_candidates)
        containment_matrix = compute_masked_containment_matrix(
            self.track_state.boxes,
            self.track_state.active,
            candidates.boxes,
            candidates.valid_mask,
        )  # (n_slots, n_candidates)

        is_detector = candidates.source == SOURCE_DETECTOR
        is_motion = candidates.source == SOURCE_MOTION

        motion_dragged_row = np.array(
            [
                self.track_state.active[slot_idx]
                and self.track_state.is_motion_dragged(
                    slot_idx, self.config.max_consecutive_motion_matches
                )
                for slot_idx in range(n_slots)
            ],
            dtype=bool,
        )

        # Ma tran "hieu qua" dung cho moi cap co candidate nguon MOTION:
        # max(IoU, containment) - sua Tier-2 IoU Failure (xem docstring
        # module) - VA zero-out cac hang track dang bi khoa
        # (motion_dragged_row) de motion khong duoc tiep tuc dan dat track
        # do (Motion Priority Conflict). Cot candidate nguon DETECTOR giu
        # nguyen IoU chuan, khong bi anh huong boi khoa nay.
        motion_effective = np.maximum(iou_matrix, containment_matrix)
        motion_effective[motion_dragged_row, :] = 0.0
        effective_matrix = iou_matrix.copy()
        effective_matrix[:, is_motion] = motion_effective[:, is_motion]

        # ---- Tang 1: Middle Score Linking (detector, IoU cao, conf trung binh tro len) ----
        tier1_candidate_mask = (
            candidates.valid_mask
            & is_detector
            & (candidates.scores >= self.config.tier1_conf_min)
        )
        self._match_tier(
            iou_matrix=iou_matrix,
            iou_threshold=self.config.tier1_iou_threshold,
            candidate_eligible_mask=tier1_candidate_mask,
            tier_id=TIER_MIDDLE_SCORE,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Tang Confidence Override (MOI): detector conf rat cao chiem
        # lai track dang bi motion/Kalman keo lech. Mac dinh KHONG doi hoi
        # chong lan (override_requires_overlap=False -> xep hang theo
        # CONFIDENCE, IoU chi pha hoa) va ap dung cho MOI track active
        # (override_motion_dragged_only=False). Chay TRUOC Tang 2 de uu tien
        # tra track ve cho detector truoc khi motion kip khoa no them mot
        # khung hinh nua. ----
        override_candidate_mask = (
            candidates.valid_mask
            & is_detector
            & (candidates.scores >= self.config.override_conf_min)
            & (tier_for_candidate == TIER_UNMATCHED)
        )
        override_affinity = self._build_override_affinity(
            iou_matrix=iou_matrix,
            motion_dragged_row=motion_dragged_row,
            candidates=candidates,
            candidate_eligible_mask=override_candidate_mask,
        )
        self._match_tier(
            iou_matrix=override_affinity,
            iou_threshold=_OVERRIDE_AFFINITY_EPS,
            candidate_eligible_mask=override_candidate_mask,
            tier_id=TIER_CONFIDENCE_OVERRIDE,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Tang 2: Motion Boxes Linking (motion, nguong max(IoU,containment)
        # cao hon, track dang bi khoa motion-dragged khong duoc xet) ----
        tier2_candidate_mask = (
            candidates.valid_mask
            & is_motion
            & (tier_for_candidate == TIER_UNMATCHED)
        )
        self._match_tier(
            iou_matrix=effective_matrix,
            iou_threshold=self.config.tier2_iou_threshold,
            candidate_eligible_mask=tier2_candidate_mask,
            tier_id=TIER_MOTION_LINK,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Tang 3: High Score Initiation (detector, conf rat cao, chua khop -> track moi) ----
        newly_initiated_track_ids: list[int] = []
        tier3_candidate_mask = (
            candidates.valid_mask
            & is_detector
            & (candidates.scores >= self.config.tier3_conf_min)
            & (tier_for_candidate == TIER_UNMATCHED)
        )
        reinit_max_distance = (
            self.config.reinit_max_center_distance_ratio * self.config.image_diagonal
        )
        for cand_idx in np.nonzero(tier3_candidate_mask)[0]:
            # Gate "gan track vua mat": detector conf >= reinit_trust_conf_min
            # duoc tin tuyet doi (khoi tao o bat ky dau). Duoi muc do, box phai
            # nam gan vi tri track vua mat - xem docstring module.
            if (
                self.config.reinit_near_lost_enabled
                and float(candidates.scores[cand_idx]) < self.config.reinit_trust_conf_min
                and not self.track_state.is_near_recent_lost(
                    candidates.boxes[cand_idx], reinit_max_distance
                )
            ):
                continue
            free_slot = self.track_state.find_free_slot()
            if free_slot is None:
                # Bang track da day - bo qua, khong khoi tao duoc them.
                # Tren FPGA day tuong duong tin hieu "table full", khong
                # phai loi, chi la gioi han tai nguyen da cau hinh (N slot).
                break
            new_id = self.track_state.init_track(
                free_slot,
                candidates.boxes[cand_idx],
                float(candidates.scores[cand_idx]),
                initial_motion_streak=0,
            )
            matched_slot_for_candidate[cand_idx] = free_slot
            tier_for_candidate[cand_idx] = TIER_HIGH_SCORE_INIT
            slot_matched_source[free_slot] = SOURCE_DETECTOR
            slot_matched_this_frame[free_slot] = True
            newly_initiated_track_ids.append(new_id)

        # ---- Tang 4: Low Score Linking (du phong cuoi, nguong noi long) ----
        tier4_candidate_mask = candidates.valid_mask & (tier_for_candidate == TIER_UNMATCHED)
        self._match_tier(
            iou_matrix=effective_matrix,
            iou_threshold=self.config.tier4_iou_threshold,
            candidate_eligible_mask=tier4_candidate_mask,
            tier_id=TIER_LOW_SCORE,
            candidates=candidates,
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            slot_matched_this_frame=slot_matched_this_frame,
        )

        # ---- Cap nhat age cho cac track KHONG duoc khop khung hinh nay ----
        for slot_idx in range(n_slots):
            if self.track_state.active[slot_idx] and not slot_matched_this_frame[slot_idx]:
                self.track_state.age_track(slot_idx)

        # ---- Xoa track vuot qua max_track_age ----
        expired_slots = self.track_state.check_and_expire_stale_tracks(
            self.config.max_track_age
        )
        expired_track_ids = expired_slots  # slot_idx da bi expire, id thuc su
        # da bi ghi de EMPTY_TRACK_ID trong track_state; ta chi can bao hieu
        # SO LUONG va VI TRI slot bi xoa cho ben goi (vd de reset Kalman tuong ung).
        for slot_idx in expired_slots:
            slot_matched_source[slot_idx] = -1

        return FsmStepResult(
            matched_slot_for_candidate=matched_slot_for_candidate,
            tier_for_candidate=tier_for_candidate,
            slot_matched_source=slot_matched_source,
            newly_initiated_track_ids=newly_initiated_track_ids,
            expired_track_ids=expired_track_ids,
        )

    def _build_override_affinity(
        self,
        iou_matrix: np.ndarray,
        motion_dragged_row: np.ndarray,
        candidates: CandidateSet,
        candidate_eligible_mask: np.ndarray,
    ) -> np.ndarray:
        """Ma tran "affinity" rieng cho Tang Confidence Override.

        Tang nay KHONG xep hang bang IoU tho. Muc tieu cua no la: detector voi
        confidence >= override_conf_min PHAI duoc quyen chiem lai track ngay ca
        khi box detector va box track hien tai KHONG giao nhau chut nao
        (IoU = 0, truong hop track da troi qua xa theo motion/Kalman). Nguong
        IoU khong con dung duoc o truong hop do, va argmax tren mot ma tran
        toan 0.0 se chon cap dau tien mot cach tuy y - nen tieu chi xep hang la
        CONFIDENCE, con IoU va khoang cach tam chi la cac term PHA HOA:

            affinity = 1.0
                     + confidence                      (he so 1.0 - tieu chi chinh)
                     + 1e-3 * IoU                      (pha hoa thu nhat)
                     + 1e-6 * (1 - d_norm)             (pha hoa thu hai)

        voi d_norm = khoang cach tam / (override_max_center_distance_ratio *
        duong cheo anh), clip ve [0, 1]. Nghia la: box detector conf cao hon
        LUON thang, du no khong chong lan track trong khi doi thu conf thap hon
        co chong lan. Chi khi hai candidate co confidence gan bang nhau
        (chenh < 1e-3) thi IoU moi quyet dinh, va chi khi ca confidence lan IoU
        bang nhau thi "gan tam hon" moi quyet dinh.

        Dieu kien HOP LE cua mot cap (track slot, candidate):
            - track active, va thuoc pham vi override (moi track active, hoac
              chi track motion-dragged neu override_motion_dragged_only = True)
            - candidate da qua candidate_eligible_mask (detector, chua khop,
              confidence >= override_conf_min)
            - VA mot trong hai:
                * co chong lan that: IoU > 0 va IoU >= override_iou_threshold
                * khoang cach tam trong gioi han
                  override_max_center_distance_ratio * duong cheo anh - chi khi
                  override_requires_overlap = False

        Moi cap KHONG hop le bang dung 0.0, nen _match_tier() voi nguong
        _OVERRIDE_AFFINITY_EPS khong the tao ra khop "ao".

        Ghi chu FPGA: van chi la so sanh so hoc tren du lieu kich thuoc co
        dinh. Gioi han khoang cach duoc kiem tra o dang BINH PHUONG (khong can
        can bac hai trong duong dan QUYET DINH); phep can bac hai o day chi
        phuc vu term pha hoa bac hai, co the bo hoan toan tren RTL neu chap
        nhan xep hang tuy y giua cac cap bang diem tuyet doi.
        """
        n_slots, n_candidates = iou_matrix.shape
        affinity = np.zeros((n_slots, n_candidates), dtype=np.float64)
        if n_slots == 0 or n_candidates == 0:
            return affinity

        # Hang track duoc phep la muc tieu cua tang nay.
        row_allowed = np.asarray(self.track_state.active, dtype=bool).copy()
        if self.config.override_motion_dragged_only:
            row_allowed &= motion_dragged_row

        pair_allowed = row_allowed[:, None] & candidate_eligible_mask[None, :]

        overlap_ok = (
            pair_allowed
            & (iou_matrix > 0.0)
            & (iou_matrix >= self.config.override_iou_threshold)
        )

        # ---- Khoang cach tam: vua la DIEU KIEN (gate khi khong chong lan),
        # vua la term pha hoa bac hai. ----
        track_cx = (self.track_state.boxes[:, 0] + self.track_state.boxes[:, 2]) / 2.0
        track_cy = (self.track_state.boxes[:, 1] + self.track_state.boxes[:, 3]) / 2.0
        cand_cx = (candidates.boxes[:, 0] + candidates.boxes[:, 2]) / 2.0
        cand_cy = (candidates.boxes[:, 1] + candidates.boxes[:, 3]) / 2.0
        dx = track_cx[:, None] - cand_cx[None, :]
        dy = track_cy[:, None] - cand_cy[None, :]
        dist_sq = dx * dx + dy * dy

        max_dist = (
            self.config.override_max_center_distance_ratio * self.config.image_diagonal
        )
        within_gate = pair_allowed & (dist_sq <= max_dist * max_dist)

        if self.config.override_requires_overlap:
            pair_valid = overlap_ok
        else:
            pair_valid = overlap_ok | within_gate
        if not np.any(pair_valid):
            return affinity

        d_norm = np.clip(np.sqrt(dist_sq) / max(max_dist, 1e-9), 0.0, 1.0)
        affinity[pair_valid] = (
            _OVERRIDE_AFFINITY_BASE
            + candidates.scores[None, :].repeat(n_slots, axis=0)[pair_valid]
            + _OVERRIDE_IOU_TIE_BREAK * iou_matrix[pair_valid]
            + _OVERRIDE_DISTANCE_TIE_BREAK * (1.0 - d_norm[pair_valid])
        )
        return affinity

    def _match_tier(
        self,
        iou_matrix: np.ndarray,
        iou_threshold: float,
        candidate_eligible_mask: np.ndarray,
        tier_id: int,
        candidates: CandidateSet,
        matched_slot_for_candidate: np.ndarray,
        tier_for_candidate: np.ndarray,
        slot_matched_source: np.ndarray,
        slot_matched_this_frame: np.ndarray,
    ) -> None:
        """
        Thuc hien khop GREEDY trong pham vi mot tang: lap qua cac cap
        (track, candidate) theo thu tu IoU giam dan, chon cap hop le dau
        tien (ca track va candidate deu CHUA duoc khop trong khung hinh nay),
        cho den khi khong con cap nao vuot nguong.

        Khi candidate duoc khop la nguon MOTION (source == SOURCE_MOTION),
        box thuc su commit vao track_state duoc dieu chinh qua
        _apply_motion_size_preservation(): giu nguyen kich thuoc track cu,
        chi cap nhat vi tri theo tam cua candidate. Candidate tu DETECTOR
        van duoc tin tuong hoan toan (ca vi tri lan kich thuoc) nhu cu.

        So vong lap toi da = n_slots * n_candidates (CO DINH, biet truoc tai
        thoi diem cau hinh) - phu hop anh xa thanh mot may trang thai co so
        buoc gioi han ro rang, khong phai vong lap "while co dieu kien dong".
        """
        n_slots, n_candidates = iou_matrix.shape

        # Ap dung mask: track chua khop khung hinh nay VA candidate hop le cho tang nay.
        working_iou = iou_matrix.copy()
        working_iou[~(~slot_matched_this_frame)[:, None].repeat(n_candidates, axis=1)] = 0.0
        # (dong tren giu lai chi cac hang track CHUA matched; slot_matched_this_frame
        #  co the da True do Tang truoc trong CUNG mot lan goi step(), vi du track
        #  vua duoc Tang 1 khop se khong duoc xet lai o Tang 2/4.)
        working_iou = working_iou * candidate_eligible_mask[None, :].astype(np.float32)

        max_iterations = n_slots * n_candidates
        for _ in range(max_iterations):
            if working_iou.size == 0:
                break
            best_idx = np.argmax(working_iou)
            best_slot, best_cand = np.unravel_index(best_idx, working_iou.shape)
            best_value = working_iou[best_slot, best_cand]

            if best_value < iou_threshold:
                break  # khong con cap nao du nguong

            # Khop cap (best_slot, best_cand). Neu nguon la MOTION, dieu
            # chinh box truoc khi commit: giu size track cu, chi lay vi tri
            # tu candidate (xem docstring ham nay va _apply_motion_size_preservation).
            candidate_source = int(candidates.source[best_cand])
            box_to_commit = candidates.boxes[best_cand]
            if candidate_source == SOURCE_MOTION:
                box_to_commit = _apply_motion_size_preservation(
                    box_to_commit, self.track_state.boxes[best_slot]
                )

            self.track_state.update_track(
                int(best_slot),
                box_to_commit,
                float(candidates.scores[best_cand]),
            )
            if candidate_source == SOURCE_MOTION:
                self.track_state.note_motion_match(int(best_slot))
            else:
                self.track_state.note_detector_match(int(best_slot))
            matched_slot_for_candidate[best_cand] = best_slot
            tier_for_candidate[best_cand] = tier_id
            slot_matched_source[best_slot] = candidate_source
            slot_matched_this_frame[best_slot] = True

            # Loai bo hang/cot vua dung de tranh khop lai
            working_iou[best_slot, :] = 0.0
            working_iou[:, best_cand] = 0.0
