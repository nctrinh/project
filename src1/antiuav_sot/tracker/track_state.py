"""
Bang trang thai track (Track state table): mang co kich thuoc co dinh N khe
(slot), moi khe luu trang thai mot track doc lap (muc 3.5, 4.1 bao cao FPGA).

Day la module QUYET DINH pipeline hoat dong o che do single-object (N=1)
hay multi-object (N>1) - khong phai detector hay motion cue. Voi N=1
(Phuong an A), bang nay chi con mot khe duy nhat, vector so sanh IoU
1xM thay vi ma tran NxM, khong con van de quan ly Track ID hay identity
switch (muc 4.2).

Thiet ke huong toi RTL: tat ca truong deu la mang numpy co dtype/shape co
dinh, KHONG dung list/dict Python dong - phan anh dung cach mot bang trong
BRAM se duoc to chuc (mang cac thanh ghi/hang co dia chi co dinh).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.utils.box_ops import INVALID_BOX, make_empty_boxes

# Track ID = -1 nghia la khe dang TRONG (chua co track nao chiem giu).
EMPTY_TRACK_ID = -1


@dataclasses.dataclass
class TrackStateTable:
    """
    boxes:      (n_slots, 4) xyxy - vi tri box gan nhat cua tung track
    track_ids:  (n_slots,) int32 - ma track, EMPTY_TRACK_ID neu khe trong
    ages:       (n_slots,) int32 - so khung hinh LIEN TIEP khong khop
                (dung cho co che het han, muc 3.5)
    active:     (n_slots,) bool - True neu khe dang chua mot track hop le
    confidence: (n_slots,) float32 - confidence cua lan khop gan nhat
                (0.0 neu track duoc du bao boi Kalman ma khong khop that)
    motion_streak: (n_slots,) int32 - so khung hinh LIEN TIEP GAN NHAT track
                nay CHI duoc cap nhat boi nguon MOTION (Tang 2/4 hoac promote
                tu Tentative Tracks), KHONG duoc detector xac nhan lai lan
                nao. Reset ve 0 moi khi khop voi mot box tu DETECTOR (Tang 1/
                3/4/Confidence Override). Day la co che chinh de "kiem soat
                chat" muc do motion duoc phep dan dat track (Motion Priority
                Conflict) - xem simple_tracker_fsm.py (max_consecutive_motion_
                matches, Tier Confidence Override).
    lost_boxes: (n_slots, 4) xyxy - vi tri CUOI CUNG cua track vua bi xoa khoi
                khe nay ("bong ma"/ghost cua track da mat). Giu lai de chan
                viec khoi tao track moi o mot vi tri XA hoan toan ngay sau khi
                mat track: mot UAV khong the dich chuyen tuc thoi, nen candidate
                motion (hoac detector conf thap) o goc khac cua anh gan nhu chac
                chan la nhieu nen, khong phai chinh vat vua mat.
    lost_ages:  (n_slots,) int32 - so khung hinh da troi qua ke tu khi ghost
                duoc ghi lai. Vuot qua gioi han cau hinh thi ghost het han va
                khong con chan gi nua (vat co the da di rat xa, hoac canh da
                doi han).
    lost_valid: (n_slots,) bool - True neu khe nay dang giu mot ghost con han.
    """

    boxes: np.ndarray
    track_ids: np.ndarray
    ages: np.ndarray
    active: np.ndarray
    confidence: np.ndarray
    motion_streak: np.ndarray
    lost_boxes: np.ndarray
    lost_ages: np.ndarray
    lost_valid: np.ndarray
    n_slots: int
    _next_track_id: int = 0

    @classmethod
    def create_empty(cls, n_slots: int) -> "TrackStateTable":
        return cls(
            boxes=make_empty_boxes(n_slots),
            track_ids=np.full((n_slots,), EMPTY_TRACK_ID, dtype=np.int32),
            ages=np.zeros((n_slots,), dtype=np.int32),
            active=np.zeros((n_slots,), dtype=bool),
            confidence=np.zeros((n_slots,), dtype=np.float32),
            motion_streak=np.zeros((n_slots,), dtype=np.int32),
            lost_boxes=make_empty_boxes(n_slots),
            lost_ages=np.zeros((n_slots,), dtype=np.int32),
            lost_valid=np.zeros((n_slots,), dtype=bool),
            n_slots=n_slots,
        )

    def num_active(self) -> int:
        return int(self.active.sum())

    def find_free_slot(self) -> int | None:
        """Tim khe TRONG dau tien de khoi tao track moi (Tang 3: High Score
        Initiation). Tra ve None neu bang da day (het cho, phai bo qua detection
        moi cho den khi co track het han)."""
        free_indices = np.nonzero(~self.active)[0]
        if free_indices.shape[0] == 0:
            return None
        return int(free_indices[0])

    def init_track(
        self,
        slot_idx: int,
        box_xyxy: np.ndarray,
        confidence: float,
        initial_motion_streak: int = 0,
    ) -> int:
        """Khoi tao mot track moi tai khe slot_idx, cap Track ID moi.
        Tra ve Track ID vua cap.

        initial_motion_streak: gia tri khoi tao cho motion_streak - Tang 3
            (High Score Initiation, tu detector) dung mac dinh 0 (track moi
            hoan toan dang tin cay). Track duoc PROMOTE tu Tentative Tracks
            (chi tu motion, xem tentative_tracks.py/pipeline.py) nen truyen
            gia tri >= max_consecutive_motion_matches ngay tu dau: track loai
            nay von di la "tu khoi tao boi motion" nen phai bi coi la da o
            trang thai motion-dragged tu khung hinh dau tien, san sang bi
            Confidence Override chiem lai boi bat ky detector box tin cay cao
            nao ngay khi xuat hien - dung tinh than "motion self-init phai bi
            kiem soat chat" (Motion Priority Conflict)."""
        if self.active[slot_idx]:
            raise ValueError(
                f"Khe {slot_idx} da co track (id={self.track_ids[slot_idx]}), "
                "khong the khoi tao de nghi tai day."
            )
        new_id = self._next_track_id
        self._next_track_id += 1

        self.boxes[slot_idx] = box_xyxy
        self.track_ids[slot_idx] = new_id
        self.ages[slot_idx] = 0
        self.active[slot_idx] = True
        self.confidence[slot_idx] = confidence
        self.motion_streak[slot_idx] = initial_motion_streak
        # Khe nay vua co track tro lai - ghost cua track cu khong con y nghia.
        self.clear_lost_memory(slot_idx)
        return new_id

    def update_track(self, slot_idx: int, box_xyxy: np.ndarray, confidence: float) -> None:
        """Cap nhat track hien co tai slot_idx voi box vua khop duoc.
        Reset age ve 0 vi vua khop thanh cong khung hinh nay.

        KHONG dong cham motion_streak o day - FSM (noi biet ro nguon
        DETECTOR/MOTION cua candidate vua khop) chiu trach nhiem goi
        note_detector_match()/note_motion_match() SAU khi goi ham nay,
        xem simple_tracker_fsm.py."""
        self.boxes[slot_idx] = box_xyxy
        self.ages[slot_idx] = 0
        self.confidence[slot_idx] = confidence

    def note_detector_match(self, slot_idx: int) -> None:
        """Bao hieu track tai slot_idx vua duoc mot box DETECTOR xac nhan
        khung hinh nay - reset motion_streak ve 0 (track duoc 'chuoc lai'
        khoi trang thai motion-dragged)."""
        self.motion_streak[slot_idx] = 0

    def note_motion_match(self, slot_idx: int) -> None:
        """Bao hieu track tai slot_idx vua duoc mot box MOTION xac nhan
        (khong phai detector) khung hinh nay - tang motion_streak them 1."""
        self.motion_streak[slot_idx] += 1

    def is_motion_dragged(self, slot_idx: int, max_consecutive_motion_matches: int) -> bool:
        """True neu track tai slot_idx da qua nhieu khung hinh LIEN TIEP chi
        duoc motion cap nhat (khong con duoc detector xac nhan) - danh dau no
        la ung vien cho Tier Confidence Override va bi loai khoi dieu kien
        du duoc Tang 2/4 tiep tuc cap nhat boi motion (xem simple_tracker_fsm.py)."""
        return int(self.motion_streak[slot_idx]) >= max_consecutive_motion_matches

    def age_track(self, slot_idx: int) -> None:
        """Tang bo dem tuoi track khi khong khop duoc box nao khung hinh nay."""
        self.ages[slot_idx] += 1

    def expire_track(self, slot_idx: int) -> None:
        """Xoa track tai slot_idx, danh dau khe tro lai trang thai trong.

        Vi tri cuoi cung cua track duoc giu lai lam "ghost" (lost_boxes/
        lost_ages/lost_valid) de cac buoc khoi tao track moi biet vat VUA o
        dau - xem is_near_recent_lost()."""
        self.lost_boxes[slot_idx] = self.boxes[slot_idx]
        self.lost_ages[slot_idx] = 0
        self.lost_valid[slot_idx] = True
        self.boxes[slot_idx] = INVALID_BOX
        self.track_ids[slot_idx] = EMPTY_TRACK_ID
        self.ages[slot_idx] = 0
        self.active[slot_idx] = False
        self.confidence[slot_idx] = 0.0
        self.motion_streak[slot_idx] = 0

    def check_and_expire_stale_tracks(self, max_track_age: int) -> list[int]:
        """Duyet toan bo N khe, xoa nhung track co age > max_track_age.
        Tra ve danh sach slot_idx vua bi xoa (de log/debug)."""
        expired_slots = []
        for slot_idx in range(self.n_slots):
            if self.active[slot_idx] and self.ages[slot_idx] > max_track_age:
                expired_slots.append(slot_idx)
                self.expire_track(slot_idx)
        return expired_slots

    def clear_lost_memory(self, slot_idx: int) -> None:
        """Xoa ghost cua mot khe (khi khe do co track moi, hoac ghost het han)."""
        self.lost_boxes[slot_idx] = INVALID_BOX
        self.lost_ages[slot_idx] = 0
        self.lost_valid[slot_idx] = False

    def age_lost_memory(self, max_lost_memory_frames: int) -> None:
        """Tang tuoi moi ghost them mot khung hinh, xoa ghost qua han.

        Goi DUNG MOT LAN moi khung hinh, o DAU buoc FSM - ghost vua duoc ghi
        trong chinh khung hinh nay (expire_track o CUOI buoc) se bat dau dem
        tu khung hinh KE TIEP."""
        for slot_idx in range(self.n_slots):
            if not self.lost_valid[slot_idx]:
                continue
            self.lost_ages[slot_idx] += 1
            if self.lost_ages[slot_idx] > max_lost_memory_frames:
                self.clear_lost_memory(slot_idx)

    def has_recent_lost(self) -> bool:
        """True neu con it nhat mot ghost con han."""
        return bool(self.lost_valid.any())

    def is_near_recent_lost(
        self, box_xyxy: np.ndarray, max_center_distance: float
    ) -> bool:
        """True neu box nam DU GAN mot ghost con han - tuc no co the la chinh
        vat vua mat, dang xuat hien lai.

        Tra ve True khi KHONG con ghost nao con han: luc do khong co thong tin
        gi de chan, moi vi tri deu hop le nhu nhau (vd dau sequence, hoac da
        mat track qua lau). Nghia la ham nay chi SIET khi thuc su biet vat vua
        o dau, khong bao gio chan vinh vien viec khoi tao track moi.

        Khoang cach do giua TAM hai box, so sanh o dang BINH PHUONG (khong can
        can bac hai - phu hop RTL)."""
        if not self.lost_valid.any():
            return True
        cx = (float(box_xyxy[0]) + float(box_xyxy[2])) / 2.0
        cy = (float(box_xyxy[1]) + float(box_xyxy[3])) / 2.0
        lost_cx = (self.lost_boxes[:, 0] + self.lost_boxes[:, 2]) / 2.0
        lost_cy = (self.lost_boxes[:, 1] + self.lost_boxes[:, 3]) / 2.0
        dist_sq = (lost_cx - cx) ** 2 + (lost_cy - cy) ** 2
        return bool((dist_sq[self.lost_valid] <= max_center_distance ** 2).any())

    def get_active_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """Tra ve (boxes, slot_indices) chi cho cac khe active - dung khi tinh
        IoU matrix voi candidate moi (khong can so voi khe trong)."""
        active_indices = np.nonzero(self.active)[0]
        return self.boxes[active_indices], active_indices
