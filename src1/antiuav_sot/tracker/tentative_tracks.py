"""
tentative_tracks.py
=====================
Bang trang thai PHU (khong phai TrackStateTable chinh thuc) theo doi cac
candidate MOTION chua khop bat ky track nao trong TrackStateTable (con
TIER_UNMATCHED sau het 4 tang cua SimpleTrackerFSM). Neu mot vi tri duoc
motion xac nhan LIEN TUC qua du so khung hinh (motion_track_confirm_frames),
de xuat promote thanh mot track THAT trong TrackStateTable.

Ly do can module nay: Tang 3 (High Score Initiation) cua SimpleTrackerFSM
CHI khoi tao track moi tu candidate DETECTOR dat tier3_conf_min. Neu mot
vat qua nho/mo nhat de detector TUNG DAT nguong do dù chỉ một lần, track se
KHONG BAO GIO duoc tao, du motion detector thay no lien tuc. Day la "Tier E"
duoc thao luan truoc do trong thiet ke fusion.

Thiet ke van theo nguyen tac fixed-size cua toan repo: mang N slot co dinh
(max_tentative_motion_tracks), khong dung list/dict Python dong.

Cung tinh than voi TrackStateTable nhung DON GIAN HON nhieu - khong can
Track ID, khong can confidence, khong can Kalman - chi can bo dem streak/
miss va vi tri gan nhat.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from antiuav_sot.tracker.iou_matrix import compute_masked_iou_matrix
from antiuav_sot.utils.box_ops import INVALID_BOX, make_empty_boxes


@dataclasses.dataclass
class TentativeTrackTable:
    """
    boxes:      (max_slots, 4) xyxy - vi tri gan nhat cua tung ung vien
    streak:     (max_slots,) int32 - so khung hinh LIEN TIEP duoc motion
                xac nhan (reset ve 0 khi bi xoa, KHONG reset khi chi mot
                lan mat khop - xem miss_count)
    miss_count: (max_slots,) int32 - so khung hinh LIEN TIEP KHONG duoc
                motion xac nhan lai; vuot qua max_miss thi xoa slot
    active:     (max_slots,) bool - True neu khe dang theo doi mot ung vien
    max_slots:  int
    """

    boxes: np.ndarray
    streak: np.ndarray
    miss_count: np.ndarray
    active: np.ndarray
    max_slots: int

    @classmethod
    def create_empty(cls, max_slots: int) -> "TentativeTrackTable":
        return cls(
            boxes=make_empty_boxes(max_slots),
            streak=np.zeros((max_slots,), dtype=np.int32),
            miss_count=np.zeros((max_slots,), dtype=np.int32),
            active=np.zeros((max_slots,), dtype=bool),
            max_slots=max_slots,
        )

    def find_free_slot(self) -> int | None:
        free_indices = np.nonzero(~self.active)[0]
        if free_indices.shape[0] == 0:
            return None
        return int(free_indices[0])

    def _clear_slot(self, slot_idx: int) -> None:
        self.boxes[slot_idx] = INVALID_BOX
        self.streak[slot_idx] = 0
        self.miss_count[slot_idx] = 0
        self.active[slot_idx] = False

    def step(
        self,
        motion_boxes: np.ndarray,
        motion_valid_mask: np.ndarray,
        iou_threshold: float,
        max_miss: int,
        confirm_frames: int,
    ) -> list[int]:
        """
        motion_boxes/motion_valid_mask: CandidateSet.boxes/valid_mask da loc
        chi con cac candidate nguon MOTION va con TIER_UNMATCHED sau
        SimpleTrackerFSM.step() (nghia la khong khop bat ky track chinh
        thuc nao trong TrackStateTable khung hinh nay).

        Tra ve danh sach slot_idx (trong bang nay) VUA DAT streak >=
        confirm_frames - ben goi (SotPipeline) tu quyet dinh co con cho
        trong TrackStateTable de promote khong, roi goi promote_and_clear().
        """
        n_cand = motion_boxes.shape[0]
        matched_slot_for_candidate = np.full((n_cand,), -1, dtype=np.int32)
        slot_matched_this_frame = np.zeros((self.max_slots,), dtype=bool)

        if self.active.any() and motion_valid_mask.any():
            iou_matrix = compute_masked_iou_matrix(
                self.boxes, self.active, motion_boxes, motion_valid_mask
            )
            working = iou_matrix.copy()
            max_iterations = self.max_slots * n_cand
            for _ in range(max_iterations):
                if working.size == 0:
                    break
                best_idx = np.argmax(working)
                best_slot, best_cand = np.unravel_index(best_idx, working.shape)
                if working[best_slot, best_cand] < iou_threshold:
                    break

                self.boxes[best_slot] = motion_boxes[best_cand]
                self.streak[best_slot] += 1
                self.miss_count[best_slot] = 0
                matched_slot_for_candidate[best_cand] = best_slot
                slot_matched_this_frame[best_slot] = True

                working[best_slot, :] = 0.0
                working[:, best_cand] = 0.0

        # Slot active nhung khong duoc khop khung hinh nay: tang miss_count,
        # xoa neu vuot qua nguong cho phep.
        for slot_idx in range(self.max_slots):
            if self.active[slot_idx] and not slot_matched_this_frame[slot_idx]:
                self.miss_count[slot_idx] += 1
                if self.miss_count[slot_idx] > max_miss:
                    self._clear_slot(slot_idx)

        # Candidate motion chua khop slot tentative nao: tao slot moi
        # (streak=1) neu con cho trong.
        for cand_idx in range(n_cand):
            if motion_valid_mask[cand_idx] and matched_slot_for_candidate[cand_idx] == -1:
                free = self.find_free_slot()
                if free is not None:
                    self.boxes[free] = motion_boxes[cand_idx]
                    self.streak[free] = 1
                    self.miss_count[free] = 0
                    self.active[free] = True

        return [
            slot_idx
            for slot_idx in range(self.max_slots)
            if self.active[slot_idx] and self.streak[slot_idx] >= confirm_frames
        ]

    def promote_and_clear(self, slot_idx: int) -> np.ndarray:
        """Lay box hien tai cua slot (de ben goi init_track() vao
        TrackStateTable), roi xoa slot nay khoi bang tentative."""
        box = self.boxes[slot_idx].copy()
        self._clear_slot(slot_idx)
        return box

    def reset(self) -> None:
        """Goi khi bat dau mot sequence moi."""
        self.boxes[:] = INVALID_BOX
        self.streak[:] = 0
        self.miss_count[:] = 0
        self.active[:] = False
