"""
adaptive_size.py
==================
Sua Issue 3 (Static Size Limits): motion.min_box_size / motion.max_box_size
trong fpga_constraints.yaml truoc day la HANG SO CO DINH, chon tay dua tren
kich thuoc UAV "trung binh" quan sat truoc o mot khoang cach/goc nhin nhat
dinh. Trong thuc te bay UAV-doi-UAV, kich thuoc pixel THAT cua muc tieu thay
doi lien tuc (tien/lui, zoom camera) - mot cap hang so co dinh se:
    - Qua chat khi UAV o gan (box that lon hon max_box_size -> bi loc bo,
      mat het motion candidate dung luc can no nhat).
    - Qua long khi UAV o xa (box nhieu nen cung lot qua min_box_size, lam
      day candidate list bang rac - xem them Issue 6).

AdaptiveBoxSizeEstimator giai quyet bang cach theo doi mot cua so truot
(deque, kich thuoc co dinh size_adapt_window) cac kich thuoc box DETECTOR
gan nhat (nguon dang tin cay nhat ve kich thuoc that cua vat, khac voi motion
box von co the chi la fragment - xem tracker/simple_tracker_fsm.py). Tu do
suy ra khoang [min_box_size, max_box_size] DONG theo tung frame:

    min_box_size_dynamic = clip(shrink_factor * p20(recent_sizes), floor, ...)
    max_box_size_dynamic = clip(grow_factor   * p80(recent_sizes), ..., ceiling)

`floor`/`ceiling` chinh la hai gia tri BAT BUOC cu (motion.min_box_size/
max_box_size trong config) - GIU NGUYEN vai tro "gioi han an toan tuyet
doi" (chan duoi/chan tren cung), khong bao gio bi vuot qua du lich su detector
noi gi - tranh truong hop mot vai detection loi (box sai lech qua lon/nho)
lam khoang dong bi keo theo sai lech cuc doan.

Khi chua du du lieu (< size_adapt_min_samples box), fallback nguyen ve cap
(floor, ceiling) tinh - dung HANH VI CU het cong dung khi vua khoi dong
sequence hoac detector chua bat duoc gi.

Thiet ke KHONG dung numpy array co dinh (deque do dai thay doi tu 0 den
size_adapt_window) vi day la thanh phan chi ton tai o GIAI DOAN HUAN
LUYEN/MO PHONG tren host (Python), khong nam tren duong di combinational cua
FPGA - tren RTL, gia tri [min_box_size, max_box_size] se la mot THANH GHI
duoc firmware/host cap nhat dinh ky (vd moi K frame) tu thong ke detector
tich luy o phia host, khong phai logic tinh trong duong pipeline framerate.
"""

from __future__ import annotations

import collections
import dataclasses

import numpy as np


@dataclasses.dataclass
class AdaptiveBoxSizeEstimator:
    """Uoc luong [min_box_size, max_box_size] dong cho nhanh motion, dua
    tren lich su kich thuoc box DETECTOR gan nhat.

    window:            so box detector gan nhat duoc giu de tinh thong ke
                        (deque tu dong loai bo box cu nhat khi day).
    floor/ceiling:      gioi han an toan tuyet doi - gia tri dong KHONG BAO
                        GIO vuot ra ngoai khoang nay (chinh la
                        motion.min_box_size/max_box_size cu trong config).
    shrink_factor:      he so co vao percentile thap (p20) de ra min_box_size
                        dong - < 1.0 de min_box_size dong THAP HON kich thuoc
                        detector gan day, chua du "bien" cho truong hop
                        motion chi bat duoc MOT PHAN (fragment) vat, von nho
                        hon box detector day du.
    grow_factor:        he so nhan vao percentile cao (p80) de ra
                        max_box_size dong - > 1.0 de chua bien cho truong hop
                        vat dang lai gan (box se lon dan giua cac frame).
    min_samples:        so box toi thieu can co truoc khi tin thong ke dong;
                        chua du thi fallback ve (floor, ceiling) tinh.
    """

    window: int
    floor: int
    ceiling: int
    shrink_factor: float = 0.6
    grow_factor: float = 1.6
    min_samples: int = 5

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("AdaptiveBoxSizeEstimator.window phai > 0")
        if self.floor <= 0 or self.ceiling <= self.floor:
            raise ValueError(
                f"floor ({self.floor}) phai > 0 va nho hon ceiling ({self.ceiling})"
            )
        self._recent_sizes: collections.deque[float] = collections.deque(maxlen=self.window)

    def update(self, detector_boxes_xyxy: np.ndarray, detector_valid_mask: np.ndarray) -> None:
        """Nap them kich thuoc (ca w LAN h, coi la hai mau doc lap - vat co
        the dai/rong lech nhau, khong muon mot chieu keo lech uoc luong cua
        chieu kia) tu cac box detector HOP LE cua frame vua roi."""
        valid_idx = np.nonzero(detector_valid_mask)[0]
        for i in valid_idx:
            x0, y0, x1, y1 = detector_boxes_xyxy[i]
            w, h = float(x1 - x0), float(y1 - y0)
            if w > 0:
                self._recent_sizes.append(w)
            if h > 0:
                self._recent_sizes.append(h)

    def estimate(self) -> tuple[int, int]:
        """Tra ve (min_box_size, max_box_size) dong cho frame hien tai, da
        clip trong [floor, ceiling]. An toan goi ngay ca khi chua co mau nao."""
        if len(self._recent_sizes) < self.min_samples:
            return self.floor, self.ceiling

        sizes = np.array(self._recent_sizes, dtype=np.float32)
        p20 = float(np.percentile(sizes, 20))
        p80 = float(np.percentile(sizes, 80))

        dynamic_min = int(round(self.shrink_factor * p20))
        dynamic_max = int(round(self.grow_factor * p80))

        dynamic_min = max(self.floor, min(dynamic_min, self.ceiling))
        dynamic_max = max(self.floor, min(dynamic_max, self.ceiling))

        # Bao ve chan cuoi: neu percentile suy bien lam dynamic_min >= dynamic_max
        # (vd toan bo mau gan nhu giong het nhau), nong ra it nhat 1px moi ben
        # de khoang loc khong bao gio rong = 0 hoac dao chieu.
        if dynamic_min >= dynamic_max:
            dynamic_min = max(self.floor, dynamic_max - 1)

        return dynamic_min, dynamic_max

    def reset(self) -> None:
        """Goi khi bat dau mot sequence moi - lich su kich thuoc cua sequence
        truoc KHONG con lien quan (UAV/goc camera moi co the hoan toan khac)."""
        self._recent_sizes.clear()
