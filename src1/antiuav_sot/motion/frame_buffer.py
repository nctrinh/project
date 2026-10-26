"""
Frame differencing: D(x, y) = I_t(x, y) - I_(t-1)(x, y), voi hai che do:

    "absdiff" (mac dinh, tuong thich nguoc): D(x,y) = |I_t - I_(t-1)|
        Bat ca vung "nong len" LAN vung "nguoi di" - dung khi khong biet
        truoc dau hieu nhiet cua vat the (background thay doi ca hai chieu,
        camera rung, v.v).

    "positive_only": D(x,y) = max(0, I_t - I_(t-1))
        CHI giu vung "nong len" (I_t > I_(t-1)), bo hoan toan vung "nguoi
        di". Phu hop khi vat the muc tieu LUON sang hon background trong
        anh IR (dung dac diem vat ly cua UAV: dong co/pin toa nhiet, luon
        nong hon nen troi/mat dat) - day la gia dinh HOP LY cho bai toan
        Anti-UAV thermal, VA co 2 loi ich ro rang:
        1. Chat luong candidate tot hon: absdiff tao ra HAI vung sang cho
           mot vat the di chuyen (noi den + noi di, "bong ma" phia sau) -
           chi mot trong hai la vi tri THAT. positive_only loai bo vung
           "noi di" (nguoi di la am, bi cat ve 0), giam mot nguon nhieu/
           candidate gia truoc khi toi buoc block-threshold.
        2. Don gian hoa RTL: phep tru co dau + clip am ve 0 (mot thao tac
           so sanh) don gian hon abs() (can dao dau neu ket qua am), giam
           mot buoc logic tren duong pipeline.

    Chon che do qua tham so `mode` - KHONG thay doi hanh vi mac dinh de
    tranh pha vo cac cau hinh dang dung "absdiff" (vd truong hop background
    thay doi hai chieu, hoac chua xac nhan duoc dac diem nhiet cua muc tieu).

Thay the ViBe (background modeling xac suat, N=20 mau/pixel, ~6.5MB, can
random update logic) bang mot phep tru don gian - hoan toan la combinational
logic khong trang thai, xu ly duoc 1+ pixel/chu ky xung nhip tuy so data
lane (muc 3.2 bao cao FPGA).

Module nay CHI giu logic frame T-1 (Frame Buffer, muc 3.1). Viec quan ly
double-buffering vat ly (BRAM/URAM) la chi tiet hien thuc hoa RTL, o day
mo phong bang cach giu don gian một numpy array trong bo nho Python.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

FrameDiffMode = Literal["absdiff", "positive_only"]


from collections import deque

class FrameBuffer:
    def __init__(self, lag: int = 5):
        self.lag = lag
        self._frames: deque[np.ndarray] = deque(maxlen=lag)

    def has_previous(self) -> bool:
        return len(self._frames) == self.lag   # đủ t-lag mới tính motion

    def get_previous(self):
        return self._frames[0] if self.has_previous() else None  # frame t-lag

    def update(self, frame):
        self._frames.append(frame.copy())

    def reset(self):
        self._frames.clear()
