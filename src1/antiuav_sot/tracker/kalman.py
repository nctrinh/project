"""
Bo loc Kalman tuyen tinh don gian, state = [x, y, vx, vy] (4 chieu, muc 3.6
bao cao FPGA), thay the vai tro "hau xu ly bang SOT" (MixFormer/OSTrack)
cua bai bao goc khi mot track bi mat khop trong vai khung hinh ngan.

Ly do thay the: SOT dua tren hoc sau can crop dong vung anh, chay hai
nhanh mang rieng biet (template/search), tinh tuong quan/attention giua
hai nhanh - kich thuoc dong va do phuc tap tinh toan lon, khong phu hop
rang buoc FPGA. Kalman chi can luu mot vector trang thai nho va thuc hien
nhan ma tran 4x4 moi khung hinh - vai chuc khoi DSP la du.

Danh doi: Kalman chi du bao theo mo hinh chuyen dong TUYEN TINH don gian,
khong "quan sat lai" anh de xac nhan hinh dang/vi tri con dung nhu SOT.
Do chinh xac se kem hon khi track bi che khuat lau hoac doi huong dot ngot -
day la danh doi co chu dich giua do chinh xac va kha nang trien khai FPGA.

QUAN TRONG ve kich thuoc (w, h): filter nay CHI theo doi TAM box trong
state, w/h duoc giu CO DINH tu lan quan sat gan nhat (_last_wh), KHONG di
qua buoc loc Kalman nao ca. Vi vay co che "giu size track cu khi khop voi
motion" (xem tracker/simple_tracker_fsm.py, _apply_motion_size_preservation)
PHAI duoc thuc hien TRUOC KHI box duoc dua vao update() o day - neu khong,
_last_wh se bi ghi de bang kich thuoc (co the sai/thieu) cua motion box
ngay lap tuc, khong co co che nao o day tu sua duoc dieu do.

CAP NHAT: update() nhan them tham so tuy chon measurement_noise de ghi de
R CHO RIENG lan goi do - dung khi quan sat den tu mot nguon kem tin cay ve
VI TRI hon binh thuong (vi du: candidate tu motion detector co the lech tam
that su cua vat neu no chi bat duoc mot phan/fragment). Khong truyen gi
(mac dinh None) thi hanh vi giong het ban goc, dung self.R co dinh tu luc
khoi tao.

Cai dat nay dung ma tran hang so (constant velocity model) viet tay bang
numpy, KHONG dung thu vien filterpy cho duong inference chinh - de giu
logic hoan toan minh bach va de doi chieu 1-1 sang RTL/HLS (4x4 matmul).
filterpy trong requirements.txt chi du phong cho muc dich doi chieu/kiem
thu nhanh, khong phai dependency bat buoc cua pipeline.
"""

from __future__ import annotations

import numpy as np


class ConstantVelocityKalmanBox:
    """Kalman filter 4 chieu [x, y, vx, vy] cho tam box (center x, center y).

    Chi theo doi TAM box, KHONG theo doi w/h - vi report chi de cap du bao
    vi tri ("du bao vi tri tiep theo cua track"), va giu w/h co dinh theo
    lan quan sat gan nhat don gian hoa dang ke ma tran trang thai (4x4 thay
    vi 8x8 neu them dw/dh).
    """

    def __init__(self, process_noise: float, measurement_noise: float):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State transition: x_k = F @ x_{k-1}, dt = 1 frame
        self.F = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # Measurement matrix: chi quan sat duoc (x, y), khong quan sat truc tiep (vx, vy)
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        self.state: np.ndarray | None = None      # (4,)
        self.covariance: np.ndarray | None = None  # (4, 4)
        self._last_wh: np.ndarray | None = None    # (2,) - w, h giu co dinh tu lan quan sat gan nhat

    def is_initialized(self) -> bool:
        return self.state is not None

    def init(self, box_xyxy: np.ndarray) -> None:
        """Khoi tao filter tu lan quan sat dau tien. Van toc ban dau = 0."""
        cx, cy, w, h = _xyxy_to_cxcywh(box_xyxy)
        self.state = np.array([cx, cy, 0.0, 0.0], dtype=np.float32)
        self.covariance = np.eye(4, dtype=np.float32) * 10.0
        self._last_wh = np.array([w, h], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """Du bao trang thai khung hinh tiep theo (khong co quan sat moi).
        Tra ve box xyxy du bao, dung khi track bi mat khop tam thoi."""
        if self.state is None:
            raise RuntimeError("Kalman filter chua duoc init() truoc khi predict()")

        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return _cxcywh_to_xyxy(self.state[0], self.state[1], *self._last_wh)

    def update(
        self,
        box_xyxy: np.ndarray,
        measurement_noise: float | None = None,
    ) -> np.ndarray:
        """Cap nhat filter voi mot quan sat that (track vua khop duoc).
        Tra ve box xyxy sau khi da loc (smoothed), va cap nhat w/h co dinh moi
        TRUC TIEP tu box_xyxy dua vao - xem ghi chu dau file ve viec nay
        phai duoc xu ly (giu-size-cu neu can) truoc khi goi ham nay.

        measurement_noise: neu khac None, GHI DE R (chi cho lan goi nay)
            bang eye(2) * measurement_noise thay vi dung self.R co dinh.
            R lon hon -> Kalman gain nho hon -> tin quan sat it hon, tin
            du bao theo van toc rieng (state hien tai) nhieu hon. Dung khi
            nguon quan sat (vd motion candidate) kem tin cay ve vi tri hon
            binh thuong.

        QUAN TRONG: luon thuc hien buoc PREDICT (state transition + tang
        covariance theo Q) TRUOC KHI ket hop quan sat moi - day la trinh tu
        BAT BUOC cua Kalman filter chuan (predict -> update moi chu ky),
        khong duoc goi thang buoc update ma bo qua predict. Neu bo qua,
        covariance khong bao gio duoc "boi" lai theo Q giua cac lan goi
        update(), Kalman gain K se co xu huong giam dan qua tung lan goi
        lien tiep -> filter ngay cang it tin quan sat moi, ngay ca khi
        detection dung moi frame - day CHINH LA nguyen nhan gay hien tuong
        track "troi" dan (IoU giam deu) da quan sat duoc qua
        scripts/eval_pipeline_stages.py, KHONG lien quan gi den viec track
        co bi mat khop hay khong.
        """
        if self.state is None:
            self.init(box_xyxy)
            return box_xyxy

        # Buoc PREDICT - bat buoc truoc update, xem ghi chu o tren.
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

        cx, cy, w, h = _xyxy_to_cxcywh(box_xyxy)
        z = np.array([cx, cy], dtype=np.float32)

        R = self.R if measurement_noise is None else np.eye(2, dtype=np.float32) * measurement_noise

        # Kalman gain
        S = self.H @ self.covariance @ self.H.T + R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.state  # innovation
        self.state = self.state + K @ y
        self.covariance = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.covariance

        self._last_wh = np.array([w, h], dtype=np.float32)
        return _cxcywh_to_xyxy(self.state[0], self.state[1], w, h)

    def reset(self) -> None:
        self.state = None
        self.covariance = None
        self._last_wh = None


def _xyxy_to_cxcywh(box_xyxy: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return float(cx), float(cy), float(w), float(h)


def _cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)
