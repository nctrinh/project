"""
Transform anh IR ve kich thuoc dau vao co dinh cho detector.

Dung "letterbox" resize (giu ty le khung hinh goc, pad them vien) thay vi
resize methanic keo dan tu do - vi:
  1. Vat the Anti-UAV rat nho (bao cao: phan lon < 2500 px tren 327680 px),
     resize khong giu ty le se lam bien dang hinh dang vat the.
  2. Padding co dinh (constant value) don gian hon nhieu de hien thuc hoa
     tren FPGA so voi cac phep noi suy phuc tap (bilinear/bicubic tren
     vung dong).

CHU Y: buoc resize/letterbox nay chi chay TREN GPU/CPU trong giai doan
train/eval. Khi port sang FPGA, camera interface se can duoc cau hinh de
xuat anh dung do phan giai target ngay tu dau (tranh phai resize on-chip).
Gia tri box tra ve LUON o he toa do ANH GOC (chua letterbox) - viec quy doi
sang he toa do sau letterbox do detector/dataset tu xu ly khi can.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray      # (H, W) hoac (H, W, 1) grayscale, kich thuoc co dinh
    scale: float            # ty le resize da ap dung (anh goc -> anh sau resize, truoc pad)
    pad_left: int
    pad_top: int
    orig_height: int
    orig_width: int


def letterbox_grayscale(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    pad_value: int = 114,
) -> LetterboxResult:
    """
    Resize anh grayscale ve dung (target_height, target_width) bang letterbox:
    giu nguyen ty le khung hinh, pad phan con lai bang pad_value co dinh.

    image: (H, W) uint8 hoac (H, W, 1)
    """
    if image.ndim == 3:
        image = image[..., 0]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    orig_h, orig_w = image.shape[:2]
    scale = min(target_height / orig_h, target_width / orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_value,
    )

    return LetterboxResult(
        image=padded,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_height=orig_h,
        orig_width=orig_w,
    )


def transform_box_to_letterboxed(
    box_xywh: np.ndarray, letterbox: LetterboxResult
) -> np.ndarray:
    """Quy doi box tu he toa do anh goc sang he toa do anh sau letterbox."""
    box = np.asarray(box_xywh, dtype=np.float32).copy()
    box[0] = box[0] * letterbox.scale + letterbox.pad_left
    box[1] = box[1] * letterbox.scale + letterbox.pad_top
    box[2] = box[2] * letterbox.scale
    box[3] = box[3] * letterbox.scale
    return box


def transform_box_from_letterboxed(
    box_xywh: np.ndarray, letterbox: LetterboxResult
) -> np.ndarray:
    """Quy doi nguoc: box trong he toa do anh sau letterbox -> anh goc.
    Dung khi ve lai ket qua detector/tracker len anh goc de hien thi."""
    box = np.asarray(box_xywh, dtype=np.float32).copy()
    box[0] = (box[0] - letterbox.pad_left) / letterbox.scale
    box[1] = (box[1] - letterbox.pad_top) / letterbox.scale
    box[2] = box[2] / letterbox.scale
    box[3] = box[3] / letterbox.scale
    return box


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. Khong tru mean/chia std theo kieu
    ImageNet vi anh IR mot kenh khong cung phan bo voi RGB tu nhien; chuan
    hoa don gian ve [0,1] de tuong thich voi INT8 quantization sau nay
    (scale/zero-point de suy ra hon)."""
    return image.astype(np.float32) / 255.0
