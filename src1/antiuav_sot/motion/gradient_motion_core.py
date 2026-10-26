"""
gradient_motion_core.py
========================
Port cua "Phuong phap 2: Direct / Gradient-based Global Motion (Lucas-Kanade
toan anh) + IRLS" tu motion_candidates.py (script nghien cuu rieng cua user,
xem memory /areas/air-to-air-tracking.md) - CHI giu phan thuc su duoc
gradient_motion_candidates() trong gradient_candidates.py su dung.

Neu can lai cac ham nay (vi du muon dung majority-vote qua 3 frame thay vi
2), co the khoi phuc tu file goc.

Thay doi duy nhat so voi ban goc: them tham so verbose (mac dinh False) de
tat cac lenh print profiling - ban goc in ra MOI LAN GOI ham, khong phu hop
khi ham nay chay moi frame trong mot pipeline streaming.
"""

from __future__ import annotations

import time

import cv2
import numba as nb
import numpy as np

# ============================================================================
# CAC HAM GIA TOC BANG NUMBA (JIT SANG C/C++)
# ============================================================================


@nb.njit(fastmath=True)
def numba_irls_affine(Ix, Iy, It, initial_params, irls_iters, huber_delta, step=4):
    """Vong lap IRLS cho mo hinh Affine (6 tham so).

    Ix/Iy/It duoc tinh MOT LAN tu anh da warp theo initial_params, nen he
    A*delta = -It mo ta INCREMENT so voi initial_params. Moi vong IRLS giai
    LAI chinh he nay voi trong so Huber moi - KHONG cong don delta qua cac
    vong (ban cu cong don khien increment bi tinh ~irls_iters lan)."""
    H, W = Ix.shape
    delta = np.zeros(6, dtype=np.float64)
    weights = np.ones((H, W), dtype=np.float64)

    for _ in range(irls_iters):
        AtA = np.zeros((6, 6), dtype=np.float64)
        Atb = np.zeros(6, dtype=np.float64)

        for y in range(0, H, step):
            for x in range(0, W, step):
                w = weights[y, x]
                ix = Ix[y, x]
                iy = Iy[y, x]
                it = It[y, x]

                a0 = ix * x
                a1 = ix * y
                a2 = ix
                a3 = iy * x
                a4 = iy * y
                a5 = iy

                Atb[0] -= w * a0 * it
                Atb[1] -= w * a1 * it
                Atb[2] -= w * a2 * it
                Atb[3] -= w * a3 * it
                Atb[4] -= w * a4 * it
                Atb[5] -= w * a5 * it

                AtA[0, 0] += w * a0 * a0
                AtA[0, 1] += w * a0 * a1
                AtA[0, 2] += w * a0 * a2
                AtA[0, 3] += w * a0 * a3
                AtA[0, 4] += w * a0 * a4
                AtA[0, 5] += w * a0 * a5
                AtA[1, 1] += w * a1 * a1
                AtA[1, 2] += w * a1 * a2
                AtA[1, 3] += w * a1 * a3
                AtA[1, 4] += w * a1 * a4
                AtA[1, 5] += w * a1 * a5
                AtA[2, 2] += w * a2 * a2
                AtA[2, 3] += w * a2 * a3
                AtA[2, 4] += w * a2 * a4
                AtA[2, 5] += w * a2 * a5
                AtA[3, 3] += w * a3 * a3
                AtA[3, 4] += w * a3 * a4
                AtA[3, 5] += w * a3 * a5
                AtA[4, 4] += w * a4 * a4
                AtA[4, 5] += w * a4 * a5
                AtA[5, 5] += w * a5 * a5

        for r in range(6):
            for c in range(r + 1, 6):
                AtA[c, r] = AtA[r, c]

        for i in range(6):
            AtA[i, i] += 1e-6

        delta = np.linalg.solve(AtA, Atb)

        for y in range(0, H, step):
            for x in range(0, W, step):
                ix = Ix[y, x]
                iy = Iy[y, x]
                it = It[y, x]
                a0 = ix * x
                a1 = ix * y
                a2 = ix
                a3 = iy * x
                a4 = iy * y
                a5 = iy

                resid = (
                    a0 * delta[0]
                    + a1 * delta[1]
                    + a2 * delta[2]
                    + a3 * delta[3]
                    + a4 * delta[4]
                    + a5 * delta[5]
                ) + it
                abs_r = abs(resid)

                if abs_r <= huber_delta:
                    weights[y, x] = 1.0
                else:
                    weights[y, x] = huber_delta / (abs_r + 1e-8)

    return initial_params + delta


@nb.njit(fastmath=True)
def numba_irls_translation(Ix, Iy, It, initial_params, irls_iters, huber_delta):
    """Vong lap IRLS cho mo hinh Tinh tien ngang/doc (2 tham so). Giai lai
    increment moi vong, khong cong don - xem numba_irls_affine."""
    H, W = Ix.shape
    delta = np.zeros(2, dtype=np.float64)
    weights = np.ones((H, W), dtype=np.float64)

    for _ in range(irls_iters):
        AtA = np.zeros((2, 2), dtype=np.float64)
        Atb = np.zeros(2, dtype=np.float64)

        for y in range(H):
            for x in range(W):
                w = weights[y, x]
                ix = Ix[y, x]
                iy = Iy[y, x]
                it = It[y, x]

                Atb[0] -= w * ix * it
                Atb[1] -= w * iy * it
                AtA[0, 0] += w * ix * ix
                AtA[0, 1] += w * ix * iy
                AtA[1, 1] += w * iy * iy

        AtA[1, 0] = AtA[0, 1]
        AtA[0, 0] += 1e-6
        AtA[1, 1] += 1e-6

        delta = np.linalg.solve(AtA, Atb)

        for y in range(H):
            for x in range(W):
                ix = Ix[y, x]
                iy = Iy[y, x]
                it = It[y, x]
                resid = (ix * delta[0] + iy * delta[1]) + it
                abs_r = abs(resid)
                weights[y, x] = 1.0 if abs_r <= huber_delta else huber_delta / (abs_r + 1e-8)

    return initial_params + delta


# ============================================================================
# CAC HAM XU LY ANH CHINH
# ============================================================================


def _rescale_affine(params, scale):
    p = params.copy()
    if len(p) == 6:
        p[2] *= scale
        p[5] *= scale
    else:
        p *= scale
    return p


def gradient_global_motion_fast(
    f1: np.ndarray,
    f2: np.ndarray,
    model: str = "affine",
    irls_iters: int = 3,
    huber_delta: float = 10.0,
    pyramid_levels: int = 2,
):
    time_stats = {"pyramid": 0.0, "warp": 0.0, "gradients": 0.0, "irls_loop": 0.0}

    t0 = time.perf_counter()
    pyr1, pyr2 = [f1.astype(np.float32)], [f2.astype(np.float32)]
    for _ in range(pyramid_levels):
        pyr1.append(cv2.resize(pyr1[-1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        pyr2.append(cv2.resize(pyr2[-1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
    pyr1, pyr2 = pyr1[::-1], pyr2[::-1]
    time_stats["pyramid"] += time.perf_counter() - t0

    n_params = 2 if model == "translation" else 6
    params = np.zeros(n_params, dtype=np.float64)

    for level, (a, b) in enumerate(zip(pyr1, pyr2)):
        H, W = a.shape
        if level > 0:
            params = params * 2.0 if model == "translation" else _rescale_affine(params, 2.0)

        t0 = time.perf_counter()
        if model == "translation":
            M = np.array([[1, 0, params[0]], [0, 1, params[1]]], dtype=np.float32)
        else:
            M = np.array(
                [[1 + params[0], params[1], params[2]], [params[3], 1 + params[4], params[5]]],
                dtype=np.float32,
            )

        a_warp = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        time_stats["warp"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        Ix = cv2.Sobel(a_warp, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.0)
        Iy = cv2.Sobel(a_warp, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.0)
        It = b - a_warp
        time_stats["gradients"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if model == "affine":
            params = numba_irls_affine(Ix, Iy, It, params, irls_iters, huber_delta)
        else:
            params = numba_irls_translation(Ix, Iy, It, params, irls_iters, huber_delta)
        time_stats["irls_loop"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    if model == "translation":
        M = np.array([[1, 0, params[0]], [0, 1, params[1]]], dtype=np.float32)
    else:
        M = np.array(
            [[1 + params[0], params[1], params[2]], [params[3], 1 + params[4], params[5]]],
            dtype=np.float32,
        )
    warped = cv2.warpAffine(
        f1.astype(np.float32), M, (f1.shape[1], f1.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    time_stats["warp"] += time.perf_counter() - t0

    H, W = f1.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if model == "translation":
        u = np.full((H, W), params[0], dtype=np.float32)
        v = np.full((H, W), params[1], dtype=np.float32)
    else:
        u = (params[0] * xx + params[1] * yy + params[2]).astype(np.float32)
        v = (params[3] * xx + params[4] * yy + params[5]).astype(np.float32)

    return params, warped, (u, v), time_stats


def motion_candidates_gradient(
    f1: np.ndarray,
    f2: np.ndarray,
    model: str = "affine",
    irls_iters: int = 3,
    huber_delta: float = 10.0,
    pyramid_levels: int = 2,
    thresh_factor: float = 2.5,
    min_area: int = 20,
    verbose: bool = False,
):
    t_total = time.perf_counter()

    params, warped, (u, v), time_stats = gradient_global_motion_fast(
        f1, f2, model=model, irls_iters=irls_iters,
        huber_delta=huber_delta, pyramid_levels=pyramid_levels,
    )

    t0 = time.perf_counter()

    # Chi giu vung "nong len" (f2 sang hon anh cu da warp): UAV IR nong hon
    # nen, nen vi tri cu cua vat (bong ma, residual am) bi loai - absdiff cu
    # tao them mot box "theo duoi" tai vi tri t-lag.
    diff = f2.astype(np.float32) - warped
    residual = np.maximum(0, diff)
    residual = cv2.blur(residual, (5, 5))

    mean_val, std_val = cv2.meanStdDev(residual)
    thresh = mean_val[0][0] + thresh_factor * std_val[0][0]
    thresh = max(thresh, 1e-3)

    _, mask = cv2.threshold(residual, thresh, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((y0, x0, y0 + h, x0 + w))

    time_stats["post_processing"] = time.perf_counter() - t0
    time_stats["TOTAL"] = time.perf_counter() - t_total

    if verbose:
        print(f"\n[{'PROFILING GRADIENT METHOD (OPTIMIZED)':^40}]")
        for k, v in time_stats.items():
            if k != "TOTAL":
                percent = (v / time_stats["TOTAL"]) * 100
                print(f" {k:<16}: {v:6.3f} s  ({percent:5.1f}%)")
        print("-" * 44)
        print(f" {'TOTAL':<16}: {time_stats['TOTAL']:6.3f} s\n")

    return {
        "mask": mask,
        "boxes": boxes,
        "params": params,
        "warped": warped,
        "flow": (u, v),
        "residual": residual,
    }