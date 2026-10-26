"""
gradient_candidates.py
========================
Adapter bien doi output cua motion_candidates_gradient() (gradient global-
motion + IRLS + connected-components tren residual, xem gradient_motion_core.py)
thanh DUNG HOP DONG fixed-size (K,4) xyxy + valid_mask (K,) ma toan bo pipeline
va Simple Tracker FSM dang doi hoi - GIONG HET chu ky goi cua
extract_motion_candidates() cu trong motion/block_threshold.py. Nho vay,
thay the trong pipeline.py chi la doi 1 dong import + 1 loi goi ham, khong
dung gi den fusion/merge.py hay tracker/*.

So voi block_threshold.py (luoi block co dinh, moi block la 1 candidate DOC
LAP, khong bao gio gop cac block ke nhau lai thanh 1 vat), phuong phap nay:
  - Tu bu chuyen dong camera (global affine motion) truoc khi tinh residual,
    nen loc duoc phan lon "motion gia" do camera pan/rung - dieu block-
    threshold KHONG lam duoc (no chi diff hai frame tho).
  - Tra ve bounding box THAT cua tung vung lien thong (connected component)
    thay vi luoi vuong co dinh -> gan voi kich thuoc that cua vat hon.
  - VAN co the phan manh (mot vat cho ra 2-3 blob rieng neu residual dut
    doan o giua) - nen VAN giu buoc gop cum (_cluster_and_merge_boxes) lam
    luoi an toan, khong phu thuoc thuat toan sinh candidate cu the nao.

CAP NHAT (xem update_log.md de biet day du boi canh/ly do):
  - min_box_size/max_box_size gio la THAM SO DONG (ben goi - pipeline.py -
    tinh moi frame tu AdaptiveBoxSizeEstimator, xem motion/adaptive_size.py)
    thay vi hang so tinh nhu truoc - ham nay KHONG thay doi gi ve mat chu ky
    goi, chi khong con gia dinh gia tri truyen vao la co dinh giua cac frame
    (Issue 3: Static Size Limits).
  - Them loc ty le canh (aspect ratio): loai bo blob qua det (vd day cap,
    rung camera tao vet keo dai 1:10) truoc khi dua vao gop cum, tranh
    "streak noise" (Issue 7: Missing Aspect Ratio Filter).
  - _cluster_and_merge_boxes doi tu "gop tat ca cap trong nguong khoang cach"
    (union-find phang, mot lan) sang gop TANG DAN co kiem tra ty le lap day
    (fill ratio) truoc khi chap nhan tung buoc gop - giai quyet Clustering
    Paradox (Issue 8): fragment CUNG mot vat (khoang cach gan, hinh dang gon,
    fill ratio cao) van duoc gop, trong khi mot fragment UAV dung canh mot
    cum nhieu nen rai rac (khoang cach cung gan nhung fill ratio thap vi
    "khoang trong" giua chung lon) se KHONG bi gop nham thanh mot box khong
    lo, meo mo.
  - Them co che "priority anchoring" khi chon top-K candidate cuoi cung:
    candidate chong lan (IoU/containment) voi mot vi tri dang duoc theo doi
    (track dang active hoac ung vien tentative, do pipeline.py truyen vao
    qua priority_boxes) LUON duoc uu tien giu lai TRUOC KHI xet den do manh
    residual - giai quyet Issue 6 (Candidate List Overflow): UAV dang lo
    lung/dung yen phat rat it residual (de bi nhieu nen manh hon danh bat
    trong danh sach K-gioi-han) nhung van dang duoc theo doi thi khong duoc
    phep bi day ra ngoai chi vi "yeu hon" ve mat tin hieu tho.

Loc kich thuoc (min_box_size/max_box_size) va gop cum la hai co che duoc
thao luan truoc do ve xu ly "kich thuoc box cua motion detector khong dam
bao dung" - xem cuoc trao doi ve fusion tier trong lich su hoi thoai.
"""

from __future__ import annotations

import numpy as np

from antiuav_sot.motion.gradient_motion_core import motion_candidates_gradient
from antiuav_sot.utils.box_ops import make_empty_boxes

BoxXYXY = tuple[float, float, float, float]


def _yxyx_to_xyxy(box_yxyx: tuple[int, int, int, int]) -> BoxXYXY:
    y0, x0, y1, x1 = box_yxyx
    return (float(x0), float(y0), float(x1), float(y1))


def _boxes_gap(a: BoxXYXY, b: BoxXYXY) -> float:
    """Khoang cach giua hai box (0 neu cham/overlap nhau), tinh tach truc
    x/y roi lay Euclid - dung de quyet dinh co gop cum khong."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _box_diag(box: BoxXYXY) -> float:
    x0, y0, x1, y1 = box
    return float(np.hypot(x1 - x0, y1 - y0))


def _box_area(box: BoxXYXY) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _aspect_ratio(w: float, h: float) -> float:
    """Ty le canh dai/canh ngan, >= 1.0. Vo cung neu mot canh = 0 (suy bien) -
    cac box do da bi loai boi buoc loc kich thuoc (min_box_size) truoc khi
    ham nay duoc goi trong thuc te, nhung van xu ly an toan o day."""
    short, long_ = (w, h) if w <= h else (h, w)
    if short <= 1e-6:
        return float("inf")
    return long_ / short


def _union_box(boxes: list[BoxXYXY]) -> BoxXYXY:
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _ring_mean(
    residual: np.ndarray, ix0: int, iy0: int, ix1: int, iy1: int, inner_sum: float
) -> float:
    """Residual trung binh tren vanh bao quanh box [ix0:ix1, iy0:iy1], vanh
    rong max(w, h) moi phia, cat theo bien anh. 0 neu vanh rong."""
    margin = max(ix1 - ix0, iy1 - iy0)
    ox0, oy0 = max(ix0 - margin, 0), max(iy0 - margin, 0)
    ox1, oy1 = min(ix1 + margin, residual.shape[1]), min(iy1 + margin, residual.shape[0])
    ring_area = (ox1 - ox0) * (oy1 - oy0) - (ix1 - ix0) * (iy1 - iy0)
    if ring_area <= 0:
        return 0.0
    return (float(residual[oy0:oy1, ox0:ox1].sum()) - inner_sum) / ring_area


def _cluster_and_merge_boxes(
    boxes_xyxy: list[BoxXYXY],
    merge_gap_ratio: float,
    min_fill_ratio: float,
    max_aspect_ratio: float,
) -> list[BoxXYXY]:
    """Gop cac box GAN NHAU (khoang cach bien < merge_gap_ratio * duong cheo
    box nho hon trong cap) thanh box hop nhat, NHUNG chi CHAP NHAN buoc gop
    do neu box hop nhat sau cung van "dac" (fill ratio = tong dien tich cac
    box thanh vien / dien tich box hop nhat, cao nghia la it khoang trong)
    va khong qua det (aspect ratio).

    Thuat toan kieu Kruskal: xet cac cap (i, j) theo thu tu khoang cach TANG
    DAN, thu gop HAI CUM (khong chi hai box) dang chua i va j; chi CHOT gop
    that su neu box hop nhat cua ca hai cum dat ca hai dieu kien tren, neu
    khong BO QUA cap nay va xet tiep cap ke - khac voi union-find "gop het
    trong mot lan" cu (mot khi hai box da o chung mot cum vi bat ky duong
    noi nao du gan, khong co co hoi "huy" du ket qua cuoi cung xau).

    Ly do can thiet ke nay (Issue 8 - Clustering Paradox): fragment cua CUNG
    mot UAV thuong dung sat nhau VA co kich thuoc/mat do tuong duong (residual
    lien tuc, chi dut quang o vai diem anh) -> fill ratio cao, duoc gop dung.
    Nguoc lai, mot fragment UAV nam ngay canh mot cum nhieu nen (rung la,
    lap song nhiet...) tuy khoang cach bien cung gan, nhung box hop nhat cua
    hai thu KHONG LIEN QUAN nay se rat rong so voi tong dien tich thuc chat
    (nhieu khoang trong xen giua cac blob nho le te) -> fill ratio thap, bi
    tu choi gop - tranh tao ra box khong lo, bien dang nhu mo ta trong Issue 8.

    merge_gap_ratio: he so nhan voi duong cheo box NHO HON trong cap de ra
        nguong khoang cach - ty le voi kich thuoc box thay vi hang so tuyet
        doi (min_box_size cu), on dinh hon khi min/max_box_size gio la GIA
        TRI DONG (Issue 3) co the thay doi manh giua cac frame.
    min_fill_ratio: nguong ty le lap day toi thieu de chap nhan gop.
    max_aspect_ratio: box hop nhat khong duoc det hon nguong nay (tranh gop
        hai blob theo duong thang keo dai thanh mot "vet" gia).
    """
    n = len(boxes_xyxy)
    if n <= 1:
        return list(boxes_xyxy)

    areas = [_box_area(b) for b in boxes_xyxy]
    parent = list(range(n))
    members: list[list[int]] = [[i] for i in range(n)]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Tat ca cap (i, j) trong nguong khoang cach TOI DA co the co (dung
    # duong cheo lon nhat trong toan bo tap hop de khong bo sot cap nao truoc
    # khi loc chinh xac hon theo tung cap ben duoi), sap theo khoang cach tang dan.
    candidate_edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = _boxes_gap(boxes_xyxy[i], boxes_xyxy[j])
            pair_threshold = merge_gap_ratio * min(_box_diag(boxes_xyxy[i]), _box_diag(boxes_xyxy[j]))
            if gap < pair_threshold:
                candidate_edges.append((gap, i, j))
    candidate_edges.sort(key=lambda e: e[0])

    for _gap, i, j in candidate_edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue  # da cung mot cum roi

        merged_members = members[ri] + members[rj]
        merged_box = _union_box([boxes_xyxy[k] for k in merged_members])
        merged_area = _box_area(merged_box)
        if merged_area <= 0:
            continue
        fill_ratio = sum(areas[k] for k in merged_members) / merged_area
        mw, mh = merged_box[2] - merged_box[0], merged_box[3] - merged_box[1]

        if fill_ratio < min_fill_ratio or _aspect_ratio(mw, mh) > max_aspect_ratio:
            continue  # gop se tao box qua rong/qua det - tu choi, giu rieng

        parent[ri] = rj
        members[rj] = merged_members

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    return [_union_box([boxes_xyxy[i] for i in idxs]) for idxs in clusters.values()]


def gradient_motion_candidates(
    frame_t: np.ndarray,
    frame_t_minus_1: np.ndarray,
    max_motion_candidates: int,
    min_box_size: int,
    max_box_size: int,
    merge_gap_ratio: float = 0.5,
    max_aspect_ratio: float = 5.0,
    merge_min_fill_ratio: float = 0.15,
    priority_boxes: np.ndarray | None = None,
    priority_iou_threshold: float = 0.05,
    model: str = "affine",
    irls_iters: int = 3,
    huber_delta: float = 10.0,
    pyramid_levels: int = 2,
    thresh_factor: float = 2.5,
    min_area: int = 20,
    strength_metric: str = "sum",
    min_strength: float = float("-inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sinh motion candidate bang gradient global-motion compensation, tra ve
    dung hop dong fixed-size (boxes (K,4) xyxy, valid_mask (K,), strengths (K,))
    nhu extract_motion_candidates() cu - thay the truc tiep tai pipeline.py.
    (them truong strengths so voi ban goc - xem motion/motion_eval.py, cac
    goi KHONG doc gia tri nay van chay binh thuong, chi can bo qua phan tu
    thu 3 cua tuple tra ve.)

    frame_t, frame_t_minus_1: (H, W) uint8 grayscale, cung kich thuoc.
    strength_metric: "sum" (mac dinh, giu nguyen hanh vi cu) = tong residual
        (muc 'nang luong motion') trong vung box SAU KHI gop cum - thien vi
        box lon (nhieu pixel hon = tong lon hon du mat do thap). "mean" =
        residual trung binh/pixel (mat do) - khong thien vi theo dien tich,
        phu hop hon khi so sanh cac box kich thuoc rat khac nhau (vd mot
        fragment nho nhung dac, so voi mot cum lon nhung loang). Dung metric
        nay CA DE XEP HANG top-K (thay "nang" nhat) LAN de ghi vao strengths
        tra ve - tranh tinh huong xep hang theo mot kieu nhung bao cao gia
        tri theo kieu khac, gay nham lan khi chon nguong min_strength.
        "contrast" = mean residual trong box TRU mean residual o vanh bao
        quanh box (rong max(w, h) moi phia) - vat noi bat tren nen yen tinh
        (UAV tren troi) diem cao, vung nam giua clutter cung chuyen dong (la
        cay, cum den, parallax toa nha) diem thap vi vanh xung quanh cung co
        residual cao. Co the am.
    min_strength: bo candidate co strength (theo strength_metric) < nguong
        nay, TRU candidate duoc neo boi priority_boxes. Mac dinh -inf = khong
        loc (contrast co the am nen khong dung 0.0 lam "tat").
    min_box_size / max_box_size: loc theo TUNG CHIEU (w va h deu phai nam
        trong [min_box_size, max_box_size]) - loai bo blob qua nho (nhieu)
        hoac qua lon (thuong la vung camera bu dong khong tot, khong phai
        UAV thuc su). CO THE la gia tri DONG tinh moi frame boi ben goi
        (xem motion/adaptive_size.py, Issue 3) - ham nay khong gia dinh gi
        ve tinh on dinh giua cac frame cua hai tham so nay.
    merge_gap_ratio: xem _cluster_and_merge_boxes.
    max_aspect_ratio: loai bo TUNG candidate (truoc khi gop cum) co ty le
        canh dai/ngan vuot nguong nay - chan "streak noise" (day cap, rung
        camera keo dai theo mot huong) truoc khi no kip lam nhieu buoc gop
        cum hoac chiem mot slot candidate (Issue 7).
    merge_min_fill_ratio: xem _cluster_and_merge_boxes (Issue 8).
    priority_boxes: (P, 4) xyxy hoac None - vi tri dang duoc theo doi (track
        active + tentative candidate, do pipeline.py cung cap) dung de "neo"
        candidate khoi bi loai khi cat con K manh nhat (Issue 6). None hoac
        mang rong = tat tinh nang nay, hanh vi giong het truoc day.
    priority_iou_threshold: nguong max(IoU, containment) voi mot priority
        box de coi mot candidate la "duoc neo".
    """
    result = motion_candidates_gradient(
        frame_t_minus_1.astype(np.float32),
        frame_t.astype(np.float32),
        model=model,
        irls_iters=irls_iters,
        huber_delta=huber_delta,
        pyramid_levels=pyramid_levels,
        thresh_factor=thresh_factor,
        min_area=min_area,
        verbose=False,
    )

    raw_boxes_xyxy = [_yxyx_to_xyxy(b) for b in result["boxes"]]

    # ---- Issue 7: loc ty le canh TRUOC khi gop cum - mot vet keo dai do
    # rung camera/day cap khong duoc phep "keo" ca mot cum hop le vao no qua
    # buoc gop khoang cach o duoi. ----
    shape_filtered = [
        box for box in raw_boxes_xyxy
        if _aspect_ratio(box[2] - box[0], box[3] - box[1]) <= max_aspect_ratio
    ]

    merged_boxes = _cluster_and_merge_boxes(
        shape_filtered,
        merge_gap_ratio=merge_gap_ratio,
        min_fill_ratio=merge_min_fill_ratio,
        max_aspect_ratio=max_aspect_ratio,
    )

    residual = result["residual"]
    filtered: list[tuple[BoxXYXY, float]] = []
    for box in merged_boxes:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if not (min_box_size <= w <= max_box_size and min_box_size <= h <= max_box_size):
            continue
        if _aspect_ratio(w, h) > max_aspect_ratio:
            # Box gop tu nhieu fragment co the tro nen det hon tung fragment
            # rieng le (vd hai blob vuong xep canh nhau theo chieu ngang) -
            # loc lai lan nua SAU gop cum, khong chi truoc do.
            continue
        ix0, iy0 = int(max(x0, 0)), int(max(y0, 0))
        ix1, iy1 = int(min(x1, residual.shape[1])), int(min(y1, residual.shape[0]))
        if ix1 > ix0 and iy1 > iy0:
            strength_sum = float(residual[iy0:iy1, ix0:ix1].sum())
            strength_mean = strength_sum / float((ix1 - ix0) * (iy1 - iy0))
        else:
            strength_sum = strength_mean = 0.0
        if strength_metric == "mean":
            strength = strength_mean
        elif strength_metric == "contrast":
            strength = strength_mean - _ring_mean(residual, ix0, iy0, ix1, iy1, strength_sum)
        else:
            strength = strength_sum
        filtered.append((box, strength))

    # ---- Issue 6: "neo uu tien" - candidate chong lan vi tri dang duoc
    # theo doi LUON duoc xep truoc, bat ke residual manh yeu the nao, truoc
    # khi cat con top-K theo do manh residual. Ngan UAV dung yen/lo lung
    # (residual yeu) bi day khoi danh sach boi nhieu nen manh hon. ----
    anchored_flags = [False] * len(filtered)
    if priority_boxes is not None and len(priority_boxes) > 0 and len(filtered) > 0:
        for idx, (box, _strength) in enumerate(filtered):
            anchored_flags[idx] = _overlaps_any_priority_box(
                box, priority_boxes, priority_iou_threshold
            )

    # min_strength ap dung TRUOC khi cat top-K (nhuong slot cho box du manh)
    # va MIEN cho candidate duoc neo - UAV dang theo doi co residual yeu van
    # phai giu duoc, dung tinh than Issue 6.
    kept = [
        i for i in range(len(filtered))
        if anchored_flags[i] or filtered[i][1] >= min_strength
    ]
    order = sorted(
        kept,
        key=lambda i: (anchored_flags[i], filtered[i][1]),
        reverse=True,
    )
    filtered = [filtered[i] for i in order[:max_motion_candidates]]

    boxes = make_empty_boxes(max_motion_candidates)
    valid_mask = np.zeros((max_motion_candidates,), dtype=bool)
    strengths = np.zeros((max_motion_candidates,), dtype=np.float32)
    for i, (box, strength) in enumerate(filtered):
        boxes[i] = box
        valid_mask[i] = True
        strengths[i] = strength

    return boxes, valid_mask, strengths


def _overlaps_any_priority_box(
    box: BoxXYXY,
    priority_boxes: np.ndarray,
    threshold: float,
) -> bool:
    """True neu box (candidate motion, xyxy) chong lan DU nhieu voi it nhat
    mot priority box, dung max(IoU, containment) - containment (Intersection
    -over-Smaller-Area, xem utils/box_ops.py) de mot candidate NHO (fragment)
    nam LOT trong mot track box LON van duoc coi la "cung vi tri", giong het
    ly do dung metric nay o Tier 2 cua Simple Tracker FSM (Issue 1)."""
    bx0, by0, bx1, by1 = box
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if b_area <= 0:
        return False

    for p in priority_boxes:
        px0, py0, px1, py1 = p[0], p[1], p[2], p[3]
        if px0 == -1.0:  # INVALID_BOX padding slot
            continue
        ix0, iy0 = max(bx0, px0), max(by0, py0)
        ix1, iy1 = min(bx1, px1), min(by1, py1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        if inter <= 0:
            continue
        p_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
        union = b_area + p_area - inter
        iou = inter / union if union > 0 else 0.0
        containment = inter / min(b_area, p_area) if min(b_area, p_area) > 0 else 0.0
        if max(iou, containment) >= threshold:
            return True
    return False