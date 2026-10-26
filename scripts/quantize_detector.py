"""
Inspect / quantize / export xmodel cho detector YOLOv8 tren DPUCZDX8G.

Chay trong docker Vitis AI GPU (conda activate vitis-ai-pytorch):

  # 1) Kiem tra kien truc map len DPU (co the doi activation tam thoi, chua fine-tune)
  python scripts/quantize_detector.py --mode inspect --weights best.pt --activation hardswish

  # 2) Calibration (model DA fine-tune voi activation DPU ho tro)
  python scripts/quantize_detector.py --mode calib --weights best_hswish.pt --calib-dir data/calib

  # 3) Test + danh gia chat luong + export xmodel (batch 1)
  #    Co label (YOLO txt) thi tinh duoc mAP that; khong co thi chi so sanh int8 vs float.
  python scripts/quantize_detector.py --mode test --weights best_hswish.pt \
         --calib-dir experiments/yolo_dataset/images/val --n-test 200
  #    (label tu dong suy ra: .../images/val -> .../labels/val, hoac chi dinh --label-dir)

  # 4) Compile
  vai_c_xir -x quantize_result/YOLOv8DPU_int.xmodel \
            -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/<BOARD>/arch.json \
            -o compiled -n yolov8_det

  Them --verbose-log de hien lai toan bo log cua Vitis AI.

Ket qua mode test:
  - console: bang so sanh output tung head, do khop int8 vs float, mAP/P/R (neu co label)
  - <out>/eval_report.json : cung noi dung o dang file
  - <out>/YOLOv8DPU_int.xmodel
"""

from __future__ import annotations

import os
import sys

# Giam log C++ (glog/UNILOG) cua Vitis AI: 0=INFO, 1=WARNING, 2=ERROR.
# PHAI dat truoc khi import torch / pytorch_nndct. Dung setdefault de van
# override duoc tu ngoai: GLOG_minloglevel=0 python ...
if "--verbose-log" not in sys.argv:
    os.environ.setdefault("GLOG_minloglevel", "1")
    os.environ.setdefault("UNILOG_MIN_LOG_LEVEL", "1")

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src1.antiuav_sot.detector.model import LightweightYoloDetector
from src1.antiuav_sot.detector.dpu_export import decode_outputs

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
IOU_THRS = np.linspace(0.5, 0.95, 10)


class _DropVaiqNoise:
    """Loc cac dong '[VAIQ_NOTE]: Find subgraph for ...' + 'node name: ...' (Python print
    cua nndct, lap lai cho moi phep cong elementwise). Cac dong khac giu nguyen."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = ""

    @staticmethod
    def _keep(line: str) -> bool:
        if "Find subgraph for" in line:
            return False
        if line.startswith("node name:") and "output shape" in line:
            return False
        return True

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if self._keep(line):
                self._stream.write(line + "\n")
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


if "--verbose-log" not in sys.argv:
    sys.stdout = _DropVaiqNoise(sys.stdout)


# ----------------------------------------------------------------------------
# Tien xu ly / doc anh
# ----------------------------------------------------------------------------
def letterbox_gray(img: np.ndarray, h: int, w: int, pad: float = 114 / 255) -> np.ndarray:
    """(H0,W0) -> (h,w) float32 [0,1], giu ti le, pad giua.
    NEN thay bang ham letterbox trong data/transforms.py de khop 100% voi luc train."""
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    r = min(h / img.shape[0], w / img.shape[1])
    nh, nw = round(img.shape[0] * r), round(img.shape[1] * r)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((h, w), pad, np.float32)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top:top + nh, left:left + nw] = img
    return out


def letterbox_params(h0: int, w0: int, h: int, w: int):
    """Tham so anh xa toa do anh goc -> toa do sau letterbox (khop letterbox_gray)."""
    r = min(h / h0, w / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    return nw / w0, nh / h0, (w - nw) // 2, (h - nh) // 2  # sx, sy, left, top


def _list_images(calib_dir: str, n: int):
    paths = sorted(p for p in glob.glob(os.path.join(calib_dir, "**", "*"), recursive=True)
                   if p.lower().endswith(IMG_EXTS))
    if not paths:
        raise FileNotFoundError(f"Khong tim thay anh trong {calib_dir}")
    rng = np.random.default_rng(0)
    if len(paths) > n:
        paths = list(rng.choice(paths, n, replace=False))
    return paths


def _read_gray(p: str):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def iter_images(calib_dir: str, n: int, h: int, w: int, in_ch: int, device):
    for p in _list_images(calib_dir, n):
        img = _read_gray(p)
        x = torch.from_numpy(letterbox_gray(img, h, w))[None, None]  # (1,1,h,w)
        if in_ch == 3:
            x = x.repeat(1, 3, 1, 1)
        yield x.to(device)


# ----------------------------------------------------------------------------
# Label (YOLO txt) -> box xyxy trong toa do letterbox
# ----------------------------------------------------------------------------
def guess_label_dir(img_dir: str):
    """.../images/val -> .../labels/val (None neu khong co thanh phan 'images')."""
    parts = Path(img_dir).parts
    if "images" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("images")
    return str(Path(*parts[:i], "labels", *parts[i + 1:]))


def load_gt(label_path: str, h0: int, w0: int, h: int, w: int):
    """Tra ve (n,4) xyxy trong toa do letterbox, hoac None neu khong co file label.
    Gia dinh bai toan 1 lop: bo qua class id."""
    if not os.path.isfile(label_path):
        return None
    sx, sy, left, top = letterbox_params(h0, w0, h, w)
    boxes = []
    with open(label_path) as f:
        for line in f:
            v = line.split()
            if len(v) < 5:
                continue
            _, cx, cy, bw, bh = (float(t) for t in v[:5])
            x1, y1 = (cx - bw / 2) * w0, (cy - bh / 2) * h0
            x2, y2 = (cx + bw / 2) * w0, (cy + bh / 2) * h0
            boxes.append([x1 * sx + left, y1 * sy + top, x2 * sx + left, y2 * sy + top])
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def iter_eval_samples(calib_dir: str, n: int, h: int, w: int, in_ch: int, device,
                      label_dir: str | None):
    """Yield (x, gt, path). gt=None neu label_dir=None hoac thieu file label."""
    for p in _list_images(calib_dir, n):
        img = _read_gray(p)
        if img is None:
            continue
        x = torch.from_numpy(letterbox_gray(img, h, w))[None, None]
        if in_ch == 3:
            x = x.repeat(1, 3, 1, 1)
        gt = None
        if label_dir is not None:
            rel = os.path.splitext(os.path.relpath(p, calib_dir))[0] + ".txt"
            gt = load_gt(os.path.join(label_dir, rel), img.shape[0], img.shape[1], h, w)
        yield x.to(device), gt, p


# ----------------------------------------------------------------------------
# Decode + metric
# ----------------------------------------------------------------------------
def _as_np(a):
    if isinstance(a, (list, tuple)) and len(a) == 1:
        a = a[0]
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float32)


def run_decode(outputs, model, conf, iou, box_format):
    """Tra ve (boxes (n,4) xyxy trong toa do letterbox, scores (n,)).
    GIA DINH decode_outputs tra ve (boxes, scores) cho batch 1. Neu khac, sua o day."""
    boxes, scores = decode_outputs(outputs, strides=model.strides, reg_max=model.reg_max,
                                   conf_thres=conf, iou_thres=iou)
    boxes = _as_np(boxes).reshape(-1, 4)
    scores = _as_np(scores).reshape(-1)
    if len(boxes) != len(scores):
        raise RuntimeError(
            f"decode_outputs tra ve {len(boxes)} box nhung {len(scores)} score; "
            "hay chinh run_decode() cho khop dinh dang tra ve.")
    if box_format == "xywh":
        cx, cy, bw, bh = boxes.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return boxes, scores


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def match_tp(pb, ps, gb, thr):
    """Greedy match theo score giam dan. Tra ve mang bool TP, cung thu tu voi pb."""
    tp = np.zeros(len(pb), bool)
    if len(pb) == 0 or len(gb) == 0:
        return tp
    iou = box_iou(pb, gb)
    used = np.zeros(len(gb), bool)
    for i in np.argsort(-ps):
        cand = np.where(used, -1.0, iou[i])
        j = int(cand.argmax())
        if cand[j] >= thr:
            tp[i] = True
            used[j] = True
    return tp


def ap_from(tp_col: np.ndarray, n_gt: int) -> float:
    """AP (noi suy kieu all-point) tu mang TP da sap theo conf giam dan."""
    if n_gt == 0:
        return float("nan")
    tpc = np.cumsum(tp_col)
    fpc = np.cumsum(~tp_col)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc + 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class DetEval:
    """Tich luy du doan qua cac anh, cuoi cung tinh mAP50, mAP50-95, P, R."""

    def __init__(self):
        self.conf, self.tp, self.n_gt = [], [], 0

    def add(self, pb, ps, gb):
        self.n_gt += len(gb)
        self.conf.append(ps)
        if len(pb):
            self.tp.append(np.stack([match_tp(pb, ps, gb, t) for t in IOU_THRS], 1))
        else:
            self.tp.append(np.zeros((0, len(IOU_THRS)), bool))

    def summarize(self, conf_thres: float) -> dict:
        conf = np.concatenate(self.conf) if self.conf else np.zeros(0, np.float32)
        tp = np.concatenate(self.tp) if self.tp else np.zeros((0, len(IOU_THRS)), bool)
        order = np.argsort(-conf)
        conf, tp = conf[order], tp[order]
        aps = [ap_from(tp[:, k], self.n_gt) for k in range(len(IOU_THRS))]
        sel = conf >= conf_thres
        n_pred = int(sel.sum())
        tp50 = int(tp[sel, 0].sum())
        p = tp50 / max(n_pred, 1)
        r = tp50 / max(self.n_gt, 1)
        return dict(map50=aps[0], map5095=float(np.mean(aps)), precision=p, recall=r,
                    f1=2 * p * r / (p + r + 1e-9), n_pred=n_pred, n_gt=self.n_gt)


class Agreement:
    """Do khop int8 so voi float (float coi nhu 'dap an'). Khong can label."""

    def __init__(self):
        self.tp = self.n_q = self.n_f = 0
        self.ious, self.dscore = [], []

    def add(self, fb, fs, qb, qs):
        self.n_f += len(fb)
        self.n_q += len(qb)
        if len(fb) == 0 or len(qb) == 0:
            return
        iou = box_iou(qb, fb)
        used = np.zeros(len(fb), bool)
        for i in np.argsort(-qs):
            cand = np.where(used, -1.0, iou[i])
            j = int(cand.argmax())
            if cand[j] >= 0.5:
                used[j] = True
                self.tp += 1
                self.ious.append(float(cand[j]))
                self.dscore.append(float(abs(qs[i] - fs[j])))

    def summarize(self) -> dict:
        p = self.tp / max(self.n_q, 1)
        r = self.tp / max(self.n_f, 1)
        return dict(n_float=self.n_f, n_int8=self.n_q, matched=self.tp, precision=p, recall=r,
                    f1=2 * p * r / (p + r + 1e-9),
                    mean_iou=float(np.mean(self.ious)) if self.ious else float("nan"),
                    mean_abs_score_diff=float(np.mean(self.dscore)) if self.dscore else float("nan"))


def _flat_outputs(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, dict):
        o = list(o.values())
    if isinstance(o, (list, tuple)):
        res = []
        for t in o:
            res.extend(_flat_outputs(t))
        return res
    return []


class OutStats:
    """So sanh output tho cua tung head: cosine, SQNR (dB), sai so tuyet doi lon nhat."""

    def __init__(self):
        self.sig, self.err = defaultdict(float), defaultdict(float)
        self.cos, self.maxerr = defaultdict(list), defaultdict(float)
        self.shape = {}
        self.fmin, self.fmax, self.qmin, self.qmax = {}, {}, {}, {}

    def add(self, fo, qo):
        for k, (f, q) in enumerate(zip(_flat_outputs(fo), _flat_outputs(qo))):
            self.shape[k] = tuple(f.shape)
            f = f.detach().float().flatten()
            q = q.detach().float().flatten()
            self.fmin[k] = min(self.fmin.get(k, math.inf), float(f.min()))
            self.fmax[k] = max(self.fmax.get(k, -math.inf), float(f.max()))
            self.qmin[k] = min(self.qmin.get(k, math.inf), float(q.min()))
            self.qmax[k] = max(self.qmax.get(k, -math.inf), float(q.max()))
            self.sig[k] += float((f * f).sum())
            self.err[k] += float(((f - q) ** 2).sum())
            self.cos[k].append(float(torch.nn.functional.cosine_similarity(f, q, dim=0)))
            self.maxerr[k] = max(self.maxerr[k], float((f - q).abs().max()))

    def summarize(self) -> list[dict]:
        rows = []
        for k in sorted(self.sig):
            name = f"head{k} {list(self.shape[k])}"
            rows.append(dict(name=name, cosine=float(np.mean(self.cos[k])),
                             sqnr_db=10 * math.log10(self.sig[k] / (self.err[k] + 1e-12) + 1e-12),
                             max_abs_err=self.maxerr[k],
                             float_range=[self.fmin[k], self.fmax[k]],
                             int8_range=[self.qmin[k], self.qmax[k]]))
        return rows


def _verdict(drop: float) -> str:
    if drop < -0.01:
        return "int8 CAO HON float -> bat thuong, thuong la do tap test qua nho/nhieu; chua ket luan duoc"
    if drop <= 0.01:
        return "TOT (giam <= 1 diem mAP50)"
    if drop <= 0.03:
        return "CHAP NHAN DUOC (giam 1-3 diem mAP50)"
    return "KEM (giam > 3 diem mAP50) -> nen tang so anh calib / QAT / doi cau hinh quantize"


def _fmt(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["inspect", "calib", "test"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--activation", default=None,
                   choices=[None, "hardswish", "relu", "leakyrelu"],
                   help="doi activation (chi de inspect nhanh; model quantize that phai da fine-tune)")
    p.add_argument("--calib-dir", default='experiments/yolo_dataset/images/val')
    p.add_argument("--n-calib", type=int, default=500)
    p.add_argument("--n-test", type=int, default=200,
                   help="so anh dung de danh gia trong mode test")
    p.add_argument("--label-dir", default=None,
                   help="thu muc label YOLO txt (mac dinh suy ra tu --calib-dir: images -> labels)")
    p.add_argument("--eval-conf", type=float, default=0.001,
                   help="nguong conf thap khi decode de tinh AP (P/R dung nguong cua detector)")
    p.add_argument("--box-format", choices=["xyxy", "xywh"], default="xyxy",
                   help="dinh dang box ma decode_outputs tra ve (toa do letterbox)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--in-ch", type=int, default=1, choices=[1, 3])
    p.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    p.add_argument("--out", default="quantize_result")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose-log", action="store_true",
                   help="hien lai toan bo log Vitis AI (VAIQ_NOTE, UNILOG...)")
    args = p.parse_args()

    device = torch.device(args.device)
    det = LightweightYoloDetector(weights_path=args.weights, activation=args.activation,
                                  device=args.device)
    print(f"[detector] activation = {det.activation}")
    model = det.to_dpu_module(in_channels=args.in_ch).to(device)
    print(f"[detector] model.training truoc khi eval() = {model.training}")
    model.eval()  # BN phai dung running stats; neu dong tren in True thi float truoc do bi sai
    dummy = torch.randn(1, args.in_ch, args.height, args.width, device=device)

    if args.mode == "inspect":
        from pytorch_nndct.apis import Inspector
        Inspector(args.target).inspect(model.cpu(), (dummy.cpu(),), device=torch.device("cpu"),
                                       output_dir="inspect", image_format="png")
        return

    from pytorch_nndct.apis import torch_quantizer
    if args.calib_dir is None:
        raise ValueError("--calib-dir bat buoc cho calib/test")

    quantizer = torch_quantizer(args.mode, model, (dummy,), output_dir=args.out,
                                device=device, target=args.target)
    qmodel = quantizer.quant_model

    with torch.no_grad():
        if args.mode == "calib":
            for k, x in enumerate(iter_images(args.calib_dir, args.n_calib, args.height,
                                              args.width, args.in_ch, device)):
                qmodel(x)
                if (k + 1) % 50 == 0:
                    print(f"[calib] {k + 1} anh")
            quantizer.export_quant_config()
            print(f"[calib] xong -> {args.out}")
            return

        # ---------------- test: danh gia float vs int8, roi export ----------------
        label_dir = args.label_dir or guess_label_dir(args.calib_dir)
        use_gt = label_dir is not None and os.path.isdir(label_dir)
        if use_gt:
            print(f"[test] label: {label_dir}")
        else:
            print("[test] KHONG tim thay thu muc label -> chi so sanh int8 vs float "
                  "(dung --label-dir de co mAP that)")

        conf_thres, nms_iou = det.conf_threshold, det.nms_iou_threshold
        out_stats, agree = OutStats(), Agreement()
        ev_f, ev_q = DetEval(), DetEval()
        n_img = n_missing = 0
        n_box_f = n_box_q = 0

        for x, gt, _ in iter_eval_samples(args.calib_dir, args.n_test, args.height, args.width,
                                          args.in_ch, device, label_dir if use_gt else None):
            fo, qo = model(x), qmodel(x)
            out_stats.add(fo, qo)
            fb, fs = run_decode(fo, model, args.eval_conf, nms_iou, args.box_format)
            qb, qs = run_decode(qo, model, args.eval_conf, nms_iou, args.box_format)
            n_img += 1

            mf, mq = fs >= conf_thres, qs >= conf_thres  # tai nguong cua detector
            n_box_f += int(mf.sum())
            n_box_q += int(mq.sum())
            agree.add(fb[mf], fs[mf], qb[mq], qs[mq])

            if use_gt:
                if gt is None:  # thieu file label -> coi la anh nen (khong co doi tuong)
                    n_missing += 1
                    gt = np.zeros((0, 4), np.float32)
                ev_f.add(fb, fs, gt)
                ev_q.add(qb, qs, gt)

        print(f"\n[test] so box (conf>={conf_thres}) tren {n_img} anh: "
              f"float={n_box_f}, int8={n_box_q}")

        # 1) Output tho tung head
        rows = out_stats.summarize()
        print("\n[test] 1) Sai lech output tho (int8 vs float), thu tu/shape theo output cua model")
        print(f"   {'head':<34}{'cosine':>10}{'SQNR(dB)':>11}{'max|err|':>11}")
        for r in rows:
            print(f"   {r['name']:<34}{r['cosine']:>10.4f}{r['sqnr_db']:>11.2f}{r['max_abs_err']:>11.4f}")
        print("   khoang gia tri output [min, max]:")
        for r in rows:
            fr, qr = r["float_range"], r["int8_range"]
            print(f"   {r['name']:<34} float=[{fr[0]:.1f}, {fr[1]:.1f}]  int8=[{qr[0]:.1f}, {qr[1]:.1f}]")
        print("   (logit tho co dai gia tri rong; sai so o vung am lon khong anh huong sau sigmoid/softmax,")
        print("    nen chi dung de dinh vi head xau; chat luong that xem muc 2 va 3)")

        # 2) Do khop detection
        ag = agree.summarize()
        print(f"\n[test] 2) Do khop detection int8 so voi float (IoU>=0.5, conf>={conf_thres})")
        print(f"   precision={_fmt(ag['precision'])} recall={_fmt(ag['recall'])} "
              f"F1={_fmt(ag['f1'])} mean IoU={_fmt(ag['mean_iou'])} "
              f"|d score|={_fmt(ag['mean_abs_score_diff'])}")
        print(f"   (float={ag['n_float']} box, int8={ag['n_int8']} box, khop={ag['matched']})")

        # 3) Chat luong that voi label
        res_f = res_q = None
        if use_gt:
            res_f, res_q = ev_f.summarize(conf_thres), ev_q.summarize(conf_thres)
            print(f"\n[test] 3) Chat luong so voi label ({res_f['n_gt']} GT, "
                  f"{n_missing} anh thieu file label coi la anh nen)")
            print(f"   {'':<8}{'mAP50':>9}{'mAP50-95':>10}{'P':>9}{'R':>9}{'F1':>9}{'#pred':>8}")
            for name, r in (("float", res_f), ("int8", res_q)):
                print(f"   {name:<8}{_fmt(r['map50']):>9}{_fmt(r['map5095']):>10}"
                      f"{_fmt(r['precision']):>9}{_fmt(r['recall']):>9}{_fmt(r['f1']):>9}"
                      f"{r['n_pred']:>8}")
            d50 = res_f["map50"] - res_q["map50"]
            d95 = res_f["map5095"] - res_q["map5095"]
            print(f"   giam do: mAP50 {d50:+.4f}, mAP50-95 {d95:+.4f}")
            if not math.isnan(d50):
                print(f"   => {_verdict(d50)}")
            if res_f["n_gt"] < 100:
                print(f"   CANH BAO: chi co {res_f['n_gt']} GT -> mAP rat nhieu, "
                      "chua du de ket luan (nen >= vai tram GT)")
            print(f"   (AP tinh voi conf>={args.eval_conf}; P/R/F1 tai conf>={conf_thres}, IoU 0.5; gia dinh 1 lop)")

        report = dict(n_images=n_img, label_dir=label_dir if use_gt else None,
                      conf_threshold=conf_thres, output_heads=rows, agreement=ag,
                      float=res_f, int8=res_q)
        os.makedirs(args.out, exist_ok=True)
        rp = os.path.join(args.out, "eval_report.json")
        with open(rp, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"\n[test] bao cao -> {rp}")

        quantizer.export_xmodel(output_dir=args.out, deploy_check=False)
        print(f"[test] xmodel -> {args.out}/YOLOv8DPU_int.xmodel")


if __name__ == "__main__":
    main()