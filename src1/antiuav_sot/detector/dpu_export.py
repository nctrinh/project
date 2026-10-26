"""
Chuyen YOLOv8 (Ultralytics DetectionModel) thanh nn.Module dap ung dieu kien
cua Vitis AI (vai_q_pytorch + vai_c_xir) cho DPUCZDX8G:

1. Khong con op ngoai DPU:
   - SiLU          -> bao loi, phai train/fine-tune voi hardswish/relu/leakyrelu
   - C2f.chunk()   -> tach cv1 thanh 2 conv (C2fDPU), tuong duong tuyet doi
   - Detect decode -> cat bo (DFL, anchors, dist2bbox, concat, sigmoid) -> CPU
2. Dau vao 1 kenh that su: conv dau tien duoc gap tu 3 kenh giong nhau -> 1 kenh.
3. Graph tinh: forward chi gom vong lap Python tinh, trace duoc.

decode_outputs(): hau xu ly numpy cho 6 output tho, dung chung cho
float, quantized (nndct test mode) va output DPU tren board.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f, Conv

DPU_UNSUPPORTED_ACTS = (nn.SiLU, nn.GELU, nn.Mish)


# ----------------------------------------------------------------------------- C2f
def split_conv(conv: Conv, s: int, e: int) -> Conv:
    """Lay cac kenh output [s:e] cua mot ultralytics Conv (conv [+ bn] + act).
    Xu ly ca truong hop model da fuse (khong con bn, conv co bias)."""
    new = copy.deepcopy(conv)
    c = conv.conv
    new.conv = nn.Conv2d(c.in_channels, e - s, c.kernel_size, c.stride, c.padding,
                         c.dilation, c.groups, bias=c.bias is not None)
    new.conv.weight.data = c.weight.data[s:e].clone()
    if c.bias is not None:
        new.conv.bias.data = c.bias.data[s:e].clone()
    if hasattr(conv, "bn"):
        bn = conv.bn
        new.bn = nn.BatchNorm2d(e - s, eps=bn.eps, momentum=bn.momentum)
        for k in ("weight", "bias", "running_mean", "running_var"):
            getattr(new.bn, k).data = getattr(bn, k).data[s:e].clone()
    return new


class C2fDPU(nn.Module):
    """C2f khong dung chunk: cv1 (2c kenh) tach thanh cv1a, cv1b (moi cai c kenh)."""

    def __init__(self, m: C2f):
        super().__init__()
        c = m.c
        self.cv1a = split_conv(m.cv1, 0, c)
        self.cv1b = split_conv(m.cv1, c, 2 * c)
        self.m = m.m
        self.cv2 = m.cv2
        self.f, self.i = m.f, m.i  # giu routing cho vong lap forward

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        for b in self.m:
            y.append(b(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ----------------------------------------------------------------------------- 1 kenh
def fold_to_single_channel(conv: Conv) -> None:
    """Conv dau vao 3 kenh giong het nhau == conv 1 kenh voi trong so cong theo truc kenh."""
    c = conv.conv
    if c.in_channels == 1:
        return
    if c.groups != 1:
        raise ValueError("Conv dau tien co groups != 1, khong gap kenh duoc")
    new = nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding,
                    c.dilation, 1, bias=c.bias is not None)
    new.weight.data = c.weight.data.sum(1, keepdim=True).clone()
    if c.bias is not None:
        new.bias.data = c.bias.data.clone()
    conv.conv = new


# ----------------------------------------------------------------------------- wrapper
class YOLOv8DPU(nn.Module):
    def __init__(self, det_model: nn.Module, in_channels: int = 1, split_c2f: bool = True):
        super().__init__()
        det = copy.deepcopy(det_model).float().eval()  # khong dong vao model goc
        self.layers = det.model                        # nn.Sequential
        self.save = det.save

        bad = [n for n, m in self.layers.named_modules() if isinstance(m, DPU_UNSUPPORTED_ACTS)]
        if bad:
            raise ValueError(
                f"Model con {len(bad)} activation DPU khong ho tro (vd: {bad[0]} "
                f"= {type(dict(self.layers.named_modules())[bad[0]]).__name__}). "
                "Tao detector voi activation='hardswish' | 'relu' | 'leakyrelu' "
                "(va fine-tune) truoc khi export."
            )

        if in_channels == 1:
            fold_to_single_channel(self.layers[0])
        elif in_channels != 3:
            raise ValueError("in_channels phai la 1 hoac 3")
        self.in_channels = in_channels

        if split_c2f:
            for idx, m in enumerate(self.layers):
                if isinstance(m, C2f):
                    self.layers[idx] = C2fDPU(m)

        head = self.layers[-1]                         # Detect
        self.head = head
        self.strides = [int(s) for s in head.stride.tolist()]
        self.reg_max = int(head.reg_max)
        self.nc = int(head.nc)

    def forward(self, x):
        y = []
        for m in self.layers[:-1]:                     # backbone + neck
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)

        h = self.head
        feats = [x if j == -1 else y[j] for j in h.f]  # P3, P4, P5
        outs = []
        for i in range(h.nl):
            outs.append(h.cv2[i](feats[i]))           # box: 4*reg_max kenh
            outs.append(h.cv3[i](feats[i]))           # cls: nc kenh
        return outs                                    # 6 tensor, KHONG decode


# ----------------------------------------------------------------------------- decode
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def decode_outputs(outs, strides=(8, 16, 32), reg_max: int = 16, nchw: bool = True,
                   conf_thres: float = 0.25, iou_thres: float = 0.5, max_det: int = 32):
    """outs: list 6 phan tu [box_P3, cls_P3, box_P4, cls_P4, box_P5, cls_P5],
    moi phan tu torch.Tensor hoac np.ndarray, shape (1,C,H,W)/(C,H,W) neu nchw,
    (1,H,W,C)/(H,W,C) neu NHWC (output VART). Gia tri float (da dequantize).
    Tra ve (boxes_xyxy (N,4) float32, scores (N,) float32), N <= max_det.
    """
    all_boxes, all_scores = [], []
    for i, s in enumerate(strides):
        box, cls = outs[2 * i], outs[2 * i + 1]
        box = box.detach().cpu().numpy() if torch.is_tensor(box) else np.asarray(box)
        cls = cls.detach().cpu().numpy() if torch.is_tensor(cls) else np.asarray(cls)
        if box.ndim == 4:
            box, cls = box[0], cls[0]
        if not nchw:                                    # HWC -> CHW
            box, cls = box.transpose(2, 0, 1), cls.transpose(2, 0, 1)
        _, H, W = box.shape

        b = box.reshape(4, reg_max, H * W).astype(np.float32)   # DFL
        b = np.exp(b - b.max(1, keepdims=True))
        b /= b.sum(1, keepdims=True)
        d = (b * np.arange(reg_max, dtype=np.float32)[None, :, None]).sum(1)  # (4, HW)

        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        ax, ay = gx.ravel() + 0.5, gy.ravel() + 0.5
        xyxy = np.stack([ax - d[0], ay - d[1], ax + d[2], ay + d[3]], 1) * s
        score = 1.0 / (1.0 + np.exp(-cls.reshape(cls.shape[0], -1).astype(np.float32)))

        all_boxes.append(xyxy)
        all_scores.append(score.max(0))                 # nc=1 -> chinh la confidence

    boxes = np.concatenate(all_boxes).astype(np.float32)
    scores = np.concatenate(all_scores).astype(np.float32)
    m = scores >= conf_thres
    boxes, scores = boxes[m], scores[m]
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    keep = _nms(boxes, scores, iou_thres, max_det)
    return boxes[keep], scores[keep]