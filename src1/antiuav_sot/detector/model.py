"""
Wrapper cho detector nhe (YOLOv8-n/s) dung Ultralytics, cau hinh rieng cho
anh IR grayscale mot kenh (muc 3.3 bao cao FPGA).

Ultralytics mac dinh ky vong anh 3 kenh RGB (backbone pretrained tren
ImageNet). De giu dung tinh chat "anh dau vao mot kenh, khong phai RGB ba
kenh" cua bai toan Anti-UAV (muc 3.1), lop wrapper nay lap lai kenh
grayscale thanh 3 kenh giong het nhau CHI o buoc dua vao Ultralytics model.
Khi port sang FPGA (to_dpu_module), 3 kenh giong het nhau duoc "gap" vao
conv dau tien (cong trong so theo truc kenh vao) -> model DPU nhan truc tiep
anh 1 kenh, tuong duong tuyet doi, KHONG can huan luyen lai.

Activation:
    DPU (DPUCZDX8G) KHONG ho tro SiLU. Chon activation qua tham so `activation`:
        "silu"      - mac dinh cua YOLOv8, KHONG map duoc len DPU
        "hardswish" - xap xi SiLU tot nhat, fine-tune hoi phuc nhanh
        "relu"
        "leakyrelu" - slope 26/256 = 0.1015625 (gia tri DPU thuc thi chinh xac)
        None        - giu nguyen activation trong checkpoint
    Activation duoc ghi vao model.yaml, nen khi goi self.model.train(...)
    trainer cua Ultralytics dung lai dung activation nay (khong bi reset ve SiLU).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import torch.nn as nn

# Chuoi duoc Ultralytics eval() trong parse_model (key "activation" cua yaml)
ACTIVATIONS = {
    "silu": "nn.SiLU()",
    "hardswish": "nn.Hardswish()",
    "relu": "nn.ReLU()",
    "leakyrelu": "nn.LeakyReLU(0.1015625)",
}
_ACT_TYPES = (nn.SiLU, nn.Hardswish, nn.ReLU, nn.LeakyReLU)


def make_activation(name: str) -> nn.Module:
    return eval(ACTIVATIONS[name], {"nn": nn})


def get_model_activation(model: nn.Module) -> str | None:
    """Tra ve ten activation dang dung trong model (theo activation dau tien gap)."""
    for m in model.modules():
        if isinstance(m, nn.LeakyReLU):
            return "leakyrelu"
        if isinstance(m, nn.Hardswish):
            return "hardswish"
        if isinstance(m, nn.ReLU):
            return "relu"
        if isinstance(m, nn.SiLU):
            return "silu"
    return None


def _replace_activation(module: nn.Module, new_act_fn) -> int:
    n = 0
    for name, child in module.named_children():
        if isinstance(child, _ACT_TYPES):
            setattr(module, name, new_act_fn())
            n += 1
        else:
            n += _replace_activation(child, new_act_fn)
    return n


def set_activation(yolo, name: str) -> int:
    """Doi activation cho:
    - module hien tai (de predict / Inspector / quantize thay ngay), va
    - yolo.model.yaml (de trainer Ultralytics dung lai khi train).
    Tra ve so module da thay.
    """
    if name not in ACTIVATIONS:
        raise ValueError(f"activation '{name}' khong hop le, chon trong {list(ACTIVATIONS)}")
    yolo.model.yaml["activation"] = ACTIVATIONS[name]
    return _replace_activation(yolo.model, lambda: make_activation(name))


def build_yolo(model_arch: str = "yolov8n", weights_path: str | None = None,
               activation: str | None = None):
    """Tao Ultralytics YOLO voi activation tuy chon. Dung chung cho
    inference (LightweightYoloDetector) va training (scripts/train_detector.py):

        yolo = build_yolo("yolov8n", "yolov8n.pt", activation="hardswish")
        yolo.train(data=..., epochs=...)
    """
    from ultralytics import YOLO  # import cuc bo de tests khong bat buoc cai ultralytics

    yolo = YOLO(weights_path if weights_path is not None else f"{model_arch}.pt")
    if activation is None:
        return yolo

    current = get_model_activation(yolo.model)
    n = set_activation(yolo, activation)
    if current != activation and weights_path is not None:
        warnings.warn(
            f"Checkpoint '{weights_path}' duoc train voi '{current}', dang doi sang "
            f"'{activation}' ({n} module). Do chinh xac se giam cho toi khi fine-tune."
        )
    return yolo


@dataclasses.dataclass
class DetectionResult:
    """Ket qua detector cho MOT anh, da qua NMS, fixed-size.

    boxes:      (max_det_boxes, 4) xyxy trong he toa do anh dau vao detector
                (sau letterbox neu co), INVALID_BOX cho slot rong
    scores:     (max_det_boxes,) float32 confidence
    valid_mask: (max_det_boxes,) bool
    """

    boxes: np.ndarray
    scores: np.ndarray
    valid_mask: np.ndarray


class LightweightYoloDetector:
    """Wrapper mong quanh Ultralytics YOLO, chi lam nhiem vu: nhan anh
    grayscale (H, W) float32 [0,1], tra ve DetectionResult da chuan hoa
    fixed-size. Toan bo phan "training loop" nam trong scripts/train_detector.py,
    KHONG nam trong class nay - class nay chi la interface inference.
    """

    def __init__(
        self,
        model_arch: str = "yolov8n",
        weights_path: str | None = None,
        max_det_boxes: int = 32,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda:0",
        activation: str | None = None,
    ):
        self.max_det_boxes = max_det_boxes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device

        self.model = build_yolo(model_arch, weights_path, activation)
        self.activation = get_model_activation(self.model.model)

    # ------------------------------------------------------------------ float inference
    def _grayscale_to_rgb_like(self, image: np.ndarray) -> np.ndarray:
        """Lap kenh grayscale (H, W) thanh (H, W, 3) CHI de tuong thich
        Ultralytics backbone. Xem docstring module ve gioi han cua cach lam nay."""
        if image.ndim == 2:
            image = image[..., None]
        return np.repeat(image, 3, axis=-1)

    def predict(self, image: np.ndarray) -> DetectionResult:
        """
        image: (H, W) float32 [0, 1] hoac uint8 [0, 255], grayscale, DA
               duoc letterbox ve dung (det_input_height, det_input_width)
               truoc khi goi ham nay (xem data/transforms.py).
        """
        rgb_like = self._grayscale_to_rgb_like(image)
        if rgb_like.dtype != np.uint8:
            rgb_like = (np.clip(rgb_like, 0, 1) * 255).astype(np.uint8)

        results = self.model.predict(
            source=rgb_like,
            conf=self.conf_threshold,
            iou=self.nms_iou_threshold,
            max_det=self.max_det_boxes,
            device=self.device,
            verbose=False,
        )

        return self._postprocess_ultralytics_result(results[0])

    def _postprocess_ultralytics_result(self, result) -> DetectionResult:
        from antiuav_sot.detector.postprocess import pad_to_fixed_size

        if result.boxes is None or len(result.boxes) == 0:
            boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)

        boxes, scores, valid_mask = pad_to_fixed_size(
            boxes_xyxy, scores, self.max_det_boxes
        )
        return DetectionResult(boxes=boxes, scores=scores, valid_mask=valid_mask)

    # ------------------------------------------------------------------ DPU
    def to_dpu_module(self, in_channels: int = 1, split_c2f: bool = True) -> nn.Module:
        """Tra ve nn.Module san sang cho vai_q_pytorch (Inspector / quantize / export xmodel):
        - Detect head bi cat: tra 6 feature map tho, decode + NMS lam o CPU
        - C2f viet lai khong dung chunk (tuong duong tuyet doi)
        - in_channels=1: conv dau nhan anh 1 kenh (gap trong so 3 kenh, tuong duong tuyet doi)
        Bao loi neu model con activation DPU khong ho tro (SiLU).
        Model cua detector KHONG bi thay doi (wrapper lam viec tren ban deepcopy).
        """
        from src1.antiuav_sot.detector.dpu_export import YOLOv8DPU

        return YOLOv8DPU(self.model.model, in_channels=in_channels, split_c2f=split_c2f)

    def postprocess_dpu_outputs(self, outs, strides=(8, 16, 32), reg_max: int = 16,
                                nchw: bool = True) -> DetectionResult:
        """Decode 6 output tho (tu YOLOv8DPU float/quantized, hoac tu DPU tren board)
        thanh DetectionResult cung dinh dang voi predict().
        Output VART tren board la NHWC va INT8 -> dequantize (x * 2**-fix_point)
        va truyen nchw=False.
        """
        from src1.antiuav_sot.detector.dpu_export import decode_outputs
        from antiuav_sot.detector.postprocess import pad_to_fixed_size

        boxes_xyxy, scores = decode_outputs(
            outs, strides=strides, reg_max=reg_max, nchw=nchw,
            conf_thres=self.conf_threshold, iou_thres=self.nms_iou_threshold,
            max_det=self.max_det_boxes,
        )
        boxes, scores, valid_mask = pad_to_fixed_size(boxes_xyxy, scores, self.max_det_boxes)
        return DetectionResult(boxes=boxes, scores=scores, valid_mask=valid_mask)