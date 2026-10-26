#!/usr/bin/env python3
"""
Export detector (FP32 hoac da quantize) sang ONNX - buoc trung gian truoc
khi dua vao toolchain FPGA (Vitis-AI, hls4ml, hoac tuong tu, muc 4.4 bao
cao FPGA "Buoc tiep theo").

Ultralytics ho tro export ONNX native (model.export(format="onnx")) cho
checkpoint FP32 - day la truong hop don gian, it rui ro. Voi checkpoint DA
quantize INT8 (tu quantize_detector.py), can dung torch.onnx.export truc
tiep vi Ultralytics .export() khong biet ve qconfig da gan thu cong.

Cach dung:
    python scripts/export_onnx.py --checkpoint experiments/detector/weights/best.pt
    python scripts/export_onnx.py --checkpoint experiments/detector/quantized_int8.pt --quantized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from antiuav_sot.utils.config import load_config


def export_fp32_checkpoint(checkpoint_path: str, imgsz: int, output_dir: str) -> str:
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
    result_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=False,   # BAT BUOC false: giu batch size va input shape CO DINH,
                          # dung nguyen tac "kich thuoc bien biet truoc" cua toan repo
    )
    return str(result_path)


def export_quantized_checkpoint(
    checkpoint_state_dict_path: str, torch_model_arch, imgsz: int, output_path: str
) -> str:
    """Export checkpoint DA quantize (tu quantize_detector.py) sang ONNX.

    torch_model_arch: instance kien truc model (da qua torch.ao.quantization.convert)
        voi state_dict duoc nap tu checkpoint_state_dict_path.

    LUU Y: day la khung code - can duoc hoan thien dong bo voi cach
    quantize_detector.py luu checkpoint (hien tai quantize_detector.py con
    NotImplementedError o phan noi du lieu, xem file do).
    """
    import torch

    torch_model_arch.load_state_dict(torch.load(checkpoint_state_dict_path))
    torch_model_arch.eval()

    dummy_input = torch.randn(1, 3, imgsz, imgsz)  # 3 kenh vi qua buoc lap kenh cua detector/model.py

    torch.onnx.export(
        torch_model_arch,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed shape, khong dung dynamic_axes
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--quantized", action="store_true",
        help="Checkpoint dau vao la mo hinh DA quantize INT8 (tu quantize_detector.py)",
    )
    parser.add_argument(
        "--output", type=str, default="experiments/detector/model.onnx"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    imgsz = max(cfg.fpga.det_input_height, cfg.fpga.det_input_width)

    output_dir = str(Path(args.output).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.quantized:
        print(
            "[export_onnx] CANH BAO: --quantized dang la khung code, can "
            "hoan thien dong bo voi quantize_detector.py truoc khi dung that."
        )
        raise NotImplementedError(
            "export_quantized_checkpoint() can duoc noi voi kien truc model "
            "thuc te sau khi quantize_detector.py hoan thien."
        )
    else:
        print(f"[export_onnx] Export checkpoint FP32: {args.checkpoint}")
        result_path = export_fp32_checkpoint(args.checkpoint, imgsz, output_dir)
        print(f"[export_onnx] Da export thanh cong: {result_path}")

        # Xac nhan nhanh: nap lai bang onnxruntime, chay thu mot input gia
        print("[export_onnx] Kiem tra nhanh voi onnxruntime...")
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(result_path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"[export_onnx] Input shape theo ONNX: {input_shape}")

        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})
        print(f"[export_onnx] Chay thu OK, so output tensor: {len(outputs)}, "
              f"shape output dau tien: {outputs[0].shape}")


if __name__ == "__main__":
    main()
