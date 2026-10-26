#!/usr/bin/env python3
"""
Chay pipeline tren MOT sequence duy nhat va xuat video overlay de kiem tra
truc quan (debug) - dung khi ban muon xem chi tiet pipeline hoat dong the
nao tren mot truong hop cu the, thay vi chi nhin con so tong hop tu
eval_tracker.py.

Cach dung:
    python scripts/visualize_sequence.py --config configs/default.yaml \
        --seq-name 20190925_101846_1_2 \
        --detector-checkpoint experiments/detector/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src1"))

from antiuav_sot.data.dataset import AntiUAVDataset, AntiUAVDatasetTest
from antiuav_sot.detector.model import LightweightYoloDetector
from antiuav_sot.pipeline.sot_pipeline import SotPipeline
from antiuav_sot.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs1/default.yaml")
    parser.add_argument("--seq-name", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default='experiments/detector/weights/best.pt')
    parser.add_argument(
        "--output", type=str, default=None,
        help="Duong dan video output (mac dinh: experiments/eval_viz/<seq_name>_debug.mp4)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    root_dir = cfg.get("data", "root_dir")
    test_root_dir = cfg.get("data", "test_root_dir", default=None)
    image_glob = cfg.get("data", "image_glob", default=["*.jpg", "*.jpeg", "*.png"])
    dataset = AntiUAVDataset(root_dir, image_glob=image_glob)
    try:
        seq = dataset.load_sequence(args.seq_name)
        is_test = False
    except:
        dataset = AntiUAVDatasetTest(test_root_dir, image_glob=image_glob)
        seq = dataset.load_sequence(args.seq_name)
        is_test = True
    print(f"[visualize_sequence] Sequence '{args.seq_name}': {len(seq)} frame")

    detector = LightweightYoloDetector(
        weights_path=args.detector_checkpoint,
        max_det_boxes=cfg.fpga.max_det_boxes,
        conf_threshold=cfg.fpga.det_conf_threshold,
        nms_iou_threshold=cfg.fpga.det_nms_iou_threshold,
        device=cfg.get("tracker_eval", "device", default="cuda:0"),
    )
    pipeline = SotPipeline(cfg.fpga, detector)

    output_path = args.output or f"experiments/eval_viz/{args.seq_name}.mp4"

    # Import truc tiep tu file eval_tracker.py trong cung thu muc scripts/,
    # tranh phu thuoc vao scripts/ duoc coi la mot Python package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_tracker_motion import run_pipeline_on_sequence, run_pipeline_on_test_sequence

    if is_test:
        run_pipeline_on_test_sequence(pipeline, seq, save_viz_path=output_path)
    else:
        result, _latencies = run_pipeline_on_sequence(pipeline, seq, save_viz_path=output_path)
        print(f"[visualize_sequence] Mean IoU tren sequence nay: {result.ious.mean():.4f}")

    print(f"[visualize_sequence] Da luu video: {output_path}")


if __name__ == "__main__":
    main()