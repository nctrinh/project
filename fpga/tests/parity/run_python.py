#!/usr/bin/env python3
"""Python reference run for the parity test (no torch / ultralytics needed).

  run_python.py pipeline --seq-dir D --detections J --constraints Y --out out.json
      SotPipeline (src1/antiuav_sot) with a replay detector -> per-frame dump + metrics
  run_python.py decode --meta meta.json --out out.json
      detector/dpu_export.py::decode_outputs(nchw=False) on dequantised int8 tensors

torch / ultralytics are replaced by stub modules ONLY inside this test process:
the code paths exercised here never call them (they are imported at module level).
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src1"))
sys.path.insert(0, str(REPO))


def install_stubs():
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    for name in ("Module", "SiLU", "Hardswish", "ReLU", "LeakyReLU", "GELU", "Mish", "Conv2d", "BatchNorm2d"):
        setattr(nn, name, type(name, (), {}))
    torch.nn = nn
    torch.is_tensor = lambda x: False
    ultra = types.ModuleType("ultralytics")
    ultra_nn = types.ModuleType("ultralytics.nn")
    ultra_mod = types.ModuleType("ultralytics.nn.modules")
    ultra_mod.C2f = type("C2f", (), {})
    ultra_mod.Conv = type("Conv", (), {})
    sys.modules.update({"torch": torch, "torch.nn": nn, "ultralytics": ultra, "ultralytics.nn": ultra_nn,
                        "ultralytics.nn.modules": ultra_mod})


install_stubs()


def run_pipeline(args):
    import cv2
    from antiuav_sot.data.dataset import SOTSequence
    from antiuav_sot.detector.model import DetectionResult
    from antiuav_sot.detector.postprocess import pad_to_fixed_size
    from antiuav_sot.pipeline.sot_pipeline import SotPipeline
    from antiuav_sot.utils.box_ops import INVALID_BOX, xyxy_to_xywh
    from antiuav_sot.utils.config import FpgaConstraints
    from antiuav_sot.utils.metrics import evaluate_sequence, summarize

    constraints = FpgaConstraints.from_yaml(args.constraints)
    replay = json.load(open(args.detections))["frames"]

    class ReplayDetector:
        def __init__(self):
            self.i = 0

        def predict(self, image):
            dets = replay[self.i] if self.i < len(replay) else []
            self.i += 1
            boxes = np.array([d["box"] for d in dets], np.float32).reshape(-1, 4)
            scores = np.array([d["score"] for d in dets], np.float32)
            b, s, v = pad_to_fixed_size(boxes, scores, constraints.max_det_boxes)
            return DetectionResult(boxes=b, scores=s, valid_mask=v)

    seq = SOTSequence(args.seq_dir, image_glob=["*.png"])
    pipeline = SotPipeline(constraints, ReplayDetector())
    pipeline.reset()
    frames, preds, active, gts, exist, res = [], [], [], [], [], []
    for sample in seq:
        raw = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        o = pipeline.step(raw)
        frames.append({
            "frame_index": sample.frame_index,
            "track_boxes": o.track_boxes.tolist(),
            "track_boxes_smoothed": o.track_boxes_smoothed.tolist(),
            "track_active": o.track_active.tolist(),
            "track_ids": o.track_ids.tolist(),
            "motion_boxes": o.motion_boxes.tolist(),
            "motion_valid_mask": o.motion_valid_mask.tolist(),
            "motion_strengths": o.motion_strengths.tolist(),
            "tier_for_candidate": o.fsm_result.tier_for_candidate.tolist(),
        })
        slot_active = bool(o.track_active[0])
        box = o.track_boxes_smoothed[0] if slot_active and o.track_boxes_smoothed[0, 0] != INVALID_BOX else None
        preds.append(box)
        active.append(slot_active and box is not None)
        gts.append(sample.gt_box_xywh)
        exist.append(bool(sample.exist))
        res.append([float(v) for v in xyxy_to_xywh(box)] if box is not None else [])
    result = evaluate_sequence(seq.seq_name, preds, active, gts, exist)
    json.dump({"frames": frames, "summary": summarize([result]), "res": res}, open(args.out, "w"))


def run_decode(args):
    from antiuav_sot.detector.dpu_export import decode_outputs

    meta = json.load(open(args.meta))
    d = Path(args.meta).parent
    outs, strides = [], []
    for lv in meta["levels"]:
        h, w = lv["h"], lv["w"]
        box = np.fromfile(d / lv["box_file"], np.int8).reshape(h, w, 4 * meta["reg_max"])
        cls = np.fromfile(d / lv["cls_file"], np.int8).reshape(h, w, meta["nc"])
        outs += [box.astype(np.float32) * 2.0 ** -lv["box_fp"], cls.astype(np.float32) * 2.0 ** -lv["cls_fp"]]
        strides.append(lv["stride"])
    boxes, scores = decode_outputs(outs, strides=strides, reg_max=meta["reg_max"], nchw=False,
                                   conf_thres=meta["conf"], iou_thres=meta["iou"], max_det=meta["max_det"])
    json.dump({"boxes": boxes.tolist(), "scores": scores.tolist()}, open(args.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pipeline")
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--constraints", required=True)
    p.add_argument("--out", required=True)
    q = sub.add_parser("decode")
    q.add_argument("--meta", required=True)
    q.add_argument("--out", required=True)
    args = ap.parse_args()
    run_pipeline(args) if args.cmd == "pipeline" else run_decode(args)


if __name__ == "__main__":
    main()
