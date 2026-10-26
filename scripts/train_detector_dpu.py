#!/usr/bin/env python3
"""
Train detector nhe (YOLOv8-n/s) tren Anti-UAV voi activation map duoc len
DPU (DPUCZDX8G khong ho tro SiLU), dung CA frame co object (exist=1, co box)
LAN frame khong co object (exist=0, negative sample) - de detector hoc duoc
phat hien "khong co gi ca" dung cach, thay vi chi tung thay anh co UAV va de
suy doan sai lech (false positive) khi trien khai thuc te tren nhung doan
video khong co UAV.

Ultralytics can du lieu o dang thu muc images/ + labels/ (YOLO txt format:
class x_center y_center width height, chuan hoa [0,1]) cung mot file
data.yaml mo ta duong dan - khac voi format IR_label.json goc cua Anti-UAV.
Script nay THUC HIEN 3 buoc:

    1. Chuyen doi (mot lan, cache lai) tu IR_label.json sang cau truc YOLO
       ma Ultralytics doc duoc, dat trong experiments/yolo_dataset/.
    2. Dung model voi activation DPU ho tro (hardswish / relu / leakyrelu),
       luu thanh checkpoint khoi tao (de ca single-GPU lan DDP multi-GPU
       deu dung dung activation), roi goi Ultralytics YOLO().train().
    3. Kiem tra best.pt: dung activation + dung duoc to_dpu_module()
       (khong con op DPU khong ho tro).

Chi co MOT class duy nhat: "uav" (class_id = 0).

Hai che do:
    - Train tu dau / tu pretrained COCO (mac dinh):
        python scripts/train_detector_dpu.py --config configs/default.yaml
    - Fine-tune tu checkpoint SiLU da train (nhanh hon, khuyen nghi):
        python scripts/train_detector_dpu.py --config configs/default.yaml \\
            --skip-conversion --init-weights experiments/detector/weights/best.pt \\
            --activation hardswish --epochs 40 --lr 0.002 --output-dir experiments/detector_hswish

Config (configs/default.yaml, muc detector_train) - cac key moi, deu tuy chon:
    activation:    hardswish | relu | leakyrelu | silu   (mac dinh hardswish)
    init_weights:  duong dan checkpoint de fine-tune     (mac dinh null)
    warmup_epochs: so epoch warmup                       (mac dinh 3)
Tham so dong lenh (neu co) ghi de config.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src1.antiuav_sot.data.dataset import AntiUAVDataset
from src1.antiuav_sot.utils.config import load_config

UAV_CLASS_ID = 0
DPU_ACTIVATIONS = ("hardswish", "relu", "leakyrelu")


def convert_to_yolo_format(
    dataset: AntiUAVDataset,
    seq_names: list[str],
    output_images_dir: Path,
    output_labels_dir: Path,
    split_name: str,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> int:
    """Chuyen doi frame sang YOLO format, ghi ra output_images_dir/output_labels_dir.
    Tra ve so anh da ghi.

    Frame co exist=1: ghi anh + file .txt chua 1 dong box (class 0 = uav).
    Frame co exist=0 (KHONG co object): ghi CHI anh, KHONG ghi file .txt -
        day la quy uoc chuan cua Ultralytics de danh dau "negative sample"/
        background.

    negative_ratio: ty le so luong frame KHONG co object duoc lay vao tap
        train SO VOI so luong frame CO object (vd 1.0 = lay ngang bang,
        0.5 = lay bang mot nua).

    Ten file dau ra: "<seq_name>__<frame_index>.<duoi_anh_goc>" de tranh
    trung ten giua cac sequence khac nhau va giu dung dinh dang anh goc.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    n_positive_written = 0
    n_negative_written = 0
    n_negative_skipped_by_ratio = 0

    for seq_name in seq_names:
        seq = dataset.load_sequence(seq_name)

        n_positive_in_seq = sum(1 for s in seq if s.exist)
        negative_budget = int(round(n_positive_in_seq * negative_ratio))

        negative_indices_in_seq = [i for i, s in enumerate(seq) if not s.exist]
        rng.shuffle(negative_indices_in_seq)
        selected_negative_indices = set(negative_indices_in_seq[:negative_budget])
        n_negative_skipped_by_ratio += max(
            0, len(negative_indices_in_seq) - negative_budget
        )

        for sample in seq:
            is_negative = not sample.exist
            if is_negative and sample.frame_index not in selected_negative_indices:
                continue

            image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(
                    f"[train_detector] CANH BAO: khong doc duoc anh "
                    f"{sample.image_path}, bo qua.",
                    file=sys.stderr,
                )
                continue

            out_name = f"{seq_name}__{sample.frame_index:06d}"
            out_image_path = output_images_dir / f"{out_name}{sample.image_path.suffix}"
            shutil.copy(sample.image_path, out_image_path)

            if is_negative:
                n_negative_written += 1
                continue

            h, w = image.shape[:2]
            x, y, bw, bh = sample.gt_box_xywh

            if bw <= 0 or bh <= 0:
                # exist=1 nhung box khong hop le -> xoa anh vua copy, bo qua
                # (tranh vo tinh bien frame co object thanh negative).
                out_image_path.unlink(missing_ok=True)
                continue

            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            norm_w = bw / w
            norm_h = bh / h

            out_label_path = output_labels_dir / f"{out_name}.txt"
            with open(out_label_path, "w", encoding="utf-8") as f:
                f.write(f"{UAV_CLASS_ID} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            n_positive_written += 1

    print(
        f"[train_detector] [{split_name}] Da ghi {n_positive_written} anh positive "
        f"(co object), {n_negative_written} anh negative (khong co object). "
        f"Bo qua {n_negative_skipped_by_ratio} anh negative do gioi han negative_ratio."
    )
    return n_positive_written + n_negative_written


def write_yolo_data_yaml(yolo_dataset_dir: Path, output_path: Path) -> None:
    content = f"""# Tu dong sinh boi scripts/train_detector_dpu.py - KHONG sua tay,
# chay lai script de cap nhat.
path: {yolo_dataset_dir.resolve()}
train: images/train
val: images/val
names:
  0: uav
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


# ----------------------------------------------------------------------------- model
def build_dpu_model(weights: str, activation: str, init_ckpt_path: Path):
    """Tao YOLO voi activation chi dinh va luu thanh checkpoint khoi tao.

    Tai sao phai luu checkpoint thay vi train truc tiep tren object:
    - Ultralytics .train() DUNG LAI model tu model.yaml -> set_activation()
      da ghi key "activation" vao yaml nen single-GPU van giu dung activation.
    - Nhung voi DDP (device="0,1"), process con dung model tu FILE trong
      args.model -> neu file la yolov8n.pt goc thi activation quay ve SiLU.
    Train tu checkpoint khoi tao nay thi CA HAI truong hop deu dung.
    """
    import torch
    from ultralytics import YOLO

    from src1.antiuav_sot.detector.model import get_model_activation, set_activation

    model = YOLO(weights)
    before = get_model_activation(model.model)
    n = set_activation(model, activation)
    after = get_model_activation(model.model)
    print(f"[train_detector] Activation: {before} -> {after} ({n} module), "
          f"yaml.activation = {model.model.yaml.get('activation')}")
    if before != activation and str(weights).endswith(".pt"):
        print(f"[train_detector] Luu y: trong so '{weights}' duoc train voi '{before}' - "
              "mAP epoch dau se thap hon, se hoi phuc trong qua trinh fine-tune.")

    init_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": copy.deepcopy(model.model).half(), "train_args": {}, "date": None},
        init_ckpt_path,
    )
    print(f"[train_detector] Checkpoint khoi tao: {init_ckpt_path}")
    return YOLO(str(init_ckpt_path))


def verify_dpu_ready(best_pt: Path, activation: str) -> None:
    """Kiem tra best.pt: dung activation va dung duoc YOLOv8DPU (khong con SiLU)."""
    from ultralytics import YOLO

    from src1.antiuav_sot.detector.model import get_model_activation
    from src1.antiuav_sot.detector.dpu_export import YOLOv8DPU

    if not best_pt.is_file():
        print(f"[train_detector] CANH BAO: khong tim thay {best_pt}, bo qua kiem tra DPU.",
              file=sys.stderr)
        return

    det = YOLO(str(best_pt)).model
    act = get_model_activation(det)
    if act != activation:
        raise RuntimeError(
            f"best.pt co activation '{act}' thay vi '{activation}' - activation da bi "
            "reset khi train. Kiem tra phien ban ultralytics / cach train tu checkpoint khoi tao."
        )
    dpu = YOLOv8DPU(det, in_channels=1)
    print(f"[train_detector] best.pt OK: activation={act}, export DPU duoc "
          f"(strides={dpu.strides}, reg_max={dpu.reg_max}, nc={dpu.nc}). "
          "Buoc tiep theo: scripts/quantize_detector.py --mode inspect.")


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs1/default.yaml")
    parser.add_argument(
        "--manifest", type=str, default="experiments/data_manifest.json",
        help="File manifest sinh boi scripts/prepare_data.py",
    )
    parser.add_argument(
        "--skip-conversion", action="store_true",
        help="Bo qua buoc chuyen doi du lieu, dung lai YOLO dataset da co san",
    )
    parser.add_argument("--activation", choices=DPU_ACTIVATIONS + ("silu",), default=None,
                        help="Ghi de detector_train.activation")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Fine-tune tu checkpoint nay (vd best.pt SiLU) thay vi pretrained COCO")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help='vd "cuda:0" hoac "0,1" (DDP)')
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay manifest tai {manifest_path}. "
            "Chay scripts/prepare_data.py truoc."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    yolo_dataset_dir = Path("experiments/yolo_dataset")
    data_yaml_path = Path("experiments/yolo_dataset.yaml")

    if not args.skip_conversion:
        dataset = AntiUAVDataset(
            manifest["root_dir"], image_glob=manifest["image_glob"]
        )
        negative_ratio = cfg.get("data", "negative_ratio", default=1.0)

        print("[train_detector] Chuyen doi tap train sang YOLO format...")
        convert_to_yolo_format(
            dataset,
            manifest["train_sequences"],
            yolo_dataset_dir / "images" / "train",
            yolo_dataset_dir / "labels" / "train",
            split_name="train",
            negative_ratio=negative_ratio,
        )

        print("[train_detector] Chuyen doi tap val sang YOLO format...")
        convert_to_yolo_format(
            dataset,
            manifest["val_sequences"],
            yolo_dataset_dir / "images" / "val",
            yolo_dataset_dir / "labels" / "val",
            split_name="val",
            negative_ratio=negative_ratio,
        )

        write_yolo_data_yaml(yolo_dataset_dir, data_yaml_path)
        print(f"[train_detector] Da ghi {data_yaml_path}")
    else:
        if not data_yaml_path.is_file():
            raise FileNotFoundError(
                f"--skip-conversion duoc bat nhung khong tim thay {data_yaml_path}. "
                "Chay lai khong co --skip-conversion truoc."
            )
        print("[train_detector] Bo qua buoc chuyen doi (dung --skip-conversion).")

    # ---- Cau hinh train (CLI ghi de config) ----
    model_arch = cfg.get("detector_train", "model_arch", default="yolov8n")
    pretrained = cfg.get("detector_train", "pretrained", default=True)
    activation = args.activation or cfg.get("detector_train", "activation", default="hardswish")
    init_weights = args.init_weights or cfg.get("detector_train", "init_weights", default=None)
    epochs = args.epochs or cfg.get("detector_train", "epochs", default=100)
    batch_size = cfg.get("detector_train", "batch_size", default=16)
    lr = args.lr or cfg.get("detector_train", "lr", default=1e-3)
    warmup_epochs = cfg.get("detector_train", "warmup_epochs", default=3)
    device = args.device or cfg.get("detector_train", "device", default="cuda:0")
    output_dir = Path(args.output_dir or cfg.get("detector_train", "output_dir",
                                                 default="experiments/detector"))

    if activation == "silu":
        print("[train_detector] CANH BAO: activation=silu KHONG map duoc len DPU "
              "(chi dung de doi chieu).", file=sys.stderr)

    # QUAN TRONG: Ultralytics train/val mode CHI chap nhan imgsz la SO
    # NGUYEN (ep anh ve hinh vuong), du dataset la anh chu nhat 512x640.
    # Day la gioi han CUA THU VIEN, khong phai lua chon thiet ke. Model la
    # fully-convolutional nen van chay/quantize o 512x640 khi export DPU.
    # Dung max(height, width) de KHONG cat mat thong tin theo chieu nao.
    imgsz = max(cfg.fpga.det_input_height, cfg.fpga.det_input_width)

    if init_weights:
        weights = init_weights
    else:
        weights = f"{model_arch}.pt" if pretrained else f"{model_arch}.yaml"

    init_ckpt_path = output_dir.parent / f"{output_dir.name}_init_{activation}.pt"
    model = build_dpu_model(weights, activation, init_ckpt_path)

    aug = cfg.get("detector_train", "augmentation", default={})
    scale = aug.get("scale", 0.5)
    fliplr = aug.get("fliplr", 0.5)
    flipud = aug.get("flipud", 0.0)
    translate = aug.get("translate", 0.1)
    mosaic = aug.get("mosaic", 1.0)
    hsv_h = aug.get("hsv_h", 0.0)
    hsv_s = aug.get("hsv_s", 0.0)
    hsv_v = aug.get("hsv_v", 0.0)

    print(f"[train_detector] Bat dau train {model_arch} (activation={activation}, "
          f"init={weights}), epochs={epochs}, batch={batch_size}, imgsz={imgsz}, "
          f"lr0={lr}, device={device}")
    print(f"[train_detector] Augmentation: scale={scale}, fliplr={fliplr}, "
          f"flipud={flipud}, translate={translate}, mosaic={mosaic}")

    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        warmup_epochs=warmup_epochs,
        device=device,
        project=str(output_dir.resolve().parent),
        name=output_dir.name,
        exist_ok=True,
        scale=scale,
        fliplr=fliplr,
        flipud=flipud,
        translate=translate,
        mosaic=mosaic,
        # Anh IR grayscale: tat augmentation mau sac.
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
    )

    best_pt = output_dir / "weights" / "best.pt"
    print(f"[train_detector] Hoan tat. Checkpoint tai: {best_pt}")
    if activation in DPU_ACTIVATIONS:
        verify_dpu_ready(best_pt, activation)


if __name__ == "__main__":
    main()