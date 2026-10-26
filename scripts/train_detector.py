#!/usr/bin/env python3
"""
Train detector nhe (YOLOv8-n/s) tren Anti-UAV, dung CA frame co object
(exist=1, co box) LAN frame khong co object (exist=0, negative sample) -
de detector hoc duoc phat hien "khong co gi ca" dung cach, thay vi chi
tung thay anh co UAV va de suy doan sai lech (false positive) khi trien
khai thuc te tren nhung doan video khong co UAV.

Ultralytics can du lieu o dang thu muc images/ + labels/ (YOLO txt format:
class x_center y_center width height, chuan hoa [0,1]) cung mot file
data.yaml mo ta duong dan - khac voi format IR_label.json goc cua Anti-UAV.
Script nay THUC HIEN 2 buoc:

    1. Chuyen doi (mot lan, cache lai) tu IR_label.json sang cau truc YOLO
       ma Ultralytics doc duoc, dat trong experiments/yolo_dataset/.
    2. Goi Ultralytics YOLO().train() voi cac hyperparameter tu default.yaml.

Chi co MOT class duy nhat: "uav" (class_id = 0) - bai toan Anti-UAV chi co
mot loai doi tuong can phat hien.

Cach dung:
    python scripts/train_detector.py --config configs/default.yaml
    python scripts/train_detector.py --config configs/default.yaml --skip-conversion
        (dung lai du lieu YOLO da convert tu lan chay truoc, tiet kiem thoi gian)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from antiuav_sot.data.dataset import AntiUAVDataset
from antiuav_sot.utils.config import load_config

UAV_CLASS_ID = 0


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
        background - detector se hoc duoc rang anh nay khong nen predict ra
        box nao, thay vi chi bao gio thay anh co object (rat de gay ra
        false positive khi trien khai that, vi model chua bao gio "thay"
        truong hop khong co UAV trong khung hinh).

    negative_ratio: ty le so luong frame KHONG co object duoc lay vao tap
        train SO VOI so luong frame CO object (vd 1.0 = lay ngang bang,
        0.5 = lay bang mot nua). Anti-UAV thuong co it frame negative hon
        positive trong moi sequence, nhung TREN TOAN BO dataset co the co
        rat nhieu frame negative (video dai, UAV chi xuat hien 1 doan) -
        khong lay het 100% se tranh mat can bang class qua muc ve phia
        negative, nhung van dam bao detector duoc thay negative sample.

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

        # Duyet 2 lan: lan 1 dem so frame positive de biet ngan sach cho
        # negative theo negative_ratio; lan 2 ghi thuc su. Anti-UAV moi
        # sequence khong qua lon nen chi phi duyet 2 lan khong dang ke.
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
                # KHONG ghi file .txt - Ultralytics hieu thu muc labels/
                # thieu file tuong ung voi mot anh la "anh nay khong co
                # object nao" (background/negative sample).
                n_negative_written += 1
                continue

            h, w = image.shape[:2]
            x, y, bw, bh = sample.gt_box_xywh

            if bw <= 0 or bh <= 0:
                # exist=1 nhung box khong hop le - day la loi du lieu that
                # su (da duoc SOTSequence bao dong o buoc doc, nhung kiem
                # tra lai o day cho an toan). Da ghi anh o tren nhung se
                # khong co label - vo tinh bien mot frame co object thanh
                # "negative" la sai, nen XOA anh vua copy va bo qua frame nay.
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


def write_yolo_data_yaml(
    yolo_dataset_dir: Path, output_path: Path
) -> None:
    content = f"""# Tu dong sinh boi scripts/train_detector.py - KHONG sua tay,
# chay lai script de cap nhat.
path: {yolo_dataset_dir.resolve()}
train: images/train
val: images/val
names:
  0: uav
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--manifest", type=str, default="experiments/data_manifest.json",
        help="File manifest sinh boi scripts/prepare_data.py",
    )
    parser.add_argument(
        "--skip-conversion", action="store_true",
        help="Bo qua buoc chuyen doi du lieu, dung lai YOLO dataset da co san",
    )
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

    # ---- Train ----
    from ultralytics import YOLO

    model_arch = cfg.get("detector_train", "model_arch", default="yolov8n")
    pretrained = cfg.get("detector_train", "pretrained", default=True)
    epochs = cfg.get("detector_train", "epochs", default=100)
    batch_size = cfg.get("detector_train", "batch_size", default=16)
    lr = cfg.get("detector_train", "lr", default=1e-3)
    device = cfg.get("detector_train", "device", default="cuda:0")
    output_dir = cfg.get("detector_train", "output_dir", default="experiments/detector")
    # QUAN TRONG: Ultralytics train/val mode CHI chap nhan imgsz la SO
    # NGUYEN (ep anh ve hinh vuong), du dataset la anh chu nhat 512x640.
    # Day la gioi han CUA THU VIEN (imgsz dang [h,w] list chi ho tro cho
    # predict/export, KHONG ho tro cho train/val - da xac nhan qua tai
    # lieu/issue tracker cua Ultralytics), khong phai lua chon thiet ke.
    #
    # He qua: anh 512x640 se bi letterbox them vien theo chieu cao khi
    # resize ve vuong, lam UAV (von da rat nho) chiem ty le pixel cang nho
    # hon trong anh training - GOP PHAN giai thich recall thap trong lan
    # train dau (~0.59-0.60). Dung max(height, width) de KHONG cat mat
    # thong tin theo chieu nao (an toan hon min, nhung ton them vien).
    #
    # Neu muon tranh hoan toan van de nay, can huan luyen bang mot vong
    # lap tuy chinh (khong dung Ultralytics .train() built-in) hoac dung
    # rect=True KET HOP voi dataset labels co cung ty le khung hinh trong
    # tung batch - ca hai deu ngoai pham vi script nay hien tai.
    imgsz = max(cfg.fpga.det_input_height, cfg.fpga.det_input_width)

    weights = f"{model_arch}.pt" if pretrained else f"{model_arch}.yaml"
    model = YOLO(weights)

    # Doc augmentation tu configs/default.yaml (detector_train.augmentation)
    # thay vi hard-code trong script - de ban chinh scale/flip/mosaic ma
    # khong can sua code. Gia tri default() o day chi la luoi an toan neu
    # config thieu field, KHONG phai gia tri khuyen nghi - xem ghi chu trong
    # configs/default.yaml ve ly do chon tung gia tri.
    aug = cfg.get("detector_train", "augmentation", default={})
    scale = aug.get("scale", 0.5)
    fliplr = aug.get("fliplr", 0.5)
    flipud = aug.get("flipud", 0.0)
    translate = aug.get("translate", 0.1)
    mosaic = aug.get("mosaic", 1.0)
    hsv_h = aug.get("hsv_h", 0.0)
    hsv_s = aug.get("hsv_s", 0.0)
    hsv_v = aug.get("hsv_v", 0.0)

    print(f"[train_detector] Bat dau train {model_arch}, epochs={epochs}, "
          f"batch={batch_size}, imgsz={imgsz}, device={device}")
    print(f"[train_detector] Augmentation: scale={scale}, fliplr={fliplr}, "
          f"flipud={flipud}, translate={translate}, mosaic={mosaic}")

    model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        lr0=lr,
        device=device,
        project=str(Path(output_dir).resolve().parent),
        name=Path(output_dir).name,
        exist_ok=True,
        scale=scale,
        fliplr=fliplr,
        flipud=flipud,
        translate=translate,
        mosaic=mosaic,
        # Anh IR grayscale: tat cac augmentation mau sac (hue/saturation)
        # vi khong co y nghia tren anh 1 kenh.
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
    )

    print(f"[train_detector] Hoan tat. Checkpoint tai: {output_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
