from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Cho phep chay script truc tiep tu repo root ma khong can pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from antiuav_sot.data.dataset import AntiUAVDataset, SequenceFormatError
from antiuav_sot.utils.config import load_config


def scan_test_sequences(test_root_dir: str | Path) -> list[str]:
    """Liet ke ten cac sequence khong nhan trong test_root_dir.

    Khac voi train/val, cac sequence nay khong co gt_rect nen KHONG the
    validate bang AntiUAVDataset.load_sequence (can gt_rect de doi chieu so
    frame). Chi liet ke ten thu muc con hop le (la thu muc, khong an) roi
    dua thang vao manifest -- khong kiem tra noi dung anh ben trong.
    """
    root = Path(test_root_dir)
    if not root.is_dir():
        print(
            f"[prepare_data] CANH BAO: test_root_dir khong ton tai: {root}",
            file=sys.stderr,
        )
        return []

    test_names = sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    return test_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/data_manifest.json",
        help="Duong dan file manifest se duoc ghi ra",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root_dir = cfg.get("data", "root_dir")
    image_glob = cfg.get("data", "image_glob", default=["*.jpg", "*.jpeg", "*.png"])
    train_ratio = cfg.get("data", "train_split", default=0.85)
    split_seed = cfg.get("data", "split_seed", default=42)
    test_root_dir = cfg.get("data", "test_root_dir", default=None)

    print(f"[prepare_data] Quet du lieu tai: {root_dir}")
    dataset = AntiUAVDataset(root_dir, image_glob=image_glob)
    print(f"[prepare_data] Tim thay {len(dataset)} sequence.")

    n_ok = 0
    n_error = 0
    error_details: list[dict] = []
    total_frames = 0
    total_frames_with_object = 0

    for seq_dir in dataset.seq_dirs:
        seq_name = seq_dir.name
        try:
            seq = dataset.load_sequence(seq_name)
            n_ok += 1
            total_frames += len(seq)
            total_frames_with_object += sum(seq.exist)
        except (SequenceFormatError, FileNotFoundError) as e:
            n_error += 1
            error_details.append({"seq_name": seq_name, "error": str(e)})
            print(f"[prepare_data] LOI o sequence '{seq_name}': {e}", file=sys.stderr)

    print(f"[prepare_data] Hop le: {n_ok} / {len(dataset)} sequence.")
    if n_error > 0:
        print(
            f"[prepare_data] CANH BAO: {n_error} sequence bi loi format, "
            "se KHONG duoc dua vao manifest. Xem chi tiet o tren."
        )
    print(f"[prepare_data] Tong so frame: {total_frames}, "
          f"co object (exist=1): {total_frames_with_object} "
          f"({100.0 * total_frames_with_object / max(total_frames, 1):.1f}%)")

    valid_seq_names = [
        d.name for d in dataset.seq_dirs
        if d.name not in {e["seq_name"] for e in error_details}
    ]
    # Xay dung lai dataset chi voi cac sequence hop le de split dung so luong
    filtered_dataset = dataset
    filtered_dataset.seq_dirs = [
        d for d in dataset.seq_dirs if d.name in valid_seq_names
    ]
    train_names, val_names = filtered_dataset.split(train_ratio=train_ratio, seed=split_seed)

    # Quet sequence khong nhan (test) neu co cau hinh duong dan
    test_names: list[str] = []
    if test_root_dir:
        print(f"[prepare_data] Quet test sequence (khong nhan) tai: {test_root_dir}")
        test_names = scan_test_sequences(test_root_dir)
        print(f"[prepare_data] Tim thay {len(test_names)} test sequence "
              "(chi liet ke ten, khong validate).")

    manifest = {
        "root_dir": str(root_dir),
        "image_glob": image_glob,
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "n_sequences_total": len(dataset.seq_dirs) + n_error,
        "n_sequences_valid": n_ok,
        "n_sequences_error": n_error,
        "total_frames": total_frames,
        "total_frames_with_object": total_frames_with_object,
        "train_sequences": train_names,
        "val_sequences": val_names,
        "error_sequences": error_details,
        "test_root_dir": str(test_root_dir) if test_root_dir else None,
        "test_sequences": test_names,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[prepare_data] Da ghi manifest ra: {output_path}")
    print(f"[prepare_data] Train: {len(train_names)} sequence, Val: {len(val_names)} sequence, "
          f"Test (khong nhan): {len(test_names)} sequence")


if __name__ == "__main__":
    main()