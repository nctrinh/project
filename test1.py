import json
import shutil
from pathlib import Path

# Cấu hình các đường dẫn
BASE_DIR = Path("/mnt/tupm/antiuav_sot")
EXP0_JSON = BASE_DIR / "experiments_motion_mean_5" / "eval_results_val.json"
EXP1_JSON = BASE_DIR / "experiments_no_motion1" / "eval_results_val.json"
VIZ_DIR = BASE_DIR / "experiments_no_motion1" / "eval_viz"
DEST_DIR = BASE_DIR / "videos"


def load_results(json_path: Path) -> dict:
    """Đọc file json và trả về dict map: seq_name -> mean_iou."""
    if not json_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tạo mapping {seq_name: mean_iou}
    return {
        item["seq_name"]: item["mean_iou"]
        for item in data.get("per_sequence", [])
        if "seq_name" in item and "mean_iou" in item
    }


def main():
    # 1. Đảm bảo folder đích tồn tại
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Thư mục đích: {DEST_DIR}")

    # 2. Đọc kết quả từ cả 2 file
    print(f"[+] Đang đọc kết quả từ:\n - {EXP0_JSON}\n - {EXP1_JSON}")
    exp0_map = load_results(EXP0_JSON)
    exp1_map = load_results(EXP1_JSON)

    copied_count = 0
    missing_video_count = 0

    print("\n--- Bắt đầu so sánh và copy ---")

    # 3. Duyệt qua các sequence của experiments1
    for seq_name, iou1 in exp1_map.items():
        if seq_name not in exp0_map:
            # Bỏ qua nếu seq không có ở experiment gốc
            continue

        iou0 = exp0_map[seq_name]

        # Kiểm tra điều kiện: experiment1 có mean_iou nhỏ hơn experiment gốc
        if iou0 - iou1 > 0.05:
            video_filename = f"{seq_name}.mp4"
            src_video = VIZ_DIR / video_filename
            dest_video = DEST_DIR / video_filename

            if src_video.exists():
                shutil.copy2(src_video, dest_video)
                print(
                    f"[COPY] {seq_name} | exp1_iou: {iou1:.4f} < exp_iou: {iou0:.4f}"
                )
                copied_count += 1
            else:
                print(
                    f"[MISSING] {seq_name} thoả điều kiện ({iou1:.4f} < {iou0:.4f}) nhưng không tìm thấy file video: {src_video}"
                )
                missing_video_count += 1

    print("\n--- Hoàn tất ---")
    print(f"Tổng số video đã copy: {copied_count}")
    if missing_video_count > 0:
        print(f"Tổng số video bị thiếu: {missing_video_count}")


if __name__ == "__main__":
    main()