from ultralytics import YOLO

model_arch = "yolov8n"  # Hoặc yolov11n, yolov8s,... tùy nhu cầu

# 1. Khởi tạo model
model = YOLO(f"{model_arch}.pt")

# 2. In cấu trúc chi tiết từng layer (dạng PyTorch Module/Sequential)
print("=== CHI TIẾT KIẾN TRÚC TỪNG LAYER ===")
print(model.model)

# 3. (Tùy chọn) In bản tóm tắt số layer, parameters và FLOPs
print("\n=== BẢNG THỐNG KÊ ARCHITECTURE ===")
model.info(detailed=True)