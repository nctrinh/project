# antiuav-fpga-sot

Pipeline Single-Object Tracking (SOT) cho bai toan Anti-UAV tren anh IR
(thermal), thiet ke theo huong **Tracking-by-Detection**: detector nhe
(YOLOv8-n/s) + motion cue (frame-diff + block threshold) + Simple Tracker
FSM 4 tang (IoU) + Kalman filter tuy chon.

Train/test tren GPU truoc, nhung **moi module deu duoc thiet ke voi kich
thuoc du lieu co dinh** ngay tu dau, de sau nay port sang FPGA ma khong can
viet lai thuat toan tu dau - chi can dich sang HLS/RTL.

## Boi canh thiet ke

Repo nay hien thuc hoa pipeline duoc de xuat trong bao cao kha thi FPGA:
rescope tu "Strong Detector with Simple Tracker" (Tang et al., 3rd Anti-UAV
Workshop, CVPRW 2023) - bo ensemble detector nang (EVA/DINO) va SOT
(MixFormer/OSTrack), giu lai Simple Tracker FSM gan nhu nguyen ban vi day
la khoi phu hop nhat voi kien truc FPGA (thuan phep so sanh IoU tren du
lieu kich thuoc bien truoc).

## Cai dat

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy du lieu Anti-UAV vao dung vi tri:

```
data/Anti-UAV/Train_Track1_2/train/<seq_name>/
    000001.jpg
    000002.jpg
    ...
    IR_label.json
```

## Cau truc repo

```
configs/
    default.yaml              # hyperparameter train/eval
    fpga_constraints.yaml     # NGUON SU THAT DUY NHAT cho moi kich thuoc co dinh
src/antiuav_sot/
    data/          # doc IR_label.json, letterbox resize
    motion/        # frame-diff + block-threshold (thay ViBe/findContours)
    detector/      # wrapper YOLOv8-n/s, NMS fixed-size
    fusion/        # merge detector + motion candidates
    tracker/       # Simple Tracker FSM 4 tang, Kalman filter
    pipeline/      # rap toan bo pipeline, chay tung frame (streaming)
    utils/         # box ops (vector hoa), config loader, metrics, viz
scripts/           # entry point CLI cho tung buoc (xem duoi)
tests/             # 44 unit test, dac biet test_pipeline_fixed_shapes.py
                    # kiem tra ky luat "kich thuoc co dinh" xuyen suot
```

## Quy trinh su dung

### 1. Kiem tra du lieu + tao manifest train/val

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Quet toan bo sequence, bao loi format neu co (vd so anh khong khop
`len(exist)`), ghi ra `experiments/data_manifest.json` voi danh sach ten
sequence cho train/val (chia theo **sequence**, khong chia theo frame, de
tranh data leakage giua cac frame lien tiep).

### 2. Train detector

```bash
python scripts/train_detector.py --config configs/default.yaml
```

Tu dong chuyen doi frame sang YOLO format (chi 1 class: `uav`), luu vao
`experiments/yolo_dataset/` - bao gom CA frame co object (exist=1, co box)
LAN frame khong co object (exist=0, negative sample - Ultralytics hieu anh
khong co file `.txt` label tuong ung la "background"), theo ty le dieu
khien boi `data.negative_ratio` trong config. Anh IR grayscale duoc lap
thanh 3 kenh CHI de tuong thich backbone Ultralytics pretrained (xem ghi
chu trong `detector/model.py`).

Augmentation (scale, flip, translate, mosaic, hsv...) doc tu
`configs/default.yaml` -> `detector_train.augmentation`, khong hard-code
trong script - chinh truc tiep trong file YAML de thu nghiem, khong can
sua code. `scale` mac dinh duoc dat cao (0.9) vi UAV la vat the rat nho,
can model thay object o nhieu kich thuoc pixel khac nhau khi train.

**Luu y ve `imgsz`**: Ultralytics train/val mode chi chap nhan `imgsz` la
so nguyen (ep anh ve hinh vuong), du du lieu Anti-UAV la anh chu nhat
512x640 - day la gioi han cua thu vien, khong phai lua chon thiet ke. He
qua la anh se bi them vien (letterbox padding) theo chieu cao khi train,
lam UAV cang nho hon trong anh - mot phan nguyen nhan co the gay recall
thap. Xem ghi chu chi tiet trong `scripts/train_detector.py`.

### 3. (Tuy chon) Quantize INT8

```bash
python scripts/quantize_detector.py --config configs/default.yaml \
    --checkpoint experiments/detector/weights/best.pt --method ptq
```

**Trang thai hien tai: KHUNG CODE, chua chay duoc end-to-end.** Phan noi
voi DataLoader/loss thuc te cua Ultralytics con `NotImplementedError` co
chu dich - xem docstring trong file de biet chinh xac can hoan thien gi
truoc khi dung (dong bo API `torch.ao.quantization` voi phien ban PyTorch
ban dang dung, va loss function thuc su cua YOLO).

### 4. Export ONNX

```bash
python scripts/export_onnx.py --checkpoint experiments/detector/weights/best.pt
```

Nhanh FP32 hoat dong day du (dung `model.export()` native cua Ultralytics).
Nhanh cho checkpoint da quantize con la khung code, phu thuoc buoc 3.

### 5. Danh gia toan bo pipeline (Precision/Success)

```bash
python scripts/eval_tracker.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val
```

Chay **toan bo** pipeline (detector + motion + FSM + Kalman) tung frame
tren moi sequence trong tap val, tinh Precision@20px va Success (AUC) theo
chuan OPE. Day la con so quyet dinh "hieu qua cao hay khong" chu khong
phai suy doan tu kien truc. Them `--save-viz` de xuat video overlay debug
cho tung sequence.

### 5b. Danh gia RIENG detector (doc lap voi tracker)

```bash
python scripts/eval_detector.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val \
    --checkpoint experiments/detector/weights/best.pt
```

Chay `detector.predict()` truc tiep, KHONG di qua motion/fusion/FSM/Kalman.
Bao cao Recall/Precision/mean IoU theo chuan detection, tach rieng frame co
object va frame khong co object (false alarm tren frame rong la loi khac
han voi bo sot muc tieu). Dung script nay TRUOC khi nghi ngo tracker co
van de - neu detector tu no da yeu, sua tracker se khong giai quyet duoc gi.

### 5c. Chan doan pipeline theo tung giai doan (detector vs FSM vs Kalman)

```bash
# Mot sequence cu the (khi da biet sequence nao co van de qua visualize_sequence.py)
python scripts/eval_pipeline_stages.py --config configs/default.yaml \
    --seq-name <ten_sequence> \
    --detector-checkpoint experiments/detector/weights/best.pt

# Toan bo tap val, tong hop
python scripts/eval_pipeline_stages.py --config configs/default.yaml \
    --manifest experiments/data_manifest.json --split val \
    --detector-checkpoint experiments/detector/weights/best.pt
```

Day la cong cu de tra loi cau hoi "detector bat trung o frame dau nhung
track sau do lai lech, loi nam o dau?" - chay full pipeline nhung ghi lai
IoU voi GT tai TUNG giai doan (detector tho / sau FSM / sau Kalman) cho moi
frame, roi tu dong gan co (flag) cho ba loai bat thuong:

- **`flag_detector_missed_but_object_visible`**: detector tu no da mat muc
  tieu (khong lien quan FSM/Kalman) - neu thay flag nay nhieu, quay lai
  muc 5b/huong dan tuning detector, khong phai sua tracker.
- **`flag_fsm_lost_good_detection`**: detector VAN dung (IoU cao) nhung FSM
  khong khop duoc vao track - thuong xay ra khi track da bi Kalman keo lech
  qua xa TRUOC DO, den luc detection dung quay lai thi IoU voi track (da
  lech) tut duoi ca nguong tier4. Day la nghi pham chinh cho hien tuong
  "object to nhung track khong bat trung theo thoi gian".
- **`flag_kalman_diverged_from_fsm`**: FSM khop dung nhung Kalman lam muot
  (smooth) keo vi tri lech khoi quan sat that.

Cung ket hop voi `predict_only_streak` (so frame lien tiep track dang chi
dua vao du bao Kalman, khong co quan sat that) - streak dai truoc mot frame
bi flag `fsm_lost` la dau hieu manh cho thay Kalman da "troi" track ra xa
truoc khi detection dung quay lai.

Ket qua chi tiet tung frame duoc luu JSON tai
`experiments/stage_diagnosis_<seq_name_hoac_split>.json` de phan tich sau.

### 6. Debug truc quan mot sequence cu the

```bash
python scripts/visualize_sequence.py --config configs/default.yaml \
    --seq-name <ten_sequence> \
    --detector-checkpoint experiments/detector/weights/best.pt
```

## Nguyen tac thiet ke xuyen suot (quan trong khi sua/mo rong code)

1. **Kich thuoc co dinh, biet truoc tai cau hinh.** Moi mang numpy trong
   pipeline (candidate boxes, track slots...) co shape CO DINH, khong phu
   thuoc du lieu runtime. Slot rong duoc danh dau bang `INVALID_BOX = -1`
   (xem `utils/box_ops.py`), khong dung `None`/list do dai thay doi.

2. **`configs/fpga_constraints.yaml` la nguon su that duy nhat.** Doi mot
   con so o day (vd `max_det_boxes`) se duoc validate tu dong khi nap
   config (`utils/config.py::FpgaConstraints.validate()`) - neu ban sua
   mot gia tri ma quen sua gia tri lien quan (vd `fusion.max_total_candidates`),
   chuong trinh se bao loi ngay khi khoi dong thay vi loi mo ho giua pipeline.

3. **Vector hoa, khong vong lap tren tung box/pixel.** Xem `utils/box_ops.py`
   (IoU matrix) va `motion/block_threshold.py` (block sum) - day la cac
   phep tinh se anh xa sang mang PE song song tren FPGA, nen phai giu tinh
   "moi phan tu doc lap" ngay tu ban Python.

4. **`tests/test_pipeline_fixed_shapes.py` la bai kiem tra ky luat.** Neu
   ban them logic moi ma vo tinh dung shape dong o dau do, test nay (dac
   biet cac test tham so hoa voi nhieu gia tri N khac nhau) se bat loi
   ngay, truoc khi ban phat hien luc thu port sang FPGA.

5. **Simple Tracker FSM (`tracker/simple_tracker_fsm.py`) la khoi lois
   nhat.** Doc ky docstring 4 tang truoc khi sua - thu tu uu tien
   (Middle Score -> Motion Link -> High Score Init -> Low Score) phan anh
   dung logic bai goc, thay doi thu tu se lam sai ket qua ca ve mat thuat
   toan lan mat tuong thich voi thiet ke FPGA da mo ta trong bao cao.

6. **`motion.diff_mode` khai thac dac diem nhiet cua UAV.** Mac dinh la
   `"positive_only"` (khac `"absdiff"` cua ban goc): vi UAV (dong co/pin)
   luon toa nhiet nong hon background trong anh IR, frame-diff CHI giu
   vung "nong len" (`I_t - I_(t-1) > 0`), bo hoan toan vung "nguoi di" -
   loai duoc "bong ma" phia sau vat the di chuyen ma `absdiff` tao ra (hai
   vung sang cho mot vat the, chi mot la vi tri THAT), dong thoi don gian
   hoa RTL (so sanh du dau thay vi tinh gia tri tuyet doi). Neu doi tuong
   dang lam viec voi khong co dac diem nhiet ro rang (background thay doi
   ca hai chieu, camera rung), doi lai `"absdiff"` trong
   `configs/fpga_constraints.yaml`. Xem `src/antiuav_sot/motion/frame_diff.py`.

## Han che da biet (thanh that voi trang thai hien tai)

- `quantize_detector.py`: khung code, chua chay duoc end-to-end (xem muc 3).
- `export_onnx.py --quantized`: phu thuoc buoc tren, cung la khung code.
- Detector wrapper lap kenh grayscale thanh 3 kenh de dung pretrained
  Ultralytics - day la thoa hiep cho giai doan train/eval GPU, **khong**
  phan anh dung kien truc 1-kenh se chay tren FPGA. Neu muon huan luyen
  backbone 1-kenh tu dau (khong pretrained), can thay `detector/model.py`
  bang kien truc tu viet.
- Chua co script chuyen doi checkpoint ONNX sang dinh dang toolchain FPGA
  cu the (Vitis-AI/hls4ml...) - ngoai pham vi giai doan train/test tren GPU.
