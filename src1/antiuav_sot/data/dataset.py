"""
Doc du lieu Anti-UAV: moi sequence la mot thu muc chua cac anh IR (thermal,
grayscale) + 1 file IR_label.json co dang:

    {
        "exist": [1, 1, 1, 0, 1, ...],              # 1 neu UAV xuat hien trong frame, 0 neu khong
        "gt_rect": [[x, y, w, h], [x, y, w, h], ...]  # box ground-truth dang xywh
    }

len(exist) == len(gt_rect) == so anh trong thu muc. Khi exist[i] == 0,
gt_rect[i] co the la [] (mang rong), [0, 0, 0, 0], hoac gia tri khong hop
le khac tuy phien ban dataset - TAT CA duoc coi la hop le va quy ve
[0, 0, 0, 0] khi doc (xem SOTSequence.__getitem__). Truong hop nay CAN
duoc loc ra khi train detector (khong co object thi khong co box duong de
hoc), nhung VAN GIU LAI khi eval tracker (de tinh dung Precision/Success
ke ca frame mat muc tieu, dung chuan benchmark OPE goc, va de model/pipeline
hoc duoc rang mot frame co the KHONG co object - khong phai loi du lieu).

Ngoai ra module nay cung cung cap AntiUAVDatasetTest/UnlabeledSequence de doc
cac sequence KHONG CO NHAN (khong co IR_label.json) - vi du tap test/submit
o data/Anti-UAV/Train_Track1_2_without_label/. Hai nhanh nay HOAN TOAN tach
biet: AntiUAVDataset/SOTSequence luon doi hoi IR_label.json, con
AntiUAVDatasetTest/UnlabeledSequence khong bao gio doc file do.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


class SequenceFormatError(Exception):
    """IR_label.json khong khop voi so luong anh, hoac cau truc sai."""


@dataclasses.dataclass
class SequenceSample:
    """Mot frame don le trong sequence, da giai nen tu SOTSequence."""

    image_path: Path
    exist: bool
    gt_box_xywh: np.ndarray  # (4,) - co the la [0,0,0,0] neu exist=False
    frame_index: int
    seq_name: str


class SOTSequence:
    """Doc mot sequence duy nhat (mot thu muc) tu Anti-UAV Train_Track1_2.

    Vi du cau truc thu muc:
        data/Anti-UAV/Train_Track1_2/train/20190925_101846_1_2/
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── IR_label.json

    Ho tro nhieu dinh dang anh trong CUNG mot lan quet (vd du lieu Anti-UAV
    thuc te lan ca .jpg lan .png giua cac sequence, hoac hiem hon la trong
    cung mot sequence). Sap xep theo TEN FILE KHONG KEM DUOI (stem), khong
    phai theo chuoi day du kem duoi - vi "000001.jpg" va "000001.png" phai
    duoc coi la CUNG mot vi tri frame khi so sanh thu tu, tranh truong hop
    sap xep lai sai khac di so voi thu tu ma IR_label.json gia dinh (index
    theo thu tu chup, khong phai theo dinh dang file).
    """

    LABEL_FILENAME = "IR_label.json"
    DEFAULT_IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

    def __init__(
        self,
        seq_dir: str | Path,
        image_glob: str | list[str] = "*.jpg",
    ):
        self.seq_dir = Path(seq_dir)
        self.seq_name = self.seq_dir.name
        self.label_path = self.seq_dir / self.LABEL_FILENAME
        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc sequence: {self.seq_dir}")
        if not self.label_path.is_file():
            raise FileNotFoundError(f"Khong tim thay {self.LABEL_FILENAME} trong {self.seq_dir}")

        patterns = [image_glob] if isinstance(image_glob, str) else list(image_glob)

        found_by_ext: dict[str, list[Path]] = {}
        for pattern in patterns:
            matches = list(self.seq_dir.glob(pattern))
            if matches:
                found_by_ext[pattern] = matches

        if len(found_by_ext) > 1:
            summary = ", ".join(f"{ext}: {len(files)}" for ext, files in found_by_ext.items())
            print(
                f"[SOTSequence] CANH BAO: sequence '{self.seq_name}' co nhieu "
                f"dinh dang anh cung luc ({summary}). Cac anh se duoc sap xep "
                "chung theo ten file (khong ke duoi) - kiem tra lai neu quy "
                "uoc dat ten khac nhau giua cac dinh dang, vi co the gay lech "
                "thu tu frame so voi IR_label.json."
            )

        all_paths = [p for paths in found_by_ext.values() for p in paths]
        self.image_paths = sorted(all_paths, key=lambda p: (p.stem, p.suffix))

        if len(self.image_paths) == 0:
            print(f"[SOTSequence] Khong tim thay anh nao khop pattern(s) {patterns} trong {self.seq_dir}")
            raise SequenceFormatError(
                f"Khong tim thay anh nao khop pattern(s) {patterns} trong {self.seq_dir}"
            )

        with open(self.label_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        if "exist" not in label or ("gt_rect" not in label and "gt" not in label):
            print(f"[SOTSequence] {self.label_path} thieu truong 'exist' hoac 'gt_rect'")
            raise SequenceFormatError(
                f"{self.label_path} thieu truong 'exist' hoac 'gt_rect' hoac 'gt'"
            )

        self.exist: list[int] = label["exist"]
        if "gt_rect" in label:
            self.gt_rect: list[list[float]] = label["gt_rect"]
        elif "gt" in label:
            self.gt_rect: list[list[float]] = label["gt"]

        if not (len(self.exist) == len(self.gt_rect) == len(self.image_paths)):
            raise SequenceFormatError(
                f"So luong khong khop trong {self.seq_name}: "
                f"so anh={len(self.image_paths)}, "
                f"len(exist)={len(self.exist)}, len(gt_rect)={len(self.gt_rect)}. "
                "IR_label.json phai co so phan tu bang chinh xac so anh trong thu muc."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> SequenceSample:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} ngoai pham vi [0, {len(self)})")

        exists = bool(self.exist[idx])
        rect = self.gt_rect[idx]

        if len(rect) != 4:
            if exists:
                raise SequenceFormatError(
                    f"{self.seq_name} frame {idx}: exist=1 nhung gt_rect khong "
                    f"hop le (can 4 gia tri [x,y,w,h], nhan duoc {rect}). "
                    "Day la loi du lieu that su, khong the tu suy ra box."
                )
            gt_box = np.zeros((4,), dtype=np.float32)
        else:
            gt_box = np.array(rect, dtype=np.float32)

        return SequenceSample(
            image_path=self.image_paths[idx],
            exist=exists,
            gt_box_xywh=gt_box,
            frame_index=idx,
            seq_name=self.seq_name,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class AntiUAVDataset:
    """Quet toan bo cac sequence trong root_dir va cung cap truy cap thong nhat.

    Dung cho ca hai muc dich:
      - Train detector: can duyet tung frame co exist=True tren toan bo dataset.
      - Eval tracker: can duyet tung sequence rieng le, giu nguyen thu tu frame
        (bao gom ca frame exist=False) de tinh Precision/Success dung chuan.
    """

    def __init__(self, root_dir: str | Path, image_glob: str | list[str] = "*.jpg"):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay root_dir: {self.root_dir}")

        self.image_glob = image_glob
        self.seq_dirs = sorted(
            d for d in self.root_dir.iterdir()
            if d.is_dir() and (d / SOTSequence.LABEL_FILENAME).is_file()
        )
        if len(self.seq_dirs) == 0:
            raise SequenceFormatError(
                f"Khong tim thay sequence nao (khong co thu muc con chua "
                f"{SOTSequence.LABEL_FILENAME}) trong {self.root_dir}"
            )

    def __len__(self) -> int:
        return len(self.seq_dirs)

    def sequence_names(self) -> list[str]:
        return [d.name for d in self.seq_dirs]

    def load_sequence(self, seq_name: str) -> SOTSequence:
        seq_dir = self.root_dir / seq_name
        return SOTSequence(seq_dir, image_glob=self.image_glob)

    def __iter__(self):
        for seq_dir in self.seq_dirs:
            yield SOTSequence(seq_dir, image_glob=self.image_glob)

    def iter_detector_samples(self):
        """Duyet tat ca frame co exist=True tren MOI sequence - dung cho
        train detector (khong yield frame khong co object)."""
        for seq in self:
            for sample in seq:
                if sample.exist:
                    yield sample

    def split(
        self, train_ratio: float = 0.85, seed: int = 42
    ) -> tuple[list[str], list[str]]:
        """Chia danh sach TEN SEQUENCE (khong phai frame) thanh train/val,
        tranh data leakage giua cac frame lien tiep cua cung mot sequence."""
        names = self.sequence_names()
        rng = np.random.default_rng(seed)
        shuffled = list(names)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_ratio))
        return shuffled[:n_train], shuffled[n_train:]


@dataclasses.dataclass
class UnlabeledSequenceSample:
    """Mot frame don le trong sequence KHONG CO NHAN (khong co IR_label.json).

    Khac voi SequenceSample, KHONG co truong exist/gt_box_xywh vi khong co
    file nhan de doc - day la du lieu dung de predict/submit, khong dung de
    tinh metric so voi ground-truth."""

    image_path: Path
    frame_index: int
    seq_name: str


class UnlabeledSequence:
    """Doc mot sequence KHONG CO NHAN (khong co IR_label.json) - dung cho tap
    test/submit, vi du data/Anti-UAV/Train_Track1_2_without_label/<seq_name>/.

    Chi doc danh sach anh trong thu muc, sap xep theo cung quy uoc voi
    SOTSequence (theo stem, khong ke duoi file, de nhat quan neu sau nay can
    so sanh thu tu voi cac sequence co nhan). KHONG doc/doi hoi IR_label.json
    - neu thu muc co file do thi cung bi BO QUA hoan toan, khong dung den.
    """

    DEFAULT_IMAGE_EXTENSIONS = SOTSequence.DEFAULT_IMAGE_EXTENSIONS

    def __init__(
        self,
        seq_dir: str | Path,
        image_glob: str | list[str] = "*.jpg",
    ):
        self.seq_dir = Path(seq_dir)
        self.seq_name = self.seq_dir.name

        if not self.seq_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc sequence: {self.seq_dir}")

        patterns = [image_glob] if isinstance(image_glob, str) else list(image_glob)

        found_by_ext: dict[str, list[Path]] = {}
        for pattern in patterns:
            matches = list(self.seq_dir.glob(pattern))
            if matches:
                found_by_ext[pattern] = matches

        if len(found_by_ext) > 1:
            summary = ", ".join(f"{ext}: {len(files)}" for ext, files in found_by_ext.items())
            print(
                f"[UnlabeledSequence] CANH BAO: sequence '{self.seq_name}' co nhieu "
                f"dinh dang anh cung luc ({summary}). Cac anh se duoc sap xep "
                "chung theo ten file (khong ke duoi)."
            )

        all_paths = [p for paths in found_by_ext.values() for p in paths]
        self.image_paths = sorted(all_paths, key=lambda p: (p.stem, p.suffix))

        if len(self.image_paths) == 0:
            raise SequenceFormatError(
                f"Khong tim thay anh nao khop pattern(s) {patterns} trong {self.seq_dir}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> UnlabeledSequenceSample:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} ngoai pham vi [0, {len(self)})")

        return UnlabeledSequenceSample(
            image_path=self.image_paths[idx],
            frame_index=idx,
            seq_name=self.seq_name,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class AntiUAVDatasetTest:
    """Quet toan bo cac sequence KHONG CO NHAN trong root_dir, vi du
    data/Anti-UAV/Train_Track1_2_without_label/.

    Khac voi AntiUAVDataset:
      - Liet ke MOI thu muc con la sequence, KHONG loc theo su ton tai cua
        IR_label.json (vi cac sequence nay von di khong co file do).
      - Dung UnlabeledSequence (khong phai SOTSequence) de doc frame, nen
        khong bao gio dong cham/doi hoi IR_label.json.
      - Khong co split() train/val vi day la tap du lieu predict/submit,
        khong dung de train hay tinh metric so voi ground-truth.
    """

    def __init__(self, root_dir: str | Path, image_glob: str | list[str] = "*.jpg"):
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay root_dir: {self.root_dir}")

        self.image_glob = image_glob
        self.seq_dirs = sorted(
            d for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if len(self.seq_dirs) == 0:
            raise SequenceFormatError(
                f"Khong tim thay sequence nao (khong co thu muc con) trong {self.root_dir}"
            )

    def __len__(self) -> int:
        return len(self.seq_dirs)

    def sequence_names(self) -> list[str]:
        return [d.name for d in self.seq_dirs]

    def load_sequence(self, seq_name: str) -> UnlabeledSequence:
        seq_dir = self.root_dir / seq_name
        return UnlabeledSequence(seq_dir, image_glob=self.image_glob)

    def __iter__(self):
        for seq_dir in self.seq_dirs:
            yield UnlabeledSequence(seq_dir, image_glob=self.image_glob)