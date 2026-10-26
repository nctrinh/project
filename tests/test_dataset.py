import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from antiuav_sot.data.dataset import AntiUAVDataset, SOTSequence, SequenceFormatError


def _write_dummy_image(path: Path, size: tuple[int, int] = (16, 16)) -> None:
    img = np.zeros(size, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_label(path: Path, n_frames: int) -> None:
    label = {
        "exist": [1] * n_frames,
        "gt_rect": [[1, 1, 5, 5]] * n_frames,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label, f)


def test_sequence_reads_jpg_only(tmp_path):
    seq_dir = tmp_path / "seq_jpg"
    seq_dir.mkdir()
    for i in range(3):
        _write_dummy_image(seq_dir / f"{i:06d}.jpg")
    _write_label(seq_dir / SOTSequence.LABEL_FILENAME, n_frames=3)

    seq = SOTSequence(seq_dir, image_glob="*.jpg")
    assert len(seq) == 3


def test_sequence_reads_png_only(tmp_path):
    seq_dir = tmp_path / "seq_png"
    seq_dir.mkdir()
    for i in range(3):
        _write_dummy_image(seq_dir / f"{i:06d}.png")
    _write_label(seq_dir / SOTSequence.LABEL_FILENAME, n_frames=3)

    seq = SOTSequence(seq_dir, image_glob="*.png")
    assert len(seq) == 3


def test_sequence_reads_mixed_jpg_and_png_with_default_patterns(tmp_path):
    """Truong hop chinh cua bug report: sequence lan ca jpg lan png, dung
    danh sach pattern mac dinh cua SOTSequence."""
    seq_dir = tmp_path / "seq_mixed"
    seq_dir.mkdir()
    for i in range(2):
        _write_dummy_image(seq_dir / f"{i:06d}.jpg")
    for i in range(2, 4):
        _write_dummy_image(seq_dir / f"{i:06d}.png")
    _write_label(seq_dir / SOTSequence.LABEL_FILENAME, n_frames=4)

    seq = SOTSequence(seq_dir, image_glob=list(SOTSequence.DEFAULT_IMAGE_EXTENSIONS))
    assert len(seq) == 4
    # Thu tu phai theo stem (000000, 000001, 000002, 000003), khong phai
    # gop rieng tung dinh dang roi noi lai (se sai thu tu neu the).
    names = [p.stem for p in seq.image_paths]
    assert names == ["000000", "000001", "000002", "000003"]


def test_sequence_with_list_pattern_but_only_one_extension_present(tmp_path):
    """Truyen list pattern nhung sequence chi thuc su co 1 dinh dang -
    khong duoc bao loi format, chi duoc dung dinh dang tim thay."""
    seq_dir = tmp_path / "seq_only_png"
    seq_dir.mkdir()
    for i in range(3):
        _write_dummy_image(seq_dir / f"{i:06d}.png")
    _write_label(seq_dir / SOTSequence.LABEL_FILENAME, n_frames=3)

    seq = SOTSequence(seq_dir, image_glob=["*.jpg", "*.jpeg", "*.png"])
    assert len(seq) == 3


def test_dataset_scans_sequences_with_mixed_formats(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    seq1 = root / "seq_a"
    seq1.mkdir()
    for i in range(2):
        _write_dummy_image(seq1 / f"{i:06d}.jpg")
    _write_label(seq1 / SOTSequence.LABEL_FILENAME, n_frames=2)

    seq2 = root / "seq_b"
    seq2.mkdir()
    for i in range(2):
        _write_dummy_image(seq2 / f"{i:06d}.png")
    _write_label(seq2 / SOTSequence.LABEL_FILENAME, n_frames=2)

    dataset = AntiUAVDataset(root, image_glob=["*.jpg", "*.jpeg", "*.png"])
    assert len(dataset) == 2

    loaded_a = dataset.load_sequence("seq_a")
    loaded_b = dataset.load_sequence("seq_b")
    assert len(loaded_a) == 2
    assert len(loaded_b) == 2


def test_sequence_frame_count_mismatch_raises(tmp_path):
    seq_dir = tmp_path / "seq_bad"
    seq_dir.mkdir()
    for i in range(3):
        _write_dummy_image(seq_dir / f"{i:06d}.jpg")
    _write_label(seq_dir / SOTSequence.LABEL_FILENAME, n_frames=5)  # co ý gay lech

    with pytest.raises(SequenceFormatError):
        SOTSequence(seq_dir, image_glob="*.jpg")


def test_empty_gt_rect_accepted_when_not_exist(tmp_path):
    """Dung format that cua Anti-UAV: gt_rect=[] khi exist=0 - PHAI duoc
    chap nhan va quy ve [0,0,0,0], KHONG duoc raise loi."""
    seq_dir = tmp_path / "seq_empty_rect"
    seq_dir.mkdir()
    for i in range(3):
        _write_dummy_image(seq_dir / f"{i:06d}.jpg")

    label = {
        "exist": [1, 0, 1],
        "gt_rect": [[1, 1, 5, 5], [], [2, 2, 6, 6]],
    }
    with open(seq_dir / SOTSequence.LABEL_FILENAME, "w", encoding="utf-8") as f:
        json.dump(label, f)

    seq = SOTSequence(seq_dir, image_glob="*.jpg")
    assert len(seq) == 3

    sample0 = seq[0]
    assert sample0.exist is True
    np.testing.assert_array_equal(sample0.gt_box_xywh, [1, 1, 5, 5])

    sample1 = seq[1]
    assert sample1.exist is False
    np.testing.assert_array_equal(sample1.gt_box_xywh, [0, 0, 0, 0])

    sample2 = seq[2]
    assert sample2.exist is True
    np.testing.assert_array_equal(sample2.gt_box_xywh, [2, 2, 6, 6])


def test_malformed_rect_with_exist_true_still_raises(tmp_path):
    """exist=1 nhung rect khong hop le VAN PHAI la loi that su - khac voi
    truong hop exist=0 (duoc chap nhan)."""
    seq_dir = tmp_path / "seq_bad_positive"
    seq_dir.mkdir()
    for i in range(2):
        _write_dummy_image(seq_dir / f"{i:06d}.jpg")

    label = {
        "exist": [1, 1],
        "gt_rect": [[1, 1, 5, 5], []],  # frame 1: exist=1 nhung rect rong -> loi that su
    }
    with open(seq_dir / SOTSequence.LABEL_FILENAME, "w", encoding="utf-8") as f:
        json.dump(label, f)

    seq = SOTSequence(seq_dir, image_glob="*.jpg")
    with pytest.raises(SequenceFormatError):
        seq[1]
