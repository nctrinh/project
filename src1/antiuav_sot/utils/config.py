"""
Nap va hop nhat configs/default.yaml + configs/fpga_constraints.yaml.

Day la diem duy nhat trong repo doc file YAML truc tiep. Moi module khac
(detector, motion, fusion, tracker...) nhan cac gia tri kich thuoc co dinh
thong qua object Config nay, KHONG tu doc YAML rieng - tranh tinh trang
hai noi doc hai gia tri khac nhau cho cung mot rang buoc.

CAP NHAT (tich hop gradient motion candidates + fusion mechanisms):
  - motion.min_box_size / motion.max_box_size / motion.merge_gap_ratio:
    BAT BUOC - dung boi motion/gradient_candidates.py de loc va gop cum
    candidate. Khong co gia tri mac dinh hop ly (phu thuoc kich thuoc that
    cua UAV muc tieu trong du lieu), nen de raw["motion"][...] bat loi
    KeyError ro rang thay vi am tham dung fallback sai.
  - kalman.motion_measurement_noise: TUY CHON, fallback = measurement_noise
    (nghia la mac dinh KHONG phan biet do tin cay theo nguon, giu hanh vi
    cu). Dat lon hon measurement_noise neu muon Kalman tin vi tri tu motion
    it hon vi tri tu detector.
  - khoi tentative_motion_tracks: TUY CHON CA KHOI (enabled mac dinh False)
    - bat che do khoi tao track moi tu motion sau K khung hinh lien tiep,
    danh cho truong hop vat qua nho de detector tung dat tier3_conf_min.

CAP NHAT (8 van de fusion detector/motion - xem update_log.md o root repo
de biet day du boi canh/ly do thiet ke tung muc):
  - tracker.max_consecutive_motion_matches / override_conf_min /
    override_iou_threshold / override_requires_overlap /
    override_motion_dragged_only / override_max_center_distance_ratio:
    TUY CHON, co fallback mac dinh - kiem soat muc do motion duoc phep tu
    dan dat track (Issue 2/4) va cho phep detector conf rat cao "chiem lai"
    track dang bi keo lech (Issue 5). Mac dinh MOI:
    override_requires_overlap=False va override_motion_dragged_only=False -
    detector conf >= override_conf_min chiem lai duoc MOI track active, KHONG
    can chong lan voi vi tri track cu (IoU = 0 van khop), chi bi gioi han
    boi override_max_center_distance_ratio (ti le duong cheo anh). Trong tang
    do, tieu chi xep hang la CONFIDENCE - IoU va khoang cach tam chi pha hoa
    khi confidence gan nhu bang nhau.
  - motion.max_aspect_ratio: TUY CHON, mac dinh 5.0 - loc box qua det
    (Issue 7).
  - motion.merge_min_fill_ratio: TUY CHON, mac dinh 0.15 - dieu kien "khong
    rong" khi gop cum, tranh gop UAV voi nhieu nen thanh box khong lo
    (Issue 8).
  - motion.size_adapt_*: TUY CHON, size_adapt_enabled mac dinh False (GIU
    HANH VI CU: min_box_size/max_box_size tinh) - bat dong bo kich thuoc loc
    theo lich su detector (Issue 3).
  - motion.candidate_anchor_iou: TUY CHON, mac dinh 0.05 - nguong "neo uu
    tien" candidate trung vi tri dang theo doi khoi bi day ra khoi danh sach
    K candidate (Issue 6).
  - tracker.reinit_near_lost_enabled / reinit_max_center_distance_ratio /
    reinit_lost_memory_frames / reinit_trust_conf_min: TUY CHON, mac dinh
    True / 0.15 / 30 / tier3_conf_min - sau khi MAT mot track, vi tri cuoi
    cung cua no duoc giu lai lam "ghost" trong reinit_lost_memory_frames
    khung hinh; trong thoi gian do, track moi khoi tao tu MOTION (tentative
    promote) hoac tu detector conf < reinit_trust_conf_min phai nam trong
    ban kinh reinit_max_center_distance_ratio * duong cheo anh quanh ghost.
    Detector conf >= reinit_trust_conf_min duoc khoi tao o bat ky dau.
  - tentative_motion_tracks.active_overlap_max_iou: TUY CHON, mac dinh 0.1 -
    khong cho candidate motion trung mot track DANG active duoc tinh vao
    Tentative Tracks, tranh mot track that bi "nhan doi" thanh mot ung vien
    tu-khoi-tao gia (Issue 2).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Loi cau hinh khong hop le - vi du xung dot kich thuoc co dinh."""


@dataclasses.dataclass(frozen=True)
class FpgaConstraints:
    # image
    image_height: int
    image_width: int
    image_channels: int
    # detector
    det_input_height: int
    det_input_width: int
    max_det_boxes: int
    det_conf_threshold: float
    det_nms_iou_threshold: float
    # motion
    block_sum_threshold: int
    max_motion_candidates: int
    motion_min_box_size: int
    motion_max_box_size: int
    motion_merge_gap_ratio: float
    motion_max_aspect_ratio: float
    motion_merge_min_fill_ratio: float
    motion_size_adapt_enabled: bool
    motion_size_adapt_window: int
    motion_size_adapt_shrink_factor: float
    motion_size_adapt_grow_factor: float
    motion_size_adapt_min_samples: int
    motion_candidate_anchor_iou: float
    motion_frame_lag: int
    motion_strength_metric: str
    motion_min_strength: float
    motion_thresh_factor: float
    motion_min_area: int
    motion_model: str
    motion_irls_iters: int
    motion_huber_delta: float
    motion_pyramid_levels: int
    # fusion
    max_total_candidates: int
    # tracker
    n_track_slots: int
    tier1_iou_threshold: float
    tier1_conf_min: float
    tier2_iou_threshold: float
    tier3_conf_min: float
    tier4_iou_threshold: float
    max_track_age: int
    max_consecutive_motion_matches: int
    override_conf_min: float
    override_iou_threshold: float
    override_requires_overlap: bool
    override_motion_dragged_only: bool
    override_max_center_distance_ratio: float
    reinit_near_lost_enabled: bool
    reinit_max_center_distance_ratio: float
    reinit_lost_memory_frames: int
    reinit_trust_conf_min: float
    # kalman
    kalman_enabled: bool
    kalman_state_dim: int
    kalman_process_noise: float
    kalman_measurement_noise: float
    kalman_motion_measurement_noise: float
    kalman_max_predict_only_frames: int
    # tentative motion tracks (track moi khoi tao tu motion, tuy chon)
    tentative_motion_tracks_enabled: bool
    max_tentative_motion_tracks: int
    tentative_iou_threshold: float
    tentative_max_miss: int
    motion_track_confirm_frames: int
    tentative_active_overlap_max_iou: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FpgaConstraints":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        image = raw["image"]
        detector = raw["detector"]
        motion = raw["motion"]
        fusion = raw["fusion"]
        tracker = raw["tracker"]
        kalman = raw["kalman"]
        tentative = raw.get("tentative_motion_tracks", {})

        constraints = cls(
            image_height=image["height"],
            image_width=image["width"],
            image_channels=image["channels"],
            det_input_height=detector["input_height"],
            det_input_width=detector["input_width"],
            max_det_boxes=detector["max_det_boxes"],
            det_conf_threshold=detector["conf_threshold"],
            det_nms_iou_threshold=detector["nms_iou_threshold"],
            block_sum_threshold=motion["block_sum_threshold"],
            max_motion_candidates=motion["max_motion_candidates"],
            motion_min_box_size=motion["min_box_size"],
            motion_max_box_size=motion["max_box_size"],
            motion_merge_gap_ratio=motion["merge_gap_ratio"],
            motion_max_aspect_ratio=motion.get("max_aspect_ratio", 5.0),
            motion_merge_min_fill_ratio=motion.get("merge_min_fill_ratio", 0.15),
            motion_size_adapt_enabled=motion.get("size_adapt_enabled", False),
            motion_size_adapt_window=motion.get("size_adapt_window", 30),
            motion_size_adapt_shrink_factor=motion.get("size_adapt_shrink_factor", 0.6),
            motion_size_adapt_grow_factor=motion.get("size_adapt_grow_factor", 1.6),
            motion_size_adapt_min_samples=motion.get("size_adapt_min_samples", 5),
            motion_candidate_anchor_iou=motion.get("candidate_anchor_iou", 0.05),
            motion_frame_lag=motion.get("frame_lag", 5),
            motion_strength_metric=motion.get("strength_metric", "sum"),
            motion_min_strength=(
                float("-inf") if motion.get("min_strength") is None else motion["min_strength"]
            ),
            motion_thresh_factor=motion.get("thresh_factor", 2.5),
            motion_min_area=motion.get("min_area", 20),
            motion_model=motion.get("model", "affine"),
            motion_irls_iters=motion.get("irls_iters", 3),
            motion_huber_delta=motion.get("huber_delta", 10.0),
            motion_pyramid_levels=motion.get("pyramid_levels", 2),
            max_total_candidates=fusion["max_total_candidates"],
            n_track_slots=tracker["n_track_slots"],
            tier1_iou_threshold=tracker["tier1_iou_threshold"],
            tier1_conf_min=tracker["tier1_conf_min"],
            tier2_iou_threshold=tracker["tier2_iou_threshold"],
            tier3_conf_min=tracker["tier3_conf_min"],
            tier4_iou_threshold=tracker["tier4_iou_threshold"],
            max_track_age=tracker["max_track_age"],
            max_consecutive_motion_matches=tracker.get("max_consecutive_motion_matches", 5),
            override_conf_min=tracker.get("override_conf_min", 0.85),
            override_iou_threshold=tracker.get("override_iou_threshold", 0.05),
            override_requires_overlap=tracker.get("override_requires_overlap", False),
            override_motion_dragged_only=tracker.get(
                "override_motion_dragged_only", False
            ),
            override_max_center_distance_ratio=tracker.get(
                "override_max_center_distance_ratio", 1.0
            ),
            reinit_near_lost_enabled=tracker.get("reinit_near_lost_enabled", True),
            reinit_max_center_distance_ratio=tracker.get(
                "reinit_max_center_distance_ratio", 0.15
            ),
            reinit_lost_memory_frames=tracker.get("reinit_lost_memory_frames", 30),
            reinit_trust_conf_min=tracker.get(
                "reinit_trust_conf_min", tracker["tier3_conf_min"]
            ),
            kalman_enabled=kalman["enabled"],
            kalman_state_dim=kalman["state_dim"],
            kalman_process_noise=kalman["process_noise"],
            kalman_measurement_noise=kalman["measurement_noise"],
            kalman_motion_measurement_noise=kalman.get(
                "motion_measurement_noise", kalman["measurement_noise"]
            ),
            kalman_max_predict_only_frames=kalman["max_predict_only_frames"],
            tentative_motion_tracks_enabled=tentative.get("enabled", False),
            max_tentative_motion_tracks=tentative.get("max_slots", 0),
            tentative_iou_threshold=tentative.get("iou_threshold", 0.3),
            tentative_max_miss=tentative.get("max_miss", 3),
            motion_track_confirm_frames=tentative.get("confirm_frames", 4),
            tentative_active_overlap_max_iou=tentative.get("active_overlap_max_iou", 0.1),
        )
        constraints.validate()
        return constraints

    def validate(self) -> None:
        """Kiem tra tinh nhat quan giua cac truong - fail som thay vi de
        loi shape xuat hien mo ho o giua pipeline."""
        expected_total = self.max_det_boxes + self.max_motion_candidates
        if self.max_total_candidates != expected_total:
            raise ConfigError(
                f"fusion.max_total_candidates ({self.max_total_candidates}) phai bang "
                f"detector.max_det_boxes + motion.max_motion_candidates "
                f"({self.max_det_boxes} + {self.max_motion_candidates} = {expected_total})"
            )
        if self.image_channels != 1:
            raise ConfigError(
                "image.channels phai = 1 (grayscale IR). Pipeline nay khong ho tro RGB."
            )
        if self.n_track_slots < 1:
            raise ConfigError("tracker.n_track_slots phai >= 1")
        if self.kalman_state_dim != 4:
            raise ConfigError(
                "kalman.state_dim hien chi ho tro = 4 ([x, y, vx, vy]). "
                "Doi gia tri khac se can sua lai KalmanBoxTracker."
            )
        if self.motion_min_box_size <= 0 or self.motion_max_box_size <= self.motion_min_box_size:
            raise ConfigError(
                f"motion.min_box_size ({self.motion_min_box_size}) phai > 0 va "
                f"nho hon motion.max_box_size ({self.motion_max_box_size})"
            )
        if self.motion_merge_gap_ratio < 0:
            raise ConfigError("motion.merge_gap_ratio phai >= 0")
        if self.motion_max_aspect_ratio < 1.0:
            raise ConfigError(
                f"motion.max_aspect_ratio ({self.motion_max_aspect_ratio}) phai >= 1.0"
            )
        if not (0.0 < self.motion_merge_min_fill_ratio <= 1.0):
            raise ConfigError(
                f"motion.merge_min_fill_ratio ({self.motion_merge_min_fill_ratio}) "
                "phai nam trong (0.0, 1.0]"
            )
        if self.motion_size_adapt_enabled:
            if self.motion_size_adapt_window <= 0:
                raise ConfigError("motion.size_adapt_window phai >= 1 khi size_adapt_enabled=true")
            if self.motion_size_adapt_min_samples <= 0:
                raise ConfigError(
                    "motion.size_adapt_min_samples phai >= 1 khi size_adapt_enabled=true"
                )
            if self.motion_size_adapt_shrink_factor <= 0 or self.motion_size_adapt_grow_factor <= 0:
                raise ConfigError(
                    "motion.size_adapt_shrink_factor/grow_factor phai > 0"
                )
        if self.motion_candidate_anchor_iou < 0:
            raise ConfigError("motion.candidate_anchor_iou phai >= 0")
        if self.motion_frame_lag < 1:
            raise ConfigError("motion.frame_lag phai >= 1")
        if self.motion_strength_metric not in ("sum", "mean", "contrast"):
            raise ConfigError(
                f"motion.strength_metric ({self.motion_strength_metric!r}) phai la "
                "'sum', 'mean' hoac 'contrast'"
            )
        if self.motion_model not in ("affine", "translation"):
            raise ConfigError(
                f"motion.model ({self.motion_model!r}) phai la 'affine' hoac 'translation'"
            )
        if self.motion_irls_iters < 1:
            raise ConfigError("motion.irls_iters phai >= 1")
        if self.motion_huber_delta <= 0:
            raise ConfigError("motion.huber_delta phai > 0")
        if self.motion_pyramid_levels < 0:
            raise ConfigError("motion.pyramid_levels phai >= 0")
        if self.motion_min_area < 0:
            raise ConfigError("motion.min_area phai >= 0")
        if self.max_consecutive_motion_matches < 1:
            raise ConfigError("tracker.max_consecutive_motion_matches phai >= 1")
        if self.override_conf_min < self.tier3_conf_min:
            raise ConfigError(
                f"tracker.override_conf_min ({self.override_conf_min}) phai >= "
                f"tracker.tier3_conf_min ({self.tier3_conf_min}) - Confidence Override "
                "chi danh cho detection TIN CAY HON CA nguong khoi tao track moi."
            )
        if self.override_iou_threshold < 0:
            raise ConfigError(
                f"tracker.override_iou_threshold ({self.override_iou_threshold}) phai >= 0."
            )
        if self.override_requires_overlap and self.override_iou_threshold <= 0:
            raise ConfigError(
                f"tracker.override_iou_threshold ({self.override_iou_threshold}) phai > 0 "
                "khi tracker.override_requires_overlap = true - nguong = 0.0 khien Tang "
                "Confidence Override khop MOI cap (ke ca IoU = 0) ma khong co tieu chi xep "
                "hang nao, tuc khop 'ao' slot/candidate dau tien. Neu muc dich la 'tin "
                "detector du khong chong lan', dat override_requires_overlap = false: khi do "
                "cac cap duoc xep hang bang CONFIDENCE (IoU/khoang cach tam chi pha hoa), "
                "cap khong chong lan bi gioi han boi override_max_center_distance_ratio."
            )
        if not (0.0 < self.override_max_center_distance_ratio <= 1.5):
            raise ConfigError(
                "tracker.override_max_center_distance_ratio "
                f"({self.override_max_center_distance_ratio}) phai nam trong (0.0, 1.5] - "
                "tinh theo ti le duong cheo anh (1.0 = khong gioi han thuc te)."
            )
        if self.reinit_near_lost_enabled:
            if not (0.0 < self.reinit_max_center_distance_ratio <= 1.5):
                raise ConfigError(
                    "tracker.reinit_max_center_distance_ratio "
                    f"({self.reinit_max_center_distance_ratio}) phai nam trong (0.0, 1.5] - "
                    "tinh theo ti le duong cheo anh."
                )
            if self.reinit_lost_memory_frames < 0:
                raise ConfigError(
                    "tracker.reinit_lost_memory_frames "
                    f"({self.reinit_lost_memory_frames}) phai >= 0."
                )
            if not (0.0 <= self.reinit_trust_conf_min <= 1.0):
                raise ConfigError(
                    f"tracker.reinit_trust_conf_min ({self.reinit_trust_conf_min}) "
                    "phai nam trong [0.0, 1.0]."
                )
            if self.reinit_trust_conf_min < self.tier3_conf_min:
                raise ConfigError(
                    f"tracker.reinit_trust_conf_min ({self.reinit_trust_conf_min}) < "
                    f"tracker.tier3_conf_min ({self.tier3_conf_min}) - moi candidate cua "
                    "Tang 3 deu da co conf >= tier3_conf_min, nen gate se KHONG BAO GIO "
                    "chan duoc gi o nhanh detector. Dat >= tier3_conf_min (bang nhau = "
                    "detector khong bao gio bi gate, chi gate nhanh motion)."
                )
        if not (0.0 <= self.tentative_active_overlap_max_iou <= 1.0):
            raise ConfigError(
                "tentative_motion_tracks.active_overlap_max_iou phai nam trong [0.0, 1.0]"
            )
        if self.kalman_motion_measurement_noise < self.kalman_measurement_noise:
            raise ConfigError(
                f"kalman.motion_measurement_noise ({self.kalman_motion_measurement_noise}) "
                f"phai >= kalman.measurement_noise ({self.kalman_measurement_noise}) - "
                "quan sat tu motion khong duoc tin cay HON quan sat tu detector."
            )
        if self.tentative_motion_tracks_enabled:
            if self.max_tentative_motion_tracks <= 0:
                raise ConfigError(
                    "tentative_motion_tracks.max_slots phai >= 1 khi enabled=true"
                )
            if self.motion_track_confirm_frames <= 0:
                raise ConfigError("tentative_motion_tracks.confirm_frames phai >= 1")
            if self.tentative_max_miss < 0:
                raise ConfigError("tentative_motion_tracks.max_miss phai >= 0")


@dataclasses.dataclass(frozen=True)
class Config:
    """Toan bo cau hinh runtime: hyperparameter train/eval + fpga constraints."""

    raw: dict[str, Any]
    fpga: FpgaConstraints
    config_dir: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        fpga_path = path.parent / Path(raw["fpga_constraints_path"]).name \
            if not Path(raw["fpga_constraints_path"]).is_absolute() \
            else Path(raw["fpga_constraints_path"])
        # Neu fpga_constraints_path duoc ghi tuong doi so voi thu muc goc repo
        # (truong hop thong thuong: "configs/fpga_constraints.yaml"), uu tien
        # duong dan do truoc; fallback ve cung thu muc voi default.yaml.
        candidate = Path(raw["fpga_constraints_path"])
        if candidate.exists():
            fpga_path = candidate
        elif (path.parent / candidate.name).exists():
            fpga_path = path.parent / candidate.name
        else:
            raise ConfigError(
                f"Khong tim thay fpga_constraints_path: {raw['fpga_constraints_path']}"
            )

        fpga = FpgaConstraints.from_yaml(fpga_path)
        return cls(raw=raw, fpga=fpga, config_dir=path.parent)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Truy cap an toan vao raw config: cfg.get('detector_train', 'epochs')."""
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.from_yaml(path)