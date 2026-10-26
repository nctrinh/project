import numpy as np
import pytest

from antiuav_sot.tracker.kalman import ConstantVelocityKalmanBox


def test_init_sets_state_from_first_box():
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)
    box = np.array([0, 0, 10, 10], dtype=np.float32)  # cx=5, cy=5, w=10, h=10
    kf.init(box)

    assert kf.is_initialized()
    np.testing.assert_allclose(kf.state, [5, 5, 0, 0])


def test_predict_without_init_raises():
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)
    with pytest.raises(RuntimeError):
        kf.predict()


def test_update_first_call_initializes_and_returns_input_unchanged():
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    result = kf.update(box)
    np.testing.assert_allclose(result, box)
    assert kf.is_initialized()


def test_update_on_exact_match_stays_near_observation():
    """Neu quan sat lien tiep KHOP HOAN HAO voi vi tri du bao (chuyen dong
    tuyen tinh dung), Kalman output phai bam sat quan sat, khong duoc
    trôi ra xa dan."""
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)

    # UAV di chuyen deu +2 px/frame theo x, dung dung mo hinh constant velocity.
    boxes = [
        np.array([0 + 2 * i, 0, 10 + 2 * i, 10], dtype=np.float32) for i in range(10)
    ]

    outputs = []
    for box in boxes:
        outputs.append(kf.update(box))

    # Sau vai frame de "hoc" van toc, output phai gan voi quan sat that
    # (sai lech nho, khong duoc phinh to dan qua tung frame).
    for i in range(3, len(boxes)):
        gt_cx = (boxes[i][0] + boxes[i][2]) / 2.0
        out_cx = (outputs[i][0] + outputs[i][2]) / 2.0
        assert abs(gt_cx - out_cx) < 3.0, (
            f"Frame {i}: Kalman output lech qua xa quan sat that "
            f"(gt_cx={gt_cx}, out_cx={out_cx}) - co the la regression cua "
            "bug 'thieu predict truoc update'."
        )


def test_update_does_not_collapse_gain_over_repeated_calls():
    """REGRESSION TEST - tai hien CHINH XAC kich ban da quan sat thuc te
    qua scripts/eval_pipeline_stages.py: UAV di chuyen deu (constant
    velocity), track khop DUNG detection that MOI frame (offset=0,
    khong phai track bi mat). Neu thieu buoc predict() truoc update()
    (bug goc), IoU giua Kalman output va quan sat that se TUT DAN DEU
    ve 0 qua vai frame (dung log thuc te: 0.75 -> 0.51 -> 0.34 -> ... ->
    0.00) du detection luon dung - vi state khong bao gio duoc dich
    chuyen theo van toc (F) truoc khi so voi quan sat moi, output ngay
    cang tut lai phia sau vi tri that dang di chuyen.
    """
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # UAV di chuyen deu +8px/frame, kich thuoc ~30x25 - tuong tu ty le va
    # toc do quan sat duoc trong du lieu that (Anti-UAV sequence).
    w, h = 30, 25
    ious = []
    for i in range(12):
        cx, cy = 100 + 8 * i, 100
        obs = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        out = kf.update(obs)
        ious.append(iou(out, obs))

    # Sau vai frame de filter "hoc" van toc, IoU phai on dinh o muc CAO
    # (>0.9), KHONG duoc tut dan ve gan 0 nhu bug goc.
    for i, val in enumerate(ious[4:]):
        assert val > 0.9, (
            f"Frame {i + 4}: IoU giua Kalman output va quan sat that = {val:.3f}, "
            "qua thap - dau hieu regression cua bug 'thieu predict() truoc "
            "update()' (state khong dich chuyen theo van toc truoc khi ket "
            "hop quan sat moi, khien output tut lai phia sau vat the dang "
            "di chuyen deu)."
        )


def test_kalman_gain_does_not_shrink_to_near_zero():
    """Kiem tra truc tiep Kalman gain (K) khong bi co ve gan 0 sau nhieu
    lan update lien tiep - day la trieu chung CHINH XAC cua bug da fix."""
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)

    box = np.array([0, 0, 10, 10], dtype=np.float32)
    kf.update(box)  # lan 1: chi init

    trace_covariance_over_time = []
    for i in range(15):
        box = np.array([2.0 * i, 0, 10 + 2.0 * i, 10], dtype=np.float32)
        kf.update(box)
        # covariance sau update - neu bug ton tai, covariance se co dan ve
        # (gan) 0 vi khong bao gio duoc "boi" lai boi Q truoc khi update.
        trace_covariance_over_time.append(float(np.trace(kf.covariance)))

    # Covariance phai on dinh o mot muc duong (steady-state cua Kalman
    # filter), KHONG duoc tiep tuc giam ve gan 0 mai - kiem tra 5 gia tri
    # cuoi khong nho hon qua nhieu so voi gia tri thu 5 (sau khi da on dinh).
    settled_value = trace_covariance_over_time[5]
    final_value = trace_covariance_over_time[-1]

    assert final_value > settled_value * 0.1, (
        f"Covariance tiep tuc co ve gan 0 sau nhieu update "
        f"(settled={settled_value:.6f}, final={final_value:.6f}) - "
        "dau hieu Kalman gain dang collapse, bug chua duoc fix dung."
    )


def test_predict_extrapolates_using_velocity():
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)

    # Cho filter "hoc" van toc +2px/frame qua vai lan update.
    for i in range(5):
        box = np.array([2.0 * i, 0, 10 + 2.0 * i, 10], dtype=np.float32)
        kf.update(box)

    predicted = kf.predict()
    predicted_cx = (predicted[0] + predicted[2]) / 2.0

    # Vi tri du bao phai TIEP TUC xu huong dich chuyen (khong dung yen tai cho),
    # tuc phai lon hon vi tri cx cua lan quan sat cuoi cung (~13.0).
    last_observed_cx = 2.0 * 4 + 5.0  # cx cua box cuoi = (8+18)/2=13
    assert predicted_cx > last_observed_cx


def test_reset_clears_state():
    kf = ConstantVelocityKalmanBox(process_noise=1e-2, measurement_noise=1e-1)
    kf.update(np.array([0, 0, 10, 10], dtype=np.float32))
    assert kf.is_initialized()

    kf.reset()
    assert not kf.is_initialized()
    assert kf.covariance is None
    assert kf._last_wh is None
