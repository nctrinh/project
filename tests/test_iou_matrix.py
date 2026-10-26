import numpy as np

from antiuav_sot.tracker.iou_matrix import compute_masked_iou_matrix
from antiuav_sot.utils.box_ops import (
    box_area_xyxy,
    iou_matrix_xyxy,
    iou_vector_xyxy,
    is_valid_box,
    make_empty_boxes,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_xywh_to_xyxy_and_back():
    xywh = np.array([[10, 20, 30, 40]], dtype=np.float32)
    xyxy = xywh_to_xyxy(xywh)
    np.testing.assert_allclose(xyxy, [[10, 20, 40, 60]])

    back = xyxy_to_xywh(xyxy)
    np.testing.assert_allclose(back, xywh)


def test_box_area_xyxy():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 5, 5]], dtype=np.float32)
    areas = box_area_xyxy(boxes)
    np.testing.assert_allclose(areas, [100.0, 0.0])


def test_iou_matrix_identical_boxes():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    iou = iou_matrix_xyxy(boxes, boxes)
    np.testing.assert_allclose(iou, [[1.0]])


def test_iou_matrix_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    iou = iou_matrix_xyxy(a, b)
    np.testing.assert_allclose(iou, [[0.0]])


def test_iou_matrix_partial_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)   # area 100
    b = np.array([[5, 5, 15, 15]], dtype=np.float32)    # area 100, overlap [5,5,10,10] = 25
    iou = iou_matrix_xyxy(a, b)
    expected = 25.0 / (100.0 + 100.0 - 25.0)
    np.testing.assert_allclose(iou, [[expected]], rtol=1e-5)


def test_iou_matrix_shape_and_vectorization():
    """N=3 track, M=5 box -> ma tran (3,5), moi phan tu doc lap."""
    n, m = 3, 5
    rng = np.random.default_rng(0)
    boxes_a = rng.uniform(0, 100, size=(n, 4)).astype(np.float32)
    boxes_a[:, 2:] += boxes_a[:, :2]  # dam bao x2>x1, y2>y1
    boxes_b = rng.uniform(0, 100, size=(m, 4)).astype(np.float32)
    boxes_b[:, 2:] += boxes_b[:, :2]

    iou = iou_matrix_xyxy(boxes_a, boxes_b)
    assert iou.shape == (n, m)
    assert np.all(iou >= 0.0) and np.all(iou <= 1.0)

    # Doi chieu tung phan tu voi cach tinh doc lap (khong vector hoa) de
    # dam bao vector hoa khong lam sai ket qua so voi tinh tung cap rieng le.
    for i in range(n):
        for j in range(m):
            single = iou_matrix_xyxy(boxes_a[i:i + 1], boxes_b[j:j + 1])[0, 0]
            np.testing.assert_allclose(iou[i, j], single, rtol=1e-5)


def test_iou_vector_single_track_sot_case():
    """Truong hop N=1 (SOT): vector 1xM."""
    track_box = np.array([0, 0, 10, 10], dtype=np.float32)
    candidates = np.array(
        [[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 15, 15]], dtype=np.float32
    )
    iou = iou_vector_xyxy(track_box, candidates)
    assert iou.shape == (3,)
    np.testing.assert_allclose(iou[0], 1.0)
    np.testing.assert_allclose(iou[1], 0.0)


def test_make_empty_boxes_and_is_valid():
    empty = make_empty_boxes(5)
    assert empty.shape == (5, 4)
    assert not np.any(is_valid_box(empty))

    empty[2] = [1, 2, 3, 4]
    valid_mask = is_valid_box(empty)
    expected = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(valid_mask, expected)


def test_masked_iou_matrix_zeroes_inactive_track_rows():
    track_boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    track_active = np.array([True, False])  # slot 1 khong active
    candidate_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    candidate_valid = np.array([True])

    iou = compute_masked_iou_matrix(track_boxes, track_active, candidate_boxes, candidate_valid)

    assert iou[0, 0] == 1.0   # track active, khop hoan toan
    assert iou[1, 0] == 0.0   # track khong active -> phai bi zero-out du IoU hinh hoc = 1.0


def test_masked_iou_matrix_zeroes_invalid_candidate_cols():
    track_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    track_active = np.array([True])
    candidate_boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    candidate_valid = np.array([True, False])  # candidate 1 la padding

    iou = compute_masked_iou_matrix(track_boxes, track_active, candidate_boxes, candidate_valid)

    assert iou[0, 0] == 1.0
    assert iou[0, 1] == 0.0  # candidate khong valid -> phai bi zero-out
