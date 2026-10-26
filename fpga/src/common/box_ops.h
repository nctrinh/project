// Port of src1/antiuav_sot/utils/box_ops.py.
// Box = xyxy float32, INVALID_BOX (-1) marks an empty slot - same convention
// as the Python pipeline (fixed-size arrays, no dynamic "None" boxes).
#ifndef ANTIUAV_BOX_OPS_H
#define ANTIUAV_BOX_OPS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace antiuav {

constexpr float INVALID_BOX = -1.0f;

using Box = std::array<float, 4>;  // x1, y1, x2, y2

inline Box make_invalid_box() { return {INVALID_BOX, INVALID_BOX, INVALID_BOX, INVALID_BOX}; }
inline std::vector<Box> make_empty_boxes(size_t n) { return std::vector<Box>(n, make_invalid_box()); }
inline bool is_valid_box(const Box& b) { return b[0] != INVALID_BOX; }

inline float box_area_xyxy(const Box& b) {
    float w = std::max(b[2] - b[0], 0.0f);
    float h = std::max(b[3] - b[1], 0.0f);
    return w * h;
}

inline float intersection_area(const Box& a, const Box& b) {
    float iw = std::max(std::min(a[2], b[2]) - std::max(a[0], b[0]), 0.0f);
    float ih = std::max(std::min(a[3], b[3]) - std::max(a[1], b[1]), 0.0f);
    return iw * ih;
}

// iou_matrix_xyxy element (float32 arithmetic, union <= 0 -> 0).
inline float iou_xyxy(const Box& a, const Box& b) {
    float inter = intersection_area(a, b);
    float uni = box_area_xyxy(a) + box_area_xyxy(b) - inter;
    return uni > 0.0f ? inter / std::max(uni, 1e-9f) : 0.0f;
}

// containment_matrix_xyxy element: Intersection-over-Smaller-Area.
inline float containment_xyxy(const Box& a, const Box& b) {
    float inter = intersection_area(a, b);
    float min_area = std::min(box_area_xyxy(a), box_area_xyxy(b));
    return min_area > 0.0f ? inter / std::max(min_area, 1e-9f) : 0.0f;
}

inline Box xywh_to_xyxy(const std::array<float, 4>& b) { return {b[0], b[1], b[0] + b[2], b[1] + b[3]}; }
inline std::array<float, 4> xyxy_to_xywh(const Box& b) { return {b[0], b[1], b[2] - b[0], b[3] - b[1]}; }

// Dense row-major (rows x cols) matrix - the "compare matrix" of the FSM.
template <typename T>
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<T> data;
    Matrix() = default;
    Matrix(int r, int c, T v = T(0)) : rows(r), cols(c), data(static_cast<size_t>(r) * c, v) {}
    T& operator()(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    const T& operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }
};

}  // namespace antiuav

#endif  // ANTIUAV_BOX_OPS_H
