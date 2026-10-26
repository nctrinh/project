// Port of src1/antiuav_sot/motion/gradient_motion_core.py and
// motion/gradient_candidates.py, split in two parts so the expensive part
// can run in parallel with the DPU:
//
//   motion_core()      STATELESS - depends only on (frame t, frame t-lag):
//                      pyramid + IRLS global motion, warp, positive residual,
//                      blur/threshold/morphology, connected components.
//                      This is the block to replace with a PL (HLS) kernel
//                      later, if a custom platform is built.
//   filter_motion_candidates()
//                      needs per-frame state (dynamic min/max box size from the
//                      detector, priority boxes from the tracker): aspect filter,
//                      cluster merge, size filter, strength, anchoring, top-K.
#ifndef ANTIUAV_GRADIENT_MOTION_H
#define ANTIUAV_GRADIENT_MOTION_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "../common/box_ops.h"
#include "../common/config.h"

namespace antiuav {

struct MotionCoreResult {
    std::vector<std::array<double, 4>> boxes_xyxy;  // raw CCL boxes (area >= min_area), CCL label order
    cv::Mat residual;                                // (H, W) float32, blurred positive residual
    std::array<double, 6> params{};                  // global motion (affine: 6, translation: first 2)
};

struct MotionParams {
    std::string model = "affine";
    int irls_iters = 3;
    double huber_delta = 10.0;
    int pyramid_levels = 2;
    double thresh_factor = 2.5;
    int min_area = 20;
    static MotionParams from(const FpgaConstraints& c);
};

// motion_candidates_gradient(f1 = frame_t_minus_lag, f2 = frame_t), both uint8.
MotionCoreResult motion_core(const cv::Mat& frame_t_minus_lag, const cv::Mat& frame_t, const MotionParams& p);

struct MotionCandidates {
    std::vector<Box> boxes;        // (max_motion_candidates) xyxy, INVALID_BOX padded
    std::vector<bool> valid_mask;
    std::vector<float> strengths;
};

// Second half of gradient_motion_candidates().
MotionCandidates filter_motion_candidates(const MotionCoreResult& core, int max_motion_candidates, int min_box_size,
                                          int max_box_size, double merge_gap_ratio, double max_aspect_ratio,
                                          double merge_min_fill_ratio, const std::vector<Box>& priority_boxes,
                                          double priority_iou_threshold, const std::string& strength_metric,
                                          double min_strength);

MotionCandidates empty_motion_candidates(int max_motion_candidates);

}  // namespace antiuav

#endif  // ANTIUAV_GRADIENT_MOTION_H
