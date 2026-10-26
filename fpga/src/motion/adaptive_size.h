// Port of src1/antiuav_sot/motion/adaptive_size.py (Issue 3: dynamic
// [min_box_size, max_box_size] for the motion branch from recent detector sizes).
#ifndef ANTIUAV_ADAPTIVE_SIZE_H
#define ANTIUAV_ADAPTIVE_SIZE_H

#include <deque>
#include <utility>
#include <vector>

#include "../common/box_ops.h"

namespace antiuav {

class AdaptiveBoxSizeEstimator {
public:
    AdaptiveBoxSizeEstimator(int window, int floor, int ceiling, double shrink_factor, double grow_factor,
                             int min_samples);

    void update(const std::vector<Box>& detector_boxes_xyxy, const std::vector<bool>& detector_valid_mask);
    std::pair<int, int> estimate() const;
    void reset() { recent_sizes_.clear(); }

private:
    int window_, floor_, ceiling_;
    double shrink_factor_, grow_factor_;
    int min_samples_;
    std::deque<float> recent_sizes_;
};

}  // namespace antiuav

#endif  // ANTIUAV_ADAPTIVE_SIZE_H
