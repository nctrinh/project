#include "adaptive_size.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace antiuav {

AdaptiveBoxSizeEstimator::AdaptiveBoxSizeEstimator(int window, int floor, int ceiling, double shrink_factor,
                                                   double grow_factor, int min_samples)
    : window_(window),
      floor_(floor),
      ceiling_(ceiling),
      shrink_factor_(shrink_factor),
      grow_factor_(grow_factor),
      min_samples_(min_samples) {
    if (window_ <= 0) throw std::invalid_argument("AdaptiveBoxSizeEstimator.window must be > 0");
    if (floor_ <= 0 || ceiling_ <= floor_) throw std::invalid_argument("floor must be > 0 and < ceiling");
}

void AdaptiveBoxSizeEstimator::update(const std::vector<Box>& boxes, const std::vector<bool>& valid) {
    auto push = [&](float v) {
        recent_sizes_.push_back(v);
        if (static_cast<int>(recent_sizes_.size()) > window_) recent_sizes_.pop_front();  // deque(maxlen=window)
    };
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!valid[i]) continue;
        float w = boxes[i][2] - boxes[i][0], h = boxes[i][3] - boxes[i][1];
        if (w > 0) push(w);
        if (h > 0) push(h);
    }
}

// np.percentile(..., method="linear")
static double percentile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    double pos = q / 100.0 * static_cast<double>(v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = pos - static_cast<double>(lo);
    return static_cast<double>(v[lo]) + (static_cast<double>(v[hi]) - static_cast<double>(v[lo])) * frac;
}

std::pair<int, int> AdaptiveBoxSizeEstimator::estimate() const {
    if (static_cast<int>(recent_sizes_.size()) < min_samples_) return {floor_, ceiling_};
    std::vector<float> sizes(recent_sizes_.begin(), recent_sizes_.end());
    double p20 = percentile(sizes, 20), p80 = percentile(sizes, 80);
    int dmin = static_cast<int>(std::nearbyint(shrink_factor_ * p20));  // Python round(): half to even
    int dmax = static_cast<int>(std::nearbyint(grow_factor_ * p80));
    dmin = std::max(floor_, std::min(dmin, ceiling_));
    dmax = std::max(floor_, std::min(dmax, ceiling_));
    if (dmin >= dmax) dmin = std::max(floor_, dmax - 1);
    return {dmin, dmax};
}

}  // namespace antiuav
