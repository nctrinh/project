// Port of src1/antiuav_sot/fusion/merge.py: concatenate the fixed-size
// detector and motion candidate lists (detector first, motion score = 0).
#ifndef ANTIUAV_MERGE_H
#define ANTIUAV_MERGE_H

#include <stdexcept>
#include <vector>

#include "../common/box_ops.h"

namespace antiuav {

constexpr int SOURCE_DETECTOR = 0;
constexpr int SOURCE_MOTION = 1;

struct CandidateSet {
    std::vector<Box> boxes;
    std::vector<float> scores;
    std::vector<int> source;
    std::vector<bool> valid_mask;
    int size() const { return static_cast<int>(boxes.size()); }
};

inline CandidateSet merge_candidates(const std::vector<Box>& det_boxes, const std::vector<float>& det_scores,
                                     const std::vector<bool>& det_valid, const std::vector<Box>& motion_boxes,
                                     const std::vector<bool>& motion_valid, int max_total_candidates) {
    if (static_cast<int>(det_boxes.size() + motion_boxes.size()) != max_total_candidates)
        throw std::invalid_argument("n_det + n_mot must equal fusion.max_total_candidates");
    CandidateSet c;
    c.boxes = det_boxes;
    c.boxes.insert(c.boxes.end(), motion_boxes.begin(), motion_boxes.end());
    c.scores = det_scores;
    c.scores.insert(c.scores.end(), motion_boxes.size(), 0.0f);
    c.source.assign(det_boxes.size(), SOURCE_DETECTOR);
    c.source.insert(c.source.end(), motion_boxes.size(), SOURCE_MOTION);
    c.valid_mask = det_valid;
    c.valid_mask.insert(c.valid_mask.end(), motion_valid.begin(), motion_valid.end());
    return c;
}

}  // namespace antiuav

#endif  // ANTIUAV_MERGE_H
