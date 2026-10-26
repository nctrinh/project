// Port of src1/antiuav_sot/utils/metrics.py (OPE metrics for Anti-UAV):
// Precision@20px, Normalized Precision AUC, Success AUC, SA / mSA, FP / FN
// rates, each "all frames" and "present only".
#ifndef ANTIUAV_METRICS_H
#define ANTIUAV_METRICS_H

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "../common/box_ops.h"
#include "../common/json.h"

namespace antiuav {

struct SequenceEvalResult {
    std::string seq_name;
    std::vector<float> center_distances;  // inf when not comparable
    std::vector<float> norm_distances;
    std::vector<float> ious;
    std::vector<bool> gt_exist, pred_active;
    int n_frames = 0;
};

SequenceEvalResult evaluate_sequence(const std::string& seq_name, const std::vector<std::optional<Box>>& predicted,
                                     const std::vector<std::array<float, 4>>& gt_boxes_xywh,
                                     const std::vector<bool>& gt_exist);

// Same keys as summarize() in Python (+ curve values, since the board has no matplotlib).
Json summarize(const std::vector<SequenceEvalResult>& results, bool with_curves = false);

}  // namespace antiuav

#endif  // ANTIUAV_METRICS_H
