// Port of src1/antiuav_sot/tracker/simple_tracker_fsm.py - same tier order:
// Tier 1 (middle score) -> Confidence Override -> Tier 2 (motion link) ->
// Tier 3 (high score init, reinit-near-lost gate) -> Tier 4 (low score).
#ifndef ANTIUAV_SIMPLE_TRACKER_FSM_H
#define ANTIUAV_SIMPLE_TRACKER_FSM_H

#include <vector>

#include "../common/config.h"
#include "../fusion/merge.h"
#include "track_state.h"

namespace antiuav {

constexpr int TIER_UNMATCHED = 0;
constexpr int TIER_MIDDLE_SCORE = 1;
constexpr int TIER_MOTION_LINK = 2;
constexpr int TIER_HIGH_SCORE_INIT = 3;
constexpr int TIER_LOW_SCORE = 4;
constexpr int TIER_CONFIDENCE_OVERRIDE = 5;

struct TrackerFsmConfig {
    double tier1_iou_threshold, tier1_conf_min, tier2_iou_threshold, tier3_conf_min, tier4_iou_threshold;
    int max_track_age;
    int max_consecutive_motion_matches;
    double override_conf_min, override_iou_threshold;
    bool override_requires_overlap, override_motion_dragged_only;
    double override_max_center_distance_ratio;
    bool reinit_near_lost_enabled;
    double reinit_max_center_distance_ratio;
    int reinit_lost_memory_frames;
    double reinit_trust_conf_min;
    double image_diagonal;
    static TrackerFsmConfig from(const FpgaConstraints& c);
};

struct FsmStepResult {
    std::vector<int> matched_slot_for_candidate;
    std::vector<int> tier_for_candidate;
    std::vector<int> slot_matched_source;  // SOURCE_* or -1
    std::vector<int> newly_initiated_track_ids;
    std::vector<int> expired_track_ids;    // slot indices (same as Python)
};

class SimpleTrackerFSM {
public:
    SimpleTrackerFSM(const TrackerFsmConfig& config, TrackStateTable* track_state)
        : config_(config), ts_(track_state) {}
    FsmStepResult step(const CandidateSet& candidates);

private:
    template <typename T>
    void match_tier(Matrix<T> working, double iou_threshold, const std::vector<bool>& eligible, int tier_id,
                    const CandidateSet& candidates, FsmStepResult& r, std::vector<bool>& slot_matched);
    Matrix<double> build_override_affinity(const Matrix<float>& iou, const std::vector<bool>& motion_dragged_row,
                                           const CandidateSet& candidates, const std::vector<bool>& eligible) const;

    TrackerFsmConfig config_;
    TrackStateTable* ts_;
};

}  // namespace antiuav

#endif  // ANTIUAV_SIMPLE_TRACKER_FSM_H
