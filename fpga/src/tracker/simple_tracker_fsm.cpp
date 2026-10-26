#include "simple_tracker_fsm.h"

#include <algorithm>
#include <cmath>

namespace antiuav {

// Override tier constants (see _build_override_affinity in Python).
static constexpr double OVERRIDE_AFFINITY_EPS = 1e-6;
static constexpr float OVERRIDE_AFFINITY_BASE = 1.0f;
static constexpr float OVERRIDE_IOU_TIE_BREAK = 1e-3f;
static constexpr float OVERRIDE_DISTANCE_TIE_BREAK = 1e-6f;

TrackerFsmConfig TrackerFsmConfig::from(const FpgaConstraints& c) {
    TrackerFsmConfig f;
    f.tier1_iou_threshold = c.tier1_iou_threshold;
    f.tier1_conf_min = c.tier1_conf_min;
    f.tier2_iou_threshold = c.tier2_iou_threshold;
    f.tier3_conf_min = c.tier3_conf_min;
    f.tier4_iou_threshold = c.tier4_iou_threshold;
    f.max_track_age = c.max_track_age;
    f.max_consecutive_motion_matches = c.max_consecutive_motion_matches;
    f.override_conf_min = c.override_conf_min;
    f.override_iou_threshold = c.override_iou_threshold;
    f.override_requires_overlap = c.override_requires_overlap;
    f.override_motion_dragged_only = c.override_motion_dragged_only;
    f.override_max_center_distance_ratio = c.override_max_center_distance_ratio;
    f.reinit_near_lost_enabled = c.reinit_near_lost_enabled;
    f.reinit_max_center_distance_ratio = c.reinit_max_center_distance_ratio;
    f.reinit_lost_memory_frames = c.reinit_lost_memory_frames;
    f.reinit_trust_conf_min = c.reinit_trust_conf_min;
    f.image_diagonal = std::hypot(static_cast<double>(c.image_width), static_cast<double>(c.image_height));
    return f;
}

// Motion candidate matched: keep the OLD track size, take only the centre.
static Box apply_motion_size_preservation(const Box& cand, const Box& track) {
    float cx = (cand[0] + cand[2]) / 2.0f, cy = (cand[1] + cand[3]) / 2.0f;
    float w = track[2] - track[0], h = track[3] - track[1];
    return {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f};
}

FsmStepResult SimpleTrackerFSM::step(const CandidateSet& cands) {
    const int n_slots = ts_->n_slots;
    const int n_cand = cands.size();
    ts_->age_lost_memory(config_.reinit_lost_memory_frames);

    FsmStepResult r;
    r.matched_slot_for_candidate.assign(n_cand, -1);
    r.tier_for_candidate.assign(n_cand, TIER_UNMATCHED);
    r.slot_matched_source.assign(n_slots, -1);
    std::vector<bool> slot_matched(n_slots, false);

    // Masked IoU / containment matrices, computed ONCE from the pre-step state.
    Matrix<float> iou(n_slots, n_cand), cont(n_slots, n_cand);
    for (int s = 0; s < n_slots; ++s)
        for (int c = 0; c < n_cand; ++c)
            if (ts_->active[s] && cands.valid_mask[c]) {
                iou(s, c) = iou_xyxy(ts_->boxes[s], cands.boxes[c]);
                cont(s, c) = containment_xyxy(ts_->boxes[s], cands.boxes[c]);
            }

    std::vector<bool> motion_dragged_row(n_slots);
    for (int s = 0; s < n_slots; ++s)
        motion_dragged_row[s] = ts_->active[s] && ts_->is_motion_dragged(s, config_.max_consecutive_motion_matches);

    // Motion columns: max(IoU, containment), locked (motion-dragged) rows zeroed.
    Matrix<float> effective = iou;
    for (int s = 0; s < n_slots; ++s)
        for (int c = 0; c < n_cand; ++c)
            if (cands.source[c] == SOURCE_MOTION)
                effective(s, c) = motion_dragged_row[s] ? 0.0f : std::max(iou(s, c), cont(s, c));

    auto eligible = [&](auto pred) {
        std::vector<bool> m(n_cand);
        for (int c = 0; c < n_cand; ++c) m[c] = cands.valid_mask[c] && pred(c);
        return m;
    };
    const float tier1_conf = static_cast<float>(config_.tier1_conf_min);
    const float override_conf = static_cast<float>(config_.override_conf_min);
    const float tier3_conf = static_cast<float>(config_.tier3_conf_min);

    // ---- Tier 1: middle score linking ----
    auto t1 = eligible([&](int c) { return cands.source[c] == SOURCE_DETECTOR && cands.scores[c] >= tier1_conf; });
    match_tier(iou, config_.tier1_iou_threshold, t1, TIER_MIDDLE_SCORE, cands, r, slot_matched);

    // ---- Confidence Override ----
    auto to = eligible([&](int c) {
        return cands.source[c] == SOURCE_DETECTOR && cands.scores[c] >= override_conf &&
               r.tier_for_candidate[c] == TIER_UNMATCHED;
    });
    match_tier(build_override_affinity(iou, motion_dragged_row, cands, to), OVERRIDE_AFFINITY_EPS, to,
               TIER_CONFIDENCE_OVERRIDE, cands, r, slot_matched);

    // ---- Tier 2: motion boxes linking ----
    auto t2 = eligible([&](int c) { return cands.source[c] == SOURCE_MOTION && r.tier_for_candidate[c] == TIER_UNMATCHED; });
    match_tier(effective, config_.tier2_iou_threshold, t2, TIER_MOTION_LINK, cands, r, slot_matched);

    // ---- Tier 3: high score initiation (+ reinit-near-lost gate) ----
    auto t3 = eligible([&](int c) {
        return cands.source[c] == SOURCE_DETECTOR && cands.scores[c] >= tier3_conf &&
               r.tier_for_candidate[c] == TIER_UNMATCHED;
    });
    const double reinit_max_distance = config_.reinit_max_center_distance_ratio * config_.image_diagonal;
    for (int c = 0; c < n_cand; ++c) {
        if (!t3[c]) continue;
        if (config_.reinit_near_lost_enabled && static_cast<double>(cands.scores[c]) < config_.reinit_trust_conf_min &&
            !ts_->is_near_recent_lost(cands.boxes[c], reinit_max_distance))
            continue;
        int free_slot = ts_->find_free_slot();
        if (free_slot < 0) break;  // table full
        int id = ts_->init_track(free_slot, cands.boxes[c], cands.scores[c], 0);
        r.matched_slot_for_candidate[c] = free_slot;
        r.tier_for_candidate[c] = TIER_HIGH_SCORE_INIT;
        r.slot_matched_source[free_slot] = SOURCE_DETECTOR;
        slot_matched[free_slot] = true;
        r.newly_initiated_track_ids.push_back(id);
    }

    // ---- Tier 4: low score linking (detector or motion) ----
    auto t4 = eligible([&](int c) { return r.tier_for_candidate[c] == TIER_UNMATCHED; });
    match_tier(effective, config_.tier4_iou_threshold, t4, TIER_LOW_SCORE, cands, r, slot_matched);

    for (int s = 0; s < n_slots; ++s)
        if (ts_->active[s] && !slot_matched[s]) ts_->age_track(s);

    r.expired_track_ids = ts_->check_and_expire_stale_tracks(config_.max_track_age);
    for (int s : r.expired_track_ids) r.slot_matched_source[s] = -1;
    return r;
}

Matrix<double> SimpleTrackerFSM::build_override_affinity(const Matrix<float>& iou,
                                                         const std::vector<bool>& motion_dragged_row,
                                                         const CandidateSet& cands,
                                                         const std::vector<bool>& eligible) const {
    const int n_slots = iou.rows, n_cand = iou.cols;
    Matrix<double> affinity(n_slots, n_cand, 0.0);
    if (n_slots == 0 || n_cand == 0) return affinity;

    const double max_dist = config_.override_max_center_distance_ratio * config_.image_diagonal;
    const float max_dist_sq = static_cast<float>(max_dist * max_dist);
    const float max_dist_f = static_cast<float>(std::max(max_dist, 1e-9));
    const float override_iou = static_cast<float>(config_.override_iou_threshold);

    for (int s = 0; s < n_slots; ++s) {
        bool row_allowed = ts_->active[s] && (!config_.override_motion_dragged_only || motion_dragged_row[s]);
        if (!row_allowed) continue;
        const Box& tb = ts_->boxes[s];
        float tcx = (tb[0] + tb[2]) / 2.0f, tcy = (tb[1] + tb[3]) / 2.0f;
        for (int c = 0; c < n_cand; ++c) {
            if (!eligible[c]) continue;
            const Box& cb = cands.boxes[c];
            float dx = tcx - (cb[0] + cb[2]) / 2.0f, dy = tcy - (cb[1] + cb[3]) / 2.0f;
            float dist_sq = dx * dx + dy * dy;
            bool overlap_ok = iou(s, c) > 0.0f && iou(s, c) >= override_iou;
            bool within_gate = dist_sq <= max_dist_sq;
            bool valid = config_.override_requires_overlap ? overlap_ok : (overlap_ok || within_gate);
            if (!valid) continue;
            float d_norm = std::min(std::max(std::sqrt(dist_sq) / max_dist_f, 0.0f), 1.0f);
            // float32 arithmetic, like numpy with weak Python scalars
            float a = OVERRIDE_AFFINITY_BASE + cands.scores[c];
            a = a + OVERRIDE_IOU_TIE_BREAK * iou(s, c);
            a = a + OVERRIDE_DISTANCE_TIE_BREAK * (1.0f - d_norm);
            affinity(s, c) = a;
        }
    }
    return affinity;
}

template <typename T>
void SimpleTrackerFSM::match_tier(Matrix<T> working, double iou_threshold, const std::vector<bool>& eligible,
                                  int tier_id, const CandidateSet& cands, FsmStepResult& r,
                                  std::vector<bool>& slot_matched) {
    const int n_slots = working.rows, n_cand = working.cols;
    if (n_slots == 0 || n_cand == 0) return;
    for (int s = 0; s < n_slots; ++s)
        for (int c = 0; c < n_cand; ++c)
            if (slot_matched[s] || !eligible[c]) working(s, c) = T(0);

    const T thr = static_cast<T>(iou_threshold);
    for (int it = 0; it < n_slots * n_cand; ++it) {
        size_t best = 0;  // np.argmax: first maximum, row-major
        for (size_t k = 1; k < working.data.size(); ++k)
            if (working.data[k] > working.data[best]) best = k;
        const int bs = static_cast<int>(best / n_cand), bc = static_cast<int>(best % n_cand);
        if (working(bs, bc) < thr) break;

        const int src = cands.source[bc];
        Box commit = cands.boxes[bc];
        if (src == SOURCE_MOTION) commit = apply_motion_size_preservation(commit, ts_->boxes[bs]);
        ts_->update_track(bs, commit, cands.scores[bc]);
        if (src == SOURCE_MOTION) ts_->note_motion_match(bs);
        else ts_->note_detector_match(bs);
        r.matched_slot_for_candidate[bc] = bs;
        r.tier_for_candidate[bc] = tier_id;
        r.slot_matched_source[bs] = src;
        slot_matched[bs] = true;

        for (int c = 0; c < n_cand; ++c) working(bs, c) = T(0);
        for (int s = 0; s < n_slots; ++s) working(s, bc) = T(0);
    }
}

template void SimpleTrackerFSM::match_tier<float>(Matrix<float>, double, const std::vector<bool>&, int,
                                                  const CandidateSet&, FsmStepResult&, std::vector<bool>&);
template void SimpleTrackerFSM::match_tier<double>(Matrix<double>, double, const std::vector<bool>&, int,
                                                   const CandidateSet&, FsmStepResult&, std::vector<bool>&);

}  // namespace antiuav
