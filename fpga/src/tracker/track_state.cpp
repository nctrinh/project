#include "track_state.h"

#include <stdexcept>

namespace antiuav {

// ----------------------------------------------------------------------------- TrackStateTable
void TrackStateTable::reset(int n) {
    n_slots = n;
    boxes = make_empty_boxes(n);
    track_ids.assign(n, EMPTY_TRACK_ID);
    ages.assign(n, 0);
    active.assign(n, false);
    confidence.assign(n, 0.0f);
    motion_streak.assign(n, 0);
    lost_boxes = make_empty_boxes(n);
    lost_ages.assign(n, 0);
    lost_valid.assign(n, false);
    next_track_id_ = 0;
}

int TrackStateTable::find_free_slot() const {
    for (int i = 0; i < n_slots; ++i)
        if (!active[i]) return i;
    return -1;
}

int TrackStateTable::init_track(int slot, const Box& box, float conf, int initial_motion_streak) {
    if (active[slot]) throw std::logic_error("init_track on an occupied slot");
    int id = next_track_id_++;
    boxes[slot] = box;
    track_ids[slot] = id;
    ages[slot] = 0;
    active[slot] = true;
    confidence[slot] = conf;
    motion_streak[slot] = initial_motion_streak;
    clear_lost_memory(slot);
    return id;
}

void TrackStateTable::update_track(int slot, const Box& box, float conf) {
    boxes[slot] = box;
    ages[slot] = 0;
    confidence[slot] = conf;
}

void TrackStateTable::expire_track(int slot) {
    lost_boxes[slot] = boxes[slot];
    lost_ages[slot] = 0;
    lost_valid[slot] = true;
    boxes[slot] = make_invalid_box();
    track_ids[slot] = EMPTY_TRACK_ID;
    ages[slot] = 0;
    active[slot] = false;
    confidence[slot] = 0.0f;
    motion_streak[slot] = 0;
}

std::vector<int> TrackStateTable::check_and_expire_stale_tracks(int max_track_age) {
    std::vector<int> expired;
    for (int i = 0; i < n_slots; ++i) {
        if (active[i] && ages[i] > max_track_age) {
            expired.push_back(i);
            expire_track(i);
        }
    }
    return expired;
}

void TrackStateTable::clear_lost_memory(int slot) {
    lost_boxes[slot] = make_invalid_box();
    lost_ages[slot] = 0;
    lost_valid[slot] = false;
}

void TrackStateTable::age_lost_memory(int max_lost_memory_frames) {
    for (int i = 0; i < n_slots; ++i) {
        if (!lost_valid[i]) continue;
        lost_ages[i] += 1;
        if (lost_ages[i] > max_lost_memory_frames) clear_lost_memory(i);
    }
}

// Squared centre distance, float32 like the numpy version.
bool TrackStateTable::is_near_recent_lost(const Box& box, double max_center_distance) const {
    bool any = false;
    for (int i = 0; i < n_slots; ++i) any = any || lost_valid[i];
    if (!any) return true;
    const float cx = static_cast<float>((static_cast<double>(box[0]) + box[2]) / 2.0);
    const float cy = static_cast<float>((static_cast<double>(box[1]) + box[3]) / 2.0);
    const float max_sq = static_cast<float>(max_center_distance * max_center_distance);
    for (int i = 0; i < n_slots; ++i) {
        if (!lost_valid[i]) continue;
        float lx = (lost_boxes[i][0] + lost_boxes[i][2]) / 2.0f, ly = (lost_boxes[i][1] + lost_boxes[i][3]) / 2.0f;
        float d = (lx - cx) * (lx - cx) + (ly - cy) * (ly - cy);
        if (d <= max_sq) return true;
    }
    return false;
}

std::vector<Box> TrackStateTable::active_boxes() const {
    std::vector<Box> out;
    for (int i = 0; i < n_slots; ++i)
        if (active[i]) out.push_back(boxes[i]);
    return out;
}

// ----------------------------------------------------------------------------- TentativeTrackTable
void TentativeTrackTable::reset() {
    boxes = make_empty_boxes(max_slots);
    streak.assign(max_slots, 0);
    miss_count.assign(max_slots, 0);
    active.assign(max_slots, false);
}

int TentativeTrackTable::find_free_slot() const {
    for (int i = 0; i < max_slots; ++i)
        if (!active[i]) return i;
    return -1;
}

void TentativeTrackTable::clear_slot(int s) {
    boxes[s] = make_invalid_box();
    streak[s] = 0;
    miss_count[s] = 0;
    active[s] = false;
}

std::vector<int> TentativeTrackTable::step(const std::vector<Box>& motion_boxes, const std::vector<bool>& valid,
                                           double iou_threshold, int max_miss, int confirm_frames) {
    const int n_cand = static_cast<int>(motion_boxes.size());
    std::vector<int> matched_slot_for_candidate(n_cand, -1);
    std::vector<bool> slot_matched(max_slots, false);

    bool any_active = false, any_valid = false;
    for (bool a : active) any_active = any_active || a;
    for (bool v : valid) any_valid = any_valid || v;
    if (any_active && any_valid) {
        Matrix<float> working(max_slots, n_cand);
        for (int s = 0; s < max_slots; ++s)
            for (int c = 0; c < n_cand; ++c)
                working(s, c) = (active[s] && valid[c]) ? iou_xyxy(boxes[s], motion_boxes[c]) : 0.0f;
        const float thr = static_cast<float>(iou_threshold);
        for (int it = 0; it < max_slots * n_cand; ++it) {
            size_t best = 0;  // np.argmax: first maximum in row-major order
            for (size_t k = 1; k < working.data.size(); ++k)
                if (working.data[k] > working.data[best]) best = k;
            int bs = static_cast<int>(best / n_cand), bc = static_cast<int>(best % n_cand);
            if (working(bs, bc) < thr) break;
            boxes[bs] = motion_boxes[bc];
            streak[bs] += 1;
            miss_count[bs] = 0;
            matched_slot_for_candidate[bc] = bs;
            slot_matched[bs] = true;
            for (int c = 0; c < n_cand; ++c) working(bs, c) = 0.0f;
            for (int s = 0; s < max_slots; ++s) working(s, bc) = 0.0f;
        }
    }

    for (int s = 0; s < max_slots; ++s) {
        if (active[s] && !slot_matched[s]) {
            miss_count[s] += 1;
            if (miss_count[s] > max_miss) clear_slot(s);
        }
    }
    for (int c = 0; c < n_cand; ++c) {
        if (valid[c] && matched_slot_for_candidate[c] == -1) {
            int f = find_free_slot();
            if (f >= 0) {
                boxes[f] = motion_boxes[c];
                streak[f] = 1;
                miss_count[f] = 0;
                active[f] = true;
            }
        }
    }
    std::vector<int> ready;
    for (int s = 0; s < max_slots; ++s)
        if (active[s] && streak[s] >= confirm_frames) ready.push_back(s);
    return ready;
}

Box TentativeTrackTable::promote_and_clear(int s) {
    Box b = boxes[s];
    clear_slot(s);
    return b;
}

std::vector<Box> TentativeTrackTable::active_boxes() const {
    std::vector<Box> out;
    for (int i = 0; i < max_slots; ++i)
        if (active[i]) out.push_back(boxes[i]);
    return out;
}

}  // namespace antiuav
