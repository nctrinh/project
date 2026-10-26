// Port of src1/antiuav_sot/tracker/track_state.py (fixed N-slot table)
// and tracker/tentative_tracks.py (motion-only candidate tracks, "Tier E").
#ifndef ANTIUAV_TRACK_STATE_H
#define ANTIUAV_TRACK_STATE_H

#include <vector>

#include "../common/box_ops.h"

namespace antiuav {

constexpr int EMPTY_TRACK_ID = -1;

class TrackStateTable {
public:
    explicit TrackStateTable(int n_slots = 1) { reset(n_slots); }
    void reset(int n_slots);

    int find_free_slot() const;  // -1 when full
    int init_track(int slot, const Box& box, float confidence, int initial_motion_streak = 0);
    void update_track(int slot, const Box& box, float confidence);
    void note_detector_match(int slot) { motion_streak[slot] = 0; }
    void note_motion_match(int slot) { motion_streak[slot] += 1; }
    bool is_motion_dragged(int slot, int max_consecutive_motion_matches) const {
        return motion_streak[slot] >= max_consecutive_motion_matches;
    }
    void age_track(int slot) { ages[slot] += 1; }
    void expire_track(int slot);
    std::vector<int> check_and_expire_stale_tracks(int max_track_age);
    void clear_lost_memory(int slot);
    void age_lost_memory(int max_lost_memory_frames);
    bool is_near_recent_lost(const Box& box, double max_center_distance) const;
    std::vector<Box> active_boxes() const;

    int n_slots = 0;
    std::vector<Box> boxes;
    std::vector<int> track_ids, ages, motion_streak, lost_ages;
    std::vector<bool> active, lost_valid;
    std::vector<float> confidence;
    std::vector<Box> lost_boxes;

private:
    int next_track_id_ = 0;
};

class TentativeTrackTable {
public:
    explicit TentativeTrackTable(int max_slots = 0) : max_slots(max_slots) { reset(); }
    void reset();
    // Returns the slots whose streak >= confirm_frames.
    std::vector<int> step(const std::vector<Box>& motion_boxes, const std::vector<bool>& motion_valid_mask,
                          double iou_threshold, int max_miss, int confirm_frames);
    Box promote_and_clear(int slot);
    std::vector<Box> active_boxes() const;

    int max_slots;
    std::vector<Box> boxes;
    std::vector<int> streak, miss_count;
    std::vector<bool> active;

private:
    int find_free_slot() const;
    void clear_slot(int slot);
};

}  // namespace antiuav

#endif  // ANTIUAV_TRACK_STATE_H
