#include "sot_pipeline.h"

#include <cmath>
#include <future>
#include <stdexcept>

#include "../common/timer.h"
#include "../fusion/merge.h"

namespace antiuav {

SotPipeline::SotPipeline(const FpgaConstraints& constraints, Detector* detector)
    : c_(constraints),
      detector_(detector),
      motion_params_(MotionParams::from(constraints)),
      frame_buffer_(constraints.motion_frame_lag),
      track_state_(constraints.n_track_slots),
      fsm_(TrackerFsmConfig::from(constraints), &track_state_) {
    if (c_.motion_size_adapt_enabled)
        size_estimator_.emplace(c_.motion_size_adapt_window, c_.motion_min_box_size, c_.motion_max_box_size,
                                c_.motion_size_adapt_shrink_factor, c_.motion_size_adapt_grow_factor,
                                c_.motion_size_adapt_min_samples);
    for (int i = 0; i < c_.n_track_slots; ++i)
        kalman_filters_.emplace_back(c_.kalman_process_noise, c_.kalman_measurement_noise);
    predict_only_streak_.assign(c_.n_track_slots, 0);
    if (c_.tentative_motion_tracks_enabled) tentative_tracks_.emplace(c_.max_tentative_motion_tracks);
}

void SotPipeline::reset() {
    frame_buffer_.reset();
    track_state_.reset(c_.n_track_slots);
    for (auto& kf : kalman_filters_) kf.reset();
    std::fill(predict_only_streak_.begin(), predict_only_streak_.end(), 0);
    if (tentative_tracks_) tentative_tracks_->reset();
    if (size_estimator_) size_estimator_->reset();
}

StageAResult SotPipeline::run_parallel_stage(const cv::Mat& raw, const cv::Mat& prev, int frame_index) const {
    if (raw.rows != c_.image_height || raw.cols != c_.image_width)
        throw std::runtime_error("raw_frame is " + std::to_string(raw.rows) + "x" + std::to_string(raw.cols) +
                                 ", does not match image.height/width in fpga_constraints.yaml (" +
                                 std::to_string(c_.image_height) + "x" + std::to_string(c_.image_width) + ")");
    auto t0 = Clock::now();
    StageAResult a;
    a.frame_index = frame_index;
    a.raw = raw;
    a.has_motion = !prev.empty();

    // Motion core on the CPU, concurrently with the DPU job below.
    std::future<MotionCoreResult> motion_future;
    Clock::time_point motion_done;
    if (a.has_motion) {
        motion_future = std::async(std::launch::async, [&]() {
            auto r = motion_core(prev, raw, motion_params_);
            motion_done = Clock::now();
            return r;
        });
    }
    a.letterbox = letterbox_grayscale(raw, c_.det_input_height, c_.det_input_width);
    a.detection = detector_->detect(a.letterbox.image, frame_index);
    auto det_done = Clock::now();
    a.det_ms = elapsed_ms(t0, det_done);
    if (a.has_motion) {
        a.motion = motion_future.get();
        a.motion_ms = elapsed_ms(t0, motion_done);
    }
    a.stage_ms = elapsed_ms(t0, Clock::now());
    return a;
}

PipelineFrameOutput SotPipeline::finish_step(StageAResult&& a) {
    auto t0 = Clock::now();
    const auto& det = a.detection;
    const auto& lb = a.letterbox;

    // Detector boxes -> original image coordinates (needed by size estimator,
    // anchoring and merge).
    std::vector<Box> det_orig = det.boxes;
    for (auto& b : det_orig) {
        if (b[0] == INVALID_BOX) continue;
        b[0] = (b[0] - static_cast<float>(lb.pad_left)) / lb.scale;
        b[1] = (b[1] - static_cast<float>(lb.pad_top)) / lb.scale;
        b[2] = (b[2] - static_cast<float>(lb.pad_left)) / lb.scale;
        b[3] = (b[3] - static_cast<float>(lb.pad_top)) / lb.scale;
    }

    // Issue 3: dynamic motion size limits
    int motion_min = c_.motion_min_box_size, motion_max = c_.motion_max_box_size;
    if (size_estimator_) {
        size_estimator_->update(det_orig, det.valid_mask);
        std::tie(motion_min, motion_max) = size_estimator_->estimate();
    }

    // Issue 6: priority boxes = active tracks + active tentative candidates
    std::vector<Box> priority = track_state_.active_boxes();
    if (tentative_tracks_) {
        auto tb = tentative_tracks_->active_boxes();
        priority.insert(priority.end(), tb.begin(), tb.end());
    }

    MotionCandidates motion = a.has_motion
        ? filter_motion_candidates(a.motion, c_.max_motion_candidates, motion_min, motion_max,
                                   c_.motion_merge_gap_ratio, c_.motion_max_aspect_ratio,
                                   c_.motion_merge_min_fill_ratio, priority, c_.motion_candidate_anchor_iou,
                                   c_.motion_strength_metric, c_.motion_min_strength)
        : empty_motion_candidates(c_.max_motion_candidates);

    CandidateSet cands = merge_candidates(det_orig, det.scores, det.valid_mask, motion.boxes, motion.valid_mask,
                                          c_.max_total_candidates);
    FsmStepResult fsm_result = fsm_.step(cands);

    // Tentative tracks from motion (Tier E)
    if (tentative_tracks_) {
        std::vector<bool> unmatched(cands.size());
        for (int i = 0; i < cands.size(); ++i)
            unmatched[i] = cands.valid_mask[i] && cands.source[i] == SOURCE_MOTION &&
                           fsm_result.tier_for_candidate[i] == TIER_UNMATCHED;
        // Issue 2: drop motion candidates that overlap an ACTIVE track
        std::vector<Box> active_boxes = track_state_.active_boxes();
        bool any_unmatched = false;
        for (bool u : unmatched) any_unmatched = any_unmatched || u;
        if (any_unmatched && !active_boxes.empty()) {
            const float thr = static_cast<float>(c_.tentative_active_overlap_max_iou);
            for (int i = 0; i < cands.size(); ++i) {
                float best = 0.0f;
                for (const Box& t : active_boxes)
                    best = std::max(best, std::max(iou_xyxy(cands.boxes[i], t), containment_xyxy(cands.boxes[i], t)));
                if (best >= thr) unmatched[i] = false;
            }
        }
        std::vector<int> ready = tentative_tracks_->step(cands.boxes, unmatched, c_.tentative_iou_threshold,
                                                         c_.tentative_max_miss, c_.motion_track_confirm_frames);
        const double reinit_max_distance =
            c_.reinit_max_center_distance_ratio * std::hypot(static_cast<double>(c_.image_width), c_.image_height);
        for (int slot : ready) {
            int free_track_slot = track_state_.find_free_slot();
            if (free_track_slot < 0) continue;
            const Box pending = tentative_tracks_->boxes[slot];
            if (c_.reinit_near_lost_enabled && !track_state_.is_near_recent_lost(pending, reinit_max_distance)) continue;
            Box promoted = tentative_tracks_->promote_and_clear(slot);
            track_state_.init_track(free_track_slot, promoted, 0.0f, c_.max_consecutive_motion_matches);
            fsm_result.slot_matched_source[free_track_slot] = SOURCE_MOTION;
        }
    }

    PipelineFrameOutput out;
    out.track_boxes_smoothed = c_.kalman_enabled ? apply_kalman(fsm_result) : track_state_.boxes;
    out.frame_index = a.frame_index;
    out.track_boxes = track_state_.boxes;
    out.track_active = track_state_.active;
    out.track_ids = track_state_.track_ids;
    out.track_ages = track_state_.ages;
    out.predict_only_streak = predict_only_streak_;
    out.fsm_result = std::move(fsm_result);
    out.detection_result = det;
    out.letterbox_scale = lb.scale;
    out.letterbox_pad_left = lb.pad_left;
    out.letterbox_pad_top = lb.pad_top;
    out.motion = std::move(motion);
    out.det_ms = a.det_ms;
    out.motion_ms = a.motion_ms;
    out.stage_a_ms = a.stage_ms;
    out.stage_b_ms = elapsed_ms(t0, Clock::now());
    return out;
}

PipelineFrameOutput SotPipeline::step(const cv::Mat& raw, int frame_index) {
    cv::Mat prev = frame_buffer_.get_previous();
    StageAResult a = run_parallel_stage(raw, prev, frame_index);
    frame_buffer_.update(raw);
    return finish_step(std::move(a));
}

std::vector<Box> SotPipeline::apply_kalman(const FsmStepResult& fsm_result) {
    std::vector<Box> out = track_state_.boxes;
    for (int s = 0; s < c_.n_track_slots; ++s) {
        auto& kf = kalman_filters_[s];
        if (!track_state_.active[s]) {
            kf.reset();
            predict_only_streak_[s] = 0;
            continue;
        }
        if (track_state_.ages[s] == 0) {
            double noise = fsm_result.slot_matched_source[s] == SOURCE_MOTION ? c_.kalman_motion_measurement_noise : -1.0;
            out[s] = kf.update(track_state_.boxes[s], noise);
            predict_only_streak_[s] = 0;
        } else if (kf.is_initialized() && predict_only_streak_[s] < c_.kalman_max_predict_only_frames) {
            out[s] = kf.predict();
            predict_only_streak_[s] += 1;
        } else {
            out[s] = make_invalid_box();
        }
    }
    return out;
}

}  // namespace antiuav
