// Port of src1/antiuav_sot/pipeline/sot_pipeline.py, split into:
//
//   run_parallel_stage()  STATELESS, thread-safe. Letterbox + DPU detector
//                         (decode/NMS included) running concurrently with the
//                         motion core (frame t vs frame t-lag). Several frames
//                         can be in this stage at the same time.
//   finish_step()         STATEFUL, strictly in frame order. Everything after
//                         that in SotPipeline.step(): rescale, adaptive size,
//                         priority anchoring + motion top-K, merge, FSM,
//                         tentative tracks, Kalman.
//
// step() = both stages back to back (sequential reference behaviour).
#ifndef ANTIUAV_SOT_PIPELINE_H
#define ANTIUAV_SOT_PIPELINE_H

#include <deque>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>

#include "../common/config.h"
#include "../detector/detector.h"
#include "../motion/adaptive_size.h"
#include "../motion/gradient_motion.h"
#include "../tracker/kalman.h"
#include "../tracker/simple_tracker_fsm.h"
#include "../tracker/track_state.h"

namespace antiuav {

// motion/frame_buffer.py: keeps the last `lag` frames, previous = frame t-lag.
class FrameBuffer {
public:
    explicit FrameBuffer(int lag) : lag_(lag) {}
    bool has_previous() const { return static_cast<int>(frames_.size()) == lag_; }
    cv::Mat get_previous() const { return has_previous() ? frames_.front() : cv::Mat(); }
    void update(const cv::Mat& f) {
        frames_.push_back(f);  // frames are never modified in place, sharing the buffer is safe
        if (static_cast<int>(frames_.size()) > lag_) frames_.pop_front();
    }
    void reset() { frames_.clear(); }

private:
    int lag_;
    std::deque<cv::Mat> frames_;
};

struct StageAResult {
    int frame_index = 0;
    cv::Mat raw;
    LetterboxResult letterbox;
    DetectionResult detection;
    bool has_motion = false;
    MotionCoreResult motion;
    double det_ms = 0, motion_ms = 0, stage_ms = 0;
};

struct PipelineFrameOutput {
    int frame_index = 0;
    std::vector<Box> track_boxes;           // after FSM (before Kalman)
    std::vector<Box> track_boxes_smoothed;  // after Kalman (== track_boxes if disabled)
    std::vector<bool> track_active;
    std::vector<int> track_ids, track_ages, predict_only_streak;
    FsmStepResult fsm_result;
    DetectionResult detection_result;       // letterbox coordinates
    float letterbox_scale = 1.0f;
    int letterbox_pad_left = 0, letterbox_pad_top = 0;
    MotionCandidates motion;                // original image coordinates
    double det_ms = 0, motion_ms = 0, stage_a_ms = 0, stage_b_ms = 0;
};

class SotPipeline {
public:
    SotPipeline(const FpgaConstraints& constraints, Detector* detector);

    void reset();
    StageAResult run_parallel_stage(const cv::Mat& raw_frame, const cv::Mat& prev_frame, int frame_index) const;
    PipelineFrameOutput finish_step(StageAResult&& a);
    PipelineFrameOutput step(const cv::Mat& raw_frame, int frame_index);  // sequential, owns a FrameBuffer

    const FpgaConstraints& constraints() const { return c_; }

private:
    std::vector<Box> apply_kalman(const FsmStepResult& fsm_result);

    FpgaConstraints c_;
    Detector* detector_;
    MotionParams motion_params_;
    FrameBuffer frame_buffer_;
    TrackStateTable track_state_;
    SimpleTrackerFSM fsm_;
    std::optional<AdaptiveBoxSizeEstimator> size_estimator_;
    std::vector<ConstantVelocityKalmanBox> kalman_filters_;
    std::vector<int> predict_only_streak_;
    std::optional<TentativeTrackTable> tentative_tracks_;
};

}  // namespace antiuav

#endif  // ANTIUAV_SOT_PIPELINE_H
