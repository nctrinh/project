// Detector branch: letterbox (data/transforms.py), DPU YOLOv8 head decode
// (detector/dpu_export.py::decode_outputs + detector/postprocess.py), and a
// replay detector (pre-computed detections, for off-target parity tests).
#ifndef ANTIUAV_DETECTOR_H
#define ANTIUAV_DETECTOR_H

#include <array>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "../common/box_ops.h"
#include "../common/config.h"
#include "../dpu/dpu_runner.h"

namespace antiuav {

struct LetterboxResult {
    cv::Mat image;  // (target_h, target_w) uint8
    float scale = 1.0f;
    int pad_left = 0, pad_top = 0;
    int orig_height = 0, orig_width = 0;
};

// data/transforms.py::letterbox_grayscale (pad_value 114, INTER_LINEAR).
LetterboxResult letterbox_grayscale(const cv::Mat& image, int target_height, int target_width, int pad_value = 114);

// detector/model.py::DetectionResult - fixed size max_det_boxes, letterbox coords.
struct DetectionResult {
    std::vector<Box> boxes;
    std::vector<float> scores;
    std::vector<bool> valid_mask;
};

// detector/postprocess.py::pad_to_fixed_size
DetectionResult pad_to_fixed_size(const std::vector<Box>& boxes, const std::vector<float>& scores, int max_det_boxes);

// One YOLOv8 level, NHWC int8 as produced by the DPU.
struct HeadLevel {
    const int8_t* box = nullptr;  // (H, W, 4*reg_max)
    const int8_t* cls = nullptr;  // (H, W, nc)
    int box_fix_point = 0, cls_fix_point = 0;
    int height = 0, width = 0, stride = 0;
};

// decode_outputs(nchw=False) on int8 tensors: DFL softmax, sigmoid, conf
// filter, NMS, max_det. The conf filter is evaluated first on the int8
// class logit (sigmoid is monotonic), so the DFL softmax only runs on the
// cells that can pass - result identical to decoding everything.
void decode_dpu_outputs(const std::vector<HeadLevel>& levels, int reg_max, int num_classes, float conf_thres,
                        float iou_thres, int max_det, std::vector<Box>& out_boxes, std::vector<float>& out_scores);

class Detector {
public:
    virtual ~Detector() = default;
    // frame_index = position of the frame in the sequence (used by the replay detector only).
    virtual DetectionResult detect(const cv::Mat& det_input_u8, int frame_index) = 0;
};

// YOLOv8 on the DPU. Thread-safe: holds a pool of runners (dpu.num_runners);
// each detect() call borrows one, so frames can overlap on multi-core DPUs.
class YoloDpuDetector : public Detector {
public:
    YoloDpuDetector(const FpgaConstraints& c, const DeviceConfig& d, const std::string& xmodel_path);
    DetectionResult detect(const cv::Mat& det_input_u8, int frame_index) override;
    std::string describe() const;

private:
    struct Slot {
        std::unique_ptr<DpuRunner> runner;
        std::vector<int> level_box_out, level_cls_out;  // output index per level (stride 8, 16, 32)
        std::vector<int> level_stride;
    };
    void map_outputs(Slot& slot) const;

    int max_det_, reg_max_, num_classes_, input_h_, input_w_;
    float conf_thres_, iou_thres_;
    std::array<int8_t, 256> input_lut_{};  // uint8 pixel -> int8 DPU input (normalize_grayscale + fix_point)
    std::vector<Slot> slots_;
    std::vector<int> free_slots_;
    std::mutex mu_;
    std::condition_variable cv_;
};

// Detections loaded from JSON (letterbox coords, already NMS'ed):
// {"frames": [[{"box": [x1,y1,x2,y2], "score": s}, ...], ...]}
class ReplayDetector : public Detector {
public:
    ReplayDetector(const std::string& json_path, int max_det_boxes);
    DetectionResult detect(const cv::Mat& det_input_u8, int frame_index) override;

private:
    int max_det_;
    std::vector<std::vector<std::pair<Box, float>>> frames_;
};

}  // namespace antiuav

#endif  // ANTIUAV_DETECTOR_H
