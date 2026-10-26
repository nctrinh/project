// Port of src1/antiuav_sot/utils/viz.py: overlay (motion cyan, GT green,
// best detection yellow + conf, tracks red with ID) and a video writer.
#ifndef ANTIUAV_VIZ_H
#define ANTIUAV_VIZ_H

#include <array>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <optional>
#include <string>

#include "../pipeline/sot_pipeline.h"

namespace antiuav {

cv::Mat render_frame(const cv::Mat& raw_gray, const std::optional<std::array<float, 4>>& gt_box_xywh,
                     const PipelineFrameOutput& out);

class SequenceVideoWriter {
public:
    SequenceVideoWriter(std::string path, int fps, const std::string& fourcc)
        : path_(std::move(path)), fps_(fps), fourcc_(fourcc) {}
    void write(const cv::Mat& frame_bgr);
    void close() { writer_.release(); }

private:
    std::string path_;
    int fps_;
    std::string fourcc_;
    cv::VideoWriter writer_;
};

}  // namespace antiuav

#endif  // ANTIUAV_VIZ_H
