#include "viz.h"

#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace antiuav {

static const cv::Scalar COLOR_GT(0, 255, 0);
static const cv::Scalar COLOR_TRACK(0, 0, 255);
static const cv::Scalar COLOR_MOTION(255, 255, 0);
static const cv::Scalar COLOR_DETECTOR_BEST(0, 255, 255);

static void draw_box(cv::Mat& img, const Box& b, const cv::Scalar& color, const std::string& label = "",
                     int thickness = 2) {
    int x1 = static_cast<int>(std::nearbyint(b[0])), y1 = static_cast<int>(std::nearbyint(b[1]));
    int x2 = static_cast<int>(std::nearbyint(b[2])), y2 = static_cast<int>(std::nearbyint(b[3]));
    cv::rectangle(img, {x1, y1}, {x2, y2}, color, thickness);
    if (!label.empty())
        cv::putText(img, label, {x1, std::max(0, y1 - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

cv::Mat render_frame(const cv::Mat& raw_gray, const std::optional<std::array<float, 4>>& gt_box_xywh,
                     const PipelineFrameOutput& out) {
    cv::Mat canvas;
    cv::cvtColor(raw_gray, canvas, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < out.motion.boxes.size(); ++i)
        if (out.motion.valid_mask[i]) draw_box(canvas, out.motion.boxes[i], COLOR_MOTION, "", 1);

    if (gt_box_xywh) {
        const auto& g = *gt_box_xywh;
        if (g[0] > 0 || g[1] > 0 || g[2] > 0 || g[3] > 0) draw_box(canvas, xywh_to_xyxy(g), COLOR_GT, "GT");
    }

    const auto& det = out.detection_result;
    int best = -1;
    for (size_t i = 0; i < det.scores.size(); ++i)
        if (det.valid_mask[i] && (best < 0 || det.scores[i] > det.scores[best])) best = static_cast<int>(i);
    if (best >= 0) {
        Box b = det.boxes[best];
        b[0] = (b[0] - out.letterbox_pad_left) / out.letterbox_scale;
        b[2] = (b[2] - out.letterbox_pad_left) / out.letterbox_scale;
        b[1] = (b[1] - out.letterbox_pad_top) / out.letterbox_scale;
        b[3] = (b[3] - out.letterbox_pad_top) / out.letterbox_scale;
        draw_box(canvas, b, COLOR_DETECTOR_BEST, "DET");
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", det.scores[best]);
        int x2 = static_cast<int>(std::nearbyint(b[2])), y1 = static_cast<int>(std::nearbyint(b[1]));
        cv::putText(canvas, buf, {x2 + 5, std::max(0, y1) + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DETECTOR_BEST, 1,
                    cv::LINE_AA);
    }

    for (size_t s = 0; s < out.track_boxes_smoothed.size(); ++s) {
        if (!out.track_active[s]) continue;
        draw_box(canvas, out.track_boxes_smoothed[s], COLOR_TRACK, "ID:" + std::to_string(out.track_ids[s]));
    }
    return canvas;
}

void SequenceVideoWriter::write(const cv::Mat& frame_bgr) {
    if (!writer_.isOpened()) {
        int fourcc = cv::VideoWriter::fourcc(fourcc_[0], fourcc_[1], fourcc_[2], fourcc_[3]);
        writer_.open(path_, fourcc, fps_, frame_bgr.size());
        if (!writer_.isOpened())
            throw std::runtime_error("Cannot open video writer " + path_ + " (fourcc " + fourcc_ +
                                     ") - set output.video_fourcc/video_ext in device.yaml (e.g. MJPG + .avi)");
    }
    writer_.write(frame_bgr);
}

}  // namespace antiuav
