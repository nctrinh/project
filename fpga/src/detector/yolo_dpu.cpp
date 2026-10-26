#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>

#include "../common/json.h"
#include "detector.h"

namespace antiuav {

// ----------------------------------------------------------------------------- letterbox
LetterboxResult letterbox_grayscale(const cv::Mat& image, int target_height, int target_width, int pad_value) {
    CV_Assert(image.type() == CV_8UC1);
    int orig_h = image.rows, orig_w = image.cols;
    double scale = std::min(static_cast<double>(target_height) / orig_h, static_cast<double>(target_width) / orig_w);
    // Python int(round(x)) = round-half-to-even = std::nearbyint (default rounding mode).
    int new_h = static_cast<int>(std::nearbyint(orig_h * scale));
    int new_w = static_cast<int>(std::nearbyint(orig_w * scale));

    LetterboxResult r;
    cv::Mat resized;
    if (new_h == orig_h && new_w == orig_w) {
        resized = image;  // cv2.resize to the same size is an exact copy
    } else {
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    }
    int pad_top = (target_height - new_h) / 2;
    int pad_bottom = target_height - new_h - pad_top;
    int pad_left = (target_width - new_w) / 2;
    int pad_right = target_width - new_w - pad_left;
    if (pad_top || pad_bottom || pad_left || pad_right) {
        cv::copyMakeBorder(resized, r.image, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT,
                           cv::Scalar(pad_value));
    } else {
        r.image = resized;
    }
    r.scale = static_cast<float>(scale);
    r.pad_left = pad_left;
    r.pad_top = pad_top;
    r.orig_height = orig_h;
    r.orig_width = orig_w;
    return r;
}

// ----------------------------------------------------------------------------- pad_to_fixed_size
DetectionResult pad_to_fixed_size(const std::vector<Box>& boxes, const std::vector<float>& scores, int max_det_boxes) {
    std::vector<size_t> idx(boxes.size());
    std::iota(idx.begin(), idx.end(), 0);
    if (static_cast<int>(idx.size()) > max_det_boxes) {
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
        idx.resize(max_det_boxes);
    }
    DetectionResult r;
    r.boxes = make_empty_boxes(max_det_boxes);
    r.scores.assign(max_det_boxes, 0.0f);
    r.valid_mask.assign(max_det_boxes, false);
    for (size_t k = 0; k < idx.size(); ++k) {
        r.boxes[k] = boxes[idx[k]];
        r.scores[k] = scores[idx[k]];
        r.valid_mask[k] = true;
    }
    return r;
}

// ----------------------------------------------------------------------------- decode
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void decode_dpu_outputs(const std::vector<HeadLevel>& levels, int reg_max, int num_classes, float conf_thres,
                        float iou_thres, int max_det, std::vector<Box>& out_boxes, std::vector<float>& out_scores) {
    out_boxes.clear();
    out_scores.clear();
    std::vector<Box> boxes;
    std::vector<float> scores;
    std::vector<float> bins(reg_max);
    const float logit_thres = std::log(conf_thres / (1.0f - conf_thres));

    for (const auto& lv : levels) {
        const float cls_scale = std::ldexp(1.0f, -lv.cls_fix_point);
        const float box_scale = std::ldexp(1.0f, -lv.box_fix_point);
        // Conservative int8 pre-filter: cells below it cannot reach conf_thres.
        const int q_min = static_cast<int>(std::floor(logit_thres / cls_scale)) - 1;
        const int box_ch = 4 * reg_max;
        for (int y = 0; y < lv.height; ++y) {
            for (int x = 0; x < lv.width; ++x) {
                const int cell = y * lv.width + x;
                const int8_t* c = lv.cls + static_cast<size_t>(cell) * num_classes;
                int q_best = c[0];
                for (int k = 1; k < num_classes; ++k) q_best = std::max(q_best, static_cast<int>(c[k]));
                if (q_best < q_min) continue;
                float score = sigmoidf(static_cast<float>(c[0]) * cls_scale);
                for (int k = 1; k < num_classes; ++k) score = std::max(score, sigmoidf(static_cast<float>(c[k]) * cls_scale));
                if (!(score >= conf_thres)) continue;

                // DFL: softmax over reg_max bins, expectation (float32 like numpy).
                const int8_t* b = lv.box + static_cast<size_t>(cell) * box_ch;
                float d[4];
                for (int side = 0; side < 4; ++side) {
                    const int8_t* bb = b + side * reg_max;
                    float vmax = static_cast<float>(bb[0]) * box_scale;
                    for (int k = 1; k < reg_max; ++k) vmax = std::max(vmax, static_cast<float>(bb[k]) * box_scale);
                    float sum = 0.0f;
                    for (int k = 0; k < reg_max; ++k) {
                        bins[k] = std::exp(static_cast<float>(bb[k]) * box_scale - vmax);
                        sum += bins[k];
                    }
                    float acc = 0.0f;
                    for (int k = 0; k < reg_max; ++k) acc += (bins[k] / sum) * static_cast<float>(k);
                    d[side] = acc;
                }
                const double ax = x + 0.5, ay = y + 0.5, s = lv.stride;
                boxes.push_back({static_cast<float>((ax - d[0]) * s), static_cast<float>((ay - d[1]) * s),
                                 static_cast<float>((ax + d[2]) * s), static_cast<float>((ay + d[3]) * s)});
                scores.push_back(score);
            }
        }
    }
    if (boxes.empty()) return;

    // NMS (dpu_export.py::_nms): scores.argsort()[::-1]. Ties are frequent
    // with int8 logits; numpy's default sort is not stable for large inputs,
    // here ties are resolved as "stable ascending, then reversed".
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] < scores[b]; });
    std::reverse(order.begin(), order.end());

    std::vector<float> areas(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) areas[i] = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]);

    std::vector<int> next;
    while (!order.empty() && static_cast<int>(out_boxes.size()) < max_det) {
        const int i = order[0];
        out_boxes.push_back(boxes[i]);
        out_scores.push_back(scores[i]);
        next.clear();
        for (size_t k = 1; k < order.size(); ++k) {
            const int j = order[k];
            float xx1 = std::max(boxes[i][0], boxes[j][0]);
            float yy1 = std::max(boxes[i][1], boxes[j][1]);
            float xx2 = std::min(boxes[i][2], boxes[j][2]);
            float yy2 = std::min(boxes[i][3], boxes[j][3]);
            float inter = std::max(xx2 - xx1, 0.0f) * std::max(yy2 - yy1, 0.0f);
            float iou = inter / (areas[i] + areas[j] - inter + 1e-9f);
            if (iou <= iou_thres) next.push_back(j);
        }
        order.swap(next);
    }
}

// ----------------------------------------------------------------------------- YoloDpuDetector
YoloDpuDetector::YoloDpuDetector(const FpgaConstraints& c, const DeviceConfig& d, const std::string& xmodel_path)
    : max_det_(c.max_det_boxes),
      reg_max_(d.dpu_reg_max),
      num_classes_(d.dpu_num_classes),
      input_h_(c.det_input_height),
      input_w_(c.det_input_width),
      conf_thres_(static_cast<float>(c.det_conf_threshold)),
      iou_thres_(static_cast<float>(c.det_nms_iou_threshold)) {
    for (int i = 0; i < d.dpu_num_runners; ++i) {
        Slot slot;
        slot.runner = DpuRunner::create(xmodel_path, d.dpu_subgraph_index, d.dpu_timeout_ms);
        map_outputs(slot);
        slots_.push_back(std::move(slot));
        free_slots_.push_back(i);
    }
    const DpuTensorInfo& in = slots_[0].runner->input_info();
    if (in.shape.size() != 4 || in.shape[1] != input_h_ || in.shape[2] != input_w_ || in.shape[3] != 1) {
        std::ostringstream ss;
        ss << "xmodel input shape mismatch: expected (N," << input_h_ << "," << input_w_
           << ",1) from detector.input_height/width, got (";
        for (size_t k = 0; k < in.shape.size(); ++k) ss << (k ? "," : "") << in.shape[k];
        ss << ")";
        throw std::runtime_error(ss.str());
    }
    // normalize_grayscale (u8/255, float32) then DPU input quantisation
    // q = round(x * 2^fix_point), saturated to int8.
    // TODO(board): confirm the rounding mode matches vai_q (round-half-away here).
    const float in_scale = std::ldexp(1.0f, in.fix_point);
    for (int p = 0; p < 256; ++p) {
        float x = static_cast<float>(p) / 255.0f;
        float q = std::round(x * in_scale);
        input_lut_[p] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, q)));
    }
}

void YoloDpuDetector::map_outputs(Slot& slot) const {
    const auto& outs = slot.runner->output_info();
    std::map<int, std::pair<int, int>, std::greater<int>> by_height;  // H -> (box idx, cls idx)
    for (size_t i = 0; i < outs.size(); ++i) {
        const auto& s = outs[i].shape;
        if (s.size() != 4) throw std::runtime_error("Unexpected DPU output rank for " + outs[i].name);
        auto& entry = by_height.emplace(s[1], std::make_pair(-1, -1)).first->second;
        if (s[3] == 4 * reg_max_) entry.first = static_cast<int>(i);
        else if (s[3] == num_classes_) entry.second = static_cast<int>(i);
        else throw std::runtime_error("DPU output " + outs[i].name + " has " + std::to_string(s[3]) +
                                      " channels, expected 4*reg_max or num_classes (check dpu.reg_max/num_classes)");
    }
    if (by_height.size() != 3)
        throw std::runtime_error("Expected 3 YOLOv8 levels in the DPU outputs, found " + std::to_string(by_height.size()));
    for (const auto& kv : by_height) {
        if (kv.second.first < 0 || kv.second.second < 0)
            throw std::runtime_error("Missing box or cls output for level H=" + std::to_string(kv.first));
        const auto& box_shape = outs[kv.second.first].shape;
        int stride = input_h_ / kv.first;
        if (stride * kv.first != input_h_ || input_w_ / box_shape[2] != stride)
            throw std::runtime_error("DPU output H/W do not divide the input size");
        slot.level_box_out.push_back(kv.second.first);
        slot.level_cls_out.push_back(kv.second.second);
        slot.level_stride.push_back(stride);
    }
}

DetectionResult YoloDpuDetector::detect(const cv::Mat& det_input_u8, int) {
    CV_Assert(det_input_u8.type() == CV_8UC1 && det_input_u8.rows == input_h_ && det_input_u8.cols == input_w_);
    int slot_idx;
    {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return !free_slots_.empty(); });
        slot_idx = free_slots_.back();
        free_slots_.pop_back();
    }
    Slot& slot = slots_[slot_idx];
    std::vector<Box> boxes;
    std::vector<float> scores;
    try {
        int8_t* dst = slot.runner->input_data();
        for (int y = 0; y < input_h_; ++y) {
            const uint8_t* src = det_input_u8.ptr<uint8_t>(y);
            int8_t* row = dst + static_cast<size_t>(y) * input_w_;
            for (int x = 0; x < input_w_; ++x) row[x] = input_lut_[src[x]];
        }
        slot.runner->run();

        const auto& outs = slot.runner->output_info();
        std::vector<HeadLevel> levels;
        for (size_t l = 0; l < slot.level_stride.size(); ++l) {
            const auto& bi = outs[slot.level_box_out[l]];
            const auto& ci = outs[slot.level_cls_out[l]];
            HeadLevel lv;
            lv.box = slot.runner->output_data(slot.level_box_out[l]);
            lv.cls = slot.runner->output_data(slot.level_cls_out[l]);
            lv.box_fix_point = bi.fix_point;
            lv.cls_fix_point = ci.fix_point;
            lv.height = bi.shape[1];
            lv.width = bi.shape[2];
            lv.stride = slot.level_stride[l];
            levels.push_back(lv);
        }
        decode_dpu_outputs(levels, reg_max_, num_classes_, conf_thres_, iou_thres_, max_det_, boxes, scores);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mu_);
        free_slots_.push_back(slot_idx);
        cv_.notify_one();
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        free_slots_.push_back(slot_idx);
        cv_.notify_one();
    }
    return pad_to_fixed_size(boxes, scores, max_det_);
}

std::string YoloDpuDetector::describe() const {
    std::ostringstream ss;
    const auto& r = *slots_[0].runner;
    ss << "DPU input " << r.input_info().name << " fix_point=" << r.input_info().fix_point << "; outputs:";
    for (size_t l = 0; l < slots_[0].level_stride.size(); ++l) {
        const auto& b = r.output_info()[slots_[0].level_box_out[l]];
        const auto& c = r.output_info()[slots_[0].level_cls_out[l]];
        ss << " [stride " << slots_[0].level_stride[l] << ": box " << b.name << " fp=" << b.fix_point << ", cls "
           << c.name << " fp=" << c.fix_point << "]";
    }
    ss << "; runners=" << slots_.size();
    return ss.str();
}

// ----------------------------------------------------------------------------- ReplayDetector
ReplayDetector::ReplayDetector(const std::string& json_path, int max_det_boxes) : max_det_(max_det_boxes) {
    Json j = Json::parse_file(json_path);
    const Json& frames = j.at("frames");
    for (size_t f = 0; f < frames.size(); ++f) {
        std::vector<std::pair<Box, float>> dets;
        for (size_t k = 0; k < frames[f].size(); ++k) {
            const Json& d = frames[f][k];
            const Json& b = d.at("box");
            dets.push_back({{static_cast<float>(b[0].as_number()), static_cast<float>(b[1].as_number()),
                             static_cast<float>(b[2].as_number()), static_cast<float>(b[3].as_number())},
                            static_cast<float>(d.at("score").as_number())});
        }
        frames_.push_back(std::move(dets));
    }
}

DetectionResult ReplayDetector::detect(const cv::Mat&, int frame_index) {
    std::vector<Box> boxes;
    std::vector<float> scores;
    if (frame_index >= 0 && frame_index < static_cast<int>(frames_.size())) {
        for (const auto& d : frames_[frame_index]) {
            boxes.push_back(d.first);
            scores.push_back(d.second);
        }
    }
    return pad_to_fixed_size(boxes, scores, max_det_);
}

}  // namespace antiuav
