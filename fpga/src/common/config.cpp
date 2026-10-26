#include "config.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace antiuav {

// ----------------------------------------------------------------------------- YAML subset
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

static std::string strip_comment(const std::string& line) {
    char quote = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (quote) {
            if (c == quote) quote = 0;
        } else if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#' && (i == 0 || line[i - 1] == ' ' || line[i - 1] == '\t')) {
            return line.substr(0, i);
        }
    }
    return line;
}

YamlMap YamlMap::load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw ConfigError("Cannot open config file: " + path);
    YamlMap m;
    m.path_ = path;
    std::vector<std::pair<int, std::string>> stack;  // (indent, key)
    std::string raw;
    int line_no = 0;
    while (std::getline(f, raw)) {
        ++line_no;
        std::string line = strip_comment(raw);
        if (trim(line).empty()) continue;
        int indent = 0;
        while (indent < static_cast<int>(line.size()) && line[indent] == ' ') ++indent;
        std::string content = trim(line);
        if (content.rfind("- ", 0) == 0 || content == "-")
            throw ConfigError(path + ":" + std::to_string(line_no) + ": YAML lists are not supported");
        size_t colon = content.find(':');
        if (colon == std::string::npos)
            throw ConfigError(path + ":" + std::to_string(line_no) + ": expected 'key: value'");
        std::string key = trim(content.substr(0, colon));
        std::string value = trim(content.substr(colon + 1));

        while (!stack.empty() && stack.back().first >= indent) stack.pop_back();
        std::string full;
        for (const auto& s : stack) full += s.second + ".";
        full += key;

        if (value.empty()) {
            stack.emplace_back(indent, key);  // section header
            continue;
        }
        if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front())
            value = value.substr(1, value.size() - 2);
        m.values_[full] = value;
    }
    return m;
}

bool YamlMap::has(const std::string& key) const { return values_.count(key) > 0; }

bool YamlMap::is_null(const std::string& key) const {
    auto it = values_.find(key);
    return it == values_.end() || it->second == "null" || it->second == "~" || it->second == "Null";
}

std::string YamlMap::get_str(const std::string& key) const {
    auto it = values_.find(key);
    if (it == values_.end()) throw ConfigError(path_ + ": missing required key '" + key + "'");
    return it->second;
}

std::string YamlMap::get_str(const std::string& key, const std::string& def) const {
    return has(key) ? get_str(key) : def;
}

int YamlMap::get_int(const std::string& key) const {
    std::string v = get_str(key);
    try {
        size_t idx = 0;
        long r = std::stol(v, &idx);
        if (idx != v.size()) throw std::invalid_argument(v);
        return static_cast<int>(r);
    } catch (const std::exception&) {
        throw ConfigError(path_ + ": key '" + key + "' must be an integer, got '" + v + "'");
    }
}

int YamlMap::get_int(const std::string& key, int def) const { return has(key) ? get_int(key) : def; }

double YamlMap::get_float(const std::string& key) const {
    std::string v = get_str(key);
    try {
        size_t idx = 0;
        double r = std::stod(v, &idx);
        if (idx != v.size()) throw std::invalid_argument(v);
        return r;
    } catch (const std::exception&) {
        throw ConfigError(path_ + ": key '" + key + "' must be a number, got '" + v + "'");
    }
}

double YamlMap::get_float(const std::string& key, double def) const { return has(key) ? get_float(key) : def; }

bool YamlMap::get_bool(const std::string& key) const {
    std::string v = get_str(key);
    if (v == "true" || v == "True" || v == "TRUE" || v == "yes" || v == "on") return true;
    if (v == "false" || v == "False" || v == "FALSE" || v == "no" || v == "off") return false;
    throw ConfigError(path_ + ": key '" + key + "' must be a bool, got '" + v + "'");
}

bool YamlMap::get_bool(const std::string& key, bool def) const { return has(key) ? get_bool(key) : def; }

// ----------------------------------------------------------------------------- FpgaConstraints
// Required keys use the throwing getters (raw["..."] in Python), optional keys
// use the same defaults as FpgaConstraints.from_yaml().
FpgaConstraints FpgaConstraints::from_yaml(const std::string& path) {
    YamlMap y = YamlMap::load(path);
    FpgaConstraints c{};
    c.image_height = y.get_int("image.height");
    c.image_width = y.get_int("image.width");
    c.image_channels = y.get_int("image.channels");
    c.det_input_height = y.get_int("detector.input_height");
    c.det_input_width = y.get_int("detector.input_width");
    c.max_det_boxes = y.get_int("detector.max_det_boxes");
    c.det_conf_threshold = y.get_float("detector.conf_threshold");
    c.det_nms_iou_threshold = y.get_float("detector.nms_iou_threshold");
    c.block_sum_threshold = y.get_int("motion.block_sum_threshold");
    c.max_motion_candidates = y.get_int("motion.max_motion_candidates");
    c.motion_min_box_size = y.get_int("motion.min_box_size");
    c.motion_max_box_size = y.get_int("motion.max_box_size");
    c.motion_merge_gap_ratio = y.get_float("motion.merge_gap_ratio");
    c.motion_max_aspect_ratio = y.get_float("motion.max_aspect_ratio", 5.0);
    c.motion_merge_min_fill_ratio = y.get_float("motion.merge_min_fill_ratio", 0.15);
    c.motion_size_adapt_enabled = y.get_bool("motion.size_adapt_enabled", false);
    c.motion_size_adapt_window = y.get_int("motion.size_adapt_window", 30);
    c.motion_size_adapt_shrink_factor = y.get_float("motion.size_adapt_shrink_factor", 0.6);
    c.motion_size_adapt_grow_factor = y.get_float("motion.size_adapt_grow_factor", 1.6);
    c.motion_size_adapt_min_samples = y.get_int("motion.size_adapt_min_samples", 5);
    c.motion_candidate_anchor_iou = y.get_float("motion.candidate_anchor_iou", 0.05);
    c.motion_frame_lag = y.get_int("motion.frame_lag", 5);
    c.motion_strength_metric = y.get_str("motion.strength_metric", "sum");
    c.motion_min_strength = y.is_null("motion.min_strength") ? -std::numeric_limits<double>::infinity()
                                                              : y.get_float("motion.min_strength");
    c.motion_thresh_factor = y.get_float("motion.thresh_factor", 2.5);
    c.motion_min_area = y.get_int("motion.min_area", 20);
    c.motion_model = y.get_str("motion.model", "affine");
    c.motion_irls_iters = y.get_int("motion.irls_iters", 3);
    c.motion_huber_delta = y.get_float("motion.huber_delta", 10.0);
    c.motion_pyramid_levels = y.get_int("motion.pyramid_levels", 2);
    c.max_total_candidates = y.get_int("fusion.max_total_candidates");
    c.n_track_slots = y.get_int("tracker.n_track_slots");
    c.tier1_iou_threshold = y.get_float("tracker.tier1_iou_threshold");
    c.tier1_conf_min = y.get_float("tracker.tier1_conf_min");
    c.tier2_iou_threshold = y.get_float("tracker.tier2_iou_threshold");
    c.tier3_conf_min = y.get_float("tracker.tier3_conf_min");
    c.tier4_iou_threshold = y.get_float("tracker.tier4_iou_threshold");
    c.max_track_age = y.get_int("tracker.max_track_age");
    c.max_consecutive_motion_matches = y.get_int("tracker.max_consecutive_motion_matches", 5);
    c.override_conf_min = y.get_float("tracker.override_conf_min", 0.85);
    c.override_iou_threshold = y.get_float("tracker.override_iou_threshold", 0.05);
    c.override_requires_overlap = y.get_bool("tracker.override_requires_overlap", false);
    c.override_motion_dragged_only = y.get_bool("tracker.override_motion_dragged_only", false);
    c.override_max_center_distance_ratio = y.get_float("tracker.override_max_center_distance_ratio", 1.0);
    c.reinit_near_lost_enabled = y.get_bool("tracker.reinit_near_lost_enabled", true);
    c.reinit_max_center_distance_ratio = y.get_float("tracker.reinit_max_center_distance_ratio", 0.15);
    c.reinit_lost_memory_frames = y.get_int("tracker.reinit_lost_memory_frames", 30);
    c.reinit_trust_conf_min = y.get_float("tracker.reinit_trust_conf_min", c.tier3_conf_min);
    c.kalman_enabled = y.get_bool("kalman.enabled");
    c.kalman_state_dim = y.get_int("kalman.state_dim");
    c.kalman_process_noise = y.get_float("kalman.process_noise");
    c.kalman_measurement_noise = y.get_float("kalman.measurement_noise");
    c.kalman_motion_measurement_noise = y.get_float("kalman.motion_measurement_noise", c.kalman_measurement_noise);
    c.kalman_max_predict_only_frames = y.get_int("kalman.max_predict_only_frames");
    c.tentative_motion_tracks_enabled = y.get_bool("tentative_motion_tracks.enabled", false);
    c.max_tentative_motion_tracks = y.get_int("tentative_motion_tracks.max_slots", 0);
    c.tentative_iou_threshold = y.get_float("tentative_motion_tracks.iou_threshold", 0.3);
    c.tentative_max_miss = y.get_int("tentative_motion_tracks.max_miss", 3);
    c.motion_track_confirm_frames = y.get_int("tentative_motion_tracks.confirm_frames", 4);
    c.tentative_active_overlap_max_iou = y.get_float("tentative_motion_tracks.active_overlap_max_iou", 0.1);
    c.validate();
    return c;
}

static std::string num(double v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

void FpgaConstraints::validate() const {
    int expected_total = max_det_boxes + max_motion_candidates;
    if (max_total_candidates != expected_total)
        throw ConfigError("fusion.max_total_candidates (" + num(max_total_candidates) +
                          ") must equal detector.max_det_boxes + motion.max_motion_candidates (" +
                          num(expected_total) + ")");
    if (image_channels != 1) throw ConfigError("image.channels must be 1 (grayscale IR)");
    if (n_track_slots < 1) throw ConfigError("tracker.n_track_slots must be >= 1");
    if (kalman_state_dim != 4) throw ConfigError("kalman.state_dim only supports 4 ([x, y, vx, vy])");
    if (motion_min_box_size <= 0 || motion_max_box_size <= motion_min_box_size)
        throw ConfigError("motion.min_box_size must be > 0 and < motion.max_box_size");
    if (motion_merge_gap_ratio < 0) throw ConfigError("motion.merge_gap_ratio must be >= 0");
    if (motion_max_aspect_ratio < 1.0) throw ConfigError("motion.max_aspect_ratio must be >= 1.0");
    if (!(motion_merge_min_fill_ratio > 0.0 && motion_merge_min_fill_ratio <= 1.0))
        throw ConfigError("motion.merge_min_fill_ratio must be in (0.0, 1.0]");
    if (motion_size_adapt_enabled) {
        if (motion_size_adapt_window <= 0) throw ConfigError("motion.size_adapt_window must be >= 1");
        if (motion_size_adapt_min_samples <= 0) throw ConfigError("motion.size_adapt_min_samples must be >= 1");
        if (motion_size_adapt_shrink_factor <= 0 || motion_size_adapt_grow_factor <= 0)
            throw ConfigError("motion.size_adapt_shrink_factor/grow_factor must be > 0");
    }
    if (motion_candidate_anchor_iou < 0) throw ConfigError("motion.candidate_anchor_iou must be >= 0");
    if (motion_frame_lag < 1) throw ConfigError("motion.frame_lag must be >= 1");
    if (motion_strength_metric != "sum" && motion_strength_metric != "mean" && motion_strength_metric != "contrast")
        throw ConfigError("motion.strength_metric must be 'sum', 'mean' or 'contrast'");
    if (motion_model != "affine" && motion_model != "translation")
        throw ConfigError("motion.model must be 'affine' or 'translation'");
    if (motion_irls_iters < 1) throw ConfigError("motion.irls_iters must be >= 1");
    if (motion_huber_delta <= 0) throw ConfigError("motion.huber_delta must be > 0");
    if (motion_pyramid_levels < 0) throw ConfigError("motion.pyramid_levels must be >= 0");
    if (motion_min_area < 0) throw ConfigError("motion.min_area must be >= 0");
    if (max_consecutive_motion_matches < 1) throw ConfigError("tracker.max_consecutive_motion_matches must be >= 1");
    if (override_conf_min < tier3_conf_min) throw ConfigError("tracker.override_conf_min must be >= tracker.tier3_conf_min");
    if (override_iou_threshold < 0) throw ConfigError("tracker.override_iou_threshold must be >= 0");
    if (override_requires_overlap && override_iou_threshold <= 0)
        throw ConfigError("tracker.override_iou_threshold must be > 0 when override_requires_overlap = true");
    if (!(override_max_center_distance_ratio > 0.0 && override_max_center_distance_ratio <= 1.5))
        throw ConfigError("tracker.override_max_center_distance_ratio must be in (0.0, 1.5]");
    if (reinit_near_lost_enabled) {
        if (!(reinit_max_center_distance_ratio > 0.0 && reinit_max_center_distance_ratio <= 1.5))
            throw ConfigError("tracker.reinit_max_center_distance_ratio must be in (0.0, 1.5]");
        if (reinit_lost_memory_frames < 0) throw ConfigError("tracker.reinit_lost_memory_frames must be >= 0");
        if (!(reinit_trust_conf_min >= 0.0 && reinit_trust_conf_min <= 1.0))
            throw ConfigError("tracker.reinit_trust_conf_min must be in [0.0, 1.0]");
        if (reinit_trust_conf_min < tier3_conf_min)
            throw ConfigError("tracker.reinit_trust_conf_min must be >= tracker.tier3_conf_min");
    }
    if (!(tentative_active_overlap_max_iou >= 0.0 && tentative_active_overlap_max_iou <= 1.0))
        throw ConfigError("tentative_motion_tracks.active_overlap_max_iou must be in [0.0, 1.0]");
    if (kalman_motion_measurement_noise < kalman_measurement_noise)
        throw ConfigError("kalman.motion_measurement_noise must be >= kalman.measurement_noise");
    if (tentative_motion_tracks_enabled) {
        if (max_tentative_motion_tracks <= 0) throw ConfigError("tentative_motion_tracks.max_slots must be >= 1");
        if (motion_track_confirm_frames <= 0) throw ConfigError("tentative_motion_tracks.confirm_frames must be >= 1");
        if (tentative_max_miss < 0) throw ConfigError("tentative_motion_tracks.max_miss must be >= 0");
    }
}

// ----------------------------------------------------------------------------- DeviceConfig
DeviceConfig DeviceConfig::from_yaml(const std::string& path) {
    YamlMap y = YamlMap::load(path);
    DeviceConfig d;
    d.board_name = y.get_str("board.name", "TODO");
    d.dpu_arch = y.get_str("dpu.arch", "TODO");
    d.dpu_fingerprint = y.get_str("dpu.fingerprint", "TODO");
    d.xmodel_path = y.get_str("dpu.xmodel_path");
    d.dpu_subgraph_index = y.get_int("dpu.subgraph_index", d.dpu_subgraph_index);
    d.dpu_num_runners = y.get_int("dpu.num_runners", d.dpu_num_runners);
    d.dpu_timeout_ms = y.get_int("dpu.timeout_ms", d.dpu_timeout_ms);
    d.dpu_reg_max = y.get_int("dpu.reg_max", d.dpu_reg_max);
    d.dpu_num_classes = y.get_int("dpu.num_classes", d.dpu_num_classes);
    d.cv_num_threads = y.get_int("cpu.cv_num_threads", d.cv_num_threads);
    d.pipeline_depth = y.get_int("cpu.pipeline_depth", d.pipeline_depth);
    d.reader_queue = y.get_int("cpu.reader_queue", d.reader_queue);
    d.writer_queue = y.get_int("cpu.writer_queue", d.writer_queue);
    d.video_fourcc = y.get_str("output.video_fourcc", d.video_fourcc);
    d.video_ext = y.get_str("output.video_ext", d.video_ext);
    d.video_fps = y.get_int("output.video_fps", d.video_fps);
    if (d.dpu_num_runners < 1) throw ConfigError("dpu.num_runners must be >= 1");
    if (d.pipeline_depth < 1) throw ConfigError("cpu.pipeline_depth must be >= 1");
    if (d.video_fourcc.size() != 4) throw ConfigError("output.video_fourcc must have 4 characters");
    return d;
}

}  // namespace antiuav
