// Port of src1/antiuav_sot/utils/config.py (FpgaConstraints + validate()),
// plus DeviceConfig: the ONLY place holding board/DPU specific values
// (config/device.yaml). The YAML reader supports the subset used by
// fpga_constraints.yaml / device.yaml: nested "key: value" maps, comments,
// scalars (int/float/bool/null/string). No flow lists.
#ifndef ANTIUAV_CONFIG_H
#define ANTIUAV_CONFIG_H

#include <map>
#include <stdexcept>
#include <string>

namespace antiuav {

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Flat view of a YAML file: "section.key" -> raw scalar string.
class YamlMap {
public:
    static YamlMap load(const std::string& path);

    bool has(const std::string& key) const;
    bool is_null(const std::string& key) const;
    std::string get_str(const std::string& key) const;  // throws if missing (like raw[...] in Python)
    std::string get_str(const std::string& key, const std::string& def) const;
    int get_int(const std::string& key) const;
    int get_int(const std::string& key, int def) const;
    double get_float(const std::string& key) const;
    double get_float(const std::string& key, double def) const;
    bool get_bool(const std::string& key) const;
    bool get_bool(const std::string& key, bool def) const;

    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::map<std::string, std::string> values_;
};

struct FpgaConstraints {
    // image
    int image_height, image_width, image_channels;
    // detector
    int det_input_height, det_input_width, max_det_boxes;
    double det_conf_threshold, det_nms_iou_threshold;
    // motion
    int block_sum_threshold, max_motion_candidates, motion_min_box_size, motion_max_box_size;
    double motion_merge_gap_ratio, motion_max_aspect_ratio, motion_merge_min_fill_ratio;
    bool motion_size_adapt_enabled;
    int motion_size_adapt_window;
    double motion_size_adapt_shrink_factor, motion_size_adapt_grow_factor;
    int motion_size_adapt_min_samples;
    double motion_candidate_anchor_iou;
    int motion_frame_lag;
    std::string motion_strength_metric;
    double motion_min_strength;  // -inf when "null" in YAML
    double motion_thresh_factor;
    int motion_min_area;
    std::string motion_model;
    int motion_irls_iters;
    double motion_huber_delta;
    int motion_pyramid_levels;
    // fusion
    int max_total_candidates;
    // tracker
    int n_track_slots;
    double tier1_iou_threshold, tier1_conf_min, tier2_iou_threshold, tier3_conf_min, tier4_iou_threshold;
    int max_track_age, max_consecutive_motion_matches;
    double override_conf_min, override_iou_threshold;
    bool override_requires_overlap, override_motion_dragged_only;
    double override_max_center_distance_ratio;
    bool reinit_near_lost_enabled;
    double reinit_max_center_distance_ratio;
    int reinit_lost_memory_frames;
    double reinit_trust_conf_min;
    // kalman
    bool kalman_enabled;
    int kalman_state_dim;
    double kalman_process_noise, kalman_measurement_noise, kalman_motion_measurement_noise;
    int kalman_max_predict_only_frames;
    // tentative motion tracks
    bool tentative_motion_tracks_enabled;
    int max_tentative_motion_tracks;
    double tentative_iou_threshold;
    int tentative_max_miss, motion_track_confirm_frames;
    double tentative_active_overlap_max_iou;

    static FpgaConstraints from_yaml(const std::string& path);
    void validate() const;
};

// Everything that depends on the (not yet chosen) board / DPU build.
struct DeviceConfig {
    std::string board_name;          // TODO(board)
    std::string dpu_arch;            // TODO(board): informational, e.g. DPUCZDX8G_ISA1_B4096
    std::string dpu_fingerprint;     // TODO(board): informational, printed by xdputil query
    std::string xmodel_path;         // relative paths are resolved against the fpga/ folder
    int dpu_subgraph_index = 0;      // index into the DPU subgraphs of the xmodel
    int dpu_num_runners = 1;         // >1 only if the DPU IP has several cores
    int dpu_timeout_ms = -1;         // runner->wait() timeout, -1 = forever
    int dpu_reg_max = 16;            // YOLOv8 DFL bins (checked against the xmodel output shapes)
    int dpu_num_classes = 1;
    int cv_num_threads = -1;         // cv::setNumThreads, -1 = OpenCV default
    int pipeline_depth = 2;          // frames in flight in the parallel (stateless) stage
    int reader_queue = 4;            // prefetched decoded frames
    int writer_queue = 8;            // frames waiting for overlay/video encoding
    std::string video_fourcc = "mp4v";
    std::string video_ext = ".mp4";
    int video_fps = 15;              // same default as utils/viz.py::SequenceVideoWriter

    static DeviceConfig from_yaml(const std::string& path);
};

}  // namespace antiuav

#endif  // ANTIUAV_CONFIG_H
