// FPGA deployment of the Anti-UAV SOT pipeline (motion branch) - C++ port of
// scripts/eval_tracker_motion.py. Detector on the DPU, everything else on the CPU.
//
// Usage (see run_sequence.sh / run_eval.sh):
//   ./antiuav_fpga --manifest <data_manifest.json> [--split val|train|all] [--sequence NAME]
//                  [--max-sequences N] [--root-dir DIR] [--seq-dir DIR]
//                  [--constraints config/fpga_constraints.yaml] [--device config/device.yaml]
//                  [--xmodel PATH] [--output-dir DIR] [--no-video] [--sequential]
//                  [--dump-frames] [--det-replay detections.json]
//
// Outputs (<output-dir>):
//   sequences/<seq>/<seq><video_ext>   overlay video
//   sequences/<seq>/<seq>.json         {"res": [[x,y,w,h] | [], ...]} one entry per image
//   sequences/<seq>/eval_<seq>.json    metrics + latency of that sequence
//   eval_results_<split|seq>.json      aggregated metrics over all evaluated sequences
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "common/config.h"
#include "common/json.h"
#include "detector/detector.h"
#include "eval/dataset.h"
#include "eval/metrics.h"
#include "pipeline/sequence_runner.h"
#include "pipeline/sot_pipeline.h"

namespace fs = std::filesystem;
using namespace antiuav;

namespace {

struct Args {
    std::string manifest, split = "val", sequence, root_dir, seq_dir;
    int max_sequences = -1;
    std::string constraints, device, xmodel, output_dir = "results", det_replay;
    bool no_video = false, sequential = false, dump_frames = false;
};

void usage() {
    std::cout << "Usage: antiuav_fpga --manifest data_manifest.json [--split val|train|all] [--sequence NAME]\n"
                 "                    [--max-sequences N] [--root-dir DIR] [--seq-dir DIR]\n"
                 "                    [--constraints FILE] [--device FILE] [--xmodel FILE] [--output-dir DIR]\n"
                 "                    [--no-video] [--sequential] [--dump-frames] [--det-replay FILE]\n";
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--manifest") a.manifest = need(i);
        else if (k == "--split") a.split = need(i);
        else if (k == "--sequence") a.sequence = need(i);
        else if (k == "--max-sequences") a.max_sequences = std::stoi(need(i));
        else if (k == "--root-dir") a.root_dir = need(i);
        else if (k == "--seq-dir") a.seq_dir = need(i);
        else if (k == "--constraints") a.constraints = need(i);
        else if (k == "--device") a.device = need(i);
        else if (k == "--xmodel") a.xmodel = need(i);
        else if (k == "--output-dir") a.output_dir = need(i);
        else if (k == "--det-replay") a.det_replay = need(i);
        else if (k == "--no-video") a.no_video = true;
        else if (k == "--sequential") a.sequential = true;
        else if (k == "--dump-frames") a.dump_frames = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.manifest.empty() && a.seq_dir.empty()) throw std::runtime_error("--manifest or --seq-dir is required");
    if (a.split != "val" && a.split != "train" && a.split != "all") throw std::runtime_error("--split must be val|train|all");
    return a;
}

fs::path exe_dir() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return fs::current_path();
    buf[n] = 0;
    return fs::path(buf).parent_path();
}

Json constraints_json(const FpgaConstraints& c) {
    Json j = Json::object();
#define F(name) j[#name] = c.name
    F(image_height); F(image_width); F(image_channels);
    F(det_input_height); F(det_input_width); F(max_det_boxes); F(det_conf_threshold); F(det_nms_iou_threshold);
    F(block_sum_threshold); F(max_motion_candidates); F(motion_min_box_size); F(motion_max_box_size);
    F(motion_merge_gap_ratio); F(motion_max_aspect_ratio); F(motion_merge_min_fill_ratio);
    F(motion_size_adapt_enabled); F(motion_size_adapt_window); F(motion_size_adapt_shrink_factor);
    F(motion_size_adapt_grow_factor); F(motion_size_adapt_min_samples); F(motion_candidate_anchor_iou);
    F(motion_frame_lag); F(motion_strength_metric); F(motion_min_strength); F(motion_thresh_factor);
    F(motion_min_area); F(motion_model); F(motion_irls_iters); F(motion_huber_delta); F(motion_pyramid_levels);
    F(max_total_candidates); F(n_track_slots); F(tier1_iou_threshold); F(tier1_conf_min); F(tier2_iou_threshold);
    F(tier3_conf_min); F(tier4_iou_threshold); F(max_track_age); F(max_consecutive_motion_matches);
    F(override_conf_min); F(override_iou_threshold); F(override_requires_overlap); F(override_motion_dragged_only);
    F(override_max_center_distance_ratio); F(reinit_near_lost_enabled); F(reinit_max_center_distance_ratio);
    F(reinit_lost_memory_frames); F(reinit_trust_conf_min); F(kalman_enabled); F(kalman_state_dim);
    F(kalman_process_noise); F(kalman_measurement_noise); F(kalman_motion_measurement_noise);
    F(kalman_max_predict_only_frames); F(tentative_motion_tracks_enabled); F(max_tentative_motion_tracks);
    F(tentative_iou_threshold); F(tentative_max_miss); F(motion_track_confirm_frames);
    F(tentative_active_overlap_max_iou);
#undef F
    return j;
}

void print_summary(const std::string& title, const Json& s, const Json& lat, double throughput_fps) {
    const Json& a = s.at("all_frames");
    const Json& p = s.at("present_only");
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n[antiuav_fpga] ==== " << title << " ====\n"
              << "  sequences: " << s.at("n_sequences").as_int() << "  frames: " << s.at("n_frames").as_int() << "\n"
              << "  mSA: " << s.at("mSA").as_number() << "   SA pooled: " << s.at("SA_pooled").as_number() << "\n"
              << "  -- all frames --   P@20px " << a.at("precision_at_20px").as_number() << "  P_Norm "
              << a.at("norm_precision_auc").as_number() << "  Success " << a.at("success_auc").as_number() << "\n"
              << "  -- present only -- P@20px " << p.at("precision_at_20px").as_number() << "  P_Norm "
              << p.at("norm_precision_auc").as_number() << "  Success " << p.at("success_auc").as_number() << "\n"
              << "  FN rate " << s.at("fn_rate_on_present_frames").as_number() << " ("
              << s.at("n_false_negative_frames").as_int() << "/" << s.at("n_present_frames").as_int() << ")"
              << "  FP rate " << s.at("fp_rate_on_absent_frames").as_number() << " ("
              << s.at("n_false_positive_frames").as_int() << "/" << s.at("n_absent_frames").as_int() << ")\n";
    if (lat.contains("mean_ms"))
        std::cout << std::setprecision(2) << "  step latency: mean " << lat.at("mean_ms").as_number() << " ms, median "
                  << lat.at("median_ms").as_number() << " ms, p95 " << lat.at("p95_ms").as_number() << " ms (~"
                  << lat.at("fps_from_mean").as_number() << " FPS)\n"
                  << "  end-to-end throughput (I/O + video included): " << throughput_fps << " FPS\n";
}

Json per_sequence_entry(const SequenceRunResult& r) {
    Json s = summarize({r.eval});
    Json e = Json::object();
    e["seq_name"] = r.eval.seq_name;
    e["n_frames"] = r.eval.n_frames;
    e["SA"] = s.at("mSA");
    e["mean_iou"] = s.at("mSA");
    e["all_frames"] = s.at("all_frames");
    e["present_only"] = s.at("present_only");
    e["fp_rate_on_absent_frames"] = s.at("fp_rate_on_absent_frames");
    e["fn_rate_on_present_frames"] = s.at("fn_rate_on_present_frames");
    e["precision_at_20px"] = s.at("all_frames").at("precision_at_20px");
    Json lat = latency_stats(r.step_ms);
    if (lat.contains("mean_ms")) e["mean_latency_ms"] = lat.at("mean_ms");
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        const fs::path base = exe_dir();
        const std::string constraints_path =
            args.constraints.empty() ? (base / "config" / "fpga_constraints.yaml").string() : args.constraints;
        const std::string device_path = args.device.empty() ? (base / "config" / "device.yaml").string() : args.device;

        FpgaConstraints constraints = FpgaConstraints::from_yaml(constraints_path);
        DeviceConfig device = DeviceConfig::from_yaml(device_path);
        if (device.cv_num_threads >= 0) cv::setNumThreads(device.cv_num_threads);
        std::string xmodel = args.xmodel.empty() ? device.xmodel_path : args.xmodel;
        if (args.xmodel.empty() && fs::path(xmodel).is_relative())
            xmodel = (fs::path(device_path).parent_path() / xmodel).lexically_normal().string();

        // ---- sequences ----
        std::vector<std::string> seq_dirs;
        std::string tag;
        if (!args.seq_dir.empty()) {
            seq_dirs.push_back(args.seq_dir);
            tag = fs::path(args.seq_dir).lexically_normal().filename().string();
        } else {
            Manifest m = Manifest::load(args.manifest);
            std::vector<std::string> names;
            if (args.split == "train") names = m.train_sequences;
            else if (args.split == "val") names = m.val_sequences;
            else {
                names = m.train_sequences;
                names.insert(names.end(), m.val_sequences.begin(), m.val_sequences.end());
            }
            if (args.max_sequences >= 0 && static_cast<int>(names.size()) > args.max_sequences)
                names.resize(args.max_sequences);
            if (!args.sequence.empty()) {
                if (std::find(names.begin(), names.end(), args.sequence) == names.end()) {
                    std::cerr << "Error: sequence '" << args.sequence << "' is not in split '" << args.split << "'\n";
                    return 1;
                }
                names = {args.sequence};
            }
            const std::string root = args.root_dir.empty() ? m.root_dir : args.root_dir;
            for (const auto& n : names) seq_dirs.push_back((fs::path(root) / n).string());
            tag = args.sequence.empty() ? args.split : args.sequence;
        }
        std::vector<std::string> image_globs = {"*.jpg", "*.jpeg", "*.png"};
        if (!args.manifest.empty()) image_globs = Manifest::load(args.manifest).image_glob;

        // ---- detector ----
        std::unique_ptr<Detector> detector;
        std::string detector_desc;
        if (!args.det_replay.empty()) {
            if (seq_dirs.size() != 1) throw std::runtime_error("--det-replay needs exactly one sequence");
            detector = std::make_unique<ReplayDetector>(args.det_replay, constraints.max_det_boxes);
            detector_desc = "replay:" + args.det_replay;
        } else {
            auto d = std::make_unique<YoloDpuDetector>(constraints, device, xmodel);
            detector_desc = d->describe();
            detector = std::move(d);
        }
        std::cout << "[antiuav_fpga] board=" << device.board_name << " dpu_arch=" << device.dpu_arch << "\n"
                  << "[antiuav_fpga] xmodel=" << xmodel << "\n[antiuav_fpga] " << detector_desc << "\n"
                  << "[antiuav_fpga] evaluating " << seq_dirs.size() << " sequence(s), mode="
                  << (args.sequential ? "sequential" : "pipelined") << "\n";

        SotPipeline pipeline(constraints, detector.get());
        const fs::path out_dir(args.output_dir);
        fs::create_directories(out_dir);

        std::vector<SequenceEvalResult> all_results;
        std::vector<double> all_latency;
        Json per_seq = Json::array();
        double total_wall_ms = 0;
        size_t total_frames = 0;
        for (const auto& dir : seq_dirs) {
            SOTSequence seq(dir, image_globs);
            fs::path seq_out = out_dir / "sequences" / seq.name();
            fs::create_directories(seq_out);
            RunOptions opt;
            opt.sequential = args.sequential;
            opt.dump_frames = args.dump_frames;
            if (!args.no_video) opt.video_path = (seq_out / (seq.name() + device.video_ext)).string();

            SequenceRunResult r = run_sequence(pipeline, seq, device, opt);

            Json res = Json::array();
            for (const auto& p : r.predictions_xywh) {
                Json b = Json::array();
                for (float v : p) b.push_back(static_cast<double>(v));
                res.push_back(b);
            }
            Json pred = Json::object();
            pred["res"] = res;
            pred.dump_file((seq_out / (seq.name() + ".json")).string(), -1);
            if (args.dump_frames) r.frames_dump.dump_file((seq_out / "frames_dump.json").string(), -1);

            Json seq_summary = summarize({r.eval}, true);
            Json seq_lat = latency_stats(r.step_ms);
            Json seq_json = Json::object();
            seq_json["seq_name"] = seq.name();
            seq_json["summary"] = seq_summary;
            seq_json["latency"] = seq_lat;
            seq_json["latency_detector_ms"] = latency_stats(r.det_ms);
            seq_json["latency_motion_core_ms"] = latency_stats(r.motion_ms);
            seq_json["latency_tracker_ms"] = latency_stats(r.stage_b_ms);
            seq_json["wall_ms"] = r.wall_ms;
            seq_json["throughput_fps"] = r.wall_ms > 0 ? 1000.0 * r.step_ms.size() / r.wall_ms : 0.0;
            seq_json.dump_file((seq_out / ("eval_" + seq.name() + ".json")).string());
            if (seq_dirs.size() == 1)
                print_summary("SEQUENCE " + seq.name(), seq_summary, seq_lat, seq_json.at("throughput_fps").as_number());
            else
                std::cout << "[antiuav_fpga] " << seq.name() << ": SA=" << seq_summary.at("mSA").as_number()
                          << " frames=" << r.eval.n_frames << " mean step "
                          << (seq_lat.contains("mean_ms") ? seq_lat.at("mean_ms").as_number() : 0.0) << " ms\n";

            per_seq.push_back(per_sequence_entry(r));
            all_results.push_back(std::move(r.eval));
            all_latency.insert(all_latency.end(), r.step_ms.begin(), r.step_ms.end());
            total_wall_ms += r.wall_ms;
            total_frames += r.step_ms.size();
        }

        Json summary = summarize(all_results, true);
        Json latency = latency_stats(all_latency);
        double throughput = total_wall_ms > 0 ? 1000.0 * total_frames / total_wall_ms : 0.0;
        if (seq_dirs.size() > 1) print_summary("AGGREGATED RESULTS", summary, latency, throughput);

        Json report = Json::object();
        report["split"] = args.seq_dir.empty() ? args.split : "seq-dir";
        report["constraints"] = constraints_path;
        report["device_config"] = device_path;
        report["board"] = device.board_name;
        report["dpu_arch"] = device.dpu_arch;
        report["xmodel"] = xmodel;
        report["detector"] = detector_desc;
        report["mode"] = args.sequential ? "sequential" : "pipelined";
        report["fpga_constraints"] = constraints_json(constraints);
        report["overall_precision_at_20px"] = summary.at("all_frames").at("precision_at_20px");
        report["overall_success_auc"] = summary.at("all_frames").at("success_auc");
        report["summary"] = summary;
        report["latency"] = latency;
        report["throughput_fps"] = throughput;
        report["per_sequence"] = per_seq;
        fs::path report_path = out_dir / ("eval_results_" + tag + ".json");
        report.dump_file(report_path.string());
        std::cout << "[antiuav_fpga] results: " << report_path.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[antiuav_fpga] ERROR: " << e.what() << "\n";
        return 1;
    }
}
