#include "sequence_runner.h"

#include <algorithm>
#include <deque>
#include <future>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <thread>

#include "../common/blocking_queue.h"
#include "../common/timer.h"
#include "../eval/viz.h"

namespace antiuav {

namespace {

struct ReadFrame {
    size_t sample_idx;
    cv::Mat image;
};

struct WriteJob {
    cv::Mat raw;
    std::optional<std::array<float, 4>> gt;
    PipelineFrameOutput out;
};

Json box_json(const Box& b) {
    Json j = Json::array();
    for (float v : b) j.push_back(static_cast<double>(v));
    return j;
}

Json frame_dump(const PipelineFrameOutput& o) {
    Json j = Json::object();
    j["frame_index"] = o.frame_index;
    Json tb = Json::array(), ts = Json::array(), ta = Json::array(), ids = Json::array();
    for (size_t s = 0; s < o.track_boxes.size(); ++s) {
        tb.push_back(box_json(o.track_boxes[s]));
        ts.push_back(box_json(o.track_boxes_smoothed[s]));
        ta.push_back(static_cast<bool>(o.track_active[s]));
        ids.push_back(o.track_ids[s]);
    }
    j["track_boxes"] = tb;
    j["track_boxes_smoothed"] = ts;
    j["track_active"] = ta;
    j["track_ids"] = ids;
    Json mb = Json::array(), mv = Json::array(), ms = Json::array();
    for (size_t i = 0; i < o.motion.boxes.size(); ++i) {
        mb.push_back(box_json(o.motion.boxes[i]));
        mv.push_back(static_cast<bool>(o.motion.valid_mask[i]));
        ms.push_back(static_cast<double>(o.motion.strengths[i]));
    }
    j["motion_boxes"] = mb;
    j["motion_valid_mask"] = mv;
    j["motion_strengths"] = ms;
    Json tiers = Json::array();
    for (int t : o.fsm_result.tier_for_candidate) tiers.push_back(t);
    j["tier_for_candidate"] = tiers;
    return j;
}

}  // namespace

Json latency_stats(const std::vector<double>& ms) {
    Json j = Json::object();
    j["n_frames_timed"] = ms.size();
    if (ms.empty()) return j;
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double v : s) sum += v;
    double mean = sum / s.size();
    auto pct = [&](double q) {  // numpy linear percentile
        double pos = q / 100.0 * (s.size() - 1);
        size_t lo = static_cast<size_t>(pos), hi = std::min(lo + 1, s.size() - 1);
        return s[lo] + (s[hi] - s[lo]) * (pos - lo);
    };
    j["mean_ms"] = mean;
    j["median_ms"] = pct(50);
    j["p95_ms"] = pct(95);
    j["fps_from_mean"] = mean > 0 ? 1000.0 / mean : 0.0;
    return j;
}

SequenceRunResult run_sequence(SotPipeline& pipeline, const SOTSequence& seq, const DeviceConfig& dev,
                               const RunOptions& opt) {
    auto t_start = Clock::now();
    pipeline.reset();
    SequenceRunResult res;
    res.predictions_xywh.assign(seq.size(), {});
    if (opt.dump_frames) res.frames_dump = Json::array();

    std::vector<std::optional<Box>> predicted;
    std::vector<std::array<float, 4>> gt_boxes;
    std::vector<bool> gt_exist;

    // ---- reader ----
    BlockingQueue<ReadFrame> read_q(static_cast<size_t>(dev.reader_queue));
    std::thread reader([&] {
        for (size_t i = 0; i < seq.size(); ++i) {
            cv::Mat img = cv::imread(seq[i].image_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "[run_sequence] WARNING: cannot read " << seq[i].image_path << ", frame skipped\n";
                continue;
            }
            if (!read_q.push({i, img})) break;  // consumer aborted
        }
        read_q.close();
    });

    // ---- writer ----
    BlockingQueue<WriteJob> write_q(static_cast<size_t>(dev.writer_queue));
    std::thread writer;
    std::exception_ptr writer_error;
    if (!opt.video_path.empty()) {
        writer = std::thread([&] {
            SequenceVideoWriter vw(opt.video_path, dev.video_fps, dev.video_fourcc);
            while (auto job = write_q.pop()) {
                try {
                    if (!writer_error) vw.write(render_frame(job->raw, job->gt, job->out));
                } catch (...) {
                    writer_error = std::current_exception();
                }
            }
            vw.close();
        });
    }

    auto consume = [&](size_t sample_idx, PipelineFrameOutput&& out) {
        const SequenceSample& s = seq[sample_idx];
        // SOT: slot 0 only (same as eval_tracker_motion.py)
        std::optional<Box> pred;
        if (out.track_active[0] && out.track_boxes_smoothed[0][0] != INVALID_BOX) pred = out.track_boxes_smoothed[0];
        predicted.push_back(pred);
        gt_boxes.push_back(s.gt_box_xywh);
        gt_exist.push_back(s.exist);
        if (pred) {
            auto xywh = xyxy_to_xywh(*pred);
            res.predictions_xywh[sample_idx] = {xywh[0], xywh[1], xywh[2], xywh[3]};
        }
        res.step_ms.push_back(out.stage_a_ms + out.stage_b_ms);
        res.det_ms.push_back(out.det_ms);
        res.motion_ms.push_back(out.motion_ms);
        res.stage_b_ms.push_back(out.stage_b_ms);
        if (opt.dump_frames) res.frames_dump.push_back(frame_dump(out));
        return std::move(out);
    };

    auto handle_output = [&](size_t sample_idx, const cv::Mat& raw, PipelineFrameOutput&& out) {
        PipelineFrameOutput o = consume(sample_idx, std::move(out));
        if (!opt.video_path.empty()) {
            std::optional<std::array<float, 4>> gt;
            if (seq[sample_idx].exist) gt = seq[sample_idx].gt_box_xywh;
            write_q.push({raw, gt, std::move(o)});
        }
    };

    try {
        if (opt.sequential) {
            while (auto f = read_q.pop()) {
                PipelineFrameOutput out = pipeline.step(f->image, seq[f->sample_idx].frame_index);
                handle_output(f->sample_idx, f->image, std::move(out));
            }
        } else {
            FrameBuffer frame_buffer(pipeline.constraints().motion_frame_lag);
            struct Pending {
                size_t sample_idx;
                cv::Mat raw;
                std::future<StageAResult> fut;
            };
            std::deque<Pending> pending;
            bool reader_done = false;
            while (true) {
                while (!reader_done && static_cast<int>(pending.size()) < dev.pipeline_depth) {
                    auto f = read_q.pop();
                    if (!f) {
                        reader_done = true;
                        break;
                    }
                    cv::Mat prev = frame_buffer.get_previous();
                    frame_buffer.update(f->image);
                    int frame_index = seq[f->sample_idx].frame_index;
                    cv::Mat raw = f->image;
                    pending.push_back({f->sample_idx, raw,
                                       std::async(std::launch::async, [&pipeline, raw, prev, frame_index] {
                                           return pipeline.run_parallel_stage(raw, prev, frame_index);
                                       })});
                }
                if (pending.empty()) break;
                Pending p = std::move(pending.front());
                pending.pop_front();
                PipelineFrameOutput out = pipeline.finish_step(p.fut.get());
                handle_output(p.sample_idx, p.raw, std::move(out));
            }
        }
    } catch (...) {
        read_q.close();
        reader.join();
        write_q.close();
        if (writer.joinable()) writer.join();
        throw;
    }

    reader.join();
    write_q.close();
    if (writer.joinable()) writer.join();
    if (writer_error) std::rethrow_exception(writer_error);

    res.eval = evaluate_sequence(seq.name(), predicted, gt_boxes, gt_exist);
    res.wall_ms = elapsed_ms(t_start, Clock::now());
    return res;
}

}  // namespace antiuav
