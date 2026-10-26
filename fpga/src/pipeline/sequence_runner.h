// Runs one sequence through the pipeline with 4 concurrent parts:
//   reader thread  -> decodes JPEG/PNG frames ahead of time
//   stage A        -> up to `pipeline_depth` frames in flight: DPU || motion core
//   stage B        -> main thread, strict frame order (stateful tracker)
//   writer thread  -> overlay + video encoding
// Results are identical to calling SotPipeline::step() frame by frame
// (--sequential runs exactly that, for checking).
#ifndef ANTIUAV_SEQUENCE_RUNNER_H
#define ANTIUAV_SEQUENCE_RUNNER_H

#include <optional>
#include <string>
#include <vector>

#include "../common/config.h"
#include "../common/json.h"
#include "../eval/dataset.h"
#include "../eval/metrics.h"
#include "sot_pipeline.h"

namespace antiuav {

struct RunOptions {
    std::string video_path;  // empty = no video
    bool sequential = false;
    bool dump_frames = false;
};

struct SequenceRunResult {
    SequenceEvalResult eval;
    std::vector<std::vector<float>> predictions_xywh;  // per image ([] = no track), JSON "res"
    std::vector<double> step_ms;                       // per evaluated frame: stage A + stage B compute time
    std::vector<double> det_ms, motion_ms, stage_b_ms;
    double wall_ms = 0;                                // whole sequence incl. I/O and video
    Json frames_dump;                                  // only with dump_frames
};

SequenceRunResult run_sequence(SotPipeline& pipeline, const SOTSequence& seq, const DeviceConfig& dev,
                               const RunOptions& opt);

Json latency_stats(const std::vector<double>& ms);

}  // namespace antiuav

#endif  // ANTIUAV_SEQUENCE_RUNNER_H
