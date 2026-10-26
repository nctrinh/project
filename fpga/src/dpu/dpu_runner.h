// DPU wrapper - replaces trackerv2/src/DPU/dpu.{h,cpp} (VART_DPU).
//
// Differences from trackerv2 (see fpga/README notes in the summary):
//   * zero-copy: the runner's OWN tensor buffers are used (vart::RunnerExt::
//     get_inputs()/get_outputs() + sync_for_write/sync_for_read) instead of
//     user float buffers wrapped in CpuFlatTensorBuffer -> no per-frame copy
//     and no float<->int8 conversion inside VART.
//   * tensors stay int8; fix_point is read from the xmodel and exposed so
//     the detector quantises the input / dequantises only what it needs.
//
// Two implementations: dpu_runner_vart.cpp (board) and dpu_runner_mock.cpp
// (host build, -DDPU_MOCK: returns all-zero outputs - used only to compile
// and run the non-DPU parts of the pipeline off-target).
#ifndef ANTIUAV_DPU_RUNNER_H
#define ANTIUAV_DPU_RUNNER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace antiuav {

struct DpuTensorInfo {
    std::string name;
    std::vector<int> shape;  // as reported by xir (NHWC for 4-D tensors)
    int fix_point = 0;       // real = int8 * 2^-fix_point
    size_t size = 0;         // elements of ONE batch item
};

class DpuRunner {
public:
    virtual ~DpuRunner() = default;

    virtual const DpuTensorInfo& input_info() const = 0;
    virtual const std::vector<DpuTensorInfo>& output_info() const = 0;

    // Pointer to batch item 0 of the input buffer (write the quantised frame here).
    virtual int8_t* input_data() = 0;
    // Flush input, execute, wait, invalidate outputs.
    virtual void run() = 0;
    // Pointer to batch item 0 of output i (valid until the next run()).
    virtual const int8_t* output_data(size_t i) const = 0;

    // xmodel_path, index into the DPU subgraphs, wait() timeout in ms (-1 = forever).
    static std::unique_ptr<DpuRunner> create(const std::string& xmodel_path, int subgraph_index, int timeout_ms);
};

}  // namespace antiuav

#endif  // ANTIUAV_DPU_RUNNER_H
