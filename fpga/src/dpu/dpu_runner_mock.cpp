// Host-only stand-in for the DPU (build.sh mock). Shapes follow the
// YOLOv8DPU export at 512x640 (NHWC, 6 outputs). All outputs are filled
// with -128 (very negative logits) -> the detector reports no boxes.
// Use --det-replay to inject detections when testing off-target.
#include <cstring>

#include "dpu_runner.h"

namespace antiuav {

class MockDpuRunner : public DpuRunner {
public:
    MockDpuRunner() {
        input_info_ = {"mock_input", {1, 512, 640, 1}, 7, 512 * 640};
        const int hw[3][2] = {{64, 80}, {32, 40}, {16, 20}};
        for (int l = 0; l < 3; ++l) {
            int h = hw[l][0], w = hw[l][1];
            output_info_.push_back({"mock_box_p" + std::to_string(l + 3), {1, h, w, 64}, 4, static_cast<size_t>(h * w * 64)});
            output_info_.push_back({"mock_cls_p" + std::to_string(l + 3), {1, h, w, 1}, 4, static_cast<size_t>(h * w)});
        }
        input_.resize(input_info_.size);
        for (const auto& o : output_info_) outputs_.emplace_back(o.size, static_cast<int8_t>(-128));
    }

    const DpuTensorInfo& input_info() const override { return input_info_; }
    const std::vector<DpuTensorInfo>& output_info() const override { return output_info_; }
    int8_t* input_data() override { return input_.data(); }
    const int8_t* output_data(size_t i) const override { return outputs_.at(i).data(); }
    void run() override {}

private:
    DpuTensorInfo input_info_;
    std::vector<DpuTensorInfo> output_info_;
    std::vector<int8_t> input_;
    std::vector<std::vector<int8_t>> outputs_;
};

std::unique_ptr<DpuRunner> DpuRunner::create(const std::string&, int, int) {
    return std::make_unique<MockDpuRunner>();
}

}  // namespace antiuav
