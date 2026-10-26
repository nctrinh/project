// VART implementation of DpuRunner (board build).
#include <glog/logging.h>

#include <numeric>
#include <stdexcept>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include "dpu_runner.h"

namespace antiuav {

// Same as trackerv2/src/common/common.h::get_dpu_subgraph.
static std::vector<const xir::Subgraph*> get_dpu_subgraphs(const xir::Graph* graph) {
    auto root = graph->get_root_subgraph();
    auto children = root->children_topological_sort();
    std::vector<const xir::Subgraph*> ret;
    for (auto c : children) {
        CHECK(c->has_attr("device"));
        if (c->get_attr<std::string>("device") == "DPU") ret.emplace_back(c);
    }
    return ret;
}

static DpuTensorInfo make_info(const xir::Tensor* t) {
    DpuTensorInfo info;
    info.name = t->get_name();
    info.shape = t->get_shape();
    if (!t->has_attr("fix_point"))
        throw std::runtime_error("DPU tensor " + info.name + " has no fix_point attribute (not an int8 DPU tensor?)");
    info.fix_point = t->get_attr<int>("fix_point");
    size_t batch = info.shape.empty() ? 1 : static_cast<size_t>(info.shape[0]);
    info.size = static_cast<size_t>(t->get_element_num()) / batch;
    if (t->get_data_type().bit_width != 8)
        throw std::runtime_error("DPU tensor " + info.name + " is not 8-bit");
    return info;
}

class VartDpuRunner : public DpuRunner {
public:
    VartDpuRunner(const std::string& xmodel_path, int subgraph_index, int timeout_ms) : timeout_ms_(timeout_ms) {
        graph_ = xir::Graph::deserialize(xmodel_path);
        auto subgraphs = get_dpu_subgraphs(graph_.get());
        if (subgraphs.empty()) throw std::runtime_error("No DPU subgraph in " + xmodel_path);
        if (subgraphs.size() > 1)
            LOG(WARNING) << "xmodel has " << subgraphs.size() << " DPU subgraphs - only subgraph " << subgraph_index
                         << " is executed. Re-check Inspector/compile logs: the YOLOv8DPU export is expected to map"
                            " to ONE DPU subgraph.";
        if (subgraph_index < 0 || subgraph_index >= static_cast<int>(subgraphs.size()))
            throw std::runtime_error("dpu.subgraph_index out of range");

        runner_ = vart::Runner::create_runner(subgraphs[subgraph_index], "run");
        runner_ext_ = dynamic_cast<vart::RunnerExt*>(runner_.get());
        if (runner_ext_ == nullptr) throw std::runtime_error("DPU runner does not implement vart::RunnerExt (zero-copy)");

        inputs_ = runner_ext_->get_inputs();
        outputs_ = runner_ext_->get_outputs();
        auto in_tensors = runner_->get_input_tensors();
        auto out_tensors = runner_->get_output_tensors();
        if (in_tensors.size() != 1) throw std::runtime_error("Expected exactly 1 DPU input tensor");
        input_info_ = make_info(in_tensors[0]);
        for (auto t : out_tensors) output_info_.push_back(make_info(t));

        // Batch item 0 addresses. TODO(board): if the DPU reports batch > 1
        // (multi-batch IP), only item 0 is used here.
        input_ptr_ = reinterpret_cast<int8_t*>(inputs_[0]->data({0, 0, 0, 0}).first);
        for (size_t i = 0; i < outputs_.size(); ++i) {
            std::vector<int> idx(output_info_[i].shape.size(), 0);
            output_ptrs_.push_back(reinterpret_cast<const int8_t*>(outputs_[i]->data(idx).first));
        }
    }

    const DpuTensorInfo& input_info() const override { return input_info_; }
    const std::vector<DpuTensorInfo>& output_info() const override { return output_info_; }
    int8_t* input_data() override { return input_ptr_; }
    const int8_t* output_data(size_t i) const override { return output_ptrs_.at(i); }

    void run() override {
        for (auto* in : inputs_) {
            if (in->get_location() != vart::TensorBuffer::location_t::HOST_VIRT)
                in->sync_for_write(0, in->get_tensor()->get_data_size() / in->get_tensor()->get_shape()[0]);
        }
        auto job = runner_->execute_async(inputs_, outputs_);
        int status = runner_->wait(static_cast<int>(job.first), timeout_ms_);
        if (status != 0) throw std::runtime_error("DPU execution failed / timed out (status " + std::to_string(status) + ")");
        for (auto* out : outputs_) {
            if (out->get_location() != vart::TensorBuffer::location_t::HOST_VIRT)
                out->sync_for_read(0, out->get_tensor()->get_data_size() / out->get_tensor()->get_shape()[0]);
        }
    }

private:
    int timeout_ms_;
    std::unique_ptr<xir::Graph> graph_;
    std::unique_ptr<vart::Runner> runner_;
    vart::RunnerExt* runner_ext_ = nullptr;
    std::vector<vart::TensorBuffer*> inputs_, outputs_;
    DpuTensorInfo input_info_;
    std::vector<DpuTensorInfo> output_info_;
    int8_t* input_ptr_ = nullptr;
    std::vector<const int8_t*> output_ptrs_;
};

std::unique_ptr<DpuRunner> DpuRunner::create(const std::string& xmodel_path, int subgraph_index, int timeout_ms) {
    return std::make_unique<VartDpuRunner>(xmodel_path, subgraph_index, timeout_ms);
}

}  // namespace antiuav
