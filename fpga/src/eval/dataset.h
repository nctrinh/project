// Port of src1/antiuav_sot/data/dataset.py (SOTSequence / AntiUAVDataset):
// <root>/<seq_name>/{000001.jpg, ..., IR_label.json}, images sorted by
// (stem, suffix), IR_label.json with "exist" + "gt_rect" (or "gt").
#ifndef ANTIUAV_DATASET_H
#define ANTIUAV_DATASET_H

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace antiuav {

class SequenceFormatError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SequenceSample {
    std::string image_path;
    bool exist = false;
    std::array<float, 4> gt_box_xywh{0, 0, 0, 0};
    int frame_index = 0;
};

class SOTSequence {
public:
    SOTSequence(const std::string& seq_dir, const std::vector<std::string>& image_globs);
    size_t size() const { return samples_.size(); }
    const SequenceSample& operator[](size_t i) const { return samples_.at(i); }
    const std::string& name() const { return seq_name_; }

private:
    std::string seq_name_;
    std::vector<SequenceSample> samples_;
};

struct Manifest {
    std::string root_dir;
    std::vector<std::string> image_glob;
    std::vector<std::string> train_sequences, val_sequences;
    static Manifest load(const std::string& path);
};

}  // namespace antiuav

#endif  // ANTIUAV_DATASET_H
