#include "dataset.h"

#include <fnmatch.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <set>

#include "../common/json.h"

namespace fs = std::filesystem;

namespace antiuav {

SOTSequence::SOTSequence(const std::string& seq_dir, const std::vector<std::string>& image_globs) {
    fs::path dir(seq_dir);
    seq_name_ = dir.filename().string();
    if (seq_name_.empty()) seq_name_ = dir.parent_path().filename().string();  // trailing slash
    fs::path label_path = dir / "IR_label.json";
    if (!fs::is_directory(dir)) throw std::runtime_error("Sequence directory not found: " + seq_dir);
    if (!fs::is_regular_file(label_path)) throw std::runtime_error("IR_label.json not found in " + seq_dir);

    // glob(pattern) for each pattern (case-sensitive, like pathlib on Linux)
    std::vector<fs::path> paths;
    std::set<std::string> matched_patterns;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string fname = entry.path().filename().string();
        for (const auto& pat : image_globs) {
            if (fnmatch(pat.c_str(), fname.c_str(), 0) == 0) {
                paths.push_back(entry.path());
                matched_patterns.insert(pat);
                break;
            }
        }
    }
    if (matched_patterns.size() > 1)
        std::cerr << "[SOTSequence] WARNING: sequence '" << seq_name_
                  << "' mixes several image formats - sorted together by file stem.\n";
    std::sort(paths.begin(), paths.end(), [](const fs::path& a, const fs::path& b) {
        auto sa = a.stem().string(), sb = b.stem().string();
        if (sa != sb) return sa < sb;
        return a.extension().string() < b.extension().string();
    });
    if (paths.empty()) throw SequenceFormatError("No image matching the glob patterns in " + seq_dir);

    Json label = Json::parse_file(label_path.string());
    if (!label.contains("exist") || (!label.contains("gt_rect") && !label.contains("gt")))
        throw SequenceFormatError(label_path.string() + " is missing 'exist' or 'gt_rect'/'gt'");
    const Json& exist = label.at("exist");
    const Json& rects = label.contains("gt_rect") ? label.at("gt_rect") : label.at("gt");
    if (!(exist.size() == rects.size() && rects.size() == paths.size()))
        throw SequenceFormatError("Count mismatch in " + seq_name_ + ": images=" + std::to_string(paths.size()) +
                                  ", len(exist)=" + std::to_string(exist.size()) +
                                  ", len(gt_rect)=" + std::to_string(rects.size()));
    for (size_t i = 0; i < paths.size(); ++i) {
        SequenceSample s;
        s.image_path = paths[i].string();
        s.exist = exist[i].as_number() != 0;
        s.frame_index = static_cast<int>(i);
        const Json& r = rects[i];
        if (r.is_array() && r.size() == 4) {
            for (int k = 0; k < 4; ++k) s.gt_box_xywh[k] = static_cast<float>(r[k].as_number());
        } else if (s.exist) {
            throw SequenceFormatError(seq_name_ + " frame " + std::to_string(i) + ": exist=1 but gt_rect is invalid");
        }
        samples_.push_back(s);
    }
}

static std::vector<std::string> string_list(const Json& j) {
    std::vector<std::string> out;
    for (size_t i = 0; i < j.size(); ++i) out.push_back(j[i].as_string());
    return out;
}

Manifest Manifest::load(const std::string& path) {
    Json j = Json::parse_file(path);
    Manifest m;
    m.root_dir = j.at("root_dir").as_string();
    if (j.contains("image_glob")) {
        const Json& g = j.at("image_glob");
        m.image_glob = g.is_string() ? std::vector<std::string>{g.as_string()} : string_list(g);
    } else {
        m.image_glob = {"*.jpg", "*.jpeg", "*.png"};
    }
    if (j.contains("train_sequences")) m.train_sequences = string_list(j.at("train_sequences"));
    if (j.contains("val_sequences")) m.val_sequences = string_list(j.at("val_sequences"));
    return m;
}

}  // namespace antiuav
