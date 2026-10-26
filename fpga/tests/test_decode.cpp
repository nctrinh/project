// Runs decode_dpu_outputs() on raw int8 NHWC head tensors stored on disk, prints
// the boxes as JSON. Used by tests/parity/run_parity.sh to compare against
// the Python decode_outputs(nchw=False) of src1/antiuav_sot/detector/dpu_export.py.
//
//   test_decode meta.json
//   meta.json: {"reg_max":16,"nc":1,"conf":0.3,"iou":0.6,"max_det":12,
//               "levels":[{"h":64,"w":80,"stride":8,"box_fp":4,"cls_fp":4,
//                          "box_file":"box0.bin","cls_file":"cls0.bin"}, ...]}
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../src/common/json.h"
#include "../src/detector/detector.h"

using namespace antiuav;

static std::vector<int8_t> read_bin(const std::string& path, size_t n) {
    std::ifstream f(path, std::ios::binary);
    std::vector<int8_t> v(n);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("short read: " + path);
    return v;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_decode meta.json\n";
        return 2;
    }
    Json meta = Json::parse_file(argv[1]);
    std::filesystem::path dir = std::filesystem::path(argv[1]).parent_path();
    int reg_max = meta.at("reg_max").as_int(), nc = meta.at("nc").as_int();
    std::vector<std::vector<int8_t>> storage;
    std::vector<HeadLevel> levels;
    const Json& lv = meta.at("levels");
    storage.reserve(2 * lv.size());
    for (size_t i = 0; i < lv.size(); ++i) {
        HeadLevel l;
        l.height = lv[i].at("h").as_int();
        l.width = lv[i].at("w").as_int();
        l.stride = lv[i].at("stride").as_int();
        l.box_fix_point = lv[i].at("box_fp").as_int();
        l.cls_fix_point = lv[i].at("cls_fp").as_int();
        size_t cells = static_cast<size_t>(l.height) * l.width;
        storage.push_back(read_bin((dir / lv[i].at("box_file").as_string()).string(), cells * 4 * reg_max));
        l.box = storage.back().data();
        storage.push_back(read_bin((dir / lv[i].at("cls_file").as_string()).string(), cells * nc));
        l.cls = storage.back().data();
        levels.push_back(l);
    }
    std::vector<Box> boxes;
    std::vector<float> scores;
    decode_dpu_outputs(levels, reg_max, nc, static_cast<float>(meta.at("conf").as_number()),
                       static_cast<float>(meta.at("iou").as_number()), meta.at("max_det").as_int(), boxes, scores);
    Json out = Json::object();
    Json jb = Json::array(), js = Json::array();
    for (size_t i = 0; i < boxes.size(); ++i) {
        Json b = Json::array();
        for (float v : boxes[i]) b.push_back(static_cast<double>(v));
        jb.push_back(b);
        js.push_back(static_cast<double>(scores[i]));
    }
    out["boxes"] = jb;
    out["scores"] = js;
    std::cout << out.dump() << "\n";
    return 0;
}
