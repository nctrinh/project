#include "metrics.h"

#include <cmath>
#include <limits>

namespace antiuav {

static constexpr float INF = std::numeric_limits<float>::infinity();

SequenceEvalResult evaluate_sequence(const std::string& seq_name, const std::vector<std::optional<Box>>& predicted,
                                     const std::vector<std::array<float, 4>>& gt_boxes_xywh,
                                     const std::vector<bool>& gt_exist) {
    SequenceEvalResult r;
    r.seq_name = seq_name;
    r.n_frames = static_cast<int>(gt_exist.size());
    r.center_distances.assign(r.n_frames, INF);
    r.norm_distances.assign(r.n_frames, INF);
    r.ious.assign(r.n_frames, 0.0f);
    r.gt_exist = gt_exist;
    r.pred_active.assign(r.n_frames, false);
    for (int i = 0; i < r.n_frames; ++i) {
        const bool exists = gt_exist[i];
        const bool active = predicted[i].has_value();
        r.pred_active[i] = active;
        if (!exists && !active) {  // correct "no object"
            r.center_distances[i] = 0.0f;
            r.norm_distances[i] = 0.0f;
            r.ious[i] = 1.0f;
            continue;
        }
        if (!exists || !active) continue;  // FP / FN keep the worst values
        const Box gt = xywh_to_xyxy(gt_boxes_xywh[i]);
        const Box& p = *predicted[i];
        float dx = (p[0] + p[2]) / 2.0f - (gt[0] + gt[2]) / 2.0f;
        float dy = (p[1] + p[3]) / 2.0f - (gt[1] + gt[3]) / 2.0f;
        r.center_distances[i] = std::sqrt(dx * dx + dy * dy);
        float gt_w = std::max(gt[2] - gt[0], 1e-6f), gt_h = std::max(gt[3] - gt[1], 1e-6f);
        float ndx = ((p[0] + p[2]) - (gt[0] + gt[2])) / 2.0f, ndy = ((p[1] + p[3]) - (gt[1] + gt[3])) / 2.0f;
        r.norm_distances[i] = std::hypot(ndx / gt_w, ndy / gt_h);
        r.ious[i] = iou_xyxy(p, gt);
    }
    return r;
}

// ----------------------------------------------------------------------------- pooling helpers
static std::vector<float> pool(const std::vector<SequenceEvalResult>& results,
                               std::vector<float> SequenceEvalResult::*field, bool present_only) {
    std::vector<float> out;
    for (const auto& r : results) {
        const auto& v = r.*field;
        for (size_t i = 0; i < v.size(); ++i)
            if (!present_only || r.gt_exist[i]) out.push_back(v[i]);
    }
    return out;
}

static std::vector<double> linspace(double start, double stop, int n) {
    std::vector<double> t(n);
    double step = (stop - start) / (n - 1);
    for (int i = 0; i < n; ++i) t[i] = start + i * step;
    t[n - 1] = stop;
    return t;
}

static double trapezoid(const std::vector<float>& y, const std::vector<double>& x) {
    double s = 0.0;
    for (size_t i = 0; i + 1 < x.size(); ++i) s += (x[i + 1] - x[i]) * (static_cast<double>(y[i]) + y[i + 1]) / 2.0;
    return s;
}

static double precision_score(const std::vector<SequenceEvalResult>& res, double thr_px, bool present_only) {
    auto d = pool(res, &SequenceEvalResult::center_distances, present_only);
    if (d.empty()) return 0.0;
    size_t n = 0;
    for (float v : d) n += v <= static_cast<float>(thr_px);
    return static_cast<double>(n) / d.size();
}

static std::vector<float> rates_le(const std::vector<float>& d, const std::vector<double>& thr) {
    std::vector<float> r(thr.size(), 0.0f);
    if (d.empty()) return r;
    for (size_t k = 0; k < thr.size(); ++k) {
        size_t n = 0;
        for (float v : d) n += static_cast<double>(v) <= thr[k];
        r[k] = static_cast<float>(static_cast<double>(n) / d.size());
    }
    return r;
}

static std::vector<float> rates_ge(const std::vector<float>& d, const std::vector<double>& thr) {
    std::vector<float> r(thr.size(), 0.0f);
    if (d.empty()) return r;
    for (size_t k = 0; k < thr.size(); ++k) {
        size_t n = 0;
        for (float v : d) n += static_cast<double>(v) >= thr[k];
        r[k] = static_cast<float>(static_cast<double>(n) / d.size());
    }
    return r;
}

static double mean(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (float x : v) s += x;
    return s / v.size();
}

static Json curve_json(const std::vector<double>& x, const std::vector<float>& y) {
    Json j = Json::object();
    Json jx = Json::array(), jy = Json::array();
    for (double v : x) jx.push_back(v);
    for (float v : y) jy.push_back(static_cast<double>(v));
    j["thresholds"] = jx;
    j["rates"] = jy;
    return j;
}

Json summarize(const std::vector<SequenceEvalResult>& results, bool with_curves) {
    Json s = Json::object();
    long n_frames = 0;
    for (const auto& r : results) n_frames += r.n_frames;
    double msa = 0.0;
    for (const auto& r : results) msa += mean(r.ious);
    s["n_sequences"] = results.size();
    s["n_frames"] = n_frames;
    s["mSA"] = results.empty() ? 0.0 : msa / results.size();
    s["SA_pooled"] = mean(pool(results, &SequenceEvalResult::ious, false));

    const auto succ_t = linspace(0.0, 1.0, 21);
    const auto norm_t = linspace(0.0, 0.5, 51);
    Json curves = Json::object();
    for (bool present_only : {false, true}) {
        auto ious = pool(results, &SequenceEvalResult::ious, present_only);
        auto nd = pool(results, &SequenceEvalResult::norm_distances, present_only);
        auto succ = rates_ge(ious, succ_t);
        auto normp = rates_le(nd, norm_t);
        Json block = Json::object();
        block["precision_at_20px"] = precision_score(results, 20.0, present_only);
        block["norm_precision_auc"] = trapezoid(normp, norm_t) / 0.5;
        block["success_auc"] = trapezoid(succ, succ_t);
        const char* name = present_only ? "present_only" : "all_frames";
        s[name] = block;
        Json c = Json::object();
        c["success"] = curve_json(succ_t, succ);
        c["norm_precision"] = curve_json(norm_t, normp);
        curves[name] = c;
    }

    long n_present = 0, n_absent = 0, n_fn = 0, n_fp = 0;
    for (const auto& r : results)
        for (int i = 0; i < r.n_frames; ++i) {
            if (r.gt_exist[i]) {
                ++n_present;
                n_fn += !r.pred_active[i];
            } else {
                ++n_absent;
                n_fp += r.pred_active[i];
            }
        }
    s["fp_rate_on_absent_frames"] = n_absent ? static_cast<double>(n_fp) / n_absent : 0.0;
    s["fn_rate_on_present_frames"] = n_present ? static_cast<double>(n_fn) / n_present : 0.0;
    s["n_present_frames"] = n_present;
    s["n_absent_frames"] = n_absent;
    s["n_false_positive_frames"] = n_fp;
    s["n_false_negative_frames"] = n_fn;
    if (with_curves) s["curves"] = curves;
    return s;
}

}  // namespace antiuav
