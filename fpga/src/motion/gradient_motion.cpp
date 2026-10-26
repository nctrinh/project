#include "gradient_motion.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace antiuav {

MotionParams MotionParams::from(const FpgaConstraints& c) {
    MotionParams p;
    p.model = c.motion_model;
    p.irls_iters = c.motion_irls_iters;
    p.huber_delta = c.motion_huber_delta;
    p.pyramid_levels = c.motion_pyramid_levels;
    p.thresh_factor = c.motion_thresh_factor;
    p.min_area = c.motion_min_area;
    return p;
}

// ----------------------------------------------------------------------------- IRLS
// numba_irls_affine: Ix/Iy/It are computed ONCE from the image warped with
// `params`, each IRLS iteration re-solves the increment with new Huber weights.
static void irls_affine(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It, double params[6], int irls_iters,
                        double huber_delta, int step = 4) {
    const int H = Ix.rows, W = Ix.cols;
    double delta[6] = {0, 0, 0, 0, 0, 0};
    const int hs = (H + step - 1) / step, ws = (W + step - 1) / step;
    std::vector<double> weights(static_cast<size_t>(hs) * ws, 1.0);  // only sampled pixels are ever used

    for (int iter = 0; iter < irls_iters; ++iter) {
        double AtA[6][6] = {};
        double Atb[6] = {};
        for (int y = 0, yi = 0; y < H; y += step, ++yi) {
            const float* px = Ix.ptr<float>(y);
            const float* py = Iy.ptr<float>(y);
            const float* pt = It.ptr<float>(y);
            for (int x = 0, xi = 0; x < W; x += step, ++xi) {
                const double w = weights[static_cast<size_t>(yi) * ws + xi];
                const double ix = px[x], iy = py[x], it = pt[x];
                const double a[6] = {ix * x, ix * y, ix, iy * x, iy * y, iy};
                for (int r = 0; r < 6; ++r) {
                    Atb[r] -= w * a[r] * it;
                    for (int c = r; c < 6; ++c) AtA[r][c] += w * a[r] * a[c];
                }
            }
        }
        for (int r = 0; r < 6; ++r)
            for (int c = r + 1; c < 6; ++c) AtA[c][r] = AtA[r][c];
        for (int i = 0; i < 6; ++i) AtA[i][i] += 1e-6;

        cv::Mat A(6, 6, CV_64F, &AtA[0][0]), b(6, 1, CV_64F, Atb), d;
        cv::solve(A, b, d, cv::DECOMP_LU);
        for (int i = 0; i < 6; ++i) delta[i] = d.at<double>(i);

        for (int y = 0, yi = 0; y < H; y += step, ++yi) {
            const float* px = Ix.ptr<float>(y);
            const float* py = Iy.ptr<float>(y);
            const float* pt = It.ptr<float>(y);
            for (int x = 0, xi = 0; x < W; x += step, ++xi) {
                const double ix = px[x], iy = py[x], it = pt[x];
                const double resid = (ix * x * delta[0] + ix * y * delta[1] + ix * delta[2] + iy * x * delta[3] +
                                      iy * y * delta[4] + iy * delta[5]) + it;
                const double abs_r = std::fabs(resid);
                weights[static_cast<size_t>(yi) * ws + xi] = abs_r <= huber_delta ? 1.0 : huber_delta / (abs_r + 1e-8);
            }
        }
    }
    for (int i = 0; i < 6; ++i) params[i] += delta[i];
}

// numba_irls_translation (every pixel, no sub-sampling).
static void irls_translation(const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat& It, double params[2], int irls_iters,
                             double huber_delta) {
    const int H = Ix.rows, W = Ix.cols;
    double delta[2] = {0, 0};
    std::vector<double> weights(static_cast<size_t>(H) * W, 1.0);
    for (int iter = 0; iter < irls_iters; ++iter) {
        double a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (int y = 0; y < H; ++y) {
            const float* px = Ix.ptr<float>(y);
            const float* py = Iy.ptr<float>(y);
            const float* pt = It.ptr<float>(y);
            const double* pw = &weights[static_cast<size_t>(y) * W];
            for (int x = 0; x < W; ++x) {
                const double w = pw[x], ix = px[x], iy = py[x], it = pt[x];
                b0 -= w * ix * it;
                b1 -= w * iy * it;
                a00 += w * ix * ix;
                a01 += w * ix * iy;
                a11 += w * iy * iy;
            }
        }
        a00 += 1e-6;
        a11 += 1e-6;
        double AtA[2][2] = {{a00, a01}, {a01, a11}}, Atb[2] = {b0, b1};
        cv::Mat A(2, 2, CV_64F, &AtA[0][0]), b(2, 1, CV_64F, Atb), d;
        cv::solve(A, b, d, cv::DECOMP_LU);
        delta[0] = d.at<double>(0);
        delta[1] = d.at<double>(1);
        for (int y = 0; y < H; ++y) {
            const float* px = Ix.ptr<float>(y);
            const float* py = Iy.ptr<float>(y);
            const float* pt = It.ptr<float>(y);
            double* pw = &weights[static_cast<size_t>(y) * W];
            for (int x = 0; x < W; ++x) {
                const double resid = (px[x] * delta[0] + py[x] * delta[1]) + static_cast<double>(pt[x]);
                const double abs_r = std::fabs(resid);
                pw[x] = abs_r <= huber_delta ? 1.0 : huber_delta / (abs_r + 1e-8);
            }
        }
    }
    params[0] += delta[0];
    params[1] += delta[1];
}

static cv::Mat warp_matrix(const double* params, bool translation) {
    cv::Mat M(2, 3, CV_32F);
    if (translation) {
        M.at<float>(0, 0) = 1; M.at<float>(0, 1) = 0; M.at<float>(0, 2) = static_cast<float>(params[0]);
        M.at<float>(1, 0) = 0; M.at<float>(1, 1) = 1; M.at<float>(1, 2) = static_cast<float>(params[1]);
    } else {
        M.at<float>(0, 0) = static_cast<float>(1 + params[0]);
        M.at<float>(0, 1) = static_cast<float>(params[1]);
        M.at<float>(0, 2) = static_cast<float>(params[2]);
        M.at<float>(1, 0) = static_cast<float>(params[3]);
        M.at<float>(1, 1) = static_cast<float>(1 + params[4]);
        M.at<float>(1, 2) = static_cast<float>(params[5]);
    }
    return M;
}

// ----------------------------------------------------------------------------- core
MotionCoreResult motion_core(const cv::Mat& frame_t_minus_lag, const cv::Mat& frame_t, const MotionParams& p) {
    CV_Assert(frame_t_minus_lag.type() == CV_8UC1 && frame_t.type() == CV_8UC1 &&
              frame_t_minus_lag.size() == frame_t.size());
    const bool translation = p.model == "translation";

    // gradient_global_motion_fast
    cv::Mat f1, f2;
    frame_t_minus_lag.convertTo(f1, CV_32F);
    frame_t.convertTo(f2, CV_32F);
    std::vector<cv::Mat> pyr1{f1}, pyr2{f2};
    for (int i = 0; i < p.pyramid_levels; ++i) {
        cv::Mat a, b;
        cv::resize(pyr1.back(), a, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        cv::resize(pyr2.back(), b, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        pyr1.push_back(a);
        pyr2.push_back(b);
    }
    std::reverse(pyr1.begin(), pyr1.end());
    std::reverse(pyr2.begin(), pyr2.end());

    double params[6] = {0, 0, 0, 0, 0, 0};
    for (size_t level = 0; level < pyr1.size(); ++level) {
        const cv::Mat& a = pyr1[level];
        const cv::Mat& b = pyr2[level];
        if (level > 0) {
            if (translation) {
                params[0] *= 2.0;
                params[1] *= 2.0;
            } else {
                params[2] *= 2.0;
                params[5] *= 2.0;
            }
        }
        cv::Mat a_warp, Ix, Iy, It;
        cv::warpAffine(a, a_warp, warp_matrix(params, translation), a.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::Sobel(a_warp, Ix, CV_32F, 1, 0, 3, 1.0 / 8.0);
        cv::Sobel(a_warp, Iy, CV_32F, 0, 1, 3, 1.0 / 8.0);
        cv::subtract(b, a_warp, It);
        if (translation) irls_translation(Ix, Iy, It, params, p.irls_iters, p.huber_delta);
        else irls_affine(Ix, Iy, It, params, p.irls_iters, p.huber_delta);
    }
    cv::Mat warped;
    cv::warpAffine(f1, warped, warp_matrix(params, translation), f1.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    // (the full-frame flow field (u, v) of the Python version is never used -> not computed)

    // motion_candidates_gradient: positive residual only (UAV is hotter than background)
    MotionCoreResult r;
    std::copy(params, params + 6, r.params.begin());
    cv::Mat diff;
    cv::subtract(f2, warped, diff);
    cv::max(diff, 0.0, diff);
    cv::blur(diff, r.residual, cv::Size(5, 5));

    cv::Scalar mean_val, std_val;
    cv::meanStdDev(r.residual, mean_val, std_val);
    double thresh = std::max(mean_val[0] + p.thresh_factor * std_val[0], 1e-3);

    cv::Mat mask_f, mask;
    cv::threshold(r.residual, mask_f, thresh, 255, cv::THRESH_BINARY);
    mask_f.convertTo(mask, CV_8U);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 4);
    for (int i = 1; i < num_labels; ++i) {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < p.min_area) continue;
        double x0 = stats.at<int>(i, cv::CC_STAT_LEFT), y0 = stats.at<int>(i, cv::CC_STAT_TOP);
        double w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        r.boxes_xyxy.push_back({x0, y0, x0 + w, y0 + h});
    }
    return r;
}

// ----------------------------------------------------------------------------- filter (gradient_candidates.py)
using BoxD = std::array<double, 4>;

static double boxes_gap(const BoxD& a, const BoxD& b) {
    double dx = std::max({b[0] - a[2], a[0] - b[2], 0.0});
    double dy = std::max({b[1] - a[3], a[1] - b[3], 0.0});
    return std::hypot(dx, dy);
}
static double box_diag(const BoxD& b) { return std::hypot(b[2] - b[0], b[3] - b[1]); }
static double box_area(const BoxD& b) { return std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]); }
static double aspect_ratio(double w, double h) {
    double s = w <= h ? w : h, l = w <= h ? h : w;
    if (s <= 1e-6) return std::numeric_limits<double>::infinity();
    return l / s;
}

static std::vector<BoxD> cluster_and_merge_boxes(const std::vector<BoxD>& boxes, double merge_gap_ratio,
                                                 double min_fill_ratio, double max_aspect_ratio) {
    const int n = static_cast<int>(boxes.size());
    if (n <= 1) return boxes;
    std::vector<double> areas(n);
    for (int i = 0; i < n; ++i) areas[i] = box_area(boxes[i]);
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<std::vector<int>> members(n);
    for (int i = 0; i < n; ++i) members[i] = {i};
    auto find = [&](int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    auto union_of = [&](const std::vector<int>& idxs) {
        BoxD u = boxes[idxs[0]];
        for (int k : idxs) {
            u[0] = std::min(u[0], boxes[k][0]);
            u[1] = std::min(u[1], boxes[k][1]);
            u[2] = std::max(u[2], boxes[k][2]);
            u[3] = std::max(u[3], boxes[k][3]);
        }
        return u;
    };

    struct Edge { double gap; int i, j; };
    std::vector<Edge> edges;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double gap = boxes_gap(boxes[i], boxes[j]);
            double thr = merge_gap_ratio * std::min(box_diag(boxes[i]), box_diag(boxes[j]));
            if (gap < thr) edges.push_back({gap, i, j});
        }
    std::stable_sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return a.gap < b.gap; });

    for (const Edge& e : edges) {
        int ri = find(e.i), rj = find(e.j);
        if (ri == rj) continue;
        std::vector<int> merged = members[ri];
        merged.insert(merged.end(), members[rj].begin(), members[rj].end());
        BoxD mb = union_of(merged);
        double merged_area = box_area(mb);
        if (merged_area <= 0) continue;
        double sum_area = 0.0;
        for (int k : merged) sum_area += areas[k];
        double fill = sum_area / merged_area;
        if (fill < min_fill_ratio || aspect_ratio(mb[2] - mb[0], mb[3] - mb[1]) > max_aspect_ratio) continue;
        parent[ri] = rj;
        members[rj] = std::move(merged);
    }

    // dict insertion order of the cluster roots, members in ascending index
    std::vector<int> root_order;
    std::vector<std::vector<int>> clusters;
    for (int i = 0; i < n; ++i) {
        int r = find(i);
        auto it = std::find(root_order.begin(), root_order.end(), r);
        if (it == root_order.end()) {
            root_order.push_back(r);
            clusters.push_back({i});
        } else {
            clusters[it - root_order.begin()].push_back(i);
        }
    }
    std::vector<BoxD> out;
    for (const auto& c : clusters) out.push_back(union_of(c));
    return out;
}

static double ring_mean(const cv::Mat& residual, int ix0, int iy0, int ix1, int iy1, double inner_sum) {
    int margin = std::max(ix1 - ix0, iy1 - iy0);
    int ox0 = std::max(ix0 - margin, 0), oy0 = std::max(iy0 - margin, 0);
    int ox1 = std::min(ix1 + margin, residual.cols), oy1 = std::min(iy1 + margin, residual.rows);
    double ring_area = static_cast<double>((ox1 - ox0) * (oy1 - oy0) - (ix1 - ix0) * (iy1 - iy0));
    if (ring_area <= 0) return 0.0;
    double outer = cv::sum(residual(cv::Rect(ox0, oy0, ox1 - ox0, oy1 - oy0)))[0];
    return (outer - inner_sum) / ring_area;
}

static bool overlaps_any_priority_box(const BoxD& box, const std::vector<Box>& priority_boxes, double threshold) {
    double b_area = std::max(0.0, box[2] - box[0]) * std::max(0.0, box[3] - box[1]);
    if (b_area <= 0) return false;
    for (const Box& p : priority_boxes) {
        if (p[0] == INVALID_BOX) continue;
        double ix0 = std::max(box[0], static_cast<double>(p[0])), iy0 = std::max(box[1], static_cast<double>(p[1]));
        double ix1 = std::min(box[2], static_cast<double>(p[2])), iy1 = std::min(box[3], static_cast<double>(p[3]));
        double inter = std::max(0.0, ix1 - ix0) * std::max(0.0, iy1 - iy0);
        if (inter <= 0) continue;
        double p_area = std::max(0.0, static_cast<double>(p[2] - p[0])) * std::max(0.0, static_cast<double>(p[3] - p[1]));
        double uni = b_area + p_area - inter;
        double iou = uni > 0 ? inter / uni : 0.0;
        double mn = std::min(b_area, p_area);
        double containment = mn > 0 ? inter / mn : 0.0;
        if (std::max(iou, containment) >= threshold) return true;
    }
    return false;
}

MotionCandidates empty_motion_candidates(int k) {
    MotionCandidates m;
    m.boxes = make_empty_boxes(k);
    m.valid_mask.assign(k, false);
    m.strengths.assign(k, 0.0f);
    return m;
}

MotionCandidates filter_motion_candidates(const MotionCoreResult& core, int max_motion_candidates, int min_box_size,
                                          int max_box_size, double merge_gap_ratio, double max_aspect_ratio,
                                          double merge_min_fill_ratio, const std::vector<Box>& priority_boxes,
                                          double priority_iou_threshold, const std::string& strength_metric,
                                          double min_strength) {
    // Issue 7: aspect filter BEFORE clustering
    std::vector<BoxD> shape_filtered;
    for (const auto& b : core.boxes_xyxy)
        if (aspect_ratio(b[2] - b[0], b[3] - b[1]) <= max_aspect_ratio) shape_filtered.push_back(b);

    std::vector<BoxD> merged = cluster_and_merge_boxes(shape_filtered, merge_gap_ratio, merge_min_fill_ratio,
                                                       max_aspect_ratio);

    const cv::Mat& residual = core.residual;
    std::vector<std::pair<BoxD, double>> filtered;
    for (const BoxD& box : merged) {
        double w = box[2] - box[0], h = box[3] - box[1];
        if (!(min_box_size <= w && w <= max_box_size && min_box_size <= h && h <= max_box_size)) continue;
        if (aspect_ratio(w, h) > max_aspect_ratio) continue;
        int ix0 = static_cast<int>(std::max(box[0], 0.0)), iy0 = static_cast<int>(std::max(box[1], 0.0));
        int ix1 = static_cast<int>(std::min(box[2], static_cast<double>(residual.cols)));
        int iy1 = static_cast<int>(std::min(box[3], static_cast<double>(residual.rows)));
        double s_sum = 0.0, s_mean = 0.0;
        if (ix1 > ix0 && iy1 > iy0) {
            s_sum = cv::sum(residual(cv::Rect(ix0, iy0, ix1 - ix0, iy1 - iy0)))[0];
            s_mean = s_sum / static_cast<double>((ix1 - ix0) * (iy1 - iy0));
        }
        double strength;
        if (strength_metric == "mean") strength = s_mean;
        else if (strength_metric == "contrast") strength = s_mean - ring_mean(residual, ix0, iy0, ix1, iy1, s_sum);
        else strength = s_sum;
        filtered.emplace_back(box, strength);
    }

    // Issue 6: priority anchoring, min_strength (anchored boxes exempt), top-K
    std::vector<char> anchored(filtered.size(), 0);
    if (!priority_boxes.empty())
        for (size_t i = 0; i < filtered.size(); ++i)
            anchored[i] = overlaps_any_priority_box(filtered[i].first, priority_boxes, priority_iou_threshold);

    std::vector<size_t> kept;
    for (size_t i = 0; i < filtered.size(); ++i)
        if (anchored[i] || filtered[i].second >= min_strength) kept.push_back(i);
    // sorted(kept, key=(anchored, strength), reverse=True) - Python's sort is stable
    std::stable_sort(kept.begin(), kept.end(), [&](size_t a, size_t b) {
        if (anchored[a] != anchored[b]) return anchored[a] > anchored[b];
        return filtered[a].second > filtered[b].second;
    });
    if (static_cast<int>(kept.size()) > max_motion_candidates) kept.resize(max_motion_candidates);

    MotionCandidates out = empty_motion_candidates(max_motion_candidates);
    for (size_t i = 0; i < kept.size(); ++i) {
        const BoxD& b = filtered[kept[i]].first;
        out.boxes[i] = {static_cast<float>(b[0]), static_cast<float>(b[1]), static_cast<float>(b[2]),
                        static_cast<float>(b[3])};
        out.valid_mask[i] = true;
        out.strengths[i] = static_cast<float>(filtered[kept[i]].second);
    }
    return out;
}

}  // namespace antiuav
