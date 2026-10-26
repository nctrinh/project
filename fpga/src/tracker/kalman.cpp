#include "kalman.h"

#include <stdexcept>

namespace antiuav {

static void to_cxcywh(const Box& b, float& cx, float& cy, float& w, float& h) {
    w = b[2] - b[0];
    h = b[3] - b[1];
    cx = b[0] + w / 2.0f;
    cy = b[1] + h / 2.0f;
}

static Box from_cxcywh(float cx, float cy, float w, float h) {
    return {cx - w / 2.0f, cy - h / 2.0f, cx + w / 2.0f, cy + h / 2.0f};
}

void ConstantVelocityKalmanBox::init(const Box& box) {
    float cx, cy, w, h;
    to_cxcywh(box, cx, cy, w, h);
    x_[0] = cx; x_[1] = cy; x_[2] = 0; x_[3] = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) P_[i][j] = (i == j) ? 10.0f : 0.0f;
    last_w_ = w;
    last_h_ = h;
    initialized_ = true;
}

// x = F x ; P = F P F^T + Q with F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
void ConstantVelocityKalmanBox::predict_state() {
    x_[0] += x_[2];
    x_[1] += x_[3];
    float FP[4][4];
    for (int j = 0; j < 4; ++j) {
        FP[0][j] = P_[0][j] + P_[2][j];
        FP[1][j] = P_[1][j] + P_[3][j];
        FP[2][j] = P_[2][j];
        FP[3][j] = P_[3][j];
    }
    for (int i = 0; i < 4; ++i) {
        P_[i][0] = FP[i][0] + FP[i][2];
        P_[i][1] = FP[i][1] + FP[i][3];
        P_[i][2] = FP[i][2];
        P_[i][3] = FP[i][3];
        P_[i][i] += q_;
    }
}

Box ConstantVelocityKalmanBox::predict() {
    if (!initialized_) throw std::logic_error("Kalman predict() before init()");
    predict_state();
    return from_cxcywh(x_[0], x_[1], last_w_, last_h_);
}

Box ConstantVelocityKalmanBox::update(const Box& box, double measurement_noise) {
    if (!initialized_) {
        init(box);
        return box;
    }
    predict_state();
    float cx, cy, w, h;
    to_cxcywh(box, cx, cy, w, h);
    const float r = measurement_noise < 0 ? r_ : static_cast<float>(measurement_noise);

    // S = H P H^T + R (2x2), K = P H^T S^-1 (4x2)
    float s00 = P_[0][0] + r, s01 = P_[0][1], s10 = P_[1][0], s11 = P_[1][1] + r;
    float det = s00 * s11 - s01 * s10;
    float i00 = s11 / det, i01 = -s01 / det, i10 = -s10 / det, i11 = s00 / det;
    float K[4][2];
    for (int i = 0; i < 4; ++i) {
        K[i][0] = P_[i][0] * i00 + P_[i][1] * i10;
        K[i][1] = P_[i][0] * i01 + P_[i][1] * i11;
    }
    float y0 = cx - x_[0], y1 = cy - x_[1];
    for (int i = 0; i < 4; ++i) x_[i] += K[i][0] * y0 + K[i][1] * y1;
    // P = (I - K H) P
    float newP[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) newP[i][j] = P_[i][j] - (K[i][0] * P_[0][j] + K[i][1] * P_[1][j]);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) P_[i][j] = newP[i][j];

    last_w_ = w;
    last_h_ = h;
    return from_cxcywh(x_[0], x_[1], w, h);
}

}  // namespace antiuav
