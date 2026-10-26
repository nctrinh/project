// Port of src1/antiuav_sot/tracker/kalman.py: constant-velocity Kalman on the
// box CENTRE [x, y, vx, vy] (float32); w/h are kept from the last observation.
#ifndef ANTIUAV_KALMAN_H
#define ANTIUAV_KALMAN_H

#include "../common/box_ops.h"

namespace antiuav {

class ConstantVelocityKalmanBox {
public:
    ConstantVelocityKalmanBox(double process_noise = 1e-2, double measurement_noise = 1e-1)
        : q_(static_cast<float>(process_noise)), r_(static_cast<float>(measurement_noise)) {}

    bool is_initialized() const { return initialized_; }
    void init(const Box& box);
    Box predict();
    // measurement_noise < 0 -> use the default R.
    Box update(const Box& box, double measurement_noise = -1.0);
    void reset() { initialized_ = false; }

private:
    void predict_state();

    float q_, r_;
    bool initialized_ = false;
    float x_[4] = {0, 0, 0, 0};
    float P_[4][4] = {};
    float last_w_ = 0, last_h_ = 0;
};

}  // namespace antiuav

#endif  // ANTIUAV_KALMAN_H
