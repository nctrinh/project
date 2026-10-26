#ifndef ANTIUAV_TIMER_H
#define ANTIUAV_TIMER_H

#include <chrono>

// Same macros as trackerv2/src/tracker/AItracker.h.
#define GET_MOMENT std::chrono::high_resolution_clock::now()
#define GET_DURATION(stop, start) \
    std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()

namespace antiuav {
using Clock = std::chrono::high_resolution_clock;
inline double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}
}  // namespace antiuav

#endif  // ANTIUAV_TIMER_H
