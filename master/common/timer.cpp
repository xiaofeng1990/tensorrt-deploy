#include "timer.h"

#include <chrono>
#include <iostream>

namespace xffw {

void Timer::Start() { start_time_ = std::chrono::high_resolution_clock::now(); }
double Timer::Stop()
{
    stop_time_ = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_);
    std::chrono::duration<double> duration = stop_time_ - start_time_;
    return duration.count();
}

void Timer::Tick() { last_tick_time_ = std::chrono::high_resolution_clock::now(); }

double Timer::Tock()
{
    last_tock_time_ = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(last_tock_time_ - last_tick_time_);
    std::chrono::duration<double> duration = last_tock_time_ - last_tick_time_;
    return duration.count();
}

} // namespace xffw
