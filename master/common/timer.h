#ifndef XFLOG_TIMER_H_
#define XFLOG_TIMER_H_

#include <chrono>

namespace xffw {

class Timer final {
  public:
    Timer() = default;
    virtual ~Timer() = default;

    void Start();
    /**
     * @brief the duration from stop to start
     *
     * @return duration with unit: second(s)
     */
    double Stop();

    void Tick();
    /**
     * @brief the duration from tock to tick
     *
     * @return duration with unit: second(s)
     */
    double Tock();

  private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stop_time_;

    std::chrono::high_resolution_clock::time_point last_tick_time_;
    std::chrono::high_resolution_clock::time_point last_tock_time_;
}; // class Timer

} // namespace xffw

#endif // XTLOG_TIMER_H_
