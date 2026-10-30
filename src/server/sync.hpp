// src/server/sync.hpp - what threads of the server hand each other: a request's cancellation, and lines (from the
// engine thread, from a child process's output); the steady clock their waits are timed by.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>

namespace lamina {

/// seconds on the steady clock (time.monotonic())
inline double mono() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// the seconds since t0 on the steady clock
inline double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/// a request's cancellation: set by the HTTP side (the client left) or the server (shutdown, unload)
class Cancel {
public:
    void set() {
        flag_.store(true);
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    }
    bool is_set() const { return flag_.load(); }
    /// sleeps up to `seconds`, waking when set; true when set
    bool wait(double seconds) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait_for(lk, std::chrono::duration<double>(seconds), [&] { return flag_.load(); });
        return flag_.load();
    }
    std::string request_id;
    /// with batch slots (no turn to wait for): when the wait for the engine ends (a steady-clock second), set by
    /// Service::run and cleared once the request is sent
    std::optional<double> queue_deadline;
    std::function<void()> queue_started;          ///< the request is sent: its row reads "reading"
    std::function<void(int)> replica_selected;    ///< the replica it went to

private:
    std::atomic<bool> flag_{false};
    std::mutex mu_;
    std::condition_variable cv_;
};

/// lines from one thread to another
class LineChannel {
public:
    void push(std::string line) {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(line));
        cv_.notify_all();
    }
    /// the next line within `timeout_s` (< 0: no limit): 1 a line, 0 none in time, -1 the channel has ended
    int pop(std::string& out, double timeout_s) {
        std::unique_lock<std::mutex> lk(mu_);
        auto ready = [&] { return !q_.empty() || closed_; };
        if (timeout_s < 0) {
            cv_.wait(lk, ready);
        } else if (!cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), ready)) {
            return 0;
        }
        if (q_.empty()) return -1;
        out = std::move(q_.front());
        q_.pop_front();
        return 1;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_.notify_all();
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(mu_);
        return closed_ && q_.empty();
    }
    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    bool closed_ = false;
};

}  // namespace lamina
