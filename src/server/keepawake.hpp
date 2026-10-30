// src/server/keepawake.hpp - the config's "prevent_sleep" (#1727, Windows only): while a request runs or waits, the
// PC does not idle into sleep (ES_SYSTEM_REQUIRED, not the display). The state belongs to the thread that set it,
// so one thread of its own sets and clears it.
#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace lamina {

class KeepAwake {
public:
    /// nullptr when off, or (said once) on a system without it
    static std::shared_ptr<KeepAwake> create(bool enabled);
    ~KeepAwake();
    void acquire();
    void release();

private:
    void loop();
    std::mutex mu_;
    std::condition_variable cv_;
    int count_ = 0;
    bool closed_ = false;
    std::thread thread_;
};

}  // namespace lamina
