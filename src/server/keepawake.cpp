// src/server/keepawake.cpp - "prevent_sleep" (keepawake.hpp).
#include "keepawake.hpp"

#include "log.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lamina {

std::shared_ptr<KeepAwake> KeepAwake::create(bool enabled) {
    if (!enabled) return nullptr;
#ifdef _WIN32
    say("[lamina] prevent_sleep: Windows stays awake while a request is running or queued");
    return std::shared_ptr<KeepAwake>(new KeepAwake());
#else
    say("[lamina] \"prevent_sleep\" is for Windows only: ignored here");
    return nullptr;
#endif
}

KeepAwake::~KeepAwake() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void KeepAwake::acquire() {
    std::lock_guard<std::mutex> lk(mu_);
    if (closed_) return;
    ++count_;
    if (!thread_.joinable()) thread_ = std::thread([this] { loop(); });
    cv_.notify_all();
}

void KeepAwake::release() {
    std::lock_guard<std::mutex> lk(mu_);
    if (count_ > 0) --count_;
    cv_.notify_all();
}

void KeepAwake::loop() {
#ifdef _WIN32
    bool held = false;
    for (;;) {
        bool want;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return closed_ || (count_ > 0) != held; });
            if (closed_) break;
            want = count_ > 0;
        }
        SetThreadExecutionState(want ? ES_CONTINUOUS | ES_SYSTEM_REQUIRED : ES_CONTINUOUS);
        held = want;
    }
    SetThreadExecutionState(ES_CONTINUOUS);
#endif
}

}  // namespace lamina
