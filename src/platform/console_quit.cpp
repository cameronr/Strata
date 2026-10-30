// src/platform/console_quit.cpp - see include/lamina/platform/console_quit.hpp.
#include "lamina/platform/console_quit.hpp"

#include <mutex>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace lamina::platform {

#if defined(_WIN32)
namespace {
std::mutex mu;
std::function<void()> quit_fn;
bool installed = false;

BOOL WINAPI on_ctrl(DWORD type) {
    if (type != CTRL_C_EVENT && type != CTRL_BREAK_EVENT) return FALSE;
    std::lock_guard<std::mutex> lk(mu);
    if (!quit_fn) return FALSE;   // the default: the process ends
    quit_fn();
    return TRUE;
}
}  // namespace

void set_console_quit(std::function<void()> quit) {
    std::lock_guard<std::mutex> lk(mu);
    quit_fn = std::move(quit);
    if (quit_fn && !installed) installed = SetConsoleCtrlHandler(on_ctrl, TRUE) != 0;
}
#else
void set_console_quit(std::function<void()>) {}
#endif

}  // namespace lamina::platform
