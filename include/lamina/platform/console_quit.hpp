// include/lamina/platform/console_quit.hpp - Ctrl+C in the console as a request to quit (Windows; a no-op elsewhere).
//
// The server runs the engine in its own console, so the user's Ctrl+C reaches the engine too, whose default handler
// ends the process at once - before the server's QUIT, and so before the engine's exit work (the learned profile).
#pragma once

#include <functional>

namespace lamina::platform {

/// While `quit` is set, Ctrl+C and Ctrl+Break call it (on a thread of their own) instead of ending the process; an
/// empty function restores the default.  Returns once no call of the previous function is running.
void set_console_quit(std::function<void()> quit);

/// set_console_quit for a scope: the default again when it ends.
struct ConsoleQuit {
    explicit ConsoleQuit(std::function<void()> quit) { set_console_quit(std::move(quit)); }
    ~ConsoleQuit() { set_console_quit(nullptr); }
    ConsoleQuit(const ConsoleQuit&) = delete;
    ConsoleQuit& operator=(const ConsoleQuit&) = delete;
};

}  // namespace lamina::platform
