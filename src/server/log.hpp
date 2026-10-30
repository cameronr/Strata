// src/server/log.hpp - the server's console lines (the Python server's print(..., flush=True)).
#pragma once

#include <cstdio>
#include <string_view>

namespace lamina {

/// one line (or several) on the console, stdout, written whole
void say(std::string_view text);
/// the same on stderr
void say_err(std::string_view text);
/// where say_err writes (the console's stderr, kept when the process's stderr goes to the engine's log)
void set_err_stream(std::FILE* f);

}  // namespace lamina
