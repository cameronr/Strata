// src/server/log.hpp - the server's console lines (the Python server's print(..., flush=True)).
#pragma once

#include <string_view>

namespace lamina {

/// one line (or several) on the console, stdout, written whole
void say(std::string_view text);
/// the same on stderr
void say_err(std::string_view text);

}  // namespace lamina
