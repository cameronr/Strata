#include "log.hpp"

#include <cstdio>
#include <mutex>

namespace lamina {
namespace {

std::mutex& mu() {
    static std::mutex m;
    return m;
}

void write(FILE* f, std::string_view text) {
    std::lock_guard<std::mutex> lk(mu());
    std::fwrite(text.data(), 1, text.size(), f);
    if (text.empty() || text.back() != '\n') std::fputc('\n', f);
    std::fflush(f);
}

}  // namespace

void say(std::string_view text) { write(stdout, text); }
void say_err(std::string_view text) { write(stderr, text); }

}  // namespace lamina
