#include "log.hpp"

#include <cstdio>
#include <mutex>

namespace lamina {
namespace {

std::mutex& mu() {
    static std::mutex m;
    return m;
}

std::FILE* g_err = nullptr;

void write(FILE* f, std::string_view text) {
    std::lock_guard<std::mutex> lk(mu());
    std::fwrite(text.data(), 1, text.size(), f);
    if (text.empty() || text.back() != '\n') std::fputc('\n', f);
    std::fflush(f);
}

}  // namespace

void say(std::string_view text) { write(stdout, text); }
void say_err(std::string_view text) { write(g_err != nullptr ? g_err : stderr, text); }

void set_err_stream(std::FILE* f) {
    std::lock_guard<std::mutex> lk(mu());
    g_err = f;
}

}  // namespace lamina
