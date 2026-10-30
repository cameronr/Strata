#include "strata/program/serve_io.hpp"

#include <cstdio>
#include <iostream>
#include <mutex>
#include <vector>

namespace strata::program {
namespace {

ServeIo stdio_io() {
    ServeIo io;
    io.read_line = [](std::string& line) { return static_cast<bool>(std::getline(std::cin, line)); };
    io.write = [](std::string_view text) {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fflush(stdout);
    };
    return io;
}

ServeIo& current() {
    static ServeIo io = stdio_io();
    return io;
}

std::mutex& write_mu() {
    static std::mutex mu;
    return mu;
}

}  // namespace

void set_serve_io(ServeIo io) {
    if (!io.read_line || !io.write) io = stdio_io();
    current() = std::move(io);
}

const ServeIo& serve_io() { return current(); }

void serve_vprint(const char* fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    char small[512];
    const int n = std::vsnprintf(small, sizeof small, fmt, ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    std::string text;
    if ((size_t) n < sizeof small) {
        text.assign(small, (size_t) n);
    } else {
        std::vector<char> big((size_t) n + 1);
        std::vsnprintf(big.data(), big.size(), fmt, ap2);
        text.assign(big.data(), (size_t) n);
    }
    va_end(ap2);
    std::lock_guard<std::mutex> lk(write_mu());   // a line is written whole, whichever thread writes it
    current().write(text);
}

void serve_print(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    serve_vprint(fmt, ap);
    va_end(ap);
}

}  // namespace strata::program
