// include/lamina/program/serve_io.hpp - the --serve loop's line protocol, wherever it travels.
//
// The engine binary (`lamina --serve`) reads requests on stdin and writes its replies on stdout. lamina-server
// runs the same loop on a thread of its own process, with in-process channels set here before lamina::program::run.
#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace lamina::program {

struct ServeIo {
    /// the next line, without its line end; false once the input has ended
    std::function<bool(std::string&)> read_line;
    /// text of whole lines, written at once
    std::function<void(std::string_view)> write;
    /// in-process: the serve loop ends the input (end_input) and joins its reader thread before it returns (the
    /// reader's state is the loop's), and leaves Ctrl+C to the host process; stdin's reader ends with the process
    bool in_process = false;
    std::function<void()> end_input;
};

/// the channels the next run() uses (default: stdin and stdout)
void set_serve_io(ServeIo io);
const ServeIo& serve_io();
/// printf through serve_io().write
void serve_print(const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
void serve_vprint(const char* fmt, va_list ap);

/// The serve loop's input reader at the loop's end: in-process the reader holds the loop's state, so its input ends
/// and it is joined before that state goes; stdin's reader ends with the process.
struct ReaderEnd {
    std::thread& t;
    const ServeIo& io;
    ~ReaderEnd() {
        if (!io.in_process) {
            t.detach();
            return;
        }
        if (io.end_input) io.end_input();
        t.join();
    }
};

/// A picture of a GENI request's embeddings file: its rows (n = nx x ny) and where they start in the rows read.
struct VisionImage {
    int64_t n, nx, ny;
    size_t off;
};
/// Reads a lamina-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, a picture
/// each, in prompt order), appending the rows to `rows` and the pictures to `imgs`: "" or the error.
std::string read_vision_records(const std::string& path, int64_t n_embd, std::vector<float>& rows,
                                std::vector<VisionImage>& imgs);
/// What the conversation cache compares: a picture is its grid and the server's key for it (`key`), or else its
/// embeddings (FNV-1a, with the seed and byte order of upstream's conversation cache).
uint64_t vision_image_key(const VisionImage& im, const uint64_t* key, const float* rows, int64_t n_embd);

}  // namespace lamina::program
