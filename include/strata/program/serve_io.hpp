// include/strata/program/serve_io.hpp - the --serve loop's line protocol, wherever it travels.
//
// The engine binary (`lamina --serve`) reads requests on stdin and writes its replies on stdout. lamina-server
// runs the same loop on a thread of its own process, with in-process channels set here before strata::program::run.
#pragma once

#include <cstdarg>
#include <functional>
#include <string>
#include <string_view>

namespace strata::program {

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

}  // namespace strata::program
