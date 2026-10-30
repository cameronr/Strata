// src/server/process.hpp - the programs the server starts (the image encoder, MCP servers, sd-cli, replicas' engines,
// the before_load command), tied to the server's lifetime (serve/lifeline.py, serve/winjob.py).
//
// Windows: each contained child starts in a kill-on-close job object whose only handle is the server's, so it ends
// however the server ends (a crash, Task Manager, the console closed), and kill() ends whatever it started too; it
// is opted out of power throttling (#691). Linux: the kernel kills it when the server ends (PR_SET_PDEATHSIG), and
// it leads its own process group, which kill() ends; the signal follows the thread that forked, so every child is
// forked by one long-lived thread.
#pragma once

#include "json.hpp"
#include "sync.hpp"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lamina {

/// environment names compare as the OS compares them (case-insensitively on Windows)
struct EnvLess {
    bool operator()(const std::string& a, const std::string& b) const;
};
using Environment = std::map<std::string, std::string, EnvLess>;

/// this process's environment
Environment environment();

struct SpawnOptions {
    std::vector<std::string> args;     ///< the program and its arguments; with `shell`, one command line
    bool shell = false;                ///< args[0] runs through the shell (cmd.exe /c, /bin/sh -c)
    std::string cwd;                   ///< "": this process's
    std::optional<Environment> env;    ///< nullopt: this process's
    enum Stream { inherit, null, pipe, file };
    Stream in = inherit, out = inherit, err = inherit;
    std::string err_file;              ///< err == file: appended to this file
    bool contain = true;               ///< ends with the server; kill() ends what it started too
    bool own_group = false;            ///< out of the console's Ctrl+C (Windows: a new process group)
    std::vector<int> cpus;             ///< Linux: the CPUs it may run on (empty: any)
    bool no_window = false;            ///< Windows: no console of its own (CREATE_NO_WINDOW)
    bool cr_lines = false;             ///< out == pipe: a lone CR ends a line too (a progress bar's updates)
    std::function<void(const std::string&)> on_err_line;   ///< err == pipe: each line of its stderr, as it comes
};

class Process {
public:
    /// starts it; std::runtime_error with the OS's words when it cannot
    explicit Process(SpawnOptions o);
    /// kills it if it still runs (contained: with what it started)
    ~Process();
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    /// to its stdin (in == pipe); false once the pipe is closed
    bool write(std::string_view data);
    void close_stdin();
    /// a line of its stdout (out == pipe), without the line end: 1 a line, 0 none within timeout_s (< 0: no limit),
    /// -1 its output ended
    int read_line(std::string& line, double timeout_s);
    /// the last lines of its stderr (err == pipe), newline-joined
    std::string stderr_tail();
    /// its stderr has been read to its end (err == pipe; true otherwise)
    bool stderr_done() const;
    /// the exit code once it has ended
    std::optional<int> poll();
    /// waits up to timeout_s (< 0: no limit); the exit code, or nullopt while it still runs
    std::optional<int> wait(double timeout_s = -1);
    void kill();
    long pid() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// the before_load command: a string (through the shell) or a list (a program and its arguments), stdin from the
/// null device, its output on the server's console; its exit code, or -1 with `why` (it could not run, or it did
/// not finish in timeout_s and was ended)
int64_t run_command(const json& cmd, double timeout_s, std::string& why);
/// a program's path through PATH (and PATHEXT on Windows), as shutil.which finds it; "" when there is none
std::string which(const std::string& name, const Environment* env = nullptr);
/// opts this process out of Windows power throttling: a minimized console window made Windows 11 move the server's
/// threads to the E-cores (#691: generation 20% slower); false elsewhere
bool no_throttle_self();
/// from here on the process's stderr (the engine's messages) is appended to `path`; the server's own say_err lines
/// stay on the console
void redirect_stderr(const std::string& path);
/// a temporary directory (the system's), without a trailing separator
std::string temp_dir();
/// the directory of this program's file
std::string self_dir();
/// a path from UTF-8 text, and back
std::filesystem::path u8_path(std::string_view utf8);
std::string u8_str(const std::filesystem::path& p);

}  // namespace lamina
