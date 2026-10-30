// src/server/runconfig.cpp - #564: the Settings view of the run config (serve/runconfig.py's port): the editable keys
// and their checks, a change made to a copy that keeps every other key, and the file written as the Python writes it
// (json.dumps with indent 1, ASCII only, the platform's line ends, no newline at the end; the earlier file kept as
// <name>.bak). Paths are read as pathlib reads them, and the messages quote values with Python's repr().
#include "runconfig.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace lamina::runconfig {

const std::vector<Editable> kEditable = {
    {"sampling.temperature", "sampling", "num>=0", {},
     "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, "
     "0.7 not)"},
    {"sampling.top_p", "sampling", "0<x<=1", {}, "Default top_p for requests that send none"},
    {"sampling.top_k", "sampling", "1..64", {}, "Default top_k for requests that send none (1-64)"},
    {"sampling.min_p", "sampling", "0<=x<=1", {}, "Default min_p for requests that send none"},
    {"reasoning_budget_tokens", "int>=0", "", {},
     "Cap the thinking of every request at this many tokens (0 or empty: no cap)"},
    {"fit_max_tokens", "bool", "", {},
     "Shorten a max_tokens that does not fit the context (on by default); off: answer 400 like llama.cpp"},
    {"anthropic_thinking", "enum", "", {"model", "on_request"},
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"},
    {"effort_position", "enum", "", {"start", "end"},
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"},
    {"aliases", "names", "", {}, "Other model names the server lists and answers to (comma-separated)"},
    {"idle_unload_s", "num>=0", "", {},
     "Unload the model after this many seconds without requests (0 or empty: never)"},
    {"lazy_load", "bool", "", {}, "Start without loading the model; the first request loads it (text only)"},
    {"engine_silence_s", "num>=0", "", {},
     "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"},
    {"api_monitor", "bool", "", {}, "Keep the last 100 requests' prompts and answers in memory for /api-monitor"},
    {"open_browser", "bool", "", {}, "Open the chat page in the browser when the model is ready"},
    {"vram_reserve_mib", "arg", "--vram-reserve-mib", {},
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"},
};

namespace {

constexpr size_t npos = std::string::npos;

const Editable* spec(std::string_view key) {
    for (const auto& e : kEditable)
        if (e.key == key) return &e;
    return nullptr;
}

// ---------------------------------------------------------------------------------------------- Python's ==
/// an int, a bool (0 / 1) or an unsigned up to int64's top as int64; false for an unsigned beyond it
bool as_i64(const json& v, int64_t& out) {
    if (v.is_boolean()) {
        out = v.get<bool>() ? 1 : 0;
        return true;
    }
    if (v.is_number_unsigned()) {
        const uint64_t u = v.get<uint64_t>();
        out = (int64_t) u;
        return u <= (uint64_t) std::numeric_limits<int64_t>::max();
    }
    out = v.get<int64_t>();
    return true;
}

/// two numbers (bools count as 0 / 1) by their exact values, as Python compares int and float
bool num_eq(const json& a, const json& b) {
    if (a.is_number_float() && b.is_number_float()) return a.get<double>() == b.get<double>();
    if (a.is_number_float() || b.is_number_float()) {
        const double d = a.is_number_float() ? a.get<double>() : b.get<double>();
        const json& i = a.is_number_float() ? b : a;
        if (!std::isfinite(d) || d != std::trunc(d)) return false;
        if (i.is_number_unsigned()) return d >= 0 && d < 18446744073709551616.0 && (uint64_t) d == i.get<uint64_t>();
        int64_t x;
        as_i64(i, x);
        return d >= -9223372036854775808.0 && d < 9223372036854775808.0 && (int64_t) d == x;
    }
    int64_t x, y;
    const bool small_a = as_i64(a, x), small_b = as_i64(b, y);
    if (small_a && small_b) return x == y;
    return !small_a && !small_b && a.get<uint64_t>() == b.get<uint64_t>();
}

/// Python's ==: numbers across int, float and bool; objects whatever their keys' order
bool py_eq(const json& a, const json& b) {
    const bool an = a.is_number() || a.is_boolean(), bn = b.is_number() || b.is_boolean();
    if (an || bn) return an && bn && num_eq(a, b);
    if (a.type() != b.type()) return false;
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (!py_eq(a[i], b[i])) return false;
        return true;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) return false;
        for (auto it = a.begin(); it != a.end(); ++it) {
            const auto f = b.find(it.key());
            if (f == b.end() || !py_eq(it.value(), *f)) return false;
        }
        return true;
    }
    return a == b;   // null, string
}

// ---------------------------------------------------------------------------------------------- pathlib
#ifdef _WIN32
constexpr std::string_view kSep = "\\";
constexpr std::string_view kPathClass = "WindowsPath";
#else
constexpr std::string_view kSep = "/";
constexpr std::string_view kPathClass = "PosixPath";
#endif

/// a path as PurePath parses it: the drive, the root, the parts (no empty ones, no ".")
struct PathParts {
    std::string drv, root;
    std::vector<std::string> tail;
};

#ifdef _WIN32
/// the "X:" drive at the start of a Windows path (any first character): its length, or 0
size_t drive_len(std::string_view s) {
    if (s.empty()) return 0;
    size_t n = 1;
    utf8_at(s, 0, &n);
    return s.size() > n && s[n] == ':' ? n + 1 : 0;
}
#endif

PathParts parse_path(std::string p) {
    PathParts out;
    std::string rel;
#ifdef _WIN32
    std::replace(p.begin(), p.end(), '/', '\\');
    // ntpath.splitroot
    if (starts_with(p, "\\\\")) {   // UNC drives (\\server\share, \\?\UNC\server\share), device drives (\\.\device)
        const size_t start = lower(p.substr(0, 8)) == "\\\\?\\unc\\" ? 8 : 2;
        const size_t i = p.find('\\', start);
        const size_t j = i == npos ? npos : p.find('\\', i + 1);
        if (j == npos) {
            out.drv = p;
        } else {
            out.drv = p.substr(0, j);
            out.root = "\\";
            rel = p.substr(j + 1);
        }
    } else if (starts_with(p, "\\")) {
        out.root = "\\";
        rel = p.substr(1);
    } else if (const size_t d = drive_len(p)) {
        out.drv = p.substr(0, d);
        if (p.size() > d && p[d] == '\\') out.root = "\\";
        rel = p.substr(d + out.root.size());
    } else {
        rel = p;
    }
    if (out.root.empty() && starts_with(out.drv, "\\") && !ends_with(out.drv, "\\")) {   // a share has a root
        const auto parts = split(out.drv, "\\");
        if ((parts.size() == 4 && parts[2] != "" && parts[2] != "?" && parts[2] != "." && parts[2] != "?.") ||
            parts.size() == 6)
            out.root = "\\";
    }
#else
    // posixpath.splitroot: two leading slashes stay two, more are one
    if (!starts_with(p, "/")) {
        rel = p;
    } else if (!starts_with(p, "//") || starts_with(p, "///")) {
        out.root = "/";
        rel = p.substr(1);
    } else {
        out.root = "//";
        rel = p.substr(2);
    }
#endif
    std::vector<std::string> parts = split(rel, kSep);
    for (auto& x : parts)
        if (!x.empty() && x != ".") out.tail.push_back(std::move(x));
    return out;
}

/// str(PurePath)
std::string format_path(const PathParts& pp) {
    std::string out = pp.drv + pp.root;
    std::vector<std::string> tail = pp.tail;
#ifdef _WIN32
    if (out.empty() && !tail.empty() && drive_len(tail[0]) != 0) tail.insert(tail.begin(), ".");   // not a drive
#endif
    for (size_t i = 0; i < tail.size(); ++i) {
        if (i) out += kSep;
        out += tail[i];
    }
    return out.empty() ? "." : out;
}

std::string path_name(const PathParts& pp) { return pp.tail.empty() ? "" : pp.tail.back(); }

/// path.with_name(name), as a string
std::string with_name(PathParts pp, const std::string& name) {
    pp.tail.back() = name;
    return format_path(pp);
}

// ---------------------------------------------------------------------------------------------- files
#ifdef _WIN32

/// a Windows error's text as Python's OSError words it (FormatMessage without its trailing dot and line end)
std::string win_text(DWORD code) {
    wchar_t* buf = nullptr;
    const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                   FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code,
                                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR) &buf, 0, nullptr);
    if (n == 0 || buf == nullptr) {
        char b[32];
        std::snprintf(b, sizeof b, "Windows Error 0x%lx", (unsigned long) code);
        return b;
    }
    int len = (int) n;
    while (len > 0 && (buf[len - 1] <= L' ' || buf[len - 1] == L'.')) --len;
    std::string out;
    if (len > 0) {
        const int m = WideCharToMultiByte(CP_UTF8, 0, buf, len, nullptr, 0, nullptr, nullptr);
        out.resize((size_t) m);
        WideCharToMultiByte(CP_UTF8, 0, buf, len, out.data(), m, nullptr, nullptr);
    }
    LocalFree(buf);
    return out;
}
#endif

/// an OSError from errno: "[Errno 2] No such file or directory: 'path'"
[[noreturn]] void os_error(int err, const std::string& filename = "") {
    std::string msg = "[Errno " + std::to_string(err) + "] " + std::strerror(err);
    if (!filename.empty()) msg += ": " + py_repr_str(filename);
    throw ConfigFileError(msg);
}

struct File {
    FILE* f;
    explicit File(FILE* file) : f(file) {}
    ~File() {
        if (f != nullptr) std::fclose(f);
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    /// fclose and its error: a write that fails only as it is flushed (a full disk)
    void close() {
        FILE* g = f;
        f = nullptr;
        if (std::fclose(g) != 0) os_error(errno);
    }
};

/// open() in binary, read or write (truncated); Python's OSError when it cannot
FILE* open_file(const std::string& path, bool write) {
#ifdef _WIN32
    FILE* f = _wfopen(widen(path).c_str(), write ? L"wb" : L"rb");
#else
    FILE* f = std::fopen(path.c_str(), write ? "wb" : "rb");
    struct stat st;
    if (f != nullptr && fstat(fileno(f), &st) == 0 && S_ISDIR(st.st_mode)) {   // Python's open() refuses a directory
        std::fclose(f);
        f = nullptr;
        errno = EISDIR;
    }
#endif
    if (f == nullptr) os_error(errno, path);
    return f;
}

std::string read_file(const std::string& path) {
    File in(open_file(path, false));
    std::string data;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, in.f)) > 0) data.append(buf, n);
    if (std::ferror(in.f)) os_error(errno);
    return data;
}

/// shutil.copyfile: the bytes, not the metadata
void copy_file(const std::string& src, const std::string& dst) {
    File in(open_file(src, false));
    File out(open_file(dst, true));
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, in.f)) > 0)
        if (std::fwrite(buf, 1, n, out.f) != n) os_error(errno);
    if (std::ferror(in.f)) os_error(errno);
    out.close();
}

/// Path.write_text: a text file, so the platform's line ends ("\r\n" on Windows)
void write_text(const std::string& path, const std::string& text) {
    File out(open_file(path, true));
    if (std::fwrite(text.data(), 1, text.size(), out.f) != text.size()) os_error(errno);
    out.close();
}

/// os.replace: `src` moved over `dst`
void replace_file(const std::string& src, const std::string& dst) {
#ifdef _WIN32
    if (!MoveFileExW(widen(src).c_str(), widen(dst).c_str(), MOVEFILE_REPLACE_EXISTING)) {
        const DWORD code = GetLastError();
        throw ConfigFileError("[WinError " + std::to_string(code) + "] " + win_text(code) + ": " + py_repr_str(src) +
                              " -> " + py_repr_str(dst));
    }
#else
    if (std::rename(src.c_str(), dst.c_str()) != 0) {
        const int err = errno;
        throw ConfigFileError("[Errno " + std::to_string(err) + "] " + std::strerror(err) + ": " + py_repr_str(src) +
                              " -> " + py_repr_str(dst));
    }
#endif
}

// ---------------------------------------------------------------------------------------------- the keys
/// the position of `flag` in args but the last item (Python's `flag in a[:-1]`, then a.index(flag)), or npos
size_t flag_at(const json& a, std::string_view flag) {
    for (size_t i = 0; i + 1 < a.size(); ++i)
        if (is_str(a[i], flag)) return i;
    return npos;
}

/// int() of a string of ASCII digits; beyond 64 bits a float, as json.hpp holds such an integer
json digits_value(const std::string& v) {
    uint64_t u = 0;
    const auto r = std::from_chars(v.data(), v.data() + v.size(), u);
    if (r.ec == std::errc() && r.ptr == v.data() + v.size()) return u;
    double d = 0.0;
    std::from_chars(v.data(), v.data() + v.size(), d);
    return d;
}

/// the value after the flag in "args": an int when it is all digits (str.isdigit, ASCII digits here)
json arg_value(const json& cfg, std::string_view flag) {
    const json& a = jget(cfg, "args");
    if (!a.is_array()) return nullptr;
    const size_t i = flag_at(a, flag);
    if (i == npos) return nullptr;
    const std::string v = py_str(a[i + 1]);
    const bool digits = !v.empty() && std::all_of(v.begin(), v.end(), [](char c) { return c >= '0' && c <= '9'; });
    return digits ? digits_value(v) : json(v);
}

/// list(x or []): a list as it is, a string's characters, an object's keys; anything else reads as an empty list
/// (a number or a bool that is not 0: Python's TypeError)
json list_of(const json& x) {
    if (x.is_array()) return x;
    json a = json::array();
    if (x.is_string()) {
        const std::string& s = x.get_ref<const std::string&>();
        for (size_t i = 0; i < s.size();) {
            size_t n = 1;
            utf8_at(s, i, &n);
            a.push_back(s.substr(i, n));
            i += n;
        }
    } else if (x.is_object()) {
        for (auto it = x.begin(); it != x.end(); ++it) a.push_back(it.key());
    }
    return a;
}

/// int(d) of a whole float: beyond 64 bits it stays a float, as json.hpp holds such an integer
json whole_value(double d) {
    if (d >= -9223372036854775808.0 && d < 9223372036854775808.0) return (int64_t) d;
    if (d >= 0 && d < 18446744073709551616.0) return (uint64_t) d;
    return d;
}

/// _number: v when it is a number in range (an int for `whole`), else what is expected
json number(const std::string& key, const json& v, bool whole = false, double lo = 0.0,
            std::optional<double> hi = std::nullopt, bool lo_open = false) {
    bool bad = !v.is_number();   // a bool too
    if (!bad && whole && v.is_number_float()) {   // int(v) first
        const double d = v.get<double>();
        if (std::isnan(d)) throw RequestError("cannot convert float NaN to integer");
        if (std::isinf(d)) throw std::overflow_error("cannot convert float infinity to integer");
        bad = d != std::trunc(d);
    }
    if (!bad) {
        const double d = v.get<double>();   // exact against these bounds (0, 1, 64)
        bad = (lo_open ? d <= lo : d < lo) || (hi && d > *hi);
    }
    if (bad) {
        char rng[80];
        if (hi) std::snprintf(rng, sizeof rng, "%s %g and at most %g", lo_open ? "more than" : "at least", lo, *hi);
        else std::snprintf(rng, sizeof rng, "%s %g", lo_open ? "more than" : "at least", lo);
        throw RequestError(key + ": expected a " + (whole ? "whole " : "") + "number, " + rng + ", not " + py_repr_value(v));
    }
    return whole && v.is_number_float() ? whole_value(v.get<double>()) : v;
}

json as_float(const json& v) { return v.get<double>(); }

}  // namespace

json value_of(const json& cfg, std::string_view key) {
    const Editable* e = spec(key);
    if (e == nullptr) throw std::out_of_range(py_repr_str(key));   // Python's KeyError
    if (e->kind == "sampling") {
        const json& s = jget(cfg, "sampling");
        return s.is_object() ? jget(s, key.substr(key.find('.') + 1)) : json(nullptr);
    }
    if (e->kind == "arg") return arg_value(cfg, e->rule);
    return jget(cfg, key);
}

json view(const json& cfg, const std::string& path) {
    json keys = json::array();
    for (const auto& e : kEditable) {
        const bool numeric = e.kind == "sampling" || e.kind == "arg" || e.kind == "int>=0" || e.kind == "num>=0";
        json k = {{"key", e.key}, {"value", value_of(cfg, e.key)}, {"help", e.help},
                  {"kind", numeric ? std::string_view("number") : e.kind}};
        if (e.kind == "enum") k["choices"] = e.choices;
        keys.push_back(std::move(k));
    }
    return {{"file", path_name(parse_path(path))}, {"keys", std::move(keys)},
            {"note", "Saved to the run config; used from the next start of the model."}};
}

json check(const std::string& key, const json& v, const json& cfg) {
    const Editable* e = spec(key);
    if (e == nullptr)
        throw RequestError(py_repr_str(key) + " cannot be changed here (only the keys the Settings view lists)");
    if (v.is_null()) return nullptr;
    const std::string_view k = e->kind;
    if (k == "bool") {
        if (!v.is_boolean()) throw RequestError(key + ": expected true or false, not " + py_repr_value(v));
        if (key == "lazy_load" && v.get<bool>() && truthy(jget(cfg, "vision")))
            throw RequestError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")");
        return v;
    }
    if (k == "enum") {
        const bool known = v.is_string() && std::find(e->choices.begin(), e->choices.end(),
                                                       v.get_ref<const std::string&>()) != e->choices.end();
        if (!known) {
            std::string choices;
            for (const auto c : e->choices) choices += (choices.empty() ? "" : ", ") + std::string(c);
            throw RequestError(key + ": expected one of " + choices + ", not " + py_repr_value(v));
        }
        return v;
    }
    if (k == "names") {
        json names = v;
        if (v.is_string()) {   // comma-separated
            names = json::array();
            const std::vector<std::string> parts = split(v.get_ref<const std::string&>(), ",");
            for (const auto& x : parts)
                if (!strip(x).empty()) names.push_back(std::string(strip(x)));
        }
        bool ok = names.is_array();
        for (size_t i = 0; ok && i < names.size(); ++i) {
            const json& x = names[i];
            ok = x.is_string() && !strip(x.get_ref<const std::string&>()).empty() &&
                 utf8_len(x.get_ref<const std::string&>()) <= 128;
        }
        if (!ok) throw RequestError(key + ": expected a list of model names, not " + py_repr_value(names));
        return names.empty() ? json(nullptr) : names;
    }
    if (k == "int>=0" || k == "arg") return number(key, v, true);
    if (k == "num>=0") return number(key, v);
    const std::string_view rule = e->rule;   // a sampling key: the server's own rules
    if (rule == "num>=0") return as_float(number(key, v));
    if (rule == "0<x<=1") return as_float(number(key, v, false, 0.0, 1.0, true));
    if (rule == "0<=x<=1") return as_float(number(key, v, false, 0.0, 1.0));
    return number(key, v, true, 1.0, 64.0);
}

std::pair<json, std::vector<std::string>> apply(const json& cfg, const json& changes) {
    if (!changes.is_object() || changes.empty())
        throw RequestError("send {\"set\": {\"<key>\": <value or null>, ...}}");
    std::vector<std::pair<std::string, json>> checked;
    for (auto it = changes.begin(); it != changes.end(); ++it)
        checked.emplace_back(it.key(), check(it.key(), *it, cfg));
    json out = cfg;   // a deep copy: the caller's config is left alone
    std::vector<std::string> changed;
    for (const auto& [key, v] : checked) {
        if (py_eq(value_of(out, key), v)) continue;
        const Editable& e = *spec(key);
        if (e.kind == "sampling") {
            json s = jget(out, "sampling").is_object() ? out["sampling"] : json::object();
            const std::string sub = key.substr(key.find('.') + 1);
            if (v.is_null()) s.erase(sub);
            else s[sub] = v;
            if (!s.empty()) out["sampling"] = std::move(s);
            else out.erase("sampling");
        } else if (e.kind == "arg") {
            json a = list_of(jget(out, "args"));
            const size_t i = flag_at(a, e.rule);
            if (i != npos) {
                if (v.is_null()) a.erase(a.begin() + (std::ptrdiff_t) i, a.begin() + (std::ptrdiff_t) i + 2);
                else a[i + 1] = py_str(v);
            } else if (!v.is_null()) {
                a.push_back(e.rule);
                a.push_back(py_str(v));
            }
            out["args"] = std::move(a);
        } else if (v.is_null()) {
            out.erase(key);
        } else {
            out[key] = v;
        }
        changed.push_back(key);
    }
    return {std::move(out), std::move(changed)};
}

std::string save(const std::string& path, const json& cfg) {
    const PathParts pp = parse_path(path);
    const std::string target = format_path(pp);
    if (pp.tail.empty()) {   // with_name() on a path without a name; its repr shows as_posix()
        std::string posix = target;
#ifdef _WIN32
        std::replace(posix.begin(), posix.end(), '\\', '/');
#endif
        throw RequestError(std::string(kPathClass) + "(" + py_repr_str(posix) + ") has an empty name");
    }
    const std::string bak = with_name(pp, pp.tail.back() + ".bak");
    copy_file(target, bak);
    const std::string tmp = with_name(pp, pp.tail.back() + ".tmp");
    write_text(tmp, py_dumps(cfg, true, false, 2) + "\n");   // the form setup writes: LF, a final line end
    replace_file(tmp, target);
    return bak;
}

json load(const std::string& path) {
    const PathParts pp = parse_path(path);
    std::string text = read_file(format_path(pp));
    constexpr std::string_view bom = "\xEF\xBB\xBF";
    if (starts_with(text, bom)) text.erase(0, bom.size());   // utf-8-sig
    if (starts_with(text, bom))   // json.loads' own check
        throw RequestError("Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)");
    json cfg;
    std::string err;
    if (!py_loads(text, cfg, &err)) throw RequestError(err);   // the parser's words, not CPython's json module's
    if (!cfg.is_object()) throw RequestError(path_name(pp) + " is not a JSON object");
    return cfg;
}

}  // namespace lamina::runconfig
