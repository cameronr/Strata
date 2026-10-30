// src/server/images.cpp - image generation through stable-diffusion.cpp's sd-cli (images.hpp): the request's fields,
// the sd-cli command and environment, one process per job read for its progress and its log, the stored images.
#include "images.hpp"

#include "log.hpp"

#include "common/base64.hpp"   // llama.cpp's, header-only

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#endif

namespace lamina::images {
namespace fs = std::filesystem;
namespace {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------------------------- Python's view

/// `v or dflt`
json or_else(const json& v, json dflt) { return truthy(v) ? v : std::move(dflt); }

/// the code points of Unicode's decimal digits (Nd, Unicode 15.0 as Python 3.12 has it): runs of ten from these zeros
constexpr uint32_t kDigitZeros[] = {
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0,
    0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620,
    0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0,
    0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50,
    0x16A60, 0x16AC0, 0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0};

/// a decimal digit's value (str.isdecimal, re's \d, what int() reads), else -1
int digit_value(uint32_t cp) {
    const auto it = std::upper_bound(std::begin(kDigitZeros), std::end(kDigitZeros), cp);
    if (it == std::begin(kDigitZeros)) return -1;
    const uint32_t zero = *(it - 1);
    return cp - zero < 10 ? (int) (cp - zero) : -1;
}

/// s with Python's whitespace stripped and its decimal digits made ASCII (what int() and float() read); false when it
/// has another character outside ASCII
bool ascii_number_text(std::string_view s, std::string& out) {
    s = strip(s);
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t c = utf8_at(s, i, &n);
        i += n;
        if (c < 0x80) {
            out += (char) c;
            continue;
        }
        const int d = digit_value(c);
        if (d < 0) return false;
        out += (char) ('0' + d);
    }
    return true;
}

/// digits with single underscores between them -> the digits ("" when the underscores are misplaced)
std::string without_underscores(std::string_view s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '_') {
            out += s[i];
        } else if (i == 0 || i + 1 == s.size() || !std::isdigit((unsigned char) s[i - 1]) ||
                   !std::isdigit((unsigned char) s[i + 1])) {
            return "";
        }
    }
    return out;
}

/// int(s) of a str (beyond 64 bits it stops at the int64 bounds, where Python's ints go on)
int64_t int_of_str(const std::string& s) {
    std::string t;
    const bool ascii = ascii_number_text(s, t);
    std::string_view v = t;
    const bool neg = !v.empty() && v[0] == '-';
    if (!v.empty() && (v[0] == '-' || v[0] == '+')) v.remove_prefix(1);
    const std::string digits = without_underscores(v);
    const bool ok = ascii && !digits.empty() && std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; });
    if (!ok) throw std::invalid_argument("invalid literal for int() with base 10: " + py_repr_value(json(s)));
    uint64_t u = 0;
    for (const char c : digits) {
        if (u > (std::numeric_limits<uint64_t>::max() - 9) / 10) {
            u = std::numeric_limits<uint64_t>::max();
            break;
        }
        u = u * 10 + (uint64_t) (c - '0');
    }
    constexpr uint64_t top = (uint64_t) std::numeric_limits<int64_t>::max();
    if (neg) return u > top ? std::numeric_limits<int64_t>::min() : -(int64_t) u;
    return u > top ? std::numeric_limits<int64_t>::max() : (int64_t) u;
}

/// int(v) of a config value
int64_t py_int(const json& v) {
    switch (v.type()) {
        case json::value_t::boolean: return v.get<bool>() ? 1 : 0;
        case json::value_t::number_integer: return v.get<int64_t>();
        case json::value_t::number_unsigned: return (int64_t) std::min<uint64_t>(v.get<uint64_t>(), (uint64_t) std::numeric_limits<int64_t>::max());
        case json::value_t::number_float: {
            const double d = v.get<double>();
            if (std::isnan(d)) throw std::invalid_argument("cannot convert float NaN to integer");
            if (std::isinf(d)) throw std::overflow_error("cannot convert float infinity to integer");
            if (d >= 9223372036854775808.0) return std::numeric_limits<int64_t>::max();
            if (d < -9223372036854775808.0) return std::numeric_limits<int64_t>::min();
            return (int64_t) d;
        }
        case json::value_t::string: return int_of_str(v.get_ref<const std::string&>());
        default:
            throw ConfigTypeError("TypeError", "int() argument must be a string, a bytes-like object or a real number, not '" +
                                                   py_type_name(v) + "'");
    }
}

/// float(s) of a str
double float_of_str(const std::string& s) {
    const std::string bad = "could not convert string to float: " + py_repr_value(json(s));
    std::string t;
    if (!ascii_number_text(s, t)) throw std::invalid_argument(bad);
    std::string_view v = t;
    const bool neg = !v.empty() && v[0] == '-';
    if (!v.empty() && (v[0] == '-' || v[0] == '+')) v.remove_prefix(1);
    const std::string word = lower(v);
    if (word == "inf" || word == "infinity") return neg ? -INFINITY : INFINITY;
    if (word == "nan") return neg ? -NAN : NAN;
    // underscores only between digits; from_chars reads the rest as strtod would (no sign, no hex)
    std::string digits;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] != '_') {
            digits += v[i];
        } else if (i == 0 || i + 1 == v.size() || !std::isdigit((unsigned char) v[i - 1]) ||
                   !std::isdigit((unsigned char) v[i + 1])) {
            throw std::invalid_argument(bad);
        }
    }
    double d = 0;
    const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), d);
    if (digits.empty() || r.ec != std::errc() || r.ptr != digits.data() + digits.size() ||
        !(std::isdigit((unsigned char) digits[0]) || digits[0] == '.'))
        throw std::invalid_argument(bad);
    return neg ? -d : d;
}

/// float(v) of a config value
double py_float(const json& v) {
    switch (v.type()) {
        case json::value_t::boolean: return v.get<bool>() ? 1.0 : 0.0;
        case json::value_t::number_integer: return (double) v.get<int64_t>();
        case json::value_t::number_unsigned: return (double) v.get<uint64_t>();
        case json::value_t::number_float: return v.get<double>();
        case json::value_t::string: return float_of_str(v.get_ref<const std::string&>());
        default:
            throw ConfigTypeError("TypeError", "float() argument must be a string or a real number, not '" + py_type_name(v) + "'");
    }
}

/// what `for x in v` goes through: a list's items, a str's characters, a dict's keys
std::vector<json> py_iter(const json& v) {
    if (v.is_array()) return std::vector<json>(v.begin(), v.end());
    std::vector<json> out;
    if (v.is_string()) {
        const std::string& s = v.get_ref<const std::string&>();
        for (size_t i = 0; i < s.size();) {
            size_t n = 1;
            utf8_at(s, i, &n);
            out.emplace_back(s.substr(i, n));
            i += n;
        }
    } else if (v.is_object()) {
        for (auto it = v.begin(); it != v.end(); ++it) out.emplace_back(it.key());
    } else {
        throw ConfigTypeError("TypeError", "'" + py_type_name(v) + "' object is not iterable");
    }
    return out;
}

std::string path_type_error(const json& v) {
    return "argument should be a str or an os.PathLike object where __fspath__ returns a str, not '" + py_type_name(v) + "'";
}

// ---------------------------------------------------------------------------------------------- paths
/// str(pathlib.Path(p)): the OS's separators, empty and "." parts dropped, no trailing separator ("..": kept)
std::string path_str(std::string p) {
    std::string drv, root, rel;
#ifdef _WIN32
    const char sep = '\\';
    std::replace(p.begin(), p.end(), '/', sep);
    if (!p.empty() && p[0] == sep) {
        if (p.size() > 1 && p[1] == sep) {   // \\server\share, \\?\UNC\server\share, \\.\device
            std::string head = p.substr(0, 8);
            std::transform(head.begin(), head.end(), head.begin(), [](char c) { return (char) std::toupper((unsigned char) c); });
            const size_t i1 = p.find(sep, head == "\\\\?\\UNC\\" ? 8 : 2);
            const size_t i2 = i1 == std::string::npos ? i1 : p.find(sep, i1 + 1);
            if (i2 == std::string::npos) {
                drv = p;
            } else {
                drv = p.substr(0, i2);
                root = sep;
                rel = p.substr(i2 + 1);
            }
        } else {
            root = sep;
            rel = p.substr(1);
        }
    } else if (p.size() > 1 && p[1] == ':') {
        drv = p.substr(0, 2);
        if (p.size() > 2 && p[2] == sep) root = sep;
        rel = p.substr(root.empty() ? 2 : 3);
    } else {
        rel = p;
    }
    if (root.empty() && !drv.empty() && drv[0] == sep && drv.back() != sep) {   // \\server\share: a root of its own
        const auto parts = split(drv, "\\");
        if ((parts.size() == 4 && std::string_view("?.").find(parts[2]) == std::string_view::npos) || parts.size() == 6)
            root = sep;
    }
#else
    const char sep = '/';
    if (p.empty() || p[0] != sep) {
        rel = p;
    } else if (p.size() < 2 || p[1] != sep || (p.size() > 2 && p[2] == sep)) {
        root = "/";
        rel = p.substr(1);
    } else {
        root = "//";   // exactly two: POSIX leaves their meaning to the system
        rel = p.substr(2);
    }
#endif
    std::string tail;
    for (const auto& part : split(rel, std::string(1, sep)))
        if (!part.empty() && part != ".") tail += (tail.empty() ? "" : std::string(1, sep)) + part;
    if (!drv.empty() || !root.empty()) return drv + root + tail;
#ifdef _WIN32
    if (tail.size() > 1 && tail[1] == ':') tail = ".\\" + tail;   // not read as a drive
#endif
    return tail.empty() ? "." : tail;
}

bool has_nul(std::string_view s) { return s.find('\0') != std::string_view::npos; }

/// Path(p).exists() / .is_dir() (an unreadable path is neither)
bool path_exists(const std::string& p) {
    std::error_code ec;
    return !has_nul(p) && fs::exists(u8_path(path_str(p)), ec);
}
bool path_is_dir(const std::string& p) {
    std::error_code ec;
    return !has_nul(p) && fs::is_directory(u8_path(path_str(p)), ec);
}

/// Path(p).stat().st_size, following links (a folder's too); nullopt when the system cannot read it (an OSError)
std::optional<int64_t> stat_size(const std::string& p) {
    const std::string s = path_str(p);
#ifdef _WIN32
    const HANDLE h = CreateFileW(u8_path(s).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return std::nullopt;
    BY_HANDLE_FILE_INFORMATION info{};
    const BOOL ok = GetFileInformationByHandle(h, &info);
    CloseHandle(h);
    if (!ok) return std::nullopt;
    return (int64_t) (((uint64_t) info.nFileSizeHigh << 32) | info.nFileSizeLow);
#else
    struct stat st {};
    if (::stat(s.c_str(), &st) != 0) return std::nullopt;
    return (int64_t) st.st_size;
#endif
}

/// os.path.abspath
fs::path abspath(const fs::path& p) {
    std::error_code ec;
    fs::path a = fs::absolute(p, ec);   // Windows: GetFullPathNameW, as Python's
    if (ec) a = p;
#ifndef _WIN32
    a = a.lexically_normal();
    if (a.has_relative_path() && a.filename().empty()) a = a.parent_path();
#endif
    return a;
}

/// tempfile.mkdtemp(prefix, dir): a new folder named prefix + 8 random characters, private to this user
fs::path make_temp_dir(const fs::path& parent, const std::string& prefix) {
    static constexpr char kChars[] = "abcdefghijklmnopqrstuvwxyz0123456789_";
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> pick(0, (int) sizeof kChars - 2);
    for (int attempt = 0; attempt < 10000; ++attempt) {
        std::string name = prefix;
        for (int i = 0; i < 8; ++i) name += kChars[pick(rng)];
        const fs::path p = abspath(parent / u8_path(name));
#ifdef _WIN32
        std::error_code ec;
        if (fs::create_directory(p, ec)) return p;
        if (ec) throw std::runtime_error("cannot create the folder " + u8_str(p) + ": " + ec.message());
#else
        if (::mkdir(p.c_str(), 0700) == 0) return p;
        if (errno != EEXIST) throw std::runtime_error("cannot create the folder " + u8_str(p) + ": " + std::strerror(errno));
#endif
    }
    throw std::runtime_error("No usable temporary directory name found");
}

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + u8_str(p));
    return std::string(std::istreambuf_iterator<char>(f), {});
}

void write_file(const fs::path& p, const std::string& data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(data.data(), (std::streamsize) data.size());
    if (!f) throw std::runtime_error("cannot write " + u8_str(p));
}

/// the stored images' names: ^[0-9a-f]{32}\.png$ (re's $ also takes a last "\n")
bool stored_name(std::string_view name) {
    if (name.size() == 37 && name.back() == '\n') name.remove_suffix(1);
    return name.size() == 36 && name.substr(32) == ".png" &&
           std::all_of(name.begin(), name.begin() + 32, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

/// secrets.token_hex(nbytes): from the system's secure source
std::string token_hex(size_t nbytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::random_device rd;
    std::string out;
    while (out.size() < nbytes * 2) {
        uint32_t r = rd();
        for (int i = 0; i < 4 && out.size() < nbytes * 2; ++i, r >>= 8) {
            out += kHex[(r >> 4) & 15];
            out += kHex[r & 15];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------------------------- sd-cli's output

/// universal newlines: "\r\n" and "\r" -> "\n"
std::string newlines(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '\r') out += s[i];
        else if (i + 1 < s.size() && s[i + 1] == '\n') continue;
        else out += '\n';
    }
    return out;
}

/// sd.cpp's progress bar, "|====    | 3/8 - 14.1s/it": re.search(r"\|\s*(\d+)/(\d+)\s*-") -> (step, steps)
std::optional<std::pair<int64_t, int64_t>> step_of(std::string_view s) {
    for (size_t bar = s.find('|'); bar != std::string_view::npos; bar = s.find('|', bar + 1)) {
        size_t i = bar + 1;
        auto spaces = [&] {
            for (size_t n = 1; i < s.size() && py_isspace(utf8_at(s, i, &n)); i += n) {}
        };
        auto number = [&](int64_t& v) {   // \d+, false without a digit (or past 64 bits)
            size_t digits = 0;
            v = 0;
            for (size_t n = 1; i < s.size(); i += n, ++digits) {
                const int d = digit_value(utf8_at(s, i, &n));
                if (d < 0) break;
                if (v > (std::numeric_limits<int64_t>::max() - d) / 10) return false;
                v = v * 10 + d;
            }
            return digits > 0;
        };
        int64_t step = 0, steps = 0;
        spaces();
        if (!number(step) || i >= s.size() || s[i] != '/') continue;
        ++i;
        if (!number(steps)) continue;
        spaces();
        if (i < s.size() && s[i] == '-') return std::make_pair(step, steps);
    }
    return std::nullopt;
}

/// the last of re.findall(prefix + r"([0-9.]+) ?" + unit, text)
std::optional<std::string> last_figure(std::string_view s, std::string_view prefix, std::string_view unit) {
    std::optional<std::string> found;
    for (size_t at = s.find(prefix); at != std::string_view::npos;) {
        const size_t i = at + prefix.size();
        size_t j = i;
        while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || s[j] == '.')) ++j;
        size_t end = std::string_view::npos;
        if (j > i) {
            if (j < s.size() && s[j] == ' ' && s.substr(j + 1, unit.size()) == unit) end = j + 1 + unit.size();
            else if (s.substr(j, unit.size()) == unit) end = j + unit.size();
        }
        if (end != std::string_view::npos) found = std::string(s.substr(i, j - i));
        at = s.find(prefix, end != std::string_view::npos ? end : at + 1);
    }
    return found;
}

/// round(x, 2): the decimal correctly rounded (ties to even), read back
double round2(double x) {
    char buf[400];
    const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed, 2);
    double out = x;
    std::from_chars(buf, r.ptr, out);
    return out;
}

/// the child's stderr, a file it appends to (its stdout is a pipe: the platform layer has no stderr=STDOUT), taken into
/// the log in whole lines whenever stdout has been quiet for a moment, and at the end. Not after each stdout line: the
/// pipe's lines are read later than they were written, and sd-cli's errors follow a burst of its stdout lines
class ErrLog {
public:
    explicit ErrLog(fs::path path) : path_(std::move(path)) {}
    /// what the file got since the last call, its line ends made "\n" (whole lines unless `all`)
    std::string take(bool all) {
        if (!f_.is_open()) f_.open(path_, std::ios::binary);
        if (f_.is_open()) {
            f_.clear();
            f_.seekg(0, std::ios::end);
            const std::streamoff end = f_.tellg();
            if (end > pos_) {
                std::string chunk((size_t) (end - pos_), '\0');
                f_.seekg(pos_);
                f_.read(chunk.data(), (std::streamsize) chunk.size());
                chunk.resize((size_t) f_.gcount());
                pos_ += (std::streamoff) chunk.size();
                part_ += chunk;
            }
        }
        size_t cut = part_.size();
        if (!all) {
            const size_t nl = part_.rfind('\n');
            cut = nl == std::string::npos ? 0 : nl + 1;
        }
        std::string out = newlines(std::string_view(part_).substr(0, cut));
        part_.erase(0, cut);
        return out;
    }

private:
    fs::path path_;
    std::ifstream f_;
    std::streamoff pos_ = 0;
    std::string part_;
};

template <class F>
struct OnExit {
    F f;
    ~OnExit() { f(); }
};
template <class F>
OnExit(F) -> OnExit<F>;

}  // namespace

// ================================================================================================= the request
std::pair<int64_t, int64_t> parse_size(const json& value, int64_t max_side) {
    if (value.is_null() || is_str(value, "auto")) return {1024, 1024};
    if (!value.is_string()) throw ImageRequestError("size: a string WIDTHxHEIGHT, e.g. \"1024x1024\"");
    // re.fullmatch(r"\s*(\d{2,5})\s*[xX×]\s*(\d{2,5})\s*")
    const std::string& s = value.get_ref<const std::string&>();
    size_t i = 0;
    auto spaces = [&] {
        for (size_t n = 1; i < s.size() && py_isspace(utf8_at(s, i, &n)); i += n) {}
    };
    auto side = [&](int64_t& v) {   // \d{2,5}: a longer run of digits fails the match
        int digits = 0;
        v = 0;
        for (size_t n = 1; i < s.size() && digits <= 5; i += n, ++digits) {
            const int d = digit_value(utf8_at(s, i, &n));
            if (d < 0) break;
            v = v * 10 + d;
        }
        return digits >= 2 && digits <= 5;
    };
    int64_t w = 0, h = 0;
    bool ok = false;
    spaces();
    if (side(w)) {
        spaces();
        size_t n = 1;
        const uint32_t x = i < s.size() ? utf8_at(s, i, &n) : 0;
        if (x == 'x' || x == 'X' || x == 0xD7) {
            i += n;
            spaces();
            if (side(h)) {
                spaces();
                ok = i == s.size();
            }
        }
    }
    if (!ok) throw ImageRequestError("size: WIDTHxHEIGHT, e.g. \"1024x1024\", not " + py_repr_value(value));
    for (const int64_t v : {w, h})
        if (v < kMinSide || v > max_side || v % kSizeStep != 0)
            throw ImageRequestError("size " + std::to_string(w) + "x" + std::to_string(h) + ": each side a multiple of " +
                                    std::to_string(kSizeStep) + " from " + std::to_string(kMinSide) + " to " +
                                    std::to_string(max_side));
    return {w, h};
}

namespace {

/// a request's whole number from lo to hi, or `dflt` when it has none
std::optional<int64_t> int_field(const json& req, const char* key, int64_t lo, int64_t hi, std::optional<int64_t> dflt) {
    const json& v = jget(req, key);
    if (v.is_null()) return dflt;
    bool ok = v.is_number_integer();
    if (ok && v.is_number_unsigned()) ok = hi >= 0 && v.get<uint64_t>() <= (uint64_t) hi && (lo <= 0 || v.get<uint64_t>() >= (uint64_t) lo);
    else if (ok) ok = lo <= v.get<int64_t>() && v.get<int64_t>() <= hi;
    if (!ok)
        throw ImageRequestError(std::string(key) + ": a whole number from " + std::to_string(lo) + " to " + std::to_string(hi));
    return v.is_number_unsigned() ? (int64_t) v.get<uint64_t>() : v.get<int64_t>();
}

}  // namespace

Job parse_request(const json& req, const json& cfg) {
    const json& prompt = jget(req, "prompt");
    if (!prompt.is_string() || strip(prompt.get_ref<const std::string&>()).empty())
        throw ImageRequestError("prompt: a non-empty string");
    Job job;
    job.prompt = prompt.get<std::string>();
    if (utf8_len(job.prompt) > 16000) throw ImageRequestError("prompt: at most 16000 characters");
    if (has_nul(job.prompt)) throw ImageRequestError("prompt: no NUL characters");
    const json fmt = or_else(jget(req, "response_format"), "b64_json");
    if (!is_str(fmt, "b64_json") && !is_str(fmt, "url")) throw ImageRequestError("response_format: b64_json or url");
    const json& out_fmt = jget(req, "output_format");
    if (!out_fmt.is_null() && !is_str(out_fmt, "png")) throw ImageRequestError("output_format: only png");
    std::tie(job.width, job.height) = parse_size(jget(req, "size"), py_int(or_else(jget(cfg, "max_side"), 2048)));
    const int64_t max_n = py_int(or_else(jget(cfg, "max_n"), 4));
    job.n = *int_field(req, "n", 1, max_n, 1);
    const int64_t steps = py_int(or_else(jget(cfg, "steps"), kTurboSteps));
    job.steps = *int_field(req, "steps", 1, 100, steps);
    const auto seed = int_field(req, "seed", 0, int64_t(1) << 62, std::nullopt);
    job.seed = seed ? *seed : (int64_t) (std::random_device{}() & 0x7FFFFFFFu);   // SystemRandom().randrange(0, 2**31)
    job.response_format = fmt.get<std::string>();
    return job;
}

// ================================================================================================= the command
std::vector<std::string> sd_cli_args(const json& cfg, const Job& job, const std::string& out_pattern) {
    auto at = [&](const char* k) -> std::string {
        if (!cfg.is_object() || !cfg.contains(k)) throw ConfigTypeError("KeyError", std::string("'") + k + "'");
        return py_str(cfg[k]);
    };
    std::vector<std::string> cmd = {at("exe"), "--diffusion-model", at("diffusion_model"), "--vae", at("vae"), "--llm", at("llm"),
                                    "-p", job.prompt, "--steps", std::to_string(job.steps), "--cfg-scale", "1.0",
                                    "--sampling-method", "euler", "-W", std::to_string(job.width), "-H", std::to_string(job.height)};
    if (job.steps == kTurboSteps) {
        cmd.push_back("--sigmas");
        cmd.push_back(py_str(or_else(jget(cfg, "sigmas"), std::string(kTurboSigmas))));
    }
    cmd.insert(cmd.end(), {"-s", std::to_string(job.seed)});
    if (job.n > 1) cmd.insert(cmd.end(), {"-b", std::to_string(job.n)});
    cmd.insert(cmd.end(), {"-o", out_pattern});
    const json& args = jget(cfg, "args");
    if (truthy(args))
        for (const json& a : py_iter(args)) cmd.push_back(py_str(a));
    if (jget(cfg, "native").is_boolean() && jget(cfg, "native").get<bool>())   // Lamina's DiT runner inside sd-cli (CUDA builds)
        cmd.insert(cmd.end(), {"--model-args", "qwen_image_2_1_native=1"});
    return cmd;
}

Environment image_env(const json& cfg, const Environment* base) {
    Environment env = base != nullptr ? *base : environment();
    std::vector<std::string> dirs;
    const json& lib_dirs = jget(cfg, "lib_dirs");
    if (truthy(lib_dirs))
        for (const json& d : py_iter(lib_dirs)) {
            if (!d.is_string()) throw ConfigTypeError("TypeError", path_type_error(d));
            if (path_is_dir(d.get<std::string>())) dirs.push_back(d.get<std::string>());
        }
    if (!dirs.empty()) {
#ifdef _WIN32
        const char *var = "PATH", *pathsep = ";";
#else
        const char *var = "LD_LIBRARY_PATH", *pathsep = ":";
#endif
        std::string value;
        for (const auto& d : dirs) value += (value.empty() ? "" : pathsep) + d;
        const auto it = env.find(var);
        if (it != env.end() && !it->second.empty()) value += pathsep + it->second;
        env[var] = value;
    }
    const json& gpu = jget(cfg, "gpu");
    if (!gpu.is_null()) {
        const json& backend = jget(cfg, "backend");
        if (backend.is_array() || backend.is_object())
            throw ConfigTypeError("TypeError", "unhashable type: '" + py_type_name(backend) + "'");
        const char* var = is_str(backend, "cuda") ? "CUDA_VISIBLE_DEVICES" : is_str(backend, "vulkan") ? "GGML_VK_VISIBLE_DEVICES" : nullptr;
        if (var != nullptr) {
            env[var] = py_str(gpu);
            if (is_str(backend, "cuda")) env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
        }
    }
    const json& extra = jget(cfg, "env");
    if (truthy(extra)) {
        if (!extra.is_object()) throw ConfigTypeError("AttributeError", "'" + py_type_name(extra) + "' object has no attribute 'items'");
        for (auto it = extra.begin(); it != extra.end(); ++it) env[it.key()] = py_str(it.value());
    }
    return env;
}

int64_t need_mib(const json& cfg, int64_t width, int64_t height) {
    if (truthy(jget(cfg, "need_mib"))) return py_int(cfg["need_mib"]);
    int64_t files = 0;
    for (const char* k : {"diffusion_model", "vae", "llm"}) {
        const json& p = jget(cfg, k);
        if (!p.is_string()) continue;   // missing (KeyError) or not a path (TypeError): not counted
        if (has_nul(p.get_ref<const std::string&>())) throw std::invalid_argument("stat: embedded null character in path");
        files += stat_size(p.get<std::string>()).value_or(0);
    }
    const double area = std::clamp((double) width * (double) height * 4096.0, -0x1p61, 0x1p61);   // 4 GiB a megapixel
    const int64_t work = (int64_t(2) << 30) + (int64_t) area;
    return (files + work) >> 20;
}

// ================================================================================================= ImageGen
ImageGen::ImageGen(const json& cfg_, std::string log_) : cfg(cfg_), log(std::move(log_)) {
    const json pol = or_else(jget(cfg, "gpu_policy"), "auto");
    if (std::none_of(kPolicies.begin(), kPolicies.end(), [&](std::string_view p) { return is_str(pol, p); }))
        throw std::invalid_argument("images \"gpu_policy\" must be one of auto, keep, unload, not " + py_repr_value(pol));
    const json rel = or_else(jget(cfg, "reload"), "lazy");
    if (!is_str(rel, "lazy") && !is_str(rel, "eager"))
        throw std::invalid_argument("images \"reload\" must be lazy or eager, not " + py_repr_value(rel));
    for (const char* k : {"exe", "diffusion_model", "vae", "llm"}) {
        const json& v = jget(cfg, k);
        if (!v.is_string() || v.get_ref<const std::string&>().empty())
            throw std::invalid_argument(std::string("the config's \"images\" section needs \"") + k +
                                        "\" (run setup with --images again)");
    }
    policy = pol.get<std::string>();
    reload = rel.get<std::string>();
    model = or_else(jget(cfg, "model_name"), std::string(kModelName));
    const json& out = jget(cfg, "out_dir");
    if (truthy(out) && !out.is_string()) throw ConfigTypeError("TypeError", path_type_error(out));
    dir = truthy(out) ? u8_path(path_str(out.get<std::string>())) : make_temp_dir(u8_path(temp_dir()), "lamina-images-");
    fs::create_directories(dir);
    keep_files = py_int(or_else(jget(cfg, "keep_files"), 64));
}

std::vector<std::string> ImageGen::missing() const {
    std::vector<std::string> out;
    for (const char* k : {"exe", "diffusion_model", "vae", "llm"}) {
        const std::string p = py_str(jget(cfg, k));
        if (!path_exists(p)) out.push_back(p);
    }
    return out;
}

void ImageGen::say(const std::string& text) const {
    std::string line = "[lamina] " + text;
    if (line.back() == '\n') line += '\n';   // print() ends it with one of its own
    lamina::say(line);
    if (log.empty()) return;
    std::ofstream f(u8_path(log), std::ios::app);   // text mode, as the Python's log file
    f << "[images] " << text << "\n";
}

void ImageGen::prune() {
    std::vector<std::pair<fs::file_time_type, fs::directory_entry>> files;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!stored_name(u8_str(it->path().filename()))) continue;
        std::error_code e2;
        const auto t = fs::last_write_time(it->path(), e2);
        if (!e2) files.emplace_back(t, *it);
    }
    std::stable_sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    // files[:max(0, len(files) - keep_files)]: all of them when keep_files is below 0
    const size_t drop = keep_files < 0 ? files.size() : files.size() - std::min(files.size(), (size_t) keep_files);
    for (size_t i = 0; i < drop; ++i) {
        std::error_code e2;
        if (!files[i].second.is_directory(e2)) fs::remove(files[i].second.path(), e2);
    }
}

std::optional<fs::path> ImageGen::file(const std::string& name) const {
    if (!stored_name(name)) return std::nullopt;
    const fs::path p = dir / u8_path(name);
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return std::nullopt;
    return p;
}

std::pair<std::vector<std::string>, json> ImageGen::run(const Job& job, Cancel* cancel, const std::string& policy_used) {
    std::error_code ec;
    fs::create_directories(dir, ec);   // a temporary folder the system's cleanup took since
    const fs::path work = make_temp_dir(dir, "job-");
    // the job's folder goes however the job ends, and the progress with it
    OnExit done{[&] {
        std::error_code e;
        for (fs::directory_iterator it(work, e), end; !e && it != end; it.increment(e)) {
            std::error_code e2;
            if (!it->is_directory(e2)) fs::remove(it->path(), e2);
        }
        fs::remove(work, e);
        std::lock_guard<std::mutex> lk(state_mu_);
        progress_.reset();
    }};
    const std::vector<std::string> cmd = sd_cli_args(cfg, job, u8_str(work / "img_%d.png"));
    const double timeout = py_float(or_else(jget(cfg, "timeout_s"), 3600));
    const auto t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lk(state_mu_);
        progress_ = {0, job.steps};
    }
    std::string raw;   // the log: sd-cli's lines, "\n"-ended
    std::string why;
    std::optional<int> rc;
    {
        std::lock_guard<std::mutex> lk(run_mu_);
        SpawnOptions o;
        o.args = cmd;
        o.cwd = u8_str(work);
        o.env = image_env(cfg);
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::pipe;
        o.cr_lines = true;   // sd.cpp ends each progress update with a CR alone
        o.err = SpawnOptions::file;
        o.err_file = u8_str(work / "sd-cli.err");
        std::unique_ptr<Process> proc;
        try {
            proc = std::make_unique<Process>(std::move(o));
        } catch (const std::invalid_argument&) {
            throw;   // a command the system cannot take (Popen's ValueError)
        } catch (const std::exception& e) {
            throw ImageJobFailed("sd-cli did not start (" + cmd[0] + "): " + e.what());
        }
        ErrLog err(work / "sd-cli.err");
        auto note = [&](std::string_view text) {   // a line's progress
            if (const auto step = step_of(decode_replace(text))) {
                std::lock_guard<std::mutex> lk2(state_mu_);
                progress_ = step;
            }
        };
        auto took = [&](const std::string& text) {   // stderr's new lines
            for (const auto& line : split(text, "\n")) note(line);
            raw += text;
        };
        auto out_line = [&](const std::string& line) {   // a stdout line ("\r" ends lines too: the progress bar's)
            for (const auto& piece : split(line, "\r")) {
                note(piece);
                raw += piece;
                raw += '\n';
            }
        };
        while (!proc->poll()) {
            if (cancel != nullptr && cancel->is_set()) why = "cancelled (the client left)";
            else if (since(t0) > timeout) why = "it ran past timeout_s (" + py_g(timeout) + " s)";
            if (!why.empty()) {
                proc->kill();
                break;
            }
            std::string line;
            const int got = proc->read_line(line, 0.2);
            if (got == 1) {
                out_line(line);
                continue;
            }
            took(err.take(false));   // stdout is quiet: stderr's lines since then go here
            if (got < 0) {   // its stdout closed while it runs on
                if (cancel != nullptr) cancel->wait(0.2);
                else std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        proc->wait(30);
        const auto until = Clock::now() + std::chrono::seconds(10);   // the rest of its output, as the reader's join
        for (std::string line;;) {
            const double left = std::chrono::duration<double>(until - Clock::now()).count();
            if (left <= 0 || proc->read_line(line, left) != 1) break;
            out_line(line);
        }
        rc = proc->poll();
        proc->kill();   // whatever it started and left running
        proc.reset();
        took(err.take(true));
    }
    const double secs = since(t0);
    {
        std::lock_guard<std::mutex> lk(state_mu_);
        progress_.reset();
    }
    const std::string text = decode_replace(raw);
    std::vector<fs::path> outs;
    for (int64_t i = 0; i < job.n; ++i) outs.push_back(work / ("img_" + std::to_string(i) + ".png"));
    const bool all_there = std::all_of(outs.begin(), outs.end(), [](const fs::path& p) {
        std::error_code e;
        return fs::is_regular_file(p, e);
    });
    if (!why.empty() || !rc || *rc != 0 || !all_there) {
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            totals_["failed"] = totals_["failed"].get<int64_t>() + 1;
        }
        const auto lines = splitlines(strip(text));
        const size_t first = lines.size() > 12 ? lines.size() - 12 : 0;
        std::string tail;
        for (size_t i = first; i < lines.size(); ++i) tail += (i > first ? "\n" : "") + std::string(lines[i]);
        const std::string what = !why.empty() ? why : rc && *rc != 0 ? "sd-cli exited with code " + exit_code_text(*rc) : "sd-cli wrote no image";
        say("image job failed: " + what + "\n" + tail);
        throw ImageJobFailed(what + "; the end of its log:\n" + tail);
    }
    std::vector<std::string> images;
    for (const auto& p : outs) images.push_back(read_file(p));
    json figures = {{"seconds", round2(secs)},
                    {"steps", job.steps},
                    {"n", job.n},
                    {"size", std::to_string(job.width) + "x" + std::to_string(job.height)},
                    {"seed", job.seed},
                    {"backend", jget(cfg, "backend")},
                    {"quant", jget(cfg, "quant")}};
    static constexpr std::string_view kFigures[][3] = {
        {"text_encode_ms", "get_learned_condition completed, taking ", "ms"},
        {"sampling_s", "sampling completed, taking ", "s"},
        {"vae_decode_s", "decode_first_stage completed, taking ", "s"}};
    for (const auto& f : kFigures)
        if (const auto v = last_figure(text, f[1], f[2])) figures[std::string(f[0])] = float_of_str(*v);
    if (!policy_used.empty()) figures["gpu_policy"] = policy_used;
    std::lock_guard<std::mutex> lk(state_mu_);
    totals_["jobs"] = totals_["jobs"].get<int64_t>() + 1;
    totals_["images"] = totals_["images"].get<int64_t>() + job.n;
    totals_["seconds"] = totals_["seconds"].get<double>() + secs;
    last_ = figures;
    return {std::move(images), std::move(figures)};
}

std::string ImageGen::store(const std::string& data) {
    std::lock_guard<std::mutex> lk(files_mu_);
    const std::string name = token_hex(16) + ".png";
    write_file(dir / u8_path(name), data);
    prune();
    return name;
}

json ImageGen::response(const std::vector<std::string>& images, const Job& job, const json& figures, const std::string& base_url) {
    json data = json::array();
    for (size_t i = 0; i < images.size(); ++i) {
        json item = {{"revised_prompt", job.prompt}, {"seed", job.seed + (int64_t) i}};
        if (job.response_format == "url") item["url"] = base_url + "/v1/images/files/" + store(images[i]);
        else item["b64_json"] = base64::encode(images[i]);
        data.push_back(std::move(item));
    }
    return {{"created", (int64_t) std::time(nullptr)},
            {"data", std::move(data)},
            {"model", model},
            {"output_format", "png"},
            {"size", std::to_string(job.width) + "x" + std::to_string(job.height)},
            {"lamina", figures}};
}

json ImageGen::status() const {
    std::lock_guard<std::mutex> lk(state_mu_);
    json out = {{"enabled", true},
                {"model", model},
                {"backend", jget(cfg, "backend")},
                {"quant", jget(cfg, "quant")},
                {"gpu_policy", policy}};
    for (auto it = totals_.begin(); it != totals_.end(); ++it) out[it.key()] = it.value();
    if (progress_) {
        out["step"] = progress_->first;
        out["steps"] = progress_->second;
    }
    if (truthy(last_)) out["last"] = last_;
    return out;
}

}  // namespace lamina::images
