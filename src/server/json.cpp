#include "json.hpp"

#include "jinja/value.h"   // py_float_repr

#include <charconv>
#include <cmath>
#include <cstdio>
#include <random>

namespace lamina {
namespace {

void dump_string(std::string& out, std::string_view s, bool ensure_ascii) {
    out += '"';
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t cp = utf8_at(s, i, &n);
        const bool invalid = n == 1 && (unsigned char) s[i] >= 0x80;
        const uint32_t c = invalid ? 0xFFFD : cp;
        i += n;
        switch (c) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        if (c < 0x20 || (ensure_ascii && c >= 0x7F)) {
            if (c >= 0x10000) {
                const uint32_t v = c - 0x10000;
                std::snprintf(buf, sizeof buf, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            } else {
                std::snprintf(buf, sizeof buf, "\\u%04x", c);
            }
            out += buf;
        } else if (invalid) {
            utf8_append(out, 0xFFFD);
        } else {
            out.append(s.data() + i - n, n);
        }
    }
    out += '"';
}

void dump(std::string& out, const json& v, bool ensure_ascii, bool compact, int indent, int level) {
    const char* item_sep = indent >= 0 || compact ? "," : ", ";
    const char* key_sep = compact ? ":" : ": ";
    auto newline = [&](int lvl) {
        if (indent < 0) return;
        out += '\n';
        out.append((size_t) (indent * lvl), ' ');
    };
    switch (v.type()) {
        case json::value_t::null: out += "null"; return;
        case json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; return;
        case json::value_t::number_integer: out += std::to_string(v.get<int64_t>()); return;
        case json::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); return;
        case json::value_t::number_float: out += jinja::py_float_repr(v.get<double>(), true); return;
        case json::value_t::string: dump_string(out, v.get_ref<const std::string&>(), ensure_ascii); return;
        case json::value_t::array: {
            if (v.empty()) {
                out += "[]";
                return;
            }
            out += '[';
            bool first = true;
            for (const auto& x : v) {
                if (!first) out += item_sep;
                first = false;
                newline(level + 1);
                dump(out, x, ensure_ascii, compact, indent, level + 1);
            }
            newline(level);
            out += ']';
            return;
        }
        case json::value_t::object: {
            if (v.empty()) {
                out += "{}";
                return;
            }
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += item_sep;
                first = false;
                newline(level + 1);
                dump_string(out, it.key(), ensure_ascii);
                out += key_sep;
                dump(out, it.value(), ensure_ascii, compact, indent, level + 1);
            }
            newline(level);
            out += '}';
            return;
        }
        default: out += "null"; return;
    }
}

}  // namespace

std::string py_dumps(const json& v, bool ensure_ascii, bool compact, int indent) {
    std::string out;
    dump(out, v, ensure_ascii, compact, indent, 0);
    return out;
}

namespace {

/// Python's json.loads stops at its recursion limit (about 1000 levels).  nlohmann copies and dumps a value
/// recursively: a value nested deeper than any real one could use up a request thread's stack.
constexpr int kMaxDepth = 1000;

/// the first bracket past kMaxDepth levels ('[' or '{'), or 0
char too_deep(std::string_view text) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (const char c : text) {
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '[' || c == '{') {
            if (++depth > kMaxDepth) return c;
        } else if (c == ']' || c == '}') {
            --depth;
        }
    }
    return 0;
}

}  // namespace

bool py_loads(std::string_view text, json& out, std::string* err) {
    if (const char c = too_deep(text)) {
        if (err != nullptr)
            *err = std::string("maximum recursion depth exceeded while decoding a JSON ") +
                   (c == '[' ? "array" : "object") + " from a unicode string";
        return false;
    }
    try {
        out = json::parse(text.begin(), text.end());
        return true;
    } catch (const std::exception& e) {
        if (err != nullptr) {   // "[json.exception.parse_error.101] parse error at line 1, column 3: ..." without its tag
            *err = e.what();
            const size_t tag = err->find("] ");
            if (starts_with(*err, "[json.exception.") && tag != std::string::npos) err->erase(0, tag + 2);
        }
        return false;
    }
}

std::string py_repr(double v) { return jinja::py_float_repr(v); }

double py_round(double x, int n) {
    if (!std::isfinite(x)) return x;
    char buf[400];
    const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::fixed, n);
    double out = x;
    if (r.ec == std::errc()) std::from_chars(buf, r.ptr, out);
    return out;
}

std::string py_g(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

namespace {

/// str.isprintable's false (the categories Cc, Cf, Zl, Zp and Zs but the space), for the code points that reach a repr
bool py_unprintable(uint32_t cp) {
    return cp < 0x20 || (cp >= 0x7f && cp <= 0xa0) || cp == 0xad || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200f) ||
           (cp >= 0x2028 && cp <= 0x202f) || (cp >= 0x205f && cp <= 0x2064) || cp == 0x3000 || cp == 0xfeff ||
           (cp >= 0xd800 && cp <= 0xdfff);
}

void repr_str(const std::string& s, std::string& out) {
    const bool dq = s.find('\'') != std::string::npos && s.find('"') == std::string::npos;
    const char q = dq ? '"' : '\'';
    out += q;
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t cp = utf8_at(s, i, &n);
        char buf[16];
        if (cp == '\\') out += "\\\\";
        else if (cp == (uint32_t) q) (out += '\\') += q;
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (cp == '\t') out += "\\t";
        else if (py_unprintable(cp)) {
            if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
            else std::snprintf(buf, sizeof buf, "\\u%04x", cp);
            out += buf;
        } else {
            out.append(s, i, n);
        }
        i += n;
    }
    out += q;
}

void repr_value(const json& v, std::string& out) {
    switch (v.type()) {
        case json::value_t::null: out += "None"; break;
        case json::value_t::boolean: out += v.get<bool>() ? "True" : "False"; break;
        case json::value_t::number_integer: out += std::to_string(v.get<int64_t>()); break;
        case json::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
        case json::value_t::number_float: out += py_repr(v.get<double>()); break;
        case json::value_t::string: repr_str(v.get_ref<const std::string&>(), out); break;
        case json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto& x : v) {
                if (!first) out += ", ";
                first = false;
                repr_value(x, out);
            }
            out += ']';
            break;
        }
        case json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                repr_str(it.key(), out);
                out += ": ";
                repr_value(it.value(), out);
            }
            out += '}';
            break;
        }
        default: out += "None"; break;
    }
}

}  // namespace

std::string py_repr_value(const json& v) {
    std::string out;
    repr_value(v, out);
    return out;
}

const json& jget(const json& o, std::string_view key) {
    static const json null_value;
    if (!o.is_object()) return null_value;
    const auto it = o.find(std::string(key));
    return it == o.end() ? null_value : *it;
}

bool truthy(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return false;
        case json::value_t::boolean: return v.get<bool>();
        case json::value_t::number_integer: return v.get<int64_t>() != 0;
        case json::value_t::number_unsigned: return v.get<uint64_t>() != 0;
        case json::value_t::number_float: return v.get<double>() != 0.0;
        case json::value_t::string: return !v.get_ref<const std::string&>().empty();
        case json::value_t::array:
        case json::value_t::object: return !v.empty();
        default: return false;
    }
}

std::string jstr(const json& o, std::string_view key, std::string_view dflt) {
    const json& v = jget(o, key);
    return v.is_string() ? v.get<std::string>() : std::string(dflt);
}

bool is_int(const json& v) { return v.is_number_integer(); }
bool is_num(const json& v) { return v.is_number(); }

// ---------------------------------------------------------------------------------------------- strings
bool py_isspace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

uint32_t utf8_at(std::string_view s, size_t i, size_t* n) {
    const unsigned char c = (unsigned char) s[i];
    auto cont = [&](size_t k) { return i + k < s.size() && ((unsigned char) s[i + k] & 0xC0) == 0x80; };
    if (c < 0x80) {
        *n = 1;
        return c;
    }
    if (c >= 0xC2 && c <= 0xDF && cont(1)) {
        *n = 2;
        return ((c & 0x1Fu) << 6) | ((unsigned char) s[i + 1] & 0x3Fu);
    }
    if (c >= 0xE0 && c <= 0xEF && cont(1) && cont(2)) {
        *n = 3;
        return ((c & 0x0Fu) << 12) | (((unsigned char) s[i + 1] & 0x3Fu) << 6) | ((unsigned char) s[i + 2] & 0x3Fu);
    }
    if (c >= 0xF0 && c <= 0xF4 && cont(1) && cont(2) && cont(3)) {
        *n = 4;
        return ((c & 0x07u) << 18) | (((unsigned char) s[i + 1] & 0x3Fu) << 12) |
               (((unsigned char) s[i + 2] & 0x3Fu) << 6) | ((unsigned char) s[i + 3] & 0x3Fu);
    }
    *n = 1;
    return c;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

std::string_view lstrip(std::string_view s) {
    size_t i = 0;
    while (i < s.size()) {
        size_t n = 1;
        const uint32_t c = utf8_at(s, i, &n);
        if (!py_isspace(c)) break;
        i += n;
    }
    return s.substr(i);
}

std::string_view rstrip(std::string_view s) {
    size_t end = s.size();
    while (end > 0) {
        size_t start = end - 1;   // the last character's first byte
        while (start > 0 && ((unsigned char) s[start] & 0xC0) == 0x80 && end - start < 4) --start;
        size_t n = 1;
        const uint32_t c = utf8_at(s, start, &n);
        if (start + n != end || !py_isspace(c)) break;
        end = start;
    }
    return s.substr(0, end);
}

std::string_view strip(std::string_view s) { return rstrip(lstrip(s)); }

std::string_view lstrip_chars(std::string_view s, std::string_view chars) {
    const size_t i = s.find_first_not_of(chars);
    return i == std::string_view::npos ? std::string_view() : s.substr(i);
}

std::string_view rstrip_chars(std::string_view s, std::string_view chars) {
    const size_t i = s.find_last_not_of(chars);
    return i == std::string_view::npos ? std::string_view() : s.substr(0, i + 1);
}

std::string_view strip_chars(std::string_view s, std::string_view chars) {
    return rstrip_chars(lstrip_chars(s, chars), chars);
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.substr(s.size() - p.size()) == p; }

bool starts_with_any(std::string_view s, std::initializer_list<std::string_view> ps) {
    for (const std::string_view p : ps)
        if (starts_with(s, p)) return true;
    return false;
}

bool contains_space(std::string_view s) {
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        if (py_isspace(utf8_at(s, i, &n))) return true;
        i += n;
    }
    return false;
}

std::string replace_all(std::string s, std::string_view from, std::string_view to) {
    if (from.empty()) return s;
    size_t i = 0;
    while ((i = s.find(from, i)) != std::string::npos) {
        s.replace(i, from.size(), to);
        i += to.size();
    }
    return s;
}

std::vector<std::string> split(std::string_view s, std::string_view sep) {
    std::vector<std::string> out;
    size_t i = 0;
    while (true) {
        const size_t j = s.find(sep, i);
        if (j == std::string_view::npos) {
            out.emplace_back(s.substr(i));
            return out;
        }
        out.emplace_back(s.substr(i, j - i));
        i = j + sep.size();
    }
}

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out)
        if (c >= 'A' && c <= 'Z') c = (char) (c - 'A' + 'a');
    return out;
}

size_t utf8_len(std::string_view s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++n) {
        size_t k = 1;
        utf8_at(s, i, &k);
        i += k;
    }
    return n;
}

std::string random_hex(size_t n) {
    thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    std::string out(n, '0');
    for (size_t i = 0; i < n; ++i) out[i] = hex[rng() & 15];
    return out;
}

}  // namespace lamina
