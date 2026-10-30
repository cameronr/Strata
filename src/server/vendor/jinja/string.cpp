#include "jinja/string.h"
#include <cstdint>
#include "jinja/value.h"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace jinja {

//
// string_part
//

bool string_part::is_uppercase() const {
    for (char c : val) {
        if (std::islower(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

bool string_part::is_lowercase() const {
    for (char c : val) {
        if (std::isupper(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

//
// string
//

void string::mark_input() {
    for (auto & part : parts) {
        part.is_input = true;
    }
}

std::string string::str() const {
    if (parts.size() == 1) {
        return parts[0].val;
    }
    std::ostringstream oss;
    for (const auto & part : parts) {
        oss << part.val;
    }
    return oss.str();
}

size_t string::length() const {
    size_t len = 0;
    for (const auto & part : parts) {
        len += part.val.length();
    }
    return len;
}

void string::hash_update(hasher & hash) const noexcept {
    for (const auto & part : parts) {
        hash.update(part.val.data(), part.val.length());
    }
}

bool string::all_parts_are_input() const {
    for (const auto & part : parts) {
        if (!part.is_input) {
            return false;
        }
    }
    return true;
}

bool string::is_uppercase() const {
    for (const auto & part : parts) {
        if (!part.is_uppercase()) {
            return false;
        }
    }
    return true;
}

bool string::is_lowercase() const {
    for (const auto & part : parts) {
        if (!part.is_lowercase()) {
            return false;
        }
    }
    return true;
}

// mark this string as input if other has ALL parts as input
void string::mark_input_based_on(const string & other) {
    if (other.all_parts_are_input()) {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }
}

string & string::append(const string & other) {
    for (const auto & part : other.parts) {
        parts.push_back(part);
    }
    return *this;
}

// in-place transformation

using transform_fn = std::function<std::string(const std::string&)>;
static string apply_transform(string & self, const transform_fn & fn) {
    for (auto & part : self.parts) {
        part.val = fn(part.val);
    }
    return self;
}

string string::uppercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::toupper);
        return res;
    });
}
string string::lowercase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        std::transform(res.begin(), res.end(), res.begin(), ::tolower);
        return res;
    });
}
string string::capitalize() {
    return apply_transform(*this, [](const std::string & s) {
        if (s.empty()) return s;
        std::string res = s;
        res[0] = ::toupper(static_cast<unsigned char>(res[0]));
        std::transform(res.begin() + 1, res.end(), res.begin() + 1, ::tolower);
        return res;
    });
}
string string::titlecase() {
    return apply_transform(*this, [](const std::string & s) {
        std::string res = s;
        bool capitalize_next = true;
        for (char &c : res) {
            if (isspace(static_cast<unsigned char>(c))) {
                capitalize_next = true;
            } else if (capitalize_next) {
                c = ::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else {
                c = ::tolower(static_cast<unsigned char>(c));
            }
        }
        return res;
    });
}
// Lamina: Python's str.isspace, which str.strip() (and so jinja2's trim) strips when no characters are given
static bool py_space_at(const std::string & s, size_t i, size_t & n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = c;
    n = 1;
    if (c >= 0xC2 && c <= 0xDF && i + 1 < s.size()) {
        cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        n = 2;
    } else if (c >= 0xE0 && c <= 0xEF && i + 2 < s.size()) {
        cp = ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
             (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        n = 3;
    } else if (c >= 0x80) {
        return false;
    }
    return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

string string::strip(bool left, bool right, std::optional<const std::string_view> chars) {
    static auto strip_part = [](const std::string & s, bool left, bool right, std::optional<const std::string_view> chars) -> std::string {
        size_t start = 0;
        size_t end = s.length();
        if (chars) {
            auto match_char = [&chars](unsigned char c) -> bool { return (*chars).find(c) != std::string::npos; };
            if (left) {
                while (start < end && match_char(static_cast<unsigned char>(s[start]))) {
                    ++start;
                }
            }
            if (right) {
                while (end > start && match_char(static_cast<unsigned char>(s[end - 1]))) {
                    --end;
                }
            }
            return s.substr(start, end - start);
        }
        size_t n = 1;
        if (left) {
            while (start < end && py_space_at(s, start, n)) {
                start += n;
            }
        }
        if (right) {
            while (end > start) {
                size_t k = end - 1;   // the last character's first byte
                while (k > start && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80 && end - k < 3) {
                    --k;
                }
                if (!py_space_at(s, k, n) || k + n != end) {
                    break;
                }
                end = k;
            }
        }
        return s.substr(start, end - start);
    };
    if (parts.empty()) {
        return *this;
    }
    if (left) {
        for (size_t i = 0; i < parts.size(); ++i) {
            parts[i].val = strip_part(parts[i].val, true, false, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                --i;
                continue;
            } else {
                break;
            }
        }
    }
    if (right) {
        for (size_t i = parts.size(); i-- > 0;) {
            parts[i].val = strip_part(parts[i].val, false, true, chars);
            if (parts[i].val.empty()) {
                // remove empty part
                parts.erase(parts.begin() + i);
                continue;
            } else {
                break;
            }
        }
    }
    return *this;
}

} // namespace jinja
