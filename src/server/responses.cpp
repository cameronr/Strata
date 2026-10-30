// src/server/responses.cpp - the Responses API: a request -> the chat path's messages, tools and kwargs; the parser's
// events -> Responses output items and typed events (serve/responses.py's port; the overview is in responses.hpp).
#include "responses.hpp"

#include "src/unicode.h"   // llama.cpp's: code point classes (repr's printable characters)

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <utility>

namespace lamina::responses {
namespace {

// ---------------------------------------------------------------------------------------------- Python's view
bool is_str(const json& v, std::string_view s) { return v.is_string() && v.get_ref<const std::string&>() == s; }

bool is_one_of(const json& v, std::initializer_list<std::string_view> set) {
    if (!v.is_string()) return false;
    for (const std::string_view s : set)
        if (v.get_ref<const std::string&>() == s) return true;
    return false;
}

/// dict.get(key, dflt): the member when the key is there (also a null one)
json get_or(const json& o, const std::string& key, const json& dflt) {
    if (o.is_object()) {
        const auto it = o.find(key);
        if (it != o.end()) return *it;
    }
    return dflt;
}

/// type(v).__name__
std::string type_name(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "NoneType";
        case json::value_t::boolean: return "bool";
        case json::value_t::number_integer:
        case json::value_t::number_unsigned: return "int";
        case json::value_t::number_float: return "float";
        case json::value_t::string: return "str";
        case json::value_t::array: return "list";
        default: return "dict";
    }
}

bool printable(uint32_t cp) {   // str.isprintable() of one character (llama.cpp's table: Unicode 15.1)
    const unicode_cpt_flags f = unicode_cpt_flags_from_cpt(cp);
    return !(f.is_undefined || f.is_control || f.is_separator);
}

std::string repr(const json& v);

std::string repr_str(std::string_view s) {
    const bool single = s.find('\'') != std::string_view::npos;
    const char quote = single && s.find('"') == std::string_view::npos ? '"' : '\'';
    std::string out(1, quote);
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        size_t n = 1;
        const uint32_t c = utf8_at(s, i, &n);
        i += n;
        if (c == (uint32_t) quote || c == '\\') {
            out += '\\';
            out += (char) c;
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c < 0x20 || c == 0x7F) {
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            out += buf;
        } else if (c < 0x7F) {
            out += (char) c;
        } else if (printable(c)) {
            utf8_append(out, c);
        } else {
            if (c <= 0xFF) {
                std::snprintf(buf, sizeof buf, "\\x%02x", c);
            } else if (c <= 0xFFFF) {
                std::snprintf(buf, sizeof buf, "\\u%04x", c);
            } else {
                std::snprintf(buf, sizeof buf, "\\U%08x", c);
            }
            out += buf;
        }
    }
    out += quote;
    return out;
}

/// repr() of a JSON value, as Python writes the error messages that name one
std::string repr(const json& v) {
    switch (v.type()) {
        case json::value_t::null: return "None";
        case json::value_t::boolean: return v.get<bool>() ? "True" : "False";
        case json::value_t::number_integer: return std::to_string(v.get<int64_t>());
        case json::value_t::number_unsigned: return std::to_string(v.get<uint64_t>());
        case json::value_t::number_float: return py_repr(v.get<double>());
        case json::value_t::string: return repr_str(v.get_ref<const std::string&>());
        case json::value_t::array: {
            std::string out = "[";
            for (size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + repr(v[i]);
            return out + "]";
        }
        default: {
            std::string out = "{";
            for (auto it = v.begin(); it != v.end(); ++it)
                out += (it == v.begin() ? "" : ", ") + repr_str(it.key()) + ": " + repr(it.value());
            return out + "}";
        }
    }
}

/// str(v), as an f-string writes it
std::string str_of(const json& v) { return v.is_string() ? v.get<std::string>() : repr(v); }

/// sep.join(items): a TypeError on an item that is not a str
std::string py_join(const std::vector<json>& items, std::string_view sep = "") {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!items[i].is_string())
            throw Malformed("TypeError: sequence item " + std::to_string(i) + ": expected str instance, " +
                            type_name(items[i]) + " found");
        if (i) out += sep;
        out += items[i].get_ref<const std::string&>();
    }
    return out;
}

/// a + b where one side is a str
json py_add(const json& a, const json& b) {
    if (a.is_string() && b.is_string()) return a.get<std::string>() + b.get<std::string>();
    if (a.is_string()) throw Malformed("TypeError: can only concatenate str (not \"" + type_name(b) + "\") to str");
    if (a.is_array()) throw Malformed("TypeError: can only concatenate list (not \"" + type_name(b) + "\") to list");
    throw Malformed("TypeError: unsupported operand type(s) for +: '" + type_name(a) + "' and '" + type_name(b) + "'");
}

/// a += s: a list takes the characters of s, one by one
void py_iadd(json& a, const std::string& s) {
    if (a.is_string()) {
        a.get_ref<std::string&>() += s;
    } else if (a.is_array()) {
        for (size_t i = 0; i < s.size();) {
            size_t n = 1;
            utf8_at(s, i, &n);
            a.push_back(s.substr(i, n));
            i += n;
        }
    } else {
        throw Malformed("TypeError: unsupported operand type(s) for +=: '" + type_name(a) + "' and 'str'");
    }
}

/// a dict key's hash: lists and dicts have none
void hashable(const json& v) {
    if (v.is_array() || v.is_object()) throw Malformed("TypeError: unhashable type: '" + type_name(v) + "'");
}

/// == between dict keys: True == 1 == 1.0, a str only equals a str
bool key_eq(const json& a, const json& b) {
    if (a.is_string() || b.is_string())
        return a.is_string() && b.is_string() && a.get_ref<const std::string&>() == b.get_ref<const std::string&>();
    if (a.is_null() || b.is_null()) return a.is_null() && b.is_null();
    const auto integral = [](const json& v) { return v.is_boolean() || v.is_number_integer(); };
    if (integral(a) && integral(b)) {   // exactly, signed or not
        const auto split = [](const json& v) -> std::pair<bool, uint64_t> {
            if (v.is_boolean()) return {false, v.get<bool>() ? 1 : 0};
            if (v.is_number_unsigned()) return {false, v.get<uint64_t>()};
            const int64_t x = v.get<int64_t>();
            return {x < 0, x < 0 ? 0 - (uint64_t) x : (uint64_t) x};
        };
        return split(a) == split(b);
    }
    const auto num = [](const json& v) { return v.is_boolean() ? (v.get<bool>() ? 1.0 : 0.0) : v.get<double>(); };
    return num(a) == num(b);
}

/// a dict with hashable JSON keys, searched in order (a turn makes few calls)
struct KeyMap {
    std::vector<std::pair<json, size_t>> entries;
    const size_t* find(const json& key) const {
        for (const auto& [k, v] : entries)
            if (key_eq(k, key)) return &v;
        return nullptr;
    }
    void set(const json& key, size_t value) {
        for (auto& [k, v] : entries)
            if (key_eq(k, key)) {
                v = value;
                return;
            }
        entries.emplace_back(key, value);
    }
};

// ---------------------------------------------------------------------------------------------- strings
/// str.splitlines(): \n, \r, \r\n, \v, \f, \x1c-\x1e, \x85, U+2028 and U+2029 end a line
std::vector<std::string> splitlines(std::string_view s) {
    std::vector<std::string> lines;
    size_t start = 0, i = 0;
    while (i < s.size()) {
        size_t n = 1;
        const uint32_t c = utf8_at(s, i, &n);
        if (c == '\n' || c == '\r' || c == 0x0B || c == 0x0C || (c >= 0x1C && c <= 0x1E) || c == 0x85 ||
            c == 0x2028 || c == 0x2029) {
            lines.emplace_back(s.substr(start, i - start));
            if (c == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ++n;
            start = i + n;
        }
        i += n;
    }
    if (start < s.size()) lines.emplace_back(s.substr(start));
    return lines;
}

/// the byte length of the last character
size_t last_char_len(std::string_view s) {
    size_t start = s.size() - 1;
    while (start > 0 && ((unsigned char) s[start] & 0xC0) == 0x80 && s.size() - start < 4) --start;
    size_t n = 1;
    utf8_at(s, start, &n);
    return start + n == s.size() ? n : 1;
}

/// s[:chars]
std::string_view prefix_chars(std::string_view s, size_t chars) {
    size_t i = 0;
    for (size_t k = 0; k < chars && i < s.size(); ++k) {
        size_t n = 1;
        utf8_at(s, i, &n);
        i += n;
    }
    return s.substr(0, i);
}

constexpr std::string_view kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string b64encode(std::string_view in) {
    std::string out;
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t b0 = (unsigned char) in[i], b1 = i + 1 < in.size() ? (unsigned char) in[i + 1] : 0,
                       b2 = i + 2 < in.size() ? (unsigned char) in[i + 2] : 0;
        const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
        out += kB64[(v >> 18) & 63];
        out += kB64[(v >> 12) & 63];
        out += i + 1 < in.size() ? kB64[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? kB64[v & 63] : '=';
    }
    return out;
}

/// base64.b64decode(s, validate=True): the alphabet only, whole quads, padding only at the end of the last one
bool b64decode(std::string_view s, std::string& out) {
    size_t data = s.size();
    if (data > 0 && s[data - 1] == '=') --data;
    if (data > 0 && s.size() - data == 1 && s[data - 1] == '=') --data;
    const size_t pad = s.size() - data;
    if (data % 4 != (pad == 0 ? 0u : 4 - pad)) return false;
    uint32_t v = 0;
    int bits = 0;
    out.clear();
    for (size_t i = 0; i < data; ++i) {
        const size_t k = kB64.find(s[i]);
        if (k == std::string_view::npos) return false;
        v = (v << 6) | (uint32_t) k;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += (char) ((v >> bits) & 0xFF);
        }
    }
    return true;
}

std::string new_id(const std::string& prefix) { return prefix + "_" + random_hex(32); }

int64_t now() { return (int64_t) std::time(nullptr); }

// ---------------------------------------------------------------------------------------------- request -> chat
json text_part(const json& text) { return {{"type", "text"}, {"text", truthy(text) ? text : json("")}}; }

/// a message's (or a tool output's) content -> the template's form: a string, or text and image parts
json content_of(const json& content, const std::string& param) {
    if (content.is_null()) return "";
    if (content.is_string()) return content;
    if (!content.is_array()) throw ResponsesError("expected a string or an array of content parts", param);
    json parts = json::array();
    bool image = false;
    for (size_t j = 0; j < content.size(); ++j) {
        const json& part = content[j];
        const std::string at = param + "[" + std::to_string(j) + "]";
        if (part.is_string()) {
            parts.push_back(text_part(part));
            continue;
        }
        if (!part.is_object()) throw ResponsesError("expected a content part object", at);
        const json& kind = jget(part, "type");
        if (is_one_of(kind, {"input_text", "output_text", "text", "summary_text", "reasoning_text"})) {
            parts.push_back(text_part(jget(part, "text")));
        } else if (is_str(kind, "refusal")) {
            parts.push_back(text_part(jget(part, "refusal")));
        } else if (is_one_of(kind, {"input_image", "image_url"})) {
            if (!truthy(jget(part, "image_url")))
                throw ResponsesError("only images given as image_url (a data: or http(s) URL) are supported; "
                                     "this server keeps no files",
                                     at, "unsupported_parameter");
            parts.push_back({{"type", "input_image"}, {"image_url", part["image_url"]}});
            image = true;
        } else {
            throw ResponsesError("content parts of type " + repr(kind) + " are not supported (text and images are)",
                                 at + ".type", "unsupported_parameter");
        }
    }
    // frontend.py's _parts_of on these parts: text alone joins into a string, else text and image items in order.
    // Not frontend's parts_of: Python keeps a text or image source that is no string (or fails on it, joining),
    // which parts_of reads as "".
    if (!image) {
        std::vector<json> texts;
        for (const json& p : parts) texts.push_back(p["text"]);
        return py_join(texts);
    }
    json items = json::array();
    for (const json& p : parts) {
        if (is_str(p["type"], "input_image")) {
            json url = p["image_url"];
            if (url.is_object()) url = jget(url, "url");
            items.push_back({{"type", "image"}, {"source", truthy(url) ? url : json("")}});
        } else {
            items.push_back({{"type", "text"}, {"text", p["text"]}});
        }
    }
    return items;
}

/// the thinking a reasoning item carries: its reasoning_text content, else our own encrypted_content. A summary
/// alone, or another server's encrypted_content, carries none that can go back into the prompt.
json reasoning_text(const json& item, const std::string& param) {
    const json& content = jget(item, "content");
    std::vector<json> texts;
    if (content.is_array()) {
        for (const json& p : content)
            if (p.is_object() && is_one_of(jget(p, "type"), {"reasoning_text", "text"})) {
                const json& t = jget(p, "text");
                texts.push_back(truthy(t) ? t : json(""));
            }
    } else if (truthy(content) && !content.is_string() && !content.is_object()) {   // a str or dict: no parts
        throw Malformed("TypeError: '" + type_name(content) + "' object is not iterable");
    }
    if (!texts.empty()) return py_join(texts);
    const json& enc = jget(item, "encrypted_content");
    if (enc.is_string() && starts_with(enc.get_ref<const std::string&>(), kEncryptedPrefix)) {
        std::string raw;
        json value;
        if (!b64decode(std::string_view(enc.get_ref<const std::string&>()).substr(kEncryptedPrefix.size()), raw) ||
            !py_loads(raw, value) || !value.is_object() || !value.contains("text"))
            throw ResponsesError("this reasoning item's encrypted_content is damaged", param + ".encrypted_content",
                                 "invalid_encrypted_content");
        return value["text"];
    }
    return "";
}

/// a function_call's arguments (a JSON string) -> the mapping the template renders. Arguments that are not a JSON
/// object (a call the output ended inside) are kept as text rather than refusing the whole conversation.
json arguments(const json& raw, const std::string& param) {
    if (raw.is_object()) return raw;
    if (raw.is_null() || (raw.is_string() && strip(raw.get_ref<const std::string&>()).empty())) return json::object();
    if (!raw.is_string()) throw ResponsesError("function_call arguments must be a JSON string", param);
    json value;
    if (!py_loads(raw.get_ref<const std::string&>(), value)) return {{"arguments", raw}};
    return value.is_object() ? value : json{{"arguments", value}};
}

json tool_output(const json& item, const std::string& param) {
    const json& out = jget(item, "output");
    if (out.is_object()) return py_dumps(out, false);   // some clients send a JSON result as an object
    return out.is_string() ? out : content_of(out, param + ".output");
}

/// The template places tool results by position: each run of tool messages is put in the order of the calls of the
/// assistant turn before it (a client may send the results in the order they finished), then the private call ids
/// are dropped.
void order_tool_results(json& messages) {
    size_t i = 0;
    while (i < messages.size()) {
        if (!is_str(messages[i]["role"], "tool")) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < messages.size() && is_str(messages[j]["role"], "tool")) ++j;
        std::vector<json> order;
        if (i > 0) {
            const json& calls = jget(messages[i - 1], "tool_calls");
            if (calls.is_array())
                for (const json& c : calls) order.push_back(jget(c, "_call_id"));
        }
        KeyMap rank;   // a call id -> its call's place (the last call that has it)
        for (size_t k = 0; k < order.size(); ++k)
            if (truthy(order[k])) {
                hashable(order[k]);
                rank.set(order[k], k);
            }
        std::vector<std::pair<size_t, json>> run;
        for (size_t k = i; k < j; ++k) {
            const json& id = jget(messages[k], "_call_id");
            hashable(id);
            const size_t* at = rank.find(id);
            run.emplace_back(at ? *at : order.size(), std::move(messages[k]));
        }
        std::stable_sort(run.begin(), run.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t k = i; k < j; ++k) messages[k] = std::move(run[k - i].second);
        i = j;
    }
    for (json& m : messages) {
        m.erase("_call_id");
        const auto calls = m.find("tool_calls");
        if (calls != m.end() && calls->is_array())
            for (json& c : *calls) c.erase("_call_id");
    }
}

// ---------------------------------------------------------------------------------------------- Codex turns
/// client_metadata["x-codex-turn-metadata"] (a JSON string, Codex 0.140 and later) as an object, or null
json turn_body(const json& req) {
    const json& raw = jget(jget(req, "client_metadata"), "x-codex-turn-metadata");
    json turn;
    if (!raw.is_string() || !py_loads(raw.get_ref<const std::string&>(), turn) || !turn.is_object()) return nullptr;
    return turn;
}

bool thread_title(const json& req) { return is_str(jget(turn_body(req), "thread_source"), "thread_title"); }

using Conversation = std::pair<std::string, std::string>;   // (session_id, thread_id)

/// (request_kind, (session_id, thread_id)) from Codex's turn metadata; no conversation without both
std::pair<json, std::optional<Conversation>> codex_turn(const json& req) {
    const json turn = turn_body(req);
    const json& session = jget(turn, "session_id");
    const json& thread = jget(turn, "thread_id");
    std::optional<Conversation> conversation;
    if (session.is_string() && !session.get_ref<const std::string&>().empty() && thread.is_string() &&
        !thread.get_ref<const std::string&>().empty())
        conversation = Conversation(session.get<std::string>(), thread.get<std::string>());
    return {jget(turn, "request_kind"), conversation};
}

// Codex's local compaction (the context is full, or /compact) sends its conversation again with `tools: []`
// (codex-rs/core/src/compact.rs: `Prompt { input, base_instructions, ..Default::default() }`). The template writes
// the tools at the top of the prompt, so that prompt shares almost nothing with the one the engine holds and the
// whole conversation is read again, at its longest. So the tools a Codex conversation's last prompt was rendered with
// are kept, and that conversation's compaction prompt is rendered with them. Only the prompt: the parser and the
// response get the request's own tools, which are none. Codex says what a request is and whose it is in
// client_metadata["x-codex-turn-metadata"]: `request_kind` and the conversation's `session_id` and `thread_id` (a
// sub-agent shares its parent's session, not its thread; a fork gets its own). `prompt_cache_key` is not used: it
// groups requests for caching, and Codex gives sub-agents and ephemeral forks their parent's. A request without that
// metadata is rendered as sent and changes nothing. One entry, replaced by each prompt of a Codex conversation:
// nothing grows, and a compaction of any other conversation finds a different (session_id, thread_id) and renders its
// request as sent. Lost (a restart): the prompt is read again, as without it.
struct KeptPromptTools {
    std::mutex mu;
    std::optional<Conversation> owner;
    json tools;   ///< its prompt's tools (null: none)
};

KeptPromptTools& kept_prompt_tools() {
    static KeptPromptTools kept;
    return kept;
}

json message_item() {
    return {{"id", new_id("msg")}, {"type", "message"}, {"status", "in_progress"}, {"role", "assistant"},
            {"content", json::array()}};
}

void append(std::vector<json>& out, std::vector<json> more) {
    for (json& e : more) out.push_back(std::move(e));
}

const ToolCall& call_of(const Event& ev) {
    if (ev.call == nullptr) throw std::logic_error("a " + ev.kind + " event without its call");
    return *ev.call;
}

/// the output item is the call this event is about
bool same_call(const json& item, const Event& ev) {
    const json& id = jget(item, "call_id");
    return id.is_string() && id.get_ref<const std::string&>() == call_of(ev).id;
}

/// the user line Codex asked to name, cut to the schema's 36 characters
std::string title_text(const json& req) {
    std::vector<json> parts;
    const json& items = jget(req, "input");
    const auto take = [&](const json& item) {
        if (!item.is_object() || !is_str(jget(item, "role"), "user")) return;
        const json& content = jget(item, "content");
        if (content.is_string()) {
            parts.push_back(content);
        } else if (content.is_array()) {
            for (const json& piece : content)
                if (jget(piece, "text").is_string()) parts.push_back(piece["text"]);
        }
    };
    if (items.is_string()) take(json{{"role", "user"}, {"content", items}});
    if (items.is_array())
        for (const json& item : items) take(item);
    std::string text = py_join(parts, "\n");
    const size_t at = text.find("User prompt:");
    if (at != std::string::npos) text = text.substr(at + std::string_view("User prompt:").size());
    std::string line = "Untitled";
    for (const std::string& x : splitlines(text))
        if (!strip(x).empty()) {
            line = std::string(strip(x));
            break;
        }
    line = std::string(strip_chars(line, " \"'`"));
    static constexpr std::string_view ends[] = {".", "。", "!", "！", "?", "？", ",", "，", ";", "；", ":", "："};
    while (!line.empty()) {
        const std::string last = line.substr(line.size() - last_char_len(line));
        if (std::find(std::begin(ends), std::end(ends), last) == std::end(ends)) break;
        line = std::string(rstrip(std::string_view(line).substr(0, line.size() - last.size())));
    }
    const std::string cut(rstrip(prefix_chars(line, 36)));
    return cut.empty() ? "Untitled" : cut;
}

}  // namespace

// ---------------------------------------------------------------------------------------------- errors
ResponsesError::ResponsesError(const std::string& message_, std::string param_, std::string code_, int status_,
                               std::string kind_)
    : std::invalid_argument(message_), message(message_), param(std::move(param_)), code(std::move(code_)),
      status(status_), kind(std::move(kind_)) {}

json ResponsesError::body() const {
    return error_body(message, kind, param.empty() ? json(nullptr) : json(param),
                      code.empty() ? json(nullptr) : json(code));
}

Malformed::Malformed(const std::string& error_) : ResponsesError("the request is malformed (" + error_ + ")"),
                                                  error(error_) {}

json error_body(const std::string& message, const std::string& kind, const json& param, const json& code) {
    return {{"error", {{"message", message}, {"type", kind}, {"param", param}, {"code", code}}}};
}

// ---------------------------------------------------------------------------------------------- request -> chat
json input_messages(const json& req) {
    json messages = json::array();
    const json& instructions = jget(req, "instructions");
    if (!instructions.is_null() && !is_str(instructions, ""))
        messages.push_back({{"role", "system"},
                            {"content", instructions.is_string() ? instructions
                                                                 : content_of(instructions, "instructions")}});
    const json& given = jget(req, "input");
    if (given.is_null())
        throw ResponsesError("input is required: a string or an array of input items", "input",
                             "missing_required_parameter");
    if (!given.is_string() && !given.is_array())
        throw ResponsesError("input must be a string or an array of input items", "input");
    const json one = given.is_string() ? json::array({{{"type", "message"}, {"role", "user"}, {"content", given}}})
                                       : json::array();
    const json& items = given.is_string() ? one : given;
    std::vector<json> thinking;          // reasoning waiting for the assistant turn it belongs to
    std::optional<size_t> open_turn;     // the assistant message the next function_call joins
    const auto turn = [&](json content) {
        json t = {{"role", "assistant"}, {"content", std::move(content)}};
        if (!thinking.empty()) {
            t["reasoning_content"] = py_join(thinking);
            thinking.clear();
        }
        messages.push_back(std::move(t));
        open_turn = messages.size() - 1;
        return *open_turn;
    };
    for (size_t i = 0; i < items.size(); ++i) {
        const json& item = items[i];
        const std::string param = "input[" + std::to_string(i) + "]";
        if (!item.is_object()) throw ResponsesError("expected an input item object", param);
        json kind = jget(item, "type");
        if (!truthy(kind)) kind = item.contains("role") ? json("message") : json(nullptr);
        if (is_str(kind, "message")) {
            const json& role = jget(item, "role");
            if (!is_one_of(role, {"user", "assistant", "system", "developer"}))
                throw ResponsesError("unknown message role " + repr(role), param + ".role");
            json content = content_of(jget(item, "content"), param + ".content");
            if (is_str(role, "assistant")) {
                if (!content.is_string()) {
                    std::vector<json> texts;
                    for (const json& p : content)
                        if (is_str(jget(p, "type"), "text")) texts.push_back(get_or(p, "text", ""));
                    content = py_join(texts);
                }
                turn(std::move(content));
            } else {
                open_turn.reset();
                thinking.clear();
                messages.push_back({{"role", is_str(role, "developer") ? json("system") : role},
                                    {"content", std::move(content)}});
            }
        } else if (is_str(kind, "reasoning")) {
            json text = reasoning_text(item, param);
            if (truthy(text)) {
                if (open_turn) {   // thinking after an answer or a call starts a new model turn
                    const json& t = messages[*open_turn];
                    if (truthy(jget(t, "tool_calls")) || truthy(t["content"])) open_turn.reset();
                }
                thinking.push_back(std::move(text));
            }
        } else if (is_one_of(kind, {"function_call", "custom_tool_call"})) {
            const json& name_v = jget(item, "name");
            if (!name_v.is_string() || name_v.get_ref<const std::string&>().empty())
                throw ResponsesError("a function_call needs a name", param + ".name");
            std::string name = name_v.get<std::string>();
            if (truthy(jget(item, "namespace"))) name = str_of(item["namespace"]) + "." + name;
            json args;
            if (is_str(kind, "function_call")) {
                args = arguments(jget(item, "arguments"), param + ".arguments");
            } else {
                const json& input = jget(item, "input");
                args = {{"input", input.is_string() ? input : json("")}};
            }
            const size_t at = open_turn && thinking.empty() ? *open_turn : turn("");
            json& target = messages[at];
            if (!target.contains("tool_calls")) target["tool_calls"] = json::array();
            target["tool_calls"].push_back(
                {{"function", {{"name", name}, {"arguments", args}}}, {"_call_id", jget(item, "call_id")}});
        } else if (is_one_of(kind, {"function_call_output", "custom_tool_call_output"})) {
            messages.push_back(
                {{"role", "tool"}, {"content", tool_output(item, param)}, {"_call_id", jget(item, "call_id")}});
            open_turn.reset();
            thinking.clear();
        } else if (is_str(kind, "additional_tools")) {
            continue;   // #782 (Codex): tools the client adds as an input item; see request_tools
        } else if (is_str(kind, "item_reference")) {
            throw ResponsesError("item references need stored responses, and this server keeps none: send the items "
                                 "themselves",
                                 param, "unsupported_parameter");
        } else {
            throw ResponsesError("input items of type " + repr(kind) + " are not supported", param + ".type",
                                 "unsupported_parameter");
        }
    }
    order_tool_results(messages);
    // leading system/developer messages (Codex: instructions, then its developer message) become one system message
    // at the start; later ones become user messages in place, as on the chat path
    size_t lead = 0;
    while (lead < messages.size() && is_str(messages[lead]["role"], "system") && messages[lead]["content"].is_string())
        ++lead;
    if (lead > 1) {
        std::vector<json> texts;
        for (size_t k = 0; k < lead; ++k) texts.push_back(messages[k]["content"]);
        json merged = json::array({{{"role", "system"}, {"content", py_join(texts, "\n\n")}}});
        for (size_t k = lead; k < messages.size(); ++k) merged.push_back(std::move(messages[k]));
        messages = std::move(merged);
    }
    return late_system_to_user(std::move(messages));
}

RequestTools request_tools(const json& req) {
    RequestTools r;
    json tools = json::array();
    const json& sent = jget(req, "tools");
    if (truthy(sent) && !sent.is_array()) throw ResponsesError("tools must be an array", "tools");
    json given = truthy(sent) ? sent : json::array();
    // #782 (Codex v0.160): `additional_tools` input items carry more tool definitions (a `tools` array, as the
    // request has); they join the request's own, and an item without a usable array adds nothing
    const json& input = jget(req, "input");
    if (input.is_array())
        for (const json& item : input)
            if (item.is_object() && is_str(jget(item, "type"), "additional_tools") && jget(item, "tools").is_array())
                for (const json& t : item["tools"]) given.push_back(t);

    const auto add = [&](const json& tool, const std::string& param, const json& ns, const json& ns_description) {
        const json& kind = jget(tool, "type");
        const json& name_v = jget(tool, "name");
        if (!name_v.is_string() || name_v.get_ref<const std::string&>().empty())
            throw ResponsesError("a tool needs a name", param + ".name");
        const std::string& name = name_v.get_ref<const std::string&>();
        const std::string flat = truthy(ns) ? str_of(ns) + "." + name : name;
        json description = truthy(jget(tool, "description")) ? tool["description"] : json("");
        if (truthy(ns_description))
            description = py_add(ns_description, truthy(description) ? py_add("\n\n", description) : json(""));
        json params;
        if (is_str(kind, "function")) {
            params = truthy(jget(tool, "parameters")) ? tool["parameters"]
                                                      : json{{"type", "object"}, {"properties", json::object()}};
        } else {   // custom: free-form text, maybe in a grammar the model reads
            const json& fmt = jget(tool, "format");
            if (fmt.is_object() && is_str(jget(fmt, "type"), "grammar") && truthy(jget(fmt, "definition")))
                py_iadd(description, "\n\nThe input must follow this " + str_of(get_or(fmt, "syntax", "")) +
                                      " grammar:\n" + str_of(fmt["definition"]));
            json input = {{"type", "string"}, {"description", "the raw input"}};
            params = {{"type", "object"}, {"properties", {{"input", input}}}, {"required", json::array({"input"})}};
        }
        tools.push_back({{"name", flat}, {"description", description}, {"parameters", params}});
        const json entry = json::array({ns, name, kind});
        r.names[flat] = entry;
        if (truthy(ns)) {
            // Qwen writes Codex's MCP tools as `mcp__server__tool` (the flat name it learned), not
            // `mcp__server.tool`: map that spelling back as well
            const std::string alias = str_of(ns) + "__" + name;
            if (!r.names.contains(alias)) r.names[alias] = entry;
        }
    };

    for (size_t i = 0; i < given.size(); ++i) {
        const json& tool = given[i];
        const std::string param = "tools[" + std::to_string(i) + "]";
        if (!tool.is_object()) throw ResponsesError("expected a tool object", param);
        const json& kind = jget(tool, "type");
        if (is_one_of(kind, {"function", "custom"})) {
            add(tool, param, nullptr, "");
        } else if (is_str(kind, "namespace")) {
            const json& members = jget(tool, "tools");
            if (!truthy(members)) continue;
            if (members.is_string() || members.is_object())   // iterated, its first character or key is no tool
                throw ResponsesError("a namespace holds function tools", param + ".tools[0]");
            if (!members.is_array()) throw Malformed("TypeError: '" + type_name(members) + "' object is not iterable");
            const json& ns_description = truthy(jget(tool, "description")) ? tool["description"] : json("");
            for (size_t j = 0; j < members.size(); ++j) {
                const json& member = members[j];
                const std::string at = param + ".tools[" + std::to_string(j) + "]";
                if (!member.is_object() || !is_one_of(get_or(member, "type", "function"), {"function", "custom"}))
                    throw ResponsesError("a namespace holds function tools", at);
                json fn = {{"type", "function"}};
                for (auto it = member.begin(); it != member.end(); ++it) fn[it.key()] = it.value();
                add(fn, at, jget(tool, "name"), ns_description);
            }
        } else if (kind.is_string()) {
            r.skipped.push_back(kind.get<std::string>());   // the model cannot run OpenAI's hosted tools: left out
        } else {
            throw ResponsesError("a tool needs a type", param + ".type");
        }
    }
    r.tools = tools.empty() || is_str(jget(req, "tool_choice"), "none") ? json(nullptr) : std::move(tools);
    return r;
}

std::optional<std::vector<json>> thread_title_events(const json& req, const std::string& model) {
    // Codex 0.160 sends this beside the user turn: another session, no tools, the full instructions. Running it on
    // the one prefix cache replaces the conversation.
    if (!thread_title(req)) return std::nullopt;
    check_request(req);
    const std::string text = py_dumps(json{{"title", title_text(req)}}, false);
    Assembler a(req, model, 0, json::object(), false, false);
    std::vector<json> events = a.start();
    append(events, a.release(text));   // never blank: it is a JSON object
    a.response["usage"] = {{"input_tokens", 0},
                           {"input_tokens_details", {{"cached_tokens", 0}}},
                           {"output_tokens", 0},
                           {"output_tokens_details", {{"reasoning_tokens", 0}}},
                           {"total_tokens", 0}};
    a.response["status"] = "completed";
    a.response["completed_at"] = now();
    events.push_back(a.event("response.completed", {{"response", a.snapshot()}}));
    return events;
}

json prompt_tools(const json& req, const json& tools) {
    const auto [kind, conversation] = codex_turn(req);
    KeptPromptTools& kept = kept_prompt_tools();
    const std::lock_guard<std::mutex> lock(kept.mu);
    if (!is_str(kind, "compaction") || !conversation || kept.owner != conversation || kept.tools.is_null() ||
        truthy(jget(req, "tools")))
        return tools;
    return kept.tools;   // a copy: the kept one is nobody's to change
}

void prompt_made(const json& req, const json& tools) {
    // a thread-title turn is another session Codex sends beside the real one: it does not replace the entry
    if (thread_title(req)) return;
    const auto [kind, conversation] = codex_turn(req);
    if (!conversation || is_str(kind, "compaction")) return;
    KeptPromptTools& kept = kept_prompt_tools();
    const std::lock_guard<std::mutex> lock(kept.mu);
    kept.owner = conversation;
    kept.tools = truthy(tools) ? tools : json(nullptr);
}

json text_format(const json& req) {
    const json& text = jget(req, "text");
    if (text.is_null()) return nullptr;
    if (!text.is_object()) throw ResponsesError("text must be an object", "text");
    const json& fmt = jget(text, "format");
    if (fmt.is_null()) return nullptr;
    if (!fmt.is_object()) throw ResponsesError("text.format must be an object", "text.format");
    const json& kind = jget(fmt, "type");
    if (kind.is_null() || is_str(kind, "text")) return nullptr;
    if (is_str(kind, "json_object")) return {{"type", "json_object"}};
    if (is_str(kind, "json_schema")) {
        json spec = json::object();
        for (const char* k : {"name", "schema", "strict", "description"})
            if (fmt.contains(k)) spec[k] = fmt[k];
        return {{"type", "json_schema"}, {"json_schema", spec}};
    }
    throw ResponsesError("text.format.type must be text, json_object or json_schema", "text.format.type");
}

void check_request(const json& req) {
    if (truthy(jget(req, "previous_response_id")))
        throw ResponsesError("this server keeps no responses (stateless): send the whole conversation in input "
                             "instead of previous_response_id",
                             "previous_response_id", "unsupported_parameter");
    if (truthy(jget(req, "conversation")))
        throw ResponsesError("this server keeps no conversations (stateless): send the whole conversation in input",
                             "conversation", "unsupported_parameter");
    if (truthy(jget(req, "background")))
        throw ResponsesError("background responses are not supported: use stream or wait for the answer",
                             "background", "unsupported_parameter");
    if (truthy(jget(req, "prompt")))
        throw ResponsesError("stored prompt templates are not supported: send instructions and input", "prompt",
                             "unsupported_parameter");
    const json& n = jget(req, "max_output_tokens");
    // a whole number (not a bool, not a float), at least 1
    const bool positive = n.is_number_unsigned() ? n.get<uint64_t>() >= 1
                                                 : n.is_number_integer() && n.get<int64_t>() >= 1;
    if (!n.is_null() && !positive)
        throw ResponsesError("max_output_tokens must be a positive whole number", "max_output_tokens");
    const json& reasoning = jget(req, "reasoning");
    if (!reasoning.is_null() && !reasoning.is_object())
        throw ResponsesError("reasoning must be an object", "reasoning");
}

json template_kwargs(const json& req, const json& shared) {
    const json& reasoning = jget(req, "reasoning");   // (req.get("reasoning") or {}).get("effort")
    if (truthy(reasoning) && !reasoning.is_object())
        throw Malformed("AttributeError: '" + type_name(reasoning) + "' object has no attribute 'get'");
    json effort = jget(reasoning, "effort");
    if (effort.is_null()) effort = jget(shared, "reasoning_effort");   // the Chat settings shared with apps
    try {
        return effort_kwargs(effort);
    } catch (const RequestError& e) {
        throw ResponsesError(e.what(), "reasoning.effort");
    }
}

// ---------------------------------------------------------------------------------------------- output
Assembler::Assembler(const json& req, const std::string& model, int64_t prompt_tokens, json names, bool encrypted,
                     bool json_mode)
    : names_(std::move(names)), encrypted_(encrypted), json_mode_(json_mode), prompt_tokens_(prompt_tokens) {
    const json& reasoning = jget(req, "reasoning");   // jget reads a non-object as {}
    const json& text = jget(req, "text");
    const json& tools = jget(req, "tools");
    const json& metadata = jget(req, "metadata");
    response = json::object();
    response["id"] = new_id("resp");
    response["object"] = "response";
    response["created_at"] = now();
    response["status"] = "in_progress";
    response["background"] = false;
    response["error"] = nullptr;
    response["incomplete_details"] = nullptr;
    response["instructions"] = jget(req, "instructions");
    response["max_output_tokens"] = jget(req, "max_output_tokens");
    response["model"] = model;
    response["output"] = json::array();
    response["parallel_tool_calls"] = get_or(req, "parallel_tool_calls", true);
    response["previous_response_id"] = nullptr;
    response["reasoning"] = {{"effort", jget(reasoning, "effort")}, {"summary", jget(reasoning, "summary")}};
    response["store"] = false;
    response["temperature"] = jget(req, "temperature");
    response["text"] = text.is_object() ? text : json{{"format", {{"type", "text"}}}};
    response["tool_choice"] = get_or(req, "tool_choice", "auto");
    response["tools"] = truthy(tools) ? tools : json::array();
    response["top_p"] = jget(req, "top_p");
    response["truncation"] = "disabled";
    response["usage"] = nullptr;
    response["user"] = nullptr;
    response["metadata"] = truthy(metadata) ? metadata : json::object();
}

json Assembler::event(const std::string& kind, const json& data) {
    json e = {{"type", kind}, {"sequence_number", seq_++}};
    for (auto it = data.begin(); it != data.end(); ++it) e[it.key()] = it.value();
    return e;
}

std::vector<json> Assembler::start() {
    std::vector<json> out;
    out.push_back(event("response.created", {{"response", snapshot()}}));
    out.push_back(event("response.in_progress", {{"response", snapshot()}}));
    return out;
}

json Assembler::in_progress() { return event("response.in_progress", {{"response", snapshot()}}); }

int64_t Assembler::index() const { return (int64_t) response["output"].size() - 1; }

json& Assembler::item() { return response["output"][*item_]; }

std::vector<json> Assembler::open(json item_value) {
    json& output = response["output"];
    output.push_back(std::move(item_value));
    item_ = output.size() - 1;
    json& it = output.back();
    std::vector<json> out;
    out.push_back(event("response.output_item.added", {{"output_index", index()}, {"item", it}}));
    if (is_str(it["type"], "message")) {
        const json part = {
            {"type", "output_text"}, {"text", ""}, {"annotations", json::array()}, {"logprobs", json::array()}};
        it["content"].push_back(part);
        out.push_back(event("response.content_part.added",
                            {{"item_id", it["id"]}, {"output_index", index()}, {"content_index", 0}, {"part", part}}));
    }
    return out;
}

std::vector<json> Assembler::close(const std::string& status) {
    if (!item_) return {};
    const int64_t at = (int64_t) *item_;
    json& it = item();
    item_.reset();
    it["status"] = status;
    std::vector<json> out;
    const std::string type = jstr(it, "type");
    if (type == "message") {
        const json part = it["content"][0];
        out.push_back(event("response.output_text.done", {{"item_id", it["id"]},
                                                           {"output_index", at},
                                                           {"content_index", 0},
                                                           {"text", part["text"]},
                                                           {"logprobs", json::array()}}));
        out.push_back(event("response.content_part.done",
                            {{"item_id", it["id"]}, {"output_index", at}, {"content_index", 0}, {"part", part}}));
    } else if (type == "reasoning") {
        const json text = it["content"][0]["text"];
        out.push_back(event("response.reasoning_text.done",
                            {{"item_id", it["id"]}, {"output_index", at}, {"content_index", 0}, {"text", text}}));
        if (encrypted_)
            it["encrypted_content"] = std::string(kEncryptedPrefix) + b64encode(py_dumps(json{{"text", text}}, false));
    } else if (type == "function_call") {
        out.push_back(event("response.function_call_arguments.done", {{"item_id", it["id"]},
                                                                      {"output_index", at},
                                                                      {"name", it["name"]},
                                                                      {"arguments", it["arguments"]}}));
    } else if (type == "custom_tool_call") {
        out.push_back(event("response.custom_tool_call_input.done",
                            {{"item_id", it["id"]}, {"output_index", at}, {"input", it["input"]}}));
    }
    out.push_back(event("response.output_item.done", {{"output_index", at}, {"item", it}}));
    return out;
}

std::vector<json> Assembler::feed(const Event& ev) {
    std::vector<json> out;
    const std::string& kind = ev.kind;
    if (kind == "reasoning") {
        if (ev.text.empty()) return out;
        if (!item_ || !is_str(item()["type"], "reasoning")) {
            append(out, close());
            json content = json::array({{{"type", "reasoning_text"}, {"text", ""}}});
            append(out, open({{"id", new_id("rs")}, {"type", "reasoning"}, {"status", "in_progress"},
                              {"summary", json::array()}, {"content", content}}));
        }
        item()["content"][0]["text"].get_ref<std::string&>() += ev.text;
        out.push_back(event("response.reasoning_text.delta",
                            {{"item_id", item()["id"]},
                             {"output_index", index()},
                             {"content_index", 0},
                             {"delta", ev.text}}));
    } else if (kind == "content") {
        if (ev.text.empty()) return out;
        if (json_mode_) {   // sent once it is checked (finish)
            append(out, close());
            json_text_.push_back(ev.text);
            return out;
        }
        if (!item_ || !is_str(item()["type"], "message")) {
            append(out, close());
            append(out, open(message_item()));
        }
        item()["content"][0]["text"].get_ref<std::string&>() += ev.text;
        out.push_back(event("response.output_text.delta", {{"item_id", item()["id"]},
                                                            {"output_index", index()},
                                                            {"content_index", 0},
                                                            {"delta", ev.text},
                                                            {"logprobs", json::array()}}));
    } else if (kind == "tool_start") {
        append(out, close());
        append(out, call_started());
        append(out, open(call_item(call_of(ev))));
    } else if (kind == "tool_args") {
        if (!item_ || !same_call(item(), ev)) return out;
        json& it = item();
        if (is_str(it["type"], "function_call")) {   // a custom tool's input is sent whole, at its end
            it["arguments"].get_ref<std::string&>() += ev.text;
            out.push_back(event("response.function_call_arguments.delta",
                                {{"item_id", it["id"]}, {"output_index", index()}, {"delta", ev.text}}));
        }
    } else if (kind == "tool_call") {
        const ToolCall& call = call_of(ev);
        if (!item_ || !same_call(item(), ev)) {   // a call that was not announced while written
            append(out, close());
            append(out, call_started());
            append(out, open(call_item(call)));
            json& it = item();
            if (is_str(it["type"], "function_call")) {
                it["arguments"] = py_dumps(call.arguments, false);
                out.push_back(event("response.function_call_arguments.delta",
                                    {{"item_id", it["id"]}, {"output_index", index()}, {"delta", it["arguments"]}}));
            }
        }
        json& it = item();
        if (is_str(it["type"], "custom_tool_call")) {
            const json& value = jget(call.arguments, "input");
            it["input"] = value.is_string() ? value : json(py_dumps(call.arguments, false));
            out.push_back(event("response.custom_tool_call_input.delta",
                                {{"item_id", it["id"]}, {"output_index", index()}, {"delta", it["input"]}}));
        }
        append(out, close());
    }
    return out;
}

std::vector<json> Assembler::release(const std::string& text) {
    // a held structured answer (or the words written beside a tool call) as one message item
    if (strip(text).empty()) return {};
    std::vector<json> out = open(message_item());
    item()["content"][0]["text"] = text;
    out.push_back(event("response.output_text.delta", {{"item_id", item()["id"]},
                                                        {"output_index", index()},
                                                        {"content_index", 0},
                                                        {"delta", text},
                                                        {"logprobs", json::array()}}));
    append(out, close());
    return out;
}

std::vector<json> Assembler::call_started() {
    // the first tool call of a JSON-format turn: this turn is not the final answer, so what the model wrote before the
    // call goes out as it is, ahead of the call, and the schema is checked on the turn that ends in text (#782)
    std::vector<json> out;
    if (json_mode_ && !called_) {
        std::string text;
        for (const std::string& t : json_text_) text += t;
        json_text_.clear();
        out = release(text);
    }
    called_ = true;
    return out;
}

json Assembler::call_item(const ToolCall& call) {
    // names.get(call.name, (None, call.name, "function"))
    const bool known = names_.is_object() && names_.contains(call.name);
    const json ns = known ? names_[call.name][0] : json(nullptr);
    const json name = known ? names_[call.name][1] : json(call.name);
    const bool custom = known && is_str(names_[call.name][2], "custom");
    json it = {{"id", new_id(custom ? "ctc" : "fc")}, {"type", custom ? "custom_tool_call" : "function_call"},
               {"status", "in_progress"},           {"call_id", call.id},
               {"name", name},                      {custom ? "input" : "arguments", ""}};
    if (truthy(ns)) it["namespace"] = ns;
    return it;
}

std::vector<json> Assembler::finish(const json& done, const Validate& validate) {
    const json& reason = jget(done, "finish");
    const bool length = is_str(reason, "length");
    // a call still open was cut off (its tool_call never came, #211): incomplete, whatever the finish
    const bool unfinished = item_ && is_one_of(item()["type"], {"function_call", "custom_tool_call"});
    std::vector<json> out = close(length || unfinished ? "incomplete" : "completed");
    if (json_mode_) {
        std::string text;
        for (const std::string& t : json_text_) text += t;
        json_text_.clear();
        if (called_ || !validate) {
            append(out, release(text));   // beside a tool call: as written (#782)
        } else {
            append(out, release(validate(text, reason.is_string() ? reason.get<std::string>() : std::string())));
        }
    }
    const json n = get_or(done, "completion_tokens", 0);
    const json prompt = get_or(done, "prompt_tokens", prompt_tokens_);
    const json& reused = jget(done, "reused");
    const int64_t cached = std::min(truthy(reused) ? reused.get<int64_t>() : 0, prompt.get<int64_t>());
    response["usage"] = {{"input_tokens", prompt},
                         {"input_tokens_details", {{"cached_tokens", cached}}},
                         {"output_tokens", n},
                         {"output_tokens_details", {{"reasoning_tokens", get_or(done, "reasoning_tokens", 0)}}},
                         {"total_tokens", prompt.get<int64_t>() + n.get<int64_t>()}};
    if (length) {
        response["status"] = "incomplete";
        response["incomplete_details"] = {{"reason", "max_output_tokens"}};
        out.push_back(event("response.incomplete", {{"response", snapshot()}}));
        return out;
    }
    response["status"] = "completed";
    response["completed_at"] = now();
    out.push_back(event("response.completed", {{"response", snapshot()}}));
    return out;
}

json Assembler::failed(const std::string& message, const std::string& code) {
    if (item_) {
        item()["status"] = "incomplete";
        item_.reset();
    }
    response["status"] = "failed";
    response["error"] = {{"code", code}, {"message", message}};
    return event("response.failed", {{"response", snapshot()}});
}

json collect(const std::vector<json>& events) {
    const json* last = nullptr;
    for (const json& e : events)
        if (!e.is_null()) last = &e;
    if (last == nullptr) throw Malformed("TypeError: 'NoneType' object is not subscriptable");
    if (!last->is_object() || !last->contains("response")) throw Malformed("KeyError: 'response'");
    return (*last)["response"];
}

}  // namespace lamina::responses
