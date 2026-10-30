// src/server/check.cpp - lamina-check: the server's text layer on JSON-lines cases, for comparisons with the Python
// implementation it ports (scratch/lamina/p6_check.py).
//
//   lamina-check tokenize GGUF CASES   {"text", "parse_special", "plain": [[a, b], ...] (bytes)} -> {"ids"}
//   lamina-check decode GGUF CASES     {"ids"} -> {"text"}
//   lamina-check convert CASES         {"api": "openai"|"anthropic", "req", "think_unasked"} -> {"messages", "tools",
//                                      "kwargs", "req"} or {"error"}
//   lamina-check render GGUF CASES     {"messages", "tools", "kwargs", "add_generation_prompt"} -> {"prompt"} or {"error"}
//   lamina-check mark GGUF CASES       {"messages", "tools", "kwargs"} -> the server's prompt path: {"prompt" (unmarked),
//                                      "spans", "ids"}
//   lamina-check parse CASES           {"thinking", "tools", "stream_tools", "recover", "deltas", "reason"} -> {"events"}
//   lamina-check caps GGUF             the template's capabilities
#include "frontend.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using namespace lamina;

namespace {

int usage() {
    std::fprintf(stderr, "usage: lamina-check tokenize|decode|render|mark GGUF CASES | convert|parse CASES | caps GGUF\n");
    return 2;
}

template <class F>
int each_case(const char* path, F&& f) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        json c, out;
        if (!py_loads(line, c)) {
            out = {{"error", "bad case"}};
        } else {
            try {
                out = f(c);
            } catch (const std::exception& e) {
                out = {{"error", e.what()}};
            }
        }
        std::cout << py_dumps(out, false, true) << "\n";
    }
    return 0;
}

json events_json(const std::vector<Event>& evs) {
    json out = json::array();
    for (const Event& e : evs) {
        json j = {{"kind", e.kind}, {"text", e.text}};
        if (e.call != nullptr) {
            j["name"] = e.call->name;
            if (e.kind == "tool_call") j["arguments"] = e.call->arguments;
        }
        out.push_back(j);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    const std::string mode = argv[1];
    if (mode == "convert" || mode == "parse") {
        return each_case(argv[2], [&](json& c) -> json {
            if (mode == "convert") {
                json req = c["req"];
                const Converted r = jstr(c, "api") == "anthropic"
                                        ? anthropic_to_messages(req, !jget(c, "think_unasked").is_boolean() ||
                                                                         c["think_unasked"].get<bool>())
                                        : openai_to_messages(req);
                return {{"messages", r.messages}, {"tools", r.tools}, {"kwargs", r.kwargs}, {"req", req}};
            }
            OutputParser p(!jget(c, "thinking").is_boolean() || c["thinking"].get<bool>(), jget(c, "tools"),
                           truthy(jget(c, "stream_tools")), truthy(jget(c, "recover")));
            std::vector<Event> evs;
            for (const auto& d : c["deltas"])
                for (auto& e : p.feed(d.get<std::string>())) evs.push_back(std::move(e));
            for (auto& e : p.finish(jstr(c, "reason"))) evs.push_back(std::move(e));
            return {{"events", events_json(evs)}, {"rescued", p.rescued}, {"refused", p.refused}};
        });
    }
    if (argc < 3) return usage();
    std::unique_ptr<Tokenizer> tok;
    try {
        tok = Tokenizer::from_gguf(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    if (mode == "caps") {
        const ChatTemplate tpl(tok->chat_template);
        std::cout << py_dumps(tpl.caps(), false) << "\n";
        return 0;
    }
    if (argc < 4) return usage();
    if (mode == "tokenize")
        return each_case(argv[3], [&](json& c) -> json {
            std::vector<Span> plain;
            for (const auto& s : jget(c, "plain")) plain.push_back({s[0].get<size_t>(), s[1].get<size_t>()});
            return {{"ids", tok->encode(c["text"].get<std::string>(), truthy(jget(c, "parse_special")), plain)}};
        });
    if (mode == "decode")
        return each_case(argv[3], [&](json& c) -> json { return {{"text", tok->decode(c["ids"].get<std::vector<int32_t>>())}}; });
    const ChatTemplate tpl(tok->chat_template);
    if (mode == "render")
        return each_case(argv[3], [&](json& c) -> json {
            return {{"prompt", tpl.render(c["messages"], jget(c, "tools"),
                                          !jget(c, "add_generation_prompt").is_boolean() || c["add_generation_prompt"].get<bool>(),
                                          jget(c, "kwargs").is_object() ? c["kwargs"] : json::object())}};
        });
    if (mode == "mark") {
        const LiteralTags tags = literal_tags(tok->control_tokens());
        return each_case(argv[3], [&](json& c) -> json {
            const Marked m = mark_think_literals(c["messages"], jget(c, "tools"), tags);
            const std::string rendered = tpl.render(m.messages, m.tools, true,
                                                    jget(c, "kwargs").is_object() ? c["kwargs"] : json::object());
            auto [prompt, spans] = m.changed ? unmark_think_literals(rendered, tags)
                                             : std::make_pair(rendered, std::vector<std::pair<size_t, size_t>>{});
            std::vector<Span> plain;
            json js = json::array();
            for (const auto& [a, b] : spans) {
                plain.push_back({a, b});
                js.push_back({a, b});
            }
            return {{"prompt", prompt}, {"spans", js}, {"ids", tok->encode(prompt, true, plain)}};
        });
    }
    return usage();
}
