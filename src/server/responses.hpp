// src/server/responses.hpp - #451: OpenAI's Responses API (POST /v1/responses), stateless, on the chat path
// (serve/responses.py's port).
//
// A Responses request becomes the same template messages, tools and kwargs a Chat Completions request does, runs
// through the same Service.run, and its events (reasoning, text, tool calls) come back as Responses output items and
// typed SSE events. Nothing is stored: the client sends the whole conversation in `input` every time (`store: false`,
// as Codex CLI does), including the reasoning, message and function_call items of earlier answers.
//
// Codex CLI is the first client. What it needs, and what this does:
//   * `instructions` and `developer` messages -> the system message (leading ones merged, so the prompt start is
//     stable and the conversation cache holds across turns);
//   * function tools, also inside `namespace` tools (Codex's MCP and agent tools: the model sees `namespace.name`)
//     and `custom` tools (a free-form `input` string); hosted tools (web search, ...) are left out, the model cannot
//     run them;
//   * reasoning items: the thinking is returned as `reasoning_text` content (no summaries: this model does not write
//     them) and, when `include` asks for `reasoning.encrypted_content`, also as an opaque `encrypted_content` string
//     (base64, not encrypted: the client already holds the text). Either form sent back is put into the conversation
//     again, so the prompt matches what the model wrote and the cache is reused;
//   * the official event order: response.created, response.in_progress, response.output_item.added, deltas, ...done
//     events, response.output_item.done, and response.completed / response.incomplete / response.failed.
//
// Based on #630 by CC-David-CC (the item/event assembler, namespace tools, call_id ordering of tool results).
#pragma once

#include "frontend.hpp"
#include "json.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lamina::responses {

/// our own reasoning replay strings; others' encrypted_content is ignored
inline constexpr std::string_view kEncryptedPrefix = "lamina.r1:";

/// a request error in the Responses format: {"error": {"message", "type", "param", "code"}}
struct ResponsesError : std::invalid_argument {
    explicit ResponsesError(const std::string& message, std::string param = "", std::string code = "",
                            int status = 400, std::string kind = "invalid_request_error");
    std::string message;       ///< the whole text (what() ends at a NUL)
    std::string param, code;   ///< empty: None
    int status;
    std::string kind;
    json body() const;
};

/// a value of the wrong type where the Python server failed with a TypeError, AttributeError or KeyError (a content
/// part's text that is a number, a call_id that is a list ...): the 400 it answered, "the request is malformed
/// (<error>)"; the server also logs "[lamina] 400 malformed request (<error>):"
struct Malformed : ResponsesError {
    explicit Malformed(const std::string& error);
    std::string error;   ///< "TypeError: sequence item 0: expected str instance, int found"
};

json error_body(const std::string& message, const std::string& kind = "invalid_request_error",
                const json& param = nullptr, const json& code = nullptr);

/// `instructions` + `input` -> the chat template's messages
json input_messages(const json& req);

struct RequestTools {
    json tools;                         ///< the template's tools; null when there are none (or tool_choice "none")
    json names = json::object();        ///< template name -> [namespace (null: none), name, "function" | "custom"]
    std::vector<std::string> skipped;   ///< the hosted tools left out
};
RequestTools request_tools(const json& req);

/// Codex's thread-title turn, answered here; nullopt when this request is not one
std::optional<std::vector<json>> thread_title_events(const json& req, const std::string& model);

/// the tools the prompt is rendered with: request_tools' `tools`, or for a Codex compaction request without tools the
/// ones its own conversation's last prompt was rendered with (so its prompt starts as that one did). A copy either
/// way: a caller tells them apart by value (equal tools render the same prompt).
json prompt_tools(const json& req, const json& tools);
/// the prompt of `req` was rendered with `tools` and goes to the engine: a Codex request that is not a compaction
/// leaves its conversation and those tools for that conversation's compaction
void prompt_made(const json& req, const json& tools);

/// `text.format` -> the chat path's response_format (null: plain text)
json text_format(const json& req);
/// what this stateless server cannot do, refused before anything runs
void check_request(const json& req);
/// the request's reasoning effort (else the shared Chat settings' `reasoning_effort`) -> the template's kwargs
json template_kwargs(const json& req, const json& shared);

/// One response: its output items and its typed events, built from Service.run's events. The non-streaming answer is
/// the `response` of the last event, so both modes are the same items.
class Assembler {
public:
    /// checks a structured answer and returns its text (or throws); `finish` "" is None
    using Validate = std::function<std::string(const std::string& text, const std::string& finish)>;

    /// `names`: request_tools' names
    Assembler(const json& req, const std::string& model, int64_t prompt_tokens, json names, bool encrypted,
              bool json_mode = false);

    json event(const std::string& kind, const json& data = json::object());
    json snapshot() const { return response; }
    std::vector<json> start();
    /// a keep-alive a client counts as activity (SSE comments are not events: Codex's idle timer ignores them)
    json in_progress();
    std::vector<json> close(const std::string& status = "completed");
    std::vector<json> feed(const Event& ev);
    /// the end: close what is open, then response.completed (or .incomplete when max_output_tokens ran out);
    /// `validate` checks a structured answer (empty: it goes out as written)
    std::vector<json> finish(const json& done, const Validate& validate = {});
    /// response.failed: an error after the stream started (the HTTP status is already sent)
    json failed(const std::string& message, const std::string& code = "server_error");
    /// the last output item's index
    int64_t index() const;

    json response;

private:
    json& item();
    std::vector<json> open(json item);
    std::vector<json> release(const std::string& text);
    std::vector<json> call_started();
    json call_item(const ToolCall& call);

    json names_;
    bool encrypted_, json_mode_;
    int64_t prompt_tokens_;
    int64_t seq_ = 0;
    std::optional<size_t> item_;           ///< the open output item
    std::vector<std::string> json_text_;   ///< structured output: the answer, held until it is checked
    bool called_ = false;                  ///< a tool call was made: this turn is not the final answer

    friend std::optional<std::vector<json>> thread_title_events(const json& req, const std::string& model);
};

/// the non-streaming answer: the final response object (null events are heartbeats)
json collect(const std::vector<json>& events);

}  // namespace lamina::responses
