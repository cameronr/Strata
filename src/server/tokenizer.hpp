// src/server/tokenizer.hpp - the model's tokenizer, from its GGUF: byte-level BPE with the qwen35 pre-tokenizer.
//
// A port of tools/strata_tokenizer.py (checked against llama.cpp's tokenizer as its oracle): the pre-tokenizer split
// is llama.cpp's own (unicode_regex_split), the merges are applied lowest rank first, and the literals (GGUF token
// types 3 CONTROL and 4 USER_DEFINED) are matched on the raw text first: type 4 always, type 3 with parse_special.
// Text positions are byte offsets into UTF-8 text.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lamina {

/// [a, b) byte offsets of a text
struct Span {
    size_t a = 0, b = 0;
    bool operator==(const Span&) const = default;
    bool operator<(const Span& o) const { return a != o.a ? a < o.a : b < o.b; }
};

/// (end of a literal match, ids so far): where the encoding of a longer text with the same start can resume
struct Mark {
    size_t end = 0, count = 0;
};

class Tokenizer {
public:
    /// the vocabulary, merges and token types of a GGUF (its first shard); throws std::runtime_error
    static std::unique_ptr<Tokenizer> from_gguf(const std::string& path);

    /// `parse_special`: the CONTROL literals (<|im_start|> ...); the USER_DEFINED ones (<think> ...) are matched either
    /// way. `plain`: spans that are ordinary text even where they spell a literal; they are tokenized with the text
    /// around them. `marks` (optional) receives the resume points.
    std::vector<int32_t> encode(std::string_view text, bool parse_special = false, const std::vector<Span>& plain = {},
                                std::vector<Mark>* marks = nullptr) const;
    /// one token's raw bytes (a character can be split across tokens)
    const std::string& token_bytes(int32_t id) const;
    /// the text of these ids, an invalid UTF-8 sequence as U+FFFD (as Python's errors="replace")
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t n_vocab() const { return (int32_t) bytes_.size(); }
    /// the id of a literal's text, or -1
    int32_t literal_id(std::string_view text) const;
    /// the CONTROL literals' texts, in id order
    const std::vector<std::string>& control_tokens() const { return control_; }
    /// every literal's text (types 3 and 4)
    const std::vector<std::string>& literals() const { return literal_texts_; }
    size_t max_special_len() const { return max_special_len_; }

    std::string chat_template;                 ///< tokenizer.chat_template
    std::string pre;                           ///< tokenizer.ggml.pre
    std::map<std::string, int64_t> special_ids;   ///< tokenizer.ggml.<name>_token_id
    std::string architecture;                  ///< general.architecture

private:
    void encode_plain(std::string_view text, std::vector<int32_t>& out) const;
    void bpe(const std::string& piece, std::vector<int32_t>& out) const;

    struct Literal {
        int32_t id;
        std::string text;                      ///< as written: matched on the raw text
        bool always;                           ///< type 4: matched without parse_special
    };
    std::vector<std::string> bytes_;           ///< each token's raw bytes
    std::unordered_map<std::string, int32_t> literal_ids_;
    std::vector<std::string> literal_texts_, control_;
    std::vector<Literal> lits_;
    std::array<std::vector<uint32_t>, 256> by_first_;   ///< lits_ by their first byte, longest first
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;   ///< (left, right) -> (rank, merged id)
    std::array<int32_t, 256> byte_token_{};    ///< a byte's single-symbol token
    size_t max_special_len_ = 1;
    std::string regex_;                        ///< the pre-tokenizer's pattern (llama.cpp's for `pre`)

    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> piece_cache_;
};

/// UTF-8 decoding in pieces: complete characters out, a character split across pieces held until it completes, an
/// invalid sequence as U+FFFD (Python's incremental decoder with errors="replace")
class Utf8Decoder {
public:
    std::string push(std::string_view bytes, bool final = false);
    bool pending() const { return !held_.empty(); }

private:
    std::string held_;
};

/// the text of ids one at a time
class Detokenizer {
public:
    explicit Detokenizer(const Tokenizer& tok) : tok_(tok) {}
    std::string push(int32_t id) { return dec_.push(tok_.token_bytes(id)); }
    bool pending() const { return dec_.pending(); }

private:
    const Tokenizer& tok_;
    Utf8Decoder dec_;
};

/// `tok.encode(text, true, plain)` for chat prompts, reusing the ids of an earlier prompt up to a literal boundary
/// both share, and a reply's own ids (remember_reply). See tools/strata_tokenizer.py's PromptEncoder for why it is
/// exact.
class PromptEncoder {
public:
    explicit PromptEncoder(const Tokenizer& tok, size_t keep = 4) : tok_(tok), keep_(keep) {}
    std::vector<int32_t> encode(const std::string& text, std::vector<Span> plain = {}, bool store = true);
    /// the engine holds `prompt_ids` (a prompt this encoder gave) and then `gen_ids`: that prompt's entry becomes
    /// one of both, its cuts the reply's token boundaries where no character is split
    bool remember_reply(const std::vector<int32_t>& prompt_ids, const std::vector<int32_t>& gen_ids);
    /// the prompt with these ids was refused: it is not kept
    void forget(const std::vector<int32_t>& ids);
    size_t last_reused() const { return last_reused_; }

private:
    struct Entry {
        std::string text;
        std::vector<Span> plain;
        std::vector<int32_t> ids;
        std::vector<size_t> ends, counts;
        size_t reply_at = SIZE_MAX;            ///< where a remembered reply starts in the text (SIZE_MAX: none)
    };
    bool straddles(const std::string& text, size_t c) const;

    const Tokenizer& tok_;
    size_t keep_;
    std::mutex mu_;
    std::vector<std::shared_ptr<const Entry>> entries_;
    size_t last_reused_ = 0;
};

size_t common_prefix_len(std::string_view a, std::string_view b);
/// the first position below `limit` where the plain spans of two texts differ, or `limit`
size_t plain_agree_until(const std::vector<Span>& a, const std::vector<Span>& b, size_t limit);

}  // namespace lamina
