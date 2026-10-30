// src/core/prompt_cache.cpp - the --serve prompt cache (see include/strata/core/prompt_cache.hpp).
#include "strata/core/prompt_cache.hpp"

#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace strata::core {

namespace {

/// A replaced sequence is stashed when the switch would drop at least this many of its cells.
constexpr int64_t kStashMin = 1024;

}  // namespace

PromptCache::PromptCache(const ModelGeometry& g, SessionState& ss, std::vector<QsaState*> kv, void* stream,
                         int max_ckpts, uint64_t stash_bytes)
    : g_(g), ss_(ss), kv_(std::move(kv)), stream_(stream), max_ckpts_(max_ckpts), stash_budget_(stash_bytes),
      ckpt_bytes_(session_ckpt_bytes(g)) {}

PromptCache::~PromptCache() {
    for (const auto& [p, pinned] : owned_) {
        if (pinned) cudaFreeHost(p);
        else std::free(p);
    }
}

int64_t PromptCache::reusable(const Seq& seq, const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs,
                              bool live) const {
    const int64_t m = std::min<int64_t>((int64_t) seq.tokens.size(), (int64_t) ids.size() - 1);
    int64_t lcp = 0;
    while (lcp < m && seq.tokens[(size_t) lcp] == ids[(size_t) lcp]) ++lcp;

    // The token prefix is necessary but not sufficient for multimodal input:
    // every image that falls inside the reusable prefix must have the same
    // embedding/grid identity at the same position.  If an image was changed,
    // removed, added, or reordered, fail closed before its first pad token.
    auto matching_image = [](const std::vector<ImgKey>& all, const ImgKey& wanted) {
        for (const ImgKey& image : all)
            if (image.start == wanted.start) return image == wanted;
        return false;
    };
    for (const ImgKey& image : seq.imgs)
        if (image.start < lcp && !matching_image(imgs, image)) lcp = image.start;
    for (const ImgKey& image : imgs)
        if (image.start < lcp && !matching_image(seq.imgs, image)) lcp = image.start;

    if (live && lcp > 0 && lcp == (int64_t) seq.tokens.size()) return lcp;
    int64_t best = -1;
    for (const Ckpt& c : seq.ckpts)
        if (c.pos <= lcp && c.pos > best) best = c.pos;
    return best;
}

PromptCache::Ckpt* PromptCache::find(Seq& seq, int64_t pos) {
    for (Ckpt& c : seq.ckpts)
        if (c.pos == pos) return &c;
    return nullptr;
}

void* PromptCache::take_buffer(int64_t pos) {
    if (!free_.empty()) {
        void* h = free_.back();
        free_.pop_back();
        return h;
    }
    if (allocated_ < max_ckpts_) {
        void* h = nullptr;
        const bool pinned = cudaMallocHost(&h, (size_t) ckpt_bytes_) == cudaSuccess;
        if (!pinned) {
            (void) cudaGetLastError();
            h = std::malloc((size_t) ckpt_bytes_);   // pageable: slower copies, same result
        }
        if (h != nullptr) {
            owned_.emplace_back(h, pinned);
            ++allocated_;
        }
        return h;
    }
    // the pool is full: each sequence offers the checkpoint it can spare best (the live one's chain with the new
    // checkpoint at its end), and the least recently used offer goes
    Seq* owner = nullptr;
    size_t at = 0;
    uint64_t oldest = UINT64_MAX;
    std::vector<size_t> order;
    std::vector<int64_t> chain;
    auto offer = [&](Seq& s, bool live) {
        order.resize(s.ckpts.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return s.ckpts[a].pos < s.ckpts[b].pos; });
        chain.clear();
        size_t keep = SIZE_MAX;
        for (size_t k = 0; k < order.size(); ++k) {
            chain.push_back(s.ckpts[order[k]].pos);
            if (chain.back() == s.pin) keep = k;
        }
        if (live) chain.push_back(pos);
        const size_t k = spread_victim(chain.data(), chain.size(), keep);
        size_t pick = k < order.size() ? order[k] : SIZE_MAX;
        if (pick == SIZE_MAX)   // too few to spread: the least recently used
            for (size_t i = 0; i < s.ckpts.size(); ++i)
                if (s.ckpts[i].pos != s.pin && (pick == SIZE_MAX || s.ckpts[i].stamp < s.ckpts[pick].stamp))
                    pick = i;
        if (pick != SIZE_MAX && s.ckpts[pick].stamp < oldest) {
            oldest = s.ckpts[pick].stamp;
            owner = &s;
            at = pick;
        }
    };
    offer(live_, true);
    for (Seq& s : stash_) offer(s, false);
    if (owner == nullptr) return nullptr;
    void* h = owner->ckpts[at].host;
    owner->ckpts.erase(owner->ckpts.begin() + (ptrdiff_t) at);
    if (owner != &live_ && owner->ckpts.empty())   // a stashed sequence without a checkpoint cannot be resumed
        stash_.erase(stash_.begin() + (owner - stash_.data()));
    return h;
}

void PromptCache::release(Seq& seq, int64_t keep_upto) {
    for (size_t i = 0; i < seq.ckpts.size();) {
        if (seq.ckpts[i].pos > keep_upto) {
            free_.push_back(seq.ckpts[i].host);
            seq.ckpts.erase(seq.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
}

std::vector<PromptCache::Part> PromptCache::kv_parts(int64_t cells) const {
    // [page][kv_head][page_size][head_dim] with an identity page table: the first cells are one prefix of each
    // pool.  The indexer's pooled rows are one per idx_block cells.
    const strata::kernels::QsaShapes sh = strata::kernels::qsa_real_shapes();
    const uint64_t row = (uint64_t) g_.idx_key_dim * sizeof(float);
    std::vector<Part> parts;
    for (QsaState* st : kv_) {
        const uint64_t pages = (uint64_t) std::min<int64_t>(st->n_pages, (cells + sh.page_size - 1) / sh.page_size);
        const uint64_t c = pages * (uint64_t) g_.n_head_kv * (uint64_t) sh.page_size;
        for (const KvArray& a : qsa_kv_arrays(*st, g_)) parts.push_back({a.data, c * a.row_bytes});
        const int64_t rows = std::min<int64_t>(cells / sh.idx_block + 2, st->max_cells / sh.idx_block + 2);
        parts.push_back({st->idx_pooled, (uint64_t) rows * row});
    }
    return parts;
}

bool PromptCache::kv_copy(int64_t cells, int64_t held, uint8_t* host, bool to_host) {
    cudaStream_t cs = (cudaStream_t) stream_;
    const std::vector<Part> parts = kv_parts(cells), layout = kv_parts(held);
    for (size_t i = 0; i < parts.size(); ++i) {
        const Part& p = parts[i];
        const cudaError_t e = to_host ? cudaMemcpyAsync(host, p.dev, (size_t) p.bytes, cudaMemcpyDeviceToHost, cs)
                                      : cudaMemcpyAsync(p.dev, host, (size_t) p.bytes, cudaMemcpyHostToDevice, cs);
        if (e != cudaSuccess) return session_copy_failed(stream_);
        host += layout[i].bytes;
    }
    return cudaStreamSynchronize(cs) == cudaSuccess || session_copy_failed(stream_);
}

void PromptCache::stash_live(int64_t keep_upto) {
    const int64_t L = (int64_t) live_.tokens.size();
    if (stash_budget_ == 0 || L < kStashMin) return;
    if (ensure_kv && !ensure_kv(L)) return;
    uint64_t bytes = 0;
    for (const Part& p : kv_parts(L)) bytes += p.bytes;
    if (bytes > stash_budget_) return;
    uint64_t used = 0;
    for (const Seq& s : stash_) used += s.kv_bytes;
    while (used + bytes > stash_budget_) {   // the least recently used sequences go, but not one lending its prefix
        auto lru = stash_.end();
        for (auto it = stash_.begin(); it != stash_.end(); ++it)
            if (it->pin < 0 && (lru == stash_.end() || it->stamp < lru->stamp)) lru = it;
        if (lru == stash_.end()) return;
        used -= lru->kv_bytes;
        release(*lru, 0);
        stash_.erase(lru);
    }
    Seq s;
    s.kv.reset(new (std::nothrow) uint8_t[bytes]);
    if (!s.kv || !kv_copy(L, L, s.kv.get(), true)) return;
    checkpoint(L);   // the end state: the session is still at L
    for (size_t i = 0; i < live_.ckpts.size();) {
        if (live_.ckpts[i].pos > keep_upto) {
            s.ckpts.push_back(live_.ckpts[i]);
            live_.ckpts.erase(live_.ckpts.begin() + (ptrdiff_t) i);
        } else {
            ++i;
        }
    }
    if (s.ckpts.empty()) return;
    s.tokens = live_.tokens;
    for (const ImgKey& image : live_.imgs)
        if (image.start < L) s.imgs.push_back(image);
    s.kv_cells = L;
    s.kv_bytes = bytes;
    s.stamp = ++clock_;
    stash_.push_back(std::move(s));
}

bool PromptCache::restore(Seq& seq, int64_t pos, bool prefix) {
    Ckpt* ck = find(seq, pos);
    if (ck == nullptr) return false;
    const int64_t cells = prefix ? pos : seq.kv_cells;
    if (seq.kv && ensure_kv && !ensure_kv(cells)) return false;
    if (seq.kv && !kv_copy(cells, seq.kv_cells, seq.kv.get(), false)) return false;
    ck->stamp = ++clock_;
    return session_ckpt_load(ss_, g_, pos, ck->host, stream_);
}

int64_t PromptCache::begin(const std::vector<int64_t>& ids, const std::vector<ImgKey>& imgs) {
    const int64_t L = (int64_t) live_.tokens.size();
    auto zero = [&]() -> int64_t {
        release(live_, 0);
        live_.tokens.clear();
        live_.imgs.clear();
        session_zero(ss_, g_, nullptr, stream_);
        cudaStreamSynchronize((cudaStream_t) stream_);
        return 0;
    };
    if (!enabled()) return zero();
    int64_t best = reusable(live_, ids, imgs, true);
    int from = -1;   // -1: the live sequence, else a stashed one
    for (size_t i = 0; i < stash_.size(); ++i) {
        const int64_t r = reusable(stash_[i], ids, imgs, false);
        if (r > best) {
            best = r;
            from = (int) i;
        }
    }
    if (best <= 0) {
        stash_live(0);
        const int64_t result = zero();
        live_.imgs = imgs;
        return result;
    }
    if (from < 0) {
        // The live sequence continues.  When the prompt drops most of it (another conversation that shares a
        // prefix), the rest goes to the stash; an edit near the end just overwrites it.
        const int64_t dropped = L - best;
        if (dropped >= kStashMin && dropped > best) {
            live_.pin = best;
            stash_live(best);
            live_.pin = -1;
        }
        if (best < L && !restore(live_, best)) return zero();
        release(live_, best);
        live_.tokens.resize((size_t) best);
        live_.imgs = imgs;
        return best;
    }
    int64_t newest = 0;
    for (const Ckpt& c : stash_[(size_t) from].ckpts) newest = std::max(newest, c.pos);
    if (best < newest) {
        // Another conversation that shares the stashed one's start: it borrows the cells before `best`, and the
        // stashed sequence stays whole (pinned while the live one is stashed beside it).
        stash_[(size_t) from].pin = best;
        stash_live(0);
        release(live_, 0);
        live_.tokens.clear();
        auto donor = std::find_if(stash_.begin(), stash_.end(), [](const Seq& s) { return s.pin >= 0; });
        if (donor == stash_.end()) return zero();
        const bool ok = restore(*donor, best, true);
        donor->pin = -1;
        if (!ok) return zero();
        donor->stamp = ++clock_;
        live_.tokens.assign(donor->tokens.begin(), donor->tokens.begin() + best);
        live_.imgs = imgs;
        return best;
    }
    // That sequence continues: the live one is stashed and the chosen one becomes live.
    Seq chosen = std::move(stash_[(size_t) from]);
    stash_.erase(stash_.begin() + from);
    stash_live(0);
    release(live_, 0);
    live_.tokens.clear();
    if (!restore(chosen, best)) {
        release(chosen, 0);
        return zero();
    }
    live_.tokens.assign(chosen.tokens.begin(), chosen.tokens.begin() + best);
    live_.ckpts = std::move(chosen.ckpts);
    release(live_, best);
    live_.imgs = imgs;
    return best;
}

void PromptCache::set(int64_t pos, const int64_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    live_.tokens.insert(live_.tokens.end(), tokens, tokens + n);
}

void PromptCache::set(int64_t pos, const int32_t* tokens, int64_t n) {
    if (!enabled()) return;
    release(live_, pos);
    live_.tokens.resize((size_t) pos);
    for (int64_t i = 0; i < n; ++i) live_.tokens.push_back(tokens[i]);
}

void PromptCache::checkpoint(int64_t pos) {
    if (!enabled() || pos <= 0 || pos != (int64_t) live_.tokens.size() || find(live_, pos) != nullptr) return;
    void* h = take_buffer(pos);
    if (h == nullptr) return;
    cudaDeviceSynchronize();
    const bool split = split_from > 0 && split_save;
    if (!(split ? session_ckpt_save_layers(ss_, g_, pos, h, stream_, 0, split_from, 0)
                : session_ckpt_save(ss_, g_, pos, h, stream_))) {
        free_.push_back(h);
        return;
    }
    if (split) split_save(pos, h);
    live_.ckpts.push_back({pos, h, ++clock_});
}

void PromptCache::clear() {
    release(live_, -1);
    live_.tokens.clear();
    live_.imgs.clear();
    for (Seq& s : stash_) release(s, -1);
    stash_.clear();
}

std::string PromptCache::summary() const {
    size_t n = live_.ckpts.size();
    uint64_t kv = 0;
    for (const Seq& s : stash_) {
        n += s.ckpts.size();
        kv += s.kv_bytes;
    }
    char buf[192];
    std::snprintf(buf, sizeof buf, "live %lld cells; %zu checkpoints (%.0f MiB); %zu stashed sequences (%.0f MiB)",
                  (long long) live_.tokens.size(), n, (double) (n * ckpt_bytes_) / 1048576.0, stash_.size(),
                  (double) kv / 1048576.0);
    return buf;
}

}  // namespace strata::core
