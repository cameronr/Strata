// include/lamina/prefill/split.hpp - the batched prompt path in two stages on two GPUs (--prompt-split).
//
// One GPU alone runs a 16K-token chunk's layer about as fast as the main GPU does with the second GPU computing the
// experts its cache lacks: the RTX 5070 Ti here runs the dense prompt steps 1.4x as fast as the RTX 3090 (its MMQ
// 1.18-1.48x, its FP32 and L2 more) and streams the experts it lacks over its x16 link.  So the second GPU takes the
// last layers of each chunk whole while the main GPU runs the next chunk through the first ones:
//
//   main GPU:    chunk c, layers [0, at)  (the experts it lacks on the second GPU, ExpertRunner remote)
//   second GPU:  chunk c - 1, layers [at, n_layers)  (a Prefill of its own: its tier's experts, the rest streamed),
//                then the draft layer's K/V over the chunk's cells
//
// A chunk's residual after layer at - 1 crosses the main GPU's link once (FP32, through a pinned buffer) a sub-chunk
// at a time.  The second stage runs on a thread of its own, on its own stream, below the remote runner's priority, its
// GPU queue kept short (Prefill::set_pacing: under WDDM work queued behind a long queue waits for all of it); it works
// on copies in the second GPU's lent cache slots: the dense weights of its layers, and the session state of its
// layers - the K/V cells and the indexer of its QSA layers and its GDN states - and the draft layer's cells of one
// chunk.  The weights and the state at the prompt's start come from the main GPU's (through a staging buffer, while
// the first chunk runs there); each chunk's new cells go back after it, the GDN states and the indexers' tails at
// `finish`, which the prompt's end waits for.  A prompt-cache checkpoint after a chunk is saved by both stages into
// one buffer, each its layers.  Copies on the second GPU go out once they can run, never behind a wait: its one copy
// engine would hold every later copy behind it, the first stage's too.
#pragma once

#include "lamina/core/expert_cache.hpp"
#include "lamina/core/expert_source.hpp"
#include "lamina/core/mtp.hpp"
#include "lamina/core/session.hpp"
#include "lamina/core/weights.hpp"
#include "lamina/prefill/prefill.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lamina::prefill {

class PromptSplit final : public SplitOut {
public:
    PromptSplit();
    ~PromptSplit() override;
    PromptSplit(const PromptSplit&) = delete;
    PromptSplit& operator=(const PromptSplit&) = delete;

    /// Once, with the main GPU current: `first` is the main GPU's prompt path (its session `ss`, weights `wt`), split
    /// at layer `at` (a GDN or QSA layer past the PLE's, 2 .. n_layers - 1); the second stage runs on `device` with its
    /// cache `cache` and residency `res` (n_layers x n_expert), streaming the rest from `src`.  `mtp`: the draft layer,
    /// whose prompt K/V the second stage computes (or null).
    bool init(Prefill& first, const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
              core::MtpDrafter* mtp, int64_t at, int device, int main_device, core::ExpertSource* src,
              const core::ExpertCache* cache, const int32_t* res, int64_t max_chunk, std::string& err);
    int64_t at() const;
    /// Prefill::set_order for the second stage's copies ahead.
    void set_order(std::vector<int32_t> order);

    /// The second GPU's device bytes for a prompt in chunks of `chunk` tokens that ends at cell `cells`.
    uint64_t bytes_for(int64_t chunk, int64_t cells) const;
    /// Before a prompt [pos0, cells) whose chunks hold up to `chunk` tokens: the second stage's buffers and copies in
    /// `region` (`bytes`, from bytes_for: lent slots of its cache), the weights copied in, the session state of its
    /// layers at pos0 copied from the main GPU's (on its thread, while the first chunk runs there); `first` then runs
    /// layers [0, at) and hands each chunk on.  `checkpoints`: the prompt cache may checkpoint at the end of a run()
    /// call (`ran` says when it has decided).
    bool begin(void* region, uint64_t bytes, int64_t chunk, int64_t pos0, int64_t cells, bool checkpoints,
               std::string& err);
    bool active() const;
    /// The prompt cache's checkpoint at `pos` (core::PromptCache::split_save): the second stage saves its layers'
    /// state into `host` once its chunk ending at `pos` is done.
    void checkpoint(int64_t pos, void* host);
    /// After `first.run(...)` returned with its last chunk ending at `pos` and the prompt cache has checkpointed there
    /// or not: the second stage may go on past that chunk.
    void ran(int64_t pos);
    /// After the prompt's last run(): waits for the second stage, then the state of its layers goes back to the main
    /// GPU's session and its routing counts into `first`'s; `first` runs every layer again.
    bool finish(std::string& err);

    float* out_begin(std::string& err) override;
    bool out_end(const int64_t* tokens, int64_t T, int64_t pos0, const cudaEvent_t* landed, int64_t piece,
                 bool run_end, std::string& err) override;

    /// The second stage's prompt path (its stats, the --prefill-profile sections of its layers).
    const PrefillStats& stats() const;
    void set_profile(bool on);
    double ms_first_waited = 0;    ///< host time the first stage waited to hand a chunk on
    double ms_finish = 0;          ///< host time `finish` waited for the second stage and the state's copies
    double ms_copy_in = 0;         ///< the prefix state's copy to the second GPU (its thread)
    uint64_t weight_bytes = 0;     ///< the dense weights the second stage copies in at each `begin`

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lamina::prefill
