// include/lamina/kernels/s2_expert_grouped.hpp - R4's GPU expert tier.
//
// **WHAT THIS IS FOR.** The CPU expert pool is the engine's largest single cost - 663.6 MB of expert bytes per
// token at ~40 GB/s = 16.2 ms on a ~53 ms token - and `test_expert_pool` measures the pipeline hiding
// **1.055 ms of 19.035**. The CPU work is 96% exposed because the residual chain is strictly serial, so the
// only way past it is to stop reading those bytes on the CPU: put experts in VRAM and compute them on the GPU.
//
// **WHY GROUPED, AND WHY THAT WORD IS THE WHOLE DESIGN.** A layer routes ten experts. The obvious
// implementation is a loop over hits calling the per-matrix kernel, and it is wrong by arithmetic: five
// launches per expert at ~6.45 hits and 48 layers is **~1,550 launches per token**, which at a real-stream
// launch cost of ~3.6 us is 5.6 ms - most of the 10.4 ms the tier is supposed to save. So the hits are grouped
// into ONE launch per projection:
//
//     1. `gu_kernel`    n_hits x 2 x FF rows   - gate and up, together
//     2. `swiglu_kernel` n_hits x FF           - silu(gate) * up
//     3. `quantize_q8_0` (the existing one)    - the down weight's own activation contract
//     4. `down_kernel`   n_hits x H rows
//
// **four launches per layer, not five per expert.**
//
// `h_layer` held-out is **0.0456** - only 4.6% of (layer, token) pairs have all ten experts resident - so this
// cannot be a per-layer grouped kernel with a fixed grid, and the hit list is passed **as data** so the caller
// can keep one captured graph per layer with the maximum grid and a device-side count.
#pragma once

#include <cstdint>

namespace lamina::kernels {

/// Bytes of caller-owned scratch `moe_hit_grouped_s2` needs for `n_hits` experts.
///
/// Two regions: the gate/up activations (`n_hits x 2 x FF` floats) and the quantized intermediate
/// (`n_hits x (FF/32) x 34` bytes).  Carved from the caller's arena rather than allocated - a `cudaMalloc`
/// on the token path is forbidden by P2.T10 and illegal during stream capture, and this is called from both.
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff);

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream);
/// Plan v0.3 P6 verify window: `moe_hit_select` over `n` <= 128 routed entries (T tokens x k, flattened), and the
/// hit kernel with one activation PER TOKEN - entry `dst` reads token `dst / k_per_token`'s rows of `x_q8_0`
/// ((n_embd/32)*34 bytes each) and `x_scales` (n_embd/32 floats each).  Per hit, bitwise `moe_hit_grouped_s2_dev`.
/// Plan v0.3 P6: routed experts GROUPED.  Group g's blob is at device address grp_ptr[g] (a VRAM slot, or a mapped
/// host blob the kernel reads over PCIe); its entries are [grp_start[g], grp_start[g+1]) with ent_tok (the token
/// whose activation row of x_q8_0/x_scales it reads) and ent_dst (its row of `out`).  Each row of a blob is read
/// once for all of its entries.  Counts are read from device memory; every entry is bitwise the per-entry kernel.
void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream);
/// Plan v0.3 P6: the groups for `moe_grouped_s2` when every expert id is resident at base + id * blob (the MTP
/// layer's 512 experts): counts[0] groups, counts[1] entries; entry dst = routing index, tok = index / k_per_tok.
void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream);

}  // namespace lamina::kernels
