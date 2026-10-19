#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <climits>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace strata::core {
namespace {
bool eligible(const std::string& name, bool include_ple_key) {
    if (name.rfind("blk.", 0) != 0) return false;
    if (include_ple_key && name == "blk.1.ple_key.weight") return true;
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
struct Upload { void* dev; const uint8_t* src; uint64_t bytes; };
// The matrices' bytes into VRAM from up to 8 threads in 8 MiB pieces through pinned staging (upstream PR #2673): a thread
// copies a piece from the shard's mapping (taking its page faults) into one half of its staging buffer and sends it on
// its own stream while it fills the other half.  One synchronous cudaMemcpy a matrix took the faults one at a time
// (UD-Q4_K_XL's 2,975 MiB in ~1.5-2 s with the file cached).
bool upload_parallel(const std::vector<Upload>& jobs, std::string& err) {
    constexpr uint64_t PIECE = 8ull << 20;
    std::vector<std::pair<size_t, uint64_t>> pieces;   // (job, offset)
    for (size_t j = 0; j < jobs.size(); ++j)
        for (uint64_t o = 0; o < jobs[j].bytes; o += PIECE) pieces.emplace_back(j, o);
    if (pieces.empty()) return true;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { err = "native dense upload: no device"; return false; }
    std::atomic<size_t> next{0};
    std::atomic<bool> bad{false};
    std::atomic<int> first_err{(int) cudaSuccess};   // a worker's error (the runtime's last error is per thread)
    auto fail = [&](cudaError_t e) {
        int none = (int) cudaSuccess;
        first_err.compare_exchange_strong(none, (int) e);
        bad = true;
    };
    auto worker = [&] {
        cudaStream_t s = nullptr;
        uint8_t* stage = nullptr;
        cudaEvent_t done[2] = {nullptr, nullptr};
        bool live[2] = {false, false};
        cudaError_t e = cudaSetDevice(device);
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaHostAlloc((void**) &stage, 2 * PIECE, cudaHostAllocDefault);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming);
        if (e != cudaSuccess) fail(e);
        for (int h = 0; !bad; h ^= 1) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) break;
            const auto [j, o] = pieces[i];
            const uint64_t n = std::min(PIECE, jobs[j].bytes - o);
            if (live[h] && (e = cudaEventSynchronize(done[h])) != cudaSuccess) { fail(e); break; }
            std::memcpy(stage + h * PIECE, jobs[j].src + o, (size_t) n);
            e = cudaMemcpyAsync((uint8_t*) jobs[j].dev + o, stage + h * PIECE, (size_t) n, cudaMemcpyHostToDevice, s);
            if (e == cudaSuccess) e = cudaEventRecord(done[h], s);
            if (e != cudaSuccess) { fail(e); break; }
            live[h] = true;
        }
        if (s != nullptr && (e = cudaStreamSynchronize(s)) != cudaSuccess) fail(e);
        for (cudaEvent_t ev : done) if (ev) cudaEventDestroy(ev);
        if (stage) cudaFreeHost(stage);
        if (s) cudaStreamDestroy(s);
    };
    const size_t threads = std::min<size_t>(8, pieces.size());
    std::vector<std::thread> pool;
    for (size_t k = 1; k < threads; ++k) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    // the tails' memsets, on the legacy stream beside the workers' non-blocking ones, landed too
    if (const cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) fail(e);
    if (bad) {
        err = std::string("native dense upload: ") + cudaGetErrorString((cudaError_t) first_err.load());
        return false;
    }
    return true;
}
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor.name, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key) {
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        std::string first_arch;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            const auto* architecture = gguf.get("general.architecture");
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            // A re-split GGUF (Huihui's Q8_0) repeats general.architecture on every shard but keeps the
            // architecture's numeric keys (block_count ...) on the first one only: the full guard is the first
            // shard's (or a single file's); a later shard that names an architecture must name the same one
            if (architecture && (!number || number->u == 0)) {
                err = strata::check_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                first_arch = architecture->s;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors || (architecture && architecture->s != first_arch)) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            std::vector<Upload> jobs;
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor.name, include_ple_key)) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                // a zeroed tail: the prompt path's MMQ reads past the last row of a matrix whose rows are not whole
                // 256-value steps (the shared expert's down projection, 640), and NaN times a zero input is NaN
                constexpr uint64_t kTail = 4096;
                void* allocation = nullptr;
                auto status = cudaMalloc(&allocation, bytes + kTail);
                DevicePtr data(allocation);
                if (status == cudaSuccess) status = cudaMemset((uint8_t*) data.get() + bytes, 0, kTail);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                jobs.push_back(Upload{data.get(), (const uint8_t*) gguf.tensor_data(tensor), bytes});
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
            if (!upload_parallel(jobs, err)) return false;
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status = cudaMalloc(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
namespace {
// GCC/Clang only take the F16C intrinsics below in a function compiled for F16C (MSVC takes them anywhere); every
// CPU with the engine's AVX2 floor has F16C.
#if defined(__GNUC__) && !defined(__F16C__)
#define STRATA_F16C_FN __attribute__((target("f16c")))
#else
#define STRATA_F16C_FN
#endif
bool hc_matrix(const std::string& name) {
    if (name.rfind("blk.", 0) != 0) return name == "output_hc_down.weight" || name == "output_hc_up.weight";
    return name.ends_with(".hc_attn_down.weight") || name.ends_with(".hc_attn_up.weight") ||
           name.ends_with(".hc_ffn_down.weight") || name.ends_with(".hc_ffn_up.weight");
}
// one row's values as ggml's quantize_row_q8_0_ref: per 32, d = amax / 127 stored as f16, codes round(x / d)
STRATA_F16C_FN void q8_0_row(const float* x, int64_t n, int8_t* q, uint16_t* s) {
    for (int64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        s[b] = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(d), _MM_FROUND_TO_NEAREST_INT), 0);
        for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::roundf(x[b * 32 + j] * id);
    }
}
STRATA_F16C_FN float f16_to_f32(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
}  // namespace

bool HcQ8::names(const std::vector<std::string>& shards, std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& t : gguf.tensors())
                if (hc_matrix(t.name)) out.insert(t.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("hc q8: ") + error.what();
        return false;
    }
}

HcQ8::~HcQ8() {
    for (void* p : allocs_) cudaFree(p);
}

bool HcQ8::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err) {
    try {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { err = "hc q8: no device"; return false; }
        struct Job {
            const strata::TensorInfo* t;
            WeightRef* ref;
            bool q8_0;    // the GGUF's own Q8_0 blocks, split into codes and scales
            int kind;     // else the rows' type: 0 F32, 1 F16, 2 BF16
            void* dq = nullptr;
            void* ds = nullptr;
        };
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            std::vector<Job> jobs;
            for (const auto& t : gguf.tensors()) {
                if (!hc_matrix(t.name)) continue;
                auto found = table.table_.find(t.name);
                if (found == table.table_.end()) {
                    err = "hc q8: tensor absent from the canonical table: " + t.name;
                    return false;
                }
                WeightRef& ref = found->second;
                const std::string tn = t.type_name();
                if (t.shape.size() != 2 || t.shape[0] != (uint64_t) ref.ne0 || t.shape[1] != (uint64_t) ref.ne1 ||
                    ref.ne0 % 32 != 0 || (tn != "Q8_0" && tn != "BF16" && tn != "F16" && tn != "F32")) {
                    err = "hc q8: " + t.name + " has another shape or type (" + tn + ")";
                    return false;
                }
                Job j{&t, &ref, tn == "Q8_0", tn == "F32" ? 0 : tn == "F16" ? 1 : 2};
                const size_t n = (size_t) (ref.ne0 * ref.ne1);
                if (cudaMalloc(&j.dq, n) != cudaSuccess || cudaMalloc(&j.ds, n / 32 * 2) != cudaSuccess) {
                    if (j.dq) cudaFree(j.dq);
                    err = "hc q8: no VRAM for " + t.name;
                    return false;
                }
                allocs_.push_back(j.dq);
                allocs_.push_back(j.ds);
                jobs.push_back(j);
            }
            // a matrix at a time to each of up to 8 threads (one thread rounded IQ3_S's 194 BF16 matrices in 2 s)
            std::atomic<size_t> next{0};
            std::atomic<bool> bad{false};
            std::mutex err_mu;
            auto worker = [&] {
                std::vector<int8_t> q;
                std::vector<uint16_t> s;
                std::vector<float> row;
                if (cudaSetDevice(device) != cudaSuccess) bad = true;
                for (size_t k; !bad && (k = next++) < jobs.size();) {
                    const Job& j = jobs[k];
                    const int64_t cols = j.ref->ne0, rows = j.ref->ne1;
                    q.resize((size_t) (rows * cols));
                    s.resize((size_t) (rows * cols / 32));
                    row.resize((size_t) cols);
                    const uint8_t* src = gguf.tensor_data(*j.t);
                    for (int64_t r = 0; r < rows; ++r) {
                        int8_t* qr = q.data() + r * cols;
                        uint16_t* sr = s.data() + r * (cols / 32);
                        if (j.q8_0) {   // as stored: {f16 d, 32 codes} a block
                            const uint8_t* bl = src + (size_t) (r * (cols / 32)) * 34;
                            for (int64_t b = 0; b < cols / 32; ++b) {
                                std::memcpy(sr + b, bl + b * 34, 2);
                                std::memcpy(qr + b * 32, bl + b * 34 + 2, 32);
                            }
                            continue;
                        }
                        for (int64_t c = 0; c < cols; ++c) {
                            const size_t i = (size_t) (r * cols + c);
                            if (j.kind == 0) {
                                std::memcpy(&row[(size_t) c], src + i * 4, 4);
                            } else {
                                uint16_t h;
                                std::memcpy(&h, src + i * 2, 2);
                                if (j.kind == 1) {
                                    row[(size_t) c] = f16_to_f32(h);
                                } else {
                                    const uint32_t u = (uint32_t) h << 16;
                                    std::memcpy(&row[(size_t) c], &u, 4);
                                }
                            }
                        }
                        q8_0_row(row.data(), cols, qr, sr);
                    }
                    if (cudaMemcpy(j.dq, q.data(), q.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(j.ds, s.data(), s.size() * 2, cudaMemcpyHostToDevice) != cudaSuccess) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        if (!bad) err = "hc q8: upload of " + j.t->name + ": " + cudaGetErrorString(cudaGetLastError());
                        bad = true;
                    }
                }
            };
            const size_t threads = std::min<size_t>(8, jobs.size());
            std::vector<std::thread> pool;
            for (size_t k = 1; k < threads; ++k) pool.emplace_back(worker);
            if (threads > 0) worker();
            for (auto& th : pool) th.join();
            if (bad) {
                if (err.empty()) err = "hc q8: no device on a loader thread";
                return false;
            }
            for (const Job& j : jobs) {
                bytes_ += (uint64_t) (j.ref->ne0 * j.ref->ne1) / 32 * 34;   // the codes and an f16 scale a 32
                if (j.q8_0) ++q8_0_;
                j.ref->hc_codes = (const int8_t*) j.dq;
                j.ref->hc_scales = (const uint16_t*) j.ds;
            }
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("hc q8: ") + error.what();
        return false;
    }
}
} // namespace strata::core
