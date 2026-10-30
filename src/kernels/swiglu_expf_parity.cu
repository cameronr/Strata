// src/kernels/swiglu_expf_parity.cu - --expert-math exact's SwiGLU: the GPU's against the CPU pool's, over every float.
//
// For every 32-bit pattern x: expf_contract(x) on the device against its host form, and the SwiGLU silu(g) * u with
// g = x (u = 1 and a fixed pseudo-random u per x) in the exact gate/up kernel's operations against the CPU pool's own
// native_swiglu_exact (lamina_kernels_cpu, the function the pool calls).  Both must match bit for bit.  It also counts
// the patterns where the C runtime's expf differs from expf_contract, the reason the pool does not call it under exact.
// Usage: swiglu_expf_parity (exit 0 when both match everywhere); runs on the first visible GPU.
#include "lamina/kernels/cpu/native_expert.hpp"
#include "lamina/kernels/expf_contract.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr uint64_t kChunk = 1ull << 26;

__host__ __device__ inline float u_of(uint32_t b) {   // a u in [-4, 4) from the pattern's hash
    uint32_t h = b * 0x9e3779b1u;
    h ^= h >> 15;
    h *= 0x85ebca77u;
    h ^= h >> 13;
    return (float) (int32_t) (h >> 8) * (1.0f / 2097152.0f) - 4.0f;
}

__global__ void eval_kernel(uint32_t base, uint32_t n, uint32_t* e, uint32_t* s1, uint32_t* su) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t b = base + i;
    const float x = __uint_as_float(b);
    e[i] = __float_as_uint(lamina::kernels::expf_contract(x));
    // native_gus_contract_kernel's operations
    const float ee = lamina::kernels::expf_contract(-x);
    const float sil = __fdiv_rn(x, __fadd_rn(1.0f, ee));
    s1[i] = __float_as_uint(sil);
    su[i] = __float_as_uint(__fmul_rn(sil, u_of(b)));
}

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return u;
}
float from_bits(uint32_t u) {
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}
bool same(uint32_t a, uint32_t b) {   // NaNs compare by class: a NaN's payload is outside the model's range
    return a == b || (std::isnan(from_bits(a)) && std::isnan(from_bits(b)));
}

}  // namespace

int main() {
    namespace cpu = lamina::kernels::cpu;
    uint32_t *de = nullptr, *ds1 = nullptr, *dsu = nullptr;
    if (cudaMalloc(&de, kChunk * 4) != cudaSuccess || cudaMalloc(&ds1, kChunk * 4) != cudaSuccess ||
        cudaMalloc(&dsu, kChunk * 4) != cudaSuccess) {
        std::fprintf(stderr, "swiglu_expf_parity: cudaMalloc failed\n");
        return 2;
    }
    std::vector<uint32_t> ge(kChunk), gs1(kChunk), gsu(kChunk);
    const unsigned nth = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<uint64_t> bad_e{0}, bad_s1{0}, bad_su{0}, crt_differs{0};
    std::atomic<uint64_t> first_bad{~0ull};
    for (uint64_t base = 0; base < (1ull << 32); base += kChunk) {
        eval_kernel<<<(unsigned) (kChunk / 256), 256>>>((uint32_t) base, (uint32_t) kChunk, de, ds1, dsu);
        if (cudaMemcpy(ge.data(), de, kChunk * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(gs1.data(), ds1, kChunk * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(gsu.data(), dsu, kChunk * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "swiglu_expf_parity: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 2;
        }
        std::vector<std::thread> th;
        for (unsigned w = 0; w < nth; ++w)
            th.emplace_back([&, w] {
                uint64_t be = 0, b1 = 0, bu = 0, crt = 0;
                for (uint64_t i = w; i < kChunk; i += nth) {
                    const uint32_t b = (uint32_t) (base + i);
                    const float x = from_bits(b);
                    const uint32_t he = bits(lamina::kernels::expf_contract(x));
                    crt += !same(he, bits(std::exp(x)));
                    const bool ok = same(ge[i], he) && same(gs1[i], bits(cpu::native_swiglu_exact(x, 1.0f))) &&
                                    same(gsu[i], bits(cpu::native_swiglu_exact(x, u_of(b))));
                    if (!ok) {
                        be += !same(ge[i], he);
                        b1 += !same(gs1[i], bits(cpu::native_swiglu_exact(x, 1.0f)));
                        bu += !same(gsu[i], bits(cpu::native_swiglu_exact(x, u_of(b))));
                        uint64_t f = first_bad.load();
                        while (b < f && !first_bad.compare_exchange_weak(f, b)) {}
                    }
                }
                bad_e += be;
                bad_s1 += b1;
                bad_su += bu;
                crt_differs += crt;
            });
        for (auto& t : th) t.join();
    }
    cudaFree(de);
    cudaFree(ds1);
    cudaFree(dsu);
    std::printf("expf_contract, device against host:   %llu of 2^32 patterns differ\n",
                (unsigned long long) bad_e.load());
    std::printf("SwiGLU (u = 1), GPU against the pool:  %llu differ\n", (unsigned long long) bad_s1.load());
    std::printf("SwiGLU (u varied), GPU against the pool: %llu differ\n", (unsigned long long) bad_su.load());
    if (first_bad.load() != ~0ull) std::printf("first differing pattern 0x%08llx\n", first_bad.load());
    std::printf("the C runtime's expf differs from expf_contract at %llu patterns (NaNs by class)\n",
                (unsigned long long) crt_differs.load());
    const bool ok = bad_e == 0 && bad_s1 == 0 && bad_su == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
