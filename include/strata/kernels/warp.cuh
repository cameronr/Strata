// include/strata/kernels/warp.cuh - the kernels' warp-level helpers: sums and maxima over a warp's 32 lanes (the XOR
// butterfly: every lane gets the result, the additions always in one order), a block of 128 threads' sum, and the
// ldmatrix and cp.async instructions of the tensor-core kernels.
#pragma once

#include <cuda_runtime.h>

namespace strata::kernels {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// The sum over a block of 128 threads: `part` 4 floats of shared memory, read after one barrier (a caller that writes
/// them again first waits for every thread's read).
__device__ __forceinline__ float block128_sum(float v, float* part) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    return part[0] + part[1] + part[2] + part[3];
}

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], const void* s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned) __cvta_generic_to_shared(s)));
}
/// 16 bytes from global memory to shared memory, in the background (cp.async.cg)
__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
/// waits until at most N committed groups are still in flight
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

}  // namespace strata::kernels
