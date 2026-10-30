// src/kernels/cpu/q2_bitplane.inc - ActQ's bit-plane image for the Q2_0 rows of q2_avx2.cpp and q2_avx1.cpp: SSE2 and
// SSSE3 only.  Included INSIDE each file's anonymous namespace: each file compiles it with its own /arch flags, and
// the anonymous namespace keeps every copy to its file (one shared copy might be the AVX2 build's).

// ActQ::qp / psum / pscale from q and scale (see expert.hpp).  One 64-value block: four 16-byte groups, each
// shuffled to plane-major (4 bytes per plane), then a 4x4 dword transpose gives the four 16-byte planes.
void bitplane_image(ActQ& a) {
    if (a.nchunks % 4 != 0) { a.bp_pairs = 0; return; }
    const int nblk = a.nchunks / 2;
    const __m128i shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m128i bias = _mm_set1_epi8((char) 0x80);
    for (int b = 0; b < nblk; ++b) {
        const int8_t* q = a.q + b * 64;
        __m128i s[4];
        for (int g = 0; g < 4; ++g) {
            const __m128i v = _mm_loadu_si128((const __m128i*) (q + 16 * g));
            s[g] = _mm_shuffle_epi8(v, shuf);
            // sum of 16 signed bytes: sad of (q ^ 0x80) = sum(q + 128), two 8-byte halves
            const __m128i sad = _mm_sad_epu8(_mm_xor_si128(v, bias), _mm_setzero_si128());
            a.psum[(b >> 1) * 8 + (b & 1) * 4 + g] =
                _mm_cvtsi128_si32(sad) + _mm_extract_epi16(sad, 4) - 16 * 128;
        }
        const __m128i t0 = _mm_unpacklo_epi32(s[0], s[1]), t1 = _mm_unpacklo_epi32(s[2], s[3]);
        const __m128i t2 = _mm_unpackhi_epi32(s[0], s[1]), t3 = _mm_unpackhi_epi32(s[2], s[3]);
        int8_t* dst = a.qp + (b >> 1) * 128 + (b & 1) * 16;
        _mm_store_si128((__m128i*) (dst), _mm_unpacklo_epi64(t0, t1));
        _mm_store_si128((__m128i*) (dst + 32), _mm_unpackhi_epi64(t0, t1));
        _mm_store_si128((__m128i*) (dst + 64), _mm_unpacklo_epi64(t2, t3));
        _mm_store_si128((__m128i*) (dst + 96), _mm_unpackhi_epi64(t2, t3));
    }
    for (int c = 0; c < a.nchunks; ++c) {
        a.pscale[2 * c] = a.scale[c];
        a.pscale[2 * c + 1] = a.scale[c];
    }
    a.bp_pairs = nblk / 2;
}
