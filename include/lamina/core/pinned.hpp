// include/lamina/core/pinned.hpp - P2.S1: the host arena the expert weights live in.
//
// 33.97 GB of expert weights cannot fit in a 12 GB card, so they stay in host memory and are streamed.  That
// makes this arena the engine's real working set: it must be PAGE-LOCKED for the copy engine to reach full
// bandwidth, and it must say which backing it got, because the two options differ by more than twice in TLB
// reach:
//
//   * LARGE PAGES (2 MB) - `mmap(MAP_HUGETLB)` / hugetlbfs on Linux, `VirtualAlloc(MEM_LARGE_PAGES)` on
//     Windows.  33.97 GB at 4 KB pages is 8.3 million TLB entries, which does not fit in any TLB, so every
//     block of every expert matvec takes TLB misses.
//   * NORMAL PAGES - the fallback.  Correct, slower, and it must be REPORTED rather than silently accepted:
//     "the engine adapts to the machine it is on" is only true if the engine says what it got.  On Windows
//     large pages additionally need SeLockMemoryPrivilege, which a normal user account does not have, so this
//     fallback is the common case and not an error path.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lamina::core {

enum class PageBacking { LargePages, NormalPages, PinnedByCuda };

struct PinnedArena {
    void* base = nullptr;
    uint64_t capacity = 0;
    PageBacking backing = PageBacking::NormalPages;
    std::string note;              // why the backing is what it is, for the startup print
    uint64_t locked_bytes = 0;     // resident via the working-set lock when CUDA could not pin it
    /// Plan v0.3 P5: when the whole arena cannot be registered, it is registered in `slice`-byte pieces from the
    /// start; this is the pinned prefix (a copy that stays inside one slice can then DMA straight from the arena).
    uint64_t registered_bytes = 0;
    uint64_t slice_bytes = 0;
    int registered_slices = 0;

    PinnedArena() = default;
    /// `slice`: the piece size for the per-slice registration fallback (0 = none).
    explicit PinnedArena(uint64_t bytes, uint64_t slice = 0);
    /// Plan v0.3 P6: slices of different sizes (one per layer of a native pack), given as their start offsets
    /// followed by the end of the last one.  `slice_starts` holds the registered ones.
    PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds);
    std::vector<uint64_t> slice_starts;
    ~PinnedArena();
    PinnedArena(const PinnedArena&) = delete;
    PinnedArena& operator=(const PinnedArena&) = delete;

    bool valid() const { return base != nullptr; }
    uint8_t* data() const { return (uint8_t*) base; }
};

/// What the expert arena's load read, and how fast.
struct LoadStats {
    double seconds = 0.0;
    uint64_t bytes = 0;
    uint64_t layers = 0;
    double gib_per_second() const { return seconds > 0 ? (double) bytes / (1024.0 * 1024 * 1024) / seconds : 0.0; }
};


}  // namespace lamina::core
