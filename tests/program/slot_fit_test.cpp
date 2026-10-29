// slot_fit.hpp: how many batch slots the free VRAM can carve (synthetic: no model, no GPU).
#include "strata/program/slot_fit.hpp"

#include <cstdio>

using strata::program::slots_that_fit;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const int64_t GiB = 1LL << 30;
    // everything fits: the wanted count
    CHECK(slots_that_fit(4, GiB, GiB / 4, 10 * GiB, 0) == 4);
    // partial start: the ones that fit, not zero
    CHECK(slots_that_fit(4, GiB, GiB / 4, 3 * GiB, 0) == 2);
    CHECK(slots_that_fit(8, 600 * 1024 * 1024, 256 * 1024 * 1024, 4 * GiB, 0) == 6);
    // the exact boundary: want * each + staging (+ reserve) fits exactly, one byte over does not
    CHECK(slots_that_fit(3, GiB, GiB / 4, 3 * GiB + GiB / 4, 0) == 3);
    CHECK(slots_that_fit(3, GiB, GiB / 4, 3 * GiB + GiB / 4 - 1, 0) == 2);
    CHECK(slots_that_fit(3, GiB, 0, 3 * GiB + 100, 100) == 3);
    CHECK(slots_that_fit(3, GiB, 0, 3 * GiB + 99, 100) == 2);
    // nothing fits: no room beyond the staging, or none at all
    CHECK(slots_that_fit(4, GiB, 10 * GiB, 10 * GiB, 0) == 0);
    CHECK(slots_that_fit(4, GiB, GiB / 4, GiB / 4, 0) == 0);
    CHECK(slots_that_fit(4, GiB, GiB / 4, 0, 0) == 0);
    // degenerate inputs: never negative, never more than wanted
    CHECK(slots_that_fit(0, GiB, 0, 10 * GiB, 0) == 0);
    CHECK(slots_that_fit(-1, GiB, 0, 10 * GiB, 0) == 0);
    CHECK(slots_that_fit(2, 0, 0, 10 * GiB, 0) == 0);
    CHECK(slots_that_fit(2, -GiB, 0, 10 * GiB, 0) == 0);
    CHECK(slots_that_fit(1000, 1, 0, 10 * GiB, 0) == 1000);
    return fails ? 1 : 0;
}
