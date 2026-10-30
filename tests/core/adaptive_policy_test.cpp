#include "lamina/core/adaptive_policy.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

using lamina::core::AdaptiveMoveOrder;
using lamina::core::adaptive_move_less;

namespace {
int failures = 0;

void check(bool ok, const char* what) {
    std::printf("  %-70s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

bool same_key(const AdaptiveMoveOrder& a, const AdaptiveMoveOrder& b) {
    return a.gain == b.gain && a.layer == b.layer && a.in == b.in &&
           a.out == b.out && a.slot == b.slot;
}
}  // namespace

int main() {
    std::printf("adaptive_policy_test\n");

    const AdaptiveMoveOrder same_a{4.0f, 3, 12, 8, 40};
    const AdaptiveMoveOrder same_b{4.0f, 2, 99, 7, 41};
    check(!adaptive_move_less(same_a, same_b, false) && !adaptive_move_less(same_b, same_a, false),
          "async mode keeps equal-gain moves equivalent");
    check(adaptive_move_less(same_b, same_a, true),
          "deterministic mode orders equal gains by layer");

    std::vector<AdaptiveMoveOrder> moves = {
        {4.0f, 3, 12, 8, 40},
        {4.0f, 2, 99, 7, 41},
        {4.0f, 2, 99, 7, 39},
        {4.0f, 2, 99, 6, 99},
        {4.0f, 2, 98, 7, 41},
        {5.0f, 9, 0, -1, 1},
    };
    std::sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
        return adaptive_move_less(a, b, true);
    });
    const std::vector<AdaptiveMoveOrder> expected = {
        {5.0f, 9, 0, -1, 1},
        {4.0f, 2, 98, 7, 41},
        {4.0f, 2, 99, 6, 99},
        {4.0f, 2, 99, 7, 39},
        {4.0f, 2, 99, 7, 41},
        {4.0f, 3, 12, 8, 40},
    };
    bool sequence_ok = moves.size() == expected.size();
    for (size_t i = 0; sequence_ok && i < moves.size(); ++i) {
        sequence_ok = same_key(moves[i], expected[i]);
    }
    check(sequence_ok, "deterministic tuple orders layer, input, out, then slot");
    check(moves[2].out == 6 && moves[2].slot == 99,
          "out precedes slot when equal-gain layer/input keys match");

    return failures == 0 ? 0 : 1;
}
