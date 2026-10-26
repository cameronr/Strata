// slots.hpp: the slot registry's states, its transition table, and ownership (synthetic: no model, no GPU).
// Compile (CUDA headers for session.hpp, no CUDA runtime needed):
//   nvcc -std=c++20 -Iinclude -D__CUDA_ARCH__=800 -x c++ tests/program/slots_test.cpp src/spec/suffix_drafter.cpp -o /tmp/slots_test
// or with the toolkit's host path: g++ -std=c++20 -Iinclude -I/usr/local/cuda/include ...
#include "strata/program/slots.hpp"

#include <cstdio>
#include <string_view>

using strata::program::Slot;
using strata::program::SlotRegistry;
using strata::program::SlotState;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // ids, solo, and the empty-slot defaults
    SlotRegistry reg(3);
    CHECK(reg.size() == 3);
    CHECK(reg.solo().id == 0);
    CHECK(reg.at(2).id == 2);
    CHECK(reg.at(1).state == SlotState::free);
    CHECK(reg.solo().arena == nullptr);
    CHECK(!reg.solo().lookup.has_value());
    CHECK(reg.solo().ss.qsa_states == nullptr);   // resources are carved by the engine, not the registry

    // the transition table: only the next state of free -> admitted -> running -> finishing -> freed -> free
    CHECK(SlotRegistry::legal(SlotState::free, SlotState::admitted));
    CHECK(SlotRegistry::legal(SlotState::admitted, SlotState::running));
    CHECK(SlotRegistry::legal(SlotState::running, SlotState::finishing));
    CHECK(SlotRegistry::legal(SlotState::finishing, SlotState::freed));
    CHECK(SlotRegistry::legal(SlotState::freed, SlotState::free));
    CHECK(!SlotRegistry::legal(SlotState::free, SlotState::running));       // no skipping admitted
    CHECK(!SlotRegistry::legal(SlotState::running, SlotState::admitted));   // no going back
    CHECK(!SlotRegistry::legal(SlotState::free, SlotState::free));          // no self-loop
    CHECK(!SlotRegistry::legal(SlotState::freed, SlotState::admitted));

    // advance: the only mutator; a rejected transition changes nothing
    Slot& s = reg.solo();
    CHECK(!reg.advance(s, SlotState::running));
    CHECK(s.state == SlotState::free);
    CHECK(reg.advance(s, SlotState::admitted));
    CHECK(s.state == SlotState::admitted);
    CHECK(reg.advance(s, SlotState::running));
    CHECK(reg.advance(s, SlotState::finishing));
    CHECK(reg.advance(s, SlotState::freed));
    CHECK(reg.advance(s, SlotState::free));
    CHECK(s.state == SlotState::free);

    // a full round leaves the slot reusable, and slots are independent
    CHECK(reg.advance(s, SlotState::admitted));
    CHECK(reg.advance(reg.at(1), SlotState::admitted));
    CHECK(reg.at(1).state == SlotState::admitted);
    CHECK(s.state == SlotState::admitted);
    CHECK(!reg.advance(reg.at(1), SlotState::free));
    CHECK(reg.at(1).state == SlotState::admitted);

    // the state names the serve loop prints on an internal error
    CHECK(std::string_view("admitted") == strata::program::slot_state_name(SlotState::admitted));

    // the drafter the serve loop emplaces (host-only: the prompt-lookup history is per slot)
    s.lookup.emplace(3, 32, 1024);
    CHECK(s.lookup->size() == 0);
    s.lookup->append(7);
    CHECK(s.lookup->size() == 1);

    std::printf(fails ? "slots_test: %d FAILURES\n" : "slots_test: ok (%d failures)\n", fails);
    return fails != 0;
}
