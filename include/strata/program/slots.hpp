// include/strata/program/slots.hpp - the slot registry: one explicit owner per conversation (phase 2 of
// plans/01-engine-batching/).
//
// Today the engine serves one conversation (the solo path); batching (docs/BATCHING.md) keeps up to N of them
// open at once, and every scheduler decision and batch-window kernel argument is then an address INTO this
// table. So the per-conversation state stops being a set of locals in the serve loop and becomes one `Slot`
// per conversation, owned by a `SlotRegistry`. The solo path is slot 0 (`registry.solo()`); nothing schedules
// yet, and nothing here changes what any request computes.
//
// What lives in a slot is exactly the state two conversations must NOT share: the session (GDN recurrence,
// the QSA K/V arena, the PLE history), the prompt-lookup drafter's history, the previous prompt, the
// penalty-history buffers, and the checkpoint image a slot hands back when its conversation returns to the
// solo path. What stays engine-level (weights, the expert cache, the prompt path, the graphs, the elastic
// K/V arena) is shared by design - see the phase-2 plan for the boundary.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/session.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace strata::program {

/// The five states of the phase-2 plan, in the only order they occur:
/// free -> admitted (a request took the slot) -> running (its session state is bound) ->
/// finishing (its last window is in) -> freed (DONE was answered; it keeps the conversation) -> free.
enum class SlotState : int { free = 0, admitted, running, finishing, freed };

inline const char* slot_state_name(SlotState s) {
    switch (s) {
        case SlotState::free: return "free";
        case SlotState::admitted: return "admitted";
        case SlotState::running: return "running";
        case SlotState::finishing: return "finishing";
        case SlotState::freed: return "freed";
    }
    return "?";
}

/// One conversation's own state. The pointers are carved and owned by the engine around the registry (the
/// arena is a `cudaMalloc` the serve setup makes into `arena` and carves into `ss`); the registry only names
/// the owner. Default-constructed: an empty slot with no resources.
struct Slot {
    int id = 0;
    SlotState state = SlotState::free;

    strata::core::SessionState ss;   ///< GDN state, QSA K/V arena, PLE history (session.hpp)
    void* arena = nullptr;           ///< the device buffer `ss` was carved from
    strata::core::Doorbell db;       ///< its doorbell (upstream: a slot's doorbell is served on its own thread)

    std::optional<strata::spec::SuffixDrafter> lookup;   ///< its prompt-lookup history (serve emplaces it)
    std::vector<int64_t> prev_prompt;                    ///< the last request's prompt (checkpoint fork point)
    int32_t* d_hist = nullptr;                           ///< its penalty-history device buffer (serve allocates)
    std::vector<int32_t> hist_stage;                     ///< and the host staging that feeds it

    /// The conversation's running-state image: `conversation_checkpoint_save`/`_restore`
    /// (strata/core/conversation_snapshot.hpp) fill and apply it when a slot hands its conversation back
    /// or takes one over. Wired to the type in phase 2; the serve loop starts using it with the scheduler.
    strata::core::ConversationCheckpoint snapshot;
};

/// The slots, and the only mutator of their states. Phase 2 constructs it with `--parallel N` slots but the
/// engine still runs one request at a time on slot 0; the scheduler (phase 4) admits into the others.
class SlotRegistry {
  public:
    explicit SlotRegistry(int n_slots) : slots_(static_cast<size_t>(n_slots > 0 ? n_slots : 1)) {
        for (size_t i = 0; i < slots_.size(); ++i) slots_[i].id = static_cast<int>(i);
    }

    int size() const { return static_cast<int>(slots_.size()); }
    Slot& solo() { return slots_[0]; }              ///< the solo path's slot: always slot 0
    Slot& at(int id) { return slots_[static_cast<size_t>(id)]; }
    const Slot& at(int id) const { return slots_[static_cast<size_t>(id)]; }

    /// The transition table: only the next state of free -> admitted -> running -> finishing -> freed -> free.
    static bool legal(SlotState from, SlotState to) {
        switch (from) {
            case SlotState::free: return to == SlotState::admitted;
            case SlotState::admitted: return to == SlotState::running;
            case SlotState::running: return to == SlotState::finishing;
            case SlotState::finishing: return to == SlotState::freed;
            case SlotState::freed: return to == SlotState::free;
        }
        return false;
    }

    /// The only mutator: moves `slot` to `to` when the table allows it; returns false and changes nothing
    /// otherwise (the caller reports it - a rejected transition is an engine bug, not a request outcome).
    bool advance(Slot& slot, SlotState to) {
        if (!legal(slot.state, to)) return false;
        slot.state = to;
        return true;
    }

  private:
    std::vector<Slot> slots_;
};

}  // namespace strata::program
