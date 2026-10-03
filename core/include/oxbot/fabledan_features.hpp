#pragma once

#include "oxbot/rules.hpp"

#include <vector>

namespace oxbot {

// FableDan's independent inference contract.  These functions intentionally
// do not replace encode_state/encode_history/encode_action: the existing
// OXGDQ001 candidate model remains a supported rollback path.
std::vector<int> encode_fabledan_history(const DecisionContext& context);
std::vector<float> encode_fabledan_action(const Move& move,
                                          const DecisionContext& context,
                                          int feat_dim = 80);
// The referee-facing C++ generator exposes wildcard realization variants
// that the Python FableDan generator intentionally never emits (for example,
// a wildcard single claiming a rank absent from the hand).  This predicate
// removes only those impossible training-candidate semantics; physical suit
// and deck variants remain legal and are canonicalized by the policy.
bool fabledan_generator_candidate(const Move& move,
                                  const DecisionContext& context);
std::vector<std::vector<float>> encode_fabledan_actions(
    const std::vector<Move>& moves, const DecisionContext& context,
    int feat_dim = 80);

}  // namespace oxbot
