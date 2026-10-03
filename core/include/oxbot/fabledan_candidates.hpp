#pragma once

#include "oxbot/rules.hpp"

namespace oxbot {

// Version 2 retains bounded, deterministic suit realizations, with exactly
// the same representative set and ordering as Python's _structure_moves.
std::vector<Move> fabledan_structure_candidates(const DecisionContext& context);

}  // namespace oxbot
