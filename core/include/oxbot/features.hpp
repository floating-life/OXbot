#pragma once
#include "oxbot/rules.hpp"
#include <vector>

namespace oxbot {

// This information-set contract is mirrored by train/features.py. Neither
// encoder accepts another player's private hand or a future result.
std::vector<float> encode_state(const DecisionContext& context);
std::vector<int> encode_history(const DecisionContext& context);
std::vector<float> encode_action(const Move& move, const DecisionContext& context);

} // namespace oxbot
