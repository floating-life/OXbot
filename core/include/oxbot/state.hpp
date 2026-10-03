#pragma once

#include "oxbot/mini_json.hpp"
#include "oxbot/rules.hpp"

#include <array>
#include <string>
#include <vector>

namespace oxbot {

struct StateMirror {
    int player = -1;
    std::vector<int> hand;
    std::vector<PublicEvent> public_events;
    std::array<int, 4> remaining_counts{{-1, -1, -1, -1}};
    DecisionContext context;
    std::string error;

    bool rebuild(const json::Value& full_input);
};

}  // namespace oxbot
