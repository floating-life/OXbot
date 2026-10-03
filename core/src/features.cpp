#include "oxbot/features.hpp"
#include <algorithm>
#include <stdexcept>

namespace oxbot {
namespace {
void feature_require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void feature_faces(std::vector<float>& output, int start, const std::vector<int>& cards) {
    for (int card : cards) {
        feature_require(valid_card_id(card), "feature_card_invalid");
        output[static_cast<std::size_t>(start + card % 54)] += .5f;
    }
}
std::vector<int> sorted_feature_faces(const std::vector<int>& cards) {
    std::vector<int> result;
    for (int card : cards) {
        feature_require(valid_card_id(card), "history_card_invalid");
        result.push_back(card % 54);
    }
    std::sort(result.begin(), result.end());
    return result;
}
} // namespace

std::vector<float> encode_state(const DecisionContext& context) {
    feature_require(context.player_id >= 0 && context.player_id < 4, "feature_player_invalid");
    const int level = normalize_rank_index(context.level);
    feature_require(level >= 0, "feature_level_invalid");
    std::vector<float> result(128, 0.f);
    feature_faces(result, 0, context.hand);
    for (const auto& event : context.public_events) {
        if (event.stage == "play") feature_faces(result, 54, event.move.action);
    }
    for (int face = 0; face < 54; ++face) {
        feature_require(result[static_cast<std::size_t>(face)] <= 1.f &&
                        result[static_cast<std::size_t>(54 + face)] <= 1.f, "feature_face_count_invalid");
    }
    for (int relative = 0; relative < 4; ++relative) {
        const int count = context.remaining_counts[static_cast<std::size_t>((context.player_id + relative) % 4)];
        feature_require(count >= 0 && count <= 27, "feature_remaining_count_invalid");
        result[static_cast<std::size_t>(108 + relative)] = static_cast<float>(count) / 27.f;
    }
    result[static_cast<std::size_t>(112 + level)] = 1.f;
    result[125] = context.leading ? 1.f : 0.f;
    result[126] = static_cast<float>(context.tribute) / 2.f;
    result[127] = context.resist ? 1.f : 0.f;
    return result;
}

std::vector<int> encode_history(const DecisionContext& context) {
    feature_require(context.player_id >= 0 && context.player_id < 4, "history_player_invalid");
    std::vector<int> raw;
    for (const auto& event : context.public_events) {
        if (event.stage != "play") continue;
        feature_require(event.has_player && event.player >= 0 && event.player < 4, "history_owner_unknown");
        raw.push_back(3 + (event.player - context.player_id + 4) % 4);
        if (event.move.action.empty()) {
            raw.push_back(7);
        } else {
            for (int face : sorted_feature_faces(event.move.action)) raw.push_back(8 + face);
            raw.push_back(62);
            if (event.move.claim.empty()) raw.push_back(117);
            else for (int face : sorted_feature_faces(event.move.claim)) raw.push_back(63 + face);
        }
        raw.push_back(2);
    }
    std::vector<int> result{1};
    const std::size_t begin = raw.size() > 254 ? raw.size() - 254 : 0;
    result.insert(result.end(), raw.begin() + static_cast<std::ptrdiff_t>(begin), raw.end());
    result.push_back(118);
    return result;
}

std::vector<float> encode_action(const Move& move, const DecisionContext& context) {
    std::vector<float> result(128, 0.f);
    feature_faces(result, 0, move.action);
    feature_faces(result, 54, move.claim);
    const auto type = classify(move.claim);
    result[108 + static_cast<std::size_t>(type.kind)] = 1.f;
    result[120] = static_cast<float>(move.action.size()) / 10.f;
    result[121] = static_cast<float>(std::max(0, type.key)) / 14.f;
    result[122] = static_cast<float>(std::max(0, type.secondary)) / 14.f;
    const int level = normalize_rank_index(context.level);
    feature_require(level >= 0, "action_level_invalid");
    for (int card : move.action) {
        if (card % 54 == level * 4) result[123] += .5f;
        if (rank_index(card) == level) result[124] += .125f;
    }
    result[125] = (type.kind == PokerKind::Bomb || type.kind == PokerKind::StraightFlush || type.kind == PokerKind::Rocket) ? 1.f : 0.f;
    result[126] = move.pass() ? 1.f : 0.f;
    result[127] = !context.hand.empty() && move.action.size() == context.hand.size() ? 1.f : 0.f;
    return result;
}
} // namespace oxbot
