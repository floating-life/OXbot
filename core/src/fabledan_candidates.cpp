#include "oxbot/fabledan_candidates.hpp"
#include "oxbot/fabledan_features.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

namespace oxbot {
namespace {

using StructureKey = std::vector<int>;

int structure_rank(int card) {
    const int face = face_id(card);
    return face >= 52 ? face - 39 : face / 4;
}

StructureKey structure_semantic(const Move& move, const DecisionContext& context) {
    const auto row = encode_fabledan_action(move, context, 80);
    const auto begin = row.begin() + 38;
    const int kind = static_cast<int>(std::max_element(begin, begin + 11) - begin);
    StructureKey result{kind, static_cast<int>(std::lround(row[66] * 15.f)),
                        static_cast<int>(move.action.size())};
    std::vector<int> ranks;
    for (const int c : move.claim) ranks.push_back(structure_rank(c));
    std::sort(ranks.begin(), ranks.end());
    result.insert(result.end(), ranks.begin(), ranks.end());
    return result;
}

struct StructureHand {
    std::array<std::vector<int>, 15> natural;
    std::vector<int> wild;
};

StructureHand structure_hand(const DecisionContext& context, int rotation) {
    auto ordered = context.hand;
    const auto priority = [&](int c) {
        const int suit = suit_index(c);
        return std::make_tuple(suit < 0 ? 4 : (suit - rotation + 4) % 4,
                               structure_rank(c), face_id(c), c);
    };
    std::sort(ordered.begin(), ordered.end(), [&](int a, int b) { return priority(a) < priority(b); });
    StructureHand result;
    for (int c : ordered) {
        if (is_level_card(c, context.level)) result.wild.push_back(c);
        else result.natural[static_cast<std::size_t>(structure_rank(c))].push_back(c);
    }
    return result;
}

std::optional<StructureKey> structure_realization(const StructureKey& semantic,
                                                const StructureHand& hand, int flush_suit) {
    const int type = semantic[0];
    if (type == 0) return StructureKey{};
    std::vector<int> ranks;
    if (type == 5 || type == 9 || type == 6 || type == 7) {
        const int length = type == 6 ? 3 : (type == 7 ? 2 : 5);
        const int mult = type == 6 ? 2 : (type == 7 ? 3 : 1);
        for (int value = semantic[1]; value < semantic[1] + length; ++value) {
            for (int n = 0; n < mult; ++n) ranks.push_back((value - 1) % 13);
        }
    } else if (type == 4) {
        std::array<int, 15> counts{};
        for (auto it = semantic.begin() + 3; it != semantic.end(); ++it) {
            ++counts[static_cast<std::size_t>(*it)];
        }
        for (int amount : {3, 2}) {
            for (int rank = 0; rank < 15; ++rank) {
                if (counts[static_cast<std::size_t>(rank)] == amount) {
                    for (int j = 0; j < amount; ++j) ranks.push_back(rank);
                }
            }
        }
    } else {
        ranks.assign(semantic.begin() + 3, semantic.end());
    }
    std::array<std::size_t, 15> used{};
    std::size_t wild_used = 0;
    std::vector<int> picked;
    for (const int rank : ranks) {
        const auto& available = hand.natural[static_cast<std::size_t>(rank)];
        auto& position = used[static_cast<std::size_t>(rank)];
        if (type == 9) {
            while (position < available.size() && suit_index(available[position]) != flush_suit) ++position;
        }
        if (position < available.size()) picked.push_back(available[position++]);
        else if (rank < 13 && wild_used < hand.wild.size()) picked.push_back(hand.wild[wild_used++]);
        else return std::nullopt;
    }
    if (type == 5) {
        bool same_suit = true;
        const int suit = suit_index(picked.front());
        for (std::size_t j = 0; j < picked.size(); ++j) {
            if (structure_rank(picked[j]) != ranks[j] || suit_index(picked[j]) != suit) same_suit = false;
        }
        if (same_suit) {
            bool fixed = false;
            for (std::size_t j = 0; j < ranks.size() && !fixed; ++j) {
                for (int alternative : hand.natural[static_cast<std::size_t>(ranks[j])]) {
                    if (suit_index(alternative) != suit_index(picked[j])) {
                        picked[j] = alternative;
                        fixed = true;
                        break;
                    }
                }
            }
            if (!fixed) return std::nullopt;
        }
    }
    StructureKey faces;
    for (int c : picked) faces.push_back(face_id(c));
    std::sort(faces.begin(), faces.end());
    return faces;
}

}  // namespace

std::vector<Move> fabledan_structure_candidates(const DecisionContext& context) {
    const auto legal = legal_moves(context.hand, context.level, context.previous, context.leading);
    std::array<StructureHand, 4> hands;
    for (int rotation = 0; rotation < 4; ++rotation) {
        hands[static_cast<std::size_t>(rotation)] = structure_hand(context, rotation);
    }
    std::map<StructureKey, std::set<StructureKey>> wanted;
    std::map<StructureKey, std::map<StructureKey, Move>> groups;
    for (const auto& move : legal) {
        if (!fabledan_generator_candidate(move, context)) continue;
        const auto semantic = structure_semantic(move, context);
        auto found = wanted.find(semantic);
        if (found == wanted.end()) {
            std::set<StructureKey> representatives;
            for (const auto& hand : hands) {
                for (int suit = 0; suit < (semantic[0] == 9 ? 4 : 1); ++suit) {
                    const auto faces = structure_realization(semantic, hand, suit);
                    if (faces) representatives.insert(*faces);
                }
            }
            found = wanted.emplace(semantic, std::move(representatives)).first;
        }
        StructureKey faces;
        for (int c : move.action) faces.push_back(face_id(c));
        std::sort(faces.begin(), faces.end());
        if (found->second.count(faces)) groups[semantic].emplace(std::move(faces), move);
    }
    constexpr std::size_t candidate_budget = 256;
    const std::size_t budget = std::max(candidate_budget, groups.size());
    std::map<StructureKey, std::map<StructureKey, Move>> selected;
    std::size_t count = 0;
    for (std::size_t depth = 0; count < budget; ++depth) {
        bool added = false;
        for (const auto& group : groups) {
            if (depth >= group.second.size()) continue;
            auto variant = group.second.begin();
            std::advance(variant, static_cast<std::ptrdiff_t>(depth));
            selected[group.first].insert(*variant);
            added = true;
            if (++count == budget) break;
        }
        if (!added) break;
    }
    std::vector<Move> result;
    result.reserve(count);
    for (const auto& group : selected) {
        for (const auto& variant : group.second) result.push_back(variant.second);
    }
    return result;
}

}  // namespace oxbot
