#include "oxbot/rules.hpp"
#include "oxbot/policy.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace oxbot;

namespace {

int card(int rank, int suit = 0, int deck = 0) {
    return rank * 4 + suit + deck * kCardsPerDeck;
}

Move as_move(const std::vector<int>& cards) { return Move{cards, cards}; }

void expect_kind(const std::vector<int>& cards, PokerKind kind, int key = -1) {
    const PokerType result = classify(cards);
    assert(result.kind == kind);
    if (key >= 0) assert(result.key == key);
}

std::string semantic_key(const Move& move) {
    std::vector<int> faces;
    for (const int id : move.action) faces.push_back(face_id(id));
    std::sort(faces.begin(), faces.end());
    const PokerType type = classify(move.claim);
    std::ostringstream out;
    for (const int face : faces) out << face << ',';
    out << ':' << static_cast<int>(type.kind) << ':' << type.length << ':'
        << type.key << ':' << type.secondary;
    return out.str();
}

std::set<std::string> generated_semantics(const std::vector<int>& hand,
                                        const std::string& level) {
    std::set<std::string> result;
    for (const Move& move : legal_moves(hand, level, std::nullopt, true)) {
        assert(validate_move(move, hand, level, std::nullopt, true, nullptr));
        result.insert(semantic_key(move));
    }
    return result;
}

// For a deliberately small hand, enumerate every physical subset and every
// wildcard claim face.  Compare strategic action semantics, allowing the two
// physical copies and unimportant wildcard suits to be canonicalized.
std::set<std::string> brute_semantics(const std::vector<int>& hand,
                                    const std::string& level) {
    assert(hand.size() <= 12U);
    std::set<std::string> result;
    const unsigned int limit = 1U << static_cast<unsigned int>(hand.size());
    for (unsigned int mask = 1U; mask < limit; ++mask) {
        Move move;
        for (std::size_t i = 0; i < hand.size(); ++i) {
            if ((mask & (1U << static_cast<unsigned int>(i))) != 0U) {
                move.action.push_back(hand[i]);
                move.claim.push_back(hand[i]);
            }
        }
        if (move.action.size() > 10U) continue;
        std::function<void(std::size_t)> claims = [&](std::size_t index) {
            if (index == move.action.size()) {
                if (validate_move(move, hand, level, std::nullopt, true, nullptr)) {
                    result.insert(semantic_key(move));
                }
                return;
            }
            if (!is_level_card(move.action[index], level)) {
                claims(index + 1U);
                return;
            }
            for (int face = 0; face < 52; ++face) {
                move.claim[index] = face;
                claims(index + 1U);
            }
        };
        claims(0U);
    }
    return result;
}

void test_classification() {
    expect_kind({}, PokerKind::Pass);
    expect_kind({-1}, PokerKind::Invalid);
    expect_kind({108}, PokerKind::Invalid);
    expect_kind({0}, PokerKind::Single, 0);
    expect_kind({52}, PokerKind::Single, 13);
    expect_kind({53, 107}, PokerKind::Pair, 14);
    expect_kind({52, 53}, PokerKind::Invalid);
    expect_kind({0, 1, 2}, PokerKind::Three, 0);
    expect_kind({0, 1, 2, 3}, PokerKind::Bomb, 0);
    expect_kind({52, 106, 53, 107}, PokerKind::Rocket);
    expect_kind({0, 1, 2, 52, 106}, PokerKind::Set, 0); // AAA + pair of small jokers
    assert(classify({0, 1, 2, 52, 106}).secondary == 13);
    expect_kind({0, 5, 8, 12, 16}, PokerKind::Straight, 0);
    expect_kind({36, 41, 44, 48, 0}, PokerKind::Straight, 9);
    expect_kind({0, 4, 8, 12, 16}, PokerKind::StraightFlush, 0);
    expect_kind({40, 44, 48, 0, 4}, PokerKind::Invalid); // JQKA2
    expect_kind({36, 40, 44, 48, 52}, PokerKind::Invalid); // joker in a would-be straight
    expect_kind({0, 1, 4, 5, 8, 9}, PokerKind::TriplePairs, 0);
    expect_kind({0, 1, 44, 45, 48, 49}, PokerKind::TriplePairs, 11);
    expect_kind({48, 49, 52, 106, 53, 107}, PokerKind::Invalid);
    expect_kind({0, 1, 2, 4, 5, 6}, PokerKind::ThreeStraight, 0);
    expect_kind({0, 1, 2, 48, 49, 50}, PokerKind::ThreeStraight, 12);
    expect_kind({48, 49, 50, 52, 106, 52}, PokerKind::Invalid);

    std::mt19937 rng(17U);
    std::vector<std::vector<int>> examples{
        {0, 5, 8, 12, 16}, {36, 41, 44, 48, 0}, {0, 1, 2, 52, 106},
        {0, 1, 2, 4, 5, 6}, {0, 1, 44, 45, 48, 49}, {0, 1, 2, 3}};
    for (auto example : examples) {
        const PokerType expected = classify(example);
        for (int repeat = 0; repeat < 20; ++repeat) {
            std::shuffle(example.begin(), example.end(), rng);
            const PokerType actual = classify(example);
            assert(actual.kind == expected.kind && actual.key == expected.key &&
                   actual.secondary == expected.secondary);
        }
    }
}

void test_comparison() {
    for (int level = 0; level < 13; ++level) {
        const std::string label = natural_ranks()[static_cast<std::size_t>(level)];
        const Move level_move = as_move({card(level, 1)});
        for (int rank = 0; rank < 13; ++rank) {
            if (rank == level) continue;
            assert(beats(as_move({card(rank, 1)}), level_move, label));
            assert(!beats(level_move, as_move({card(rank, 1)}), label));
        }
        assert(beats(level_move, as_move({52}), label));
        assert(beats(as_move({52}), as_move({53}), label));
        assert(!beats(as_move({53}), level_move, label));
    }
    assert(beats(as_move({48}), as_move({0}), "2")); // natural A > K
    assert(!beats(as_move({4}), as_move({8}), "2")); // level 2 > 3
    assert(beats(as_move({8}), as_move({4}), "2"));

    const Move five_bomb = as_move({8, 9, 10, 11, 62});
    const Move six_bomb = as_move({8, 9, 10, 11, 62, 63});
    const Move straight_flush = as_move({0, 4, 8, 12, 16});
    assert(beats(as_move({53}), five_bomb, "2"));
    assert(beats(as_move({53}), straight_flush, "2"));
    assert(beats(five_bomb, straight_flush, "2"));
    assert(!beats(six_bomb, straight_flush, "2"));
    assert(beats(straight_flush, six_bomb, "2"));
    assert(beats(six_bomb, as_move({52, 106, 53, 107}), "2"));

    // Triple rank, not the first input element, decides three-with-two.
    const Move weak_set = as_move({48, 49, 8, 9, 10}); // 333KK
    const Move strong_set = as_move({0, 1, 12, 13, 14}); // 444AA
    assert(beats(weak_set, strong_set, "2"));
    assert(!beats(strong_set, weak_set, "2"));
    // Sequence comparison remains natural even when its low rank is level.
    assert(beats(as_move({20, 4, 8, 12, 16}), as_move({24, 8, 12, 16, 20}), "2"));
}

void test_claim_validation() {
    assert(is_legal_claim({0}, {54}, "2", nullptr));
    assert(is_legal_claim({0, 54}, {0, 0}, "2", nullptr));
    assert(is_legal_claim({4}, {48}, "2", nullptr));
    assert(is_legal_claim({4, 58}, {0, 1}, "2", nullptr));
    assert(!is_legal_claim({4}, {52}, "2", nullptr));
    assert(!is_legal_claim({0}, {1}, "2", nullptr));
    assert(!is_legal_claim({4}, {-1}, "2", nullptr));
    assert(!is_legal_claim({}, {0}, "2", nullptr));
    const std::vector<int> hand{0, 4, 54, 58};
    assert(validate_move(Move{{0}, {54}}, hand, "2", std::nullopt, true, nullptr));
    assert(!validate_move(Move{{0, 0}, {0, 0}}, hand, "2", std::nullopt, true, nullptr));
    assert(!validate_move(Move{{0}, {0}}, {0, 0}, "2", std::nullopt, true, nullptr));
    assert(!validate_move(Move{{1}, {1}}, hand, "2", std::nullopt, true, nullptr));
    assert(!validate_move(Move{{}, {}}, hand, "2", std::nullopt, true, nullptr));
    assert(!validate_move(Move{{}, {0}}, hand, "2", as_move({1}), false, nullptr));
    assert(validate_move(Move{{}, {}}, hand, "2", as_move({1}), false, nullptr));
}

void test_exchange() {
    const std::vector<int> hand{0, 4, 5, 8, 28, 32, 36, 40, 52, 53};
    assert(valid_tribute_card(hand, 53, "2", nullptr));
    assert(!valid_tribute_card(hand, 52, "2", nullptr));
    assert(valid_tribute_card({0, 4, 5}, 4, "2", nullptr)); // attached compat permits covering here
    assert(valid_tribute_card({0, 4, 5}, 5, "2", nullptr));
    assert(!valid_tribute_card({0, 4}, 4, "2", nullptr));
    assert(valid_tribute_card({0, 4}, 0, "2", nullptr));

    assert(!valid_return_card(hand, 0, "2", nullptr)); // A is not natural 2..10
    assert(!valid_return_card(hand, 4, "2", nullptr));
    assert(valid_return_card(hand, 8, "2", nullptr));
    assert(valid_return_card(hand, 36, "2", nullptr));
    assert(!valid_return_card(hand, 40, "2", nullptr));
    assert(!valid_return_card(hand, 0, "10", nullptr));
    // Official BotZone judge: only natural 2..9 are eligible, excluding the
    // current level; level 9 tightens the upper bound to 8.
    assert(!valid_return_card(hand, 0, "10", RuleVariant::BotzoneCompat, nullptr));
    assert(!valid_return_card(hand, 40, "10", RuleVariant::BotzoneCompat, nullptr));
    assert(valid_return_card(hand, 32, "10", RuleVariant::BotzoneCompat, nullptr));
    assert(!valid_return_card(hand, 36, "10", RuleVariant::BotzoneCompat, nullptr));
    assert(!valid_return_card(hand, 36, "9", RuleVariant::BotzoneCompat, nullptr));
    assert(!valid_return_card(hand, 32, "9", RuleVariant::BotzoneCompat, nullptr));
    assert(valid_return_card(hand, 28, "9", RuleVariant::BotzoneCompat, nullptr));
    assert(!valid_return_card(hand, 40, "10", RuleVariant::Standard, nullptr));

    DecisionContext return_context;
    return_context.stage = DecisionContext::Stage::Return;
    return_context.level = "9";
    return_context.hand = {36, 28, 40}; // 10, 8, J
    const Move official_return = RulePolicy().choose(return_context);
    assert(official_return.action == std::vector<int>({28}));

    // Official BotZone's isValidReturn uses the reordered pointorder rather
    // than the attached corrected profile.  That means 2..9 are eligible in
    // general, level 9 tightens the upper bound to 8, and the current level
    // is rejected because it is moved above the bound.  Keep every boundary
    // explicit so a future refactor cannot silently re-enable 10 or the level
    // card in an online response.
    const std::vector<int> boundary_hand{card(1), card(7), card(8), card(9),
                                         card(10), card(11)};
    assert(valid_return_card(boundary_hand, card(7), "9",
                             RuleVariant::BotzoneCompat, nullptr)); // 8
    assert(!valid_return_card(boundary_hand, card(8), "9",
                              RuleVariant::BotzoneCompat, nullptr)); // 9
    assert(!valid_return_card(boundary_hand, card(9), "9",
                              RuleVariant::BotzoneCompat, nullptr)); // 10

    assert(valid_return_card(boundary_hand, card(8), "0",
                             RuleVariant::BotzoneCompat, nullptr)); // 9
    assert(!valid_return_card(boundary_hand, card(9), "0",
                              RuleVariant::BotzoneCompat, nullptr)); // 10/current
    assert(!valid_return_card(boundary_hand, card(10), "0",
                              RuleVariant::BotzoneCompat, nullptr)); // J

    // A non-9/10 level keeps the ordinary 2..9 boundary: 9 is allowed,
    // 10 is not, and the current level is not allowed even when it is below 9.
    assert(valid_return_card(boundary_hand, card(8), "2",
                             RuleVariant::BotzoneCompat, nullptr)); // 9
    assert(!valid_return_card(boundary_hand, card(9), "2",
                              RuleVariant::BotzoneCompat, nullptr)); // 10
    assert(!valid_return_card(boundary_hand, card(1), "2",
                              RuleVariant::BotzoneCompat, nullptr)); // current 2

    // For a high current level, the same numeric 2..9 cap still applies;
    // 10 remains outside it and the current level itself is excluded.
    assert(valid_return_card(boundary_hand, card(8), "J",
                             RuleVariant::BotzoneCompat, nullptr)); // 9
    assert(!valid_return_card(boundary_hand, card(9), "J",
                              RuleVariant::BotzoneCompat, nullptr)); // 10
    assert(!valid_return_card(boundary_hand, card(10), "J",
                              RuleVariant::BotzoneCompat, nullptr)); // current J
}

void test_generation() {
    const std::vector<std::vector<int>> hands{
        {0, 1, 2, 4, 5, 8, 9, 52, 106}, // no wild when level is 7
        {4},                             // all 13 wildcard single ranks
        {4, 58},                         // all 13 wildcard pair ranks
        {4, 9, 13, 17, 21},              // wildcard flush and nonflush alternatives
        {4, 58, 0, 1, 52, 106},          // two wildcards + a joker pair
        {4, 0, 1, 2, 5, 6},
        {0, 1, 2, 52, 106},              // three-with-joker-pair generator
    };
    for (std::size_t i = 0; i < hands.size(); ++i) {
        const std::string level = i == 0U ? "7" : "2";
        const auto generated = generated_semantics(hands[i], level);
        const auto brute = brute_semantics(hands[i], level);
        if (generated != brute) {
            std::cerr << "generation mismatch case " << i << " generated=" << generated.size()
                      << " expected=" << brute.size() << '\n';
            for (const auto& item : brute) {
                if (generated.count(item) == 0U) std::cerr << "missing " << item << '\n';
            }
        }
        assert(generated == brute);
    }

    // Every generated action must remain legal while following a last move.
    const std::vector<int> full_hand{0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                                   16, 17, 20, 24, 28, 32, 36, 40, 44, 48, 52,
                                   53, 58, 106};
    const Move previous = as_move({8, 9});
    const auto follow = legal_moves(full_hand, "2", previous, false);
    assert(!follow.empty());
    for (const Move& move : follow) {
        assert(validate_move(move, full_hand, "2", previous, false, nullptr));
    }

    // Eight natural copies plus two wildcard cards must not trigger a
    // factorial enumeration of the identical rank slots.
    const std::vector<int> large_bomb{0, 1, 2, 3, 54, 55, 56, 57, 4, 58};
    const auto begin = std::chrono::steady_clock::now();
    const auto moves = legal_moves(large_bomb, "2", std::nullopt, true);
    const auto elapsed = std::chrono::steady_clock::now() - begin;
    assert(std::any_of(moves.begin(), moves.end(), [](const Move& move) {
        const PokerType type = classify(move.claim);
        return type.kind == PokerKind::Bomb && type.length == 10;
    }));
    assert(elapsed < std::chrono::seconds(2));
}

}  // namespace

int main() {
    test_classification();
    test_comparison();
    test_claim_validation();
    test_exchange();
    test_generation();
    std::cout << "rules_test ok\n";
    return 0;
}
