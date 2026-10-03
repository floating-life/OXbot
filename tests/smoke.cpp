#include "oxbot/mini_json.hpp"
#include "oxbot/policy.hpp"
#include "oxbot/protocol.hpp"
#include "oxbot/rules.hpp"

#include <cassert>
#include <iostream>

using namespace oxbot;

static void expect_kind(const std::vector<int>& cards, PokerKind kind) {
    const PokerType type = classify(cards);
    assert(type.kind == kind);
    assert(type.valid());
}

int main() {
    // Card boundary and rank encoding checks.
    assert(valid_card_id(0) && valid_card_id(107));
    assert(!valid_card_id(-1) && !valid_card_id(108));
    assert(card_name(0) == "hA");
    assert(card_name(52) == "jo");
    assert(card_name(53) == "jO");
    assert(card_name(106) == "jo");
    assert(card_name(107) == "jO");
    assert(normalize_level("10") == "0");

    expect_kind({0}, PokerKind::Single);
    expect_kind({0, 1}, PokerKind::Pair);
    expect_kind({0, 1, 2}, PokerKind::Three);
    expect_kind({0, 5, 8, 12, 16}, PokerKind::Straight); // A2345 (mixed suits)
    expect_kind({36, 41, 44, 48, 0}, PokerKind::Straight); // 10JQKA
    expect_kind({0, 1, 2, 3}, PokerKind::Bomb);
    expect_kind({52, 106, 53, 107}, PokerKind::Rocket);
    expect_kind({0, 1, 2, 4, 5}, PokerKind::Set); // AAA22
    expect_kind({0, 1, 4, 5, 8, 9}, PokerKind::TriplePairs); // AA2233

    Move previous{{4}, {4}};
    Move current{{8}, {8}};
    assert(!beats(previous, current, "2")); // A 2 is the highest natural point at level 2.
    assert(beats(current, previous, "2"));
    assert(beats(previous, current, "A"));

    std::vector<int> hand{0, 1, 2, 3, 4, 5};
    Move lead{{0}, {0}};
    assert(validate_move(lead, hand, "2", std::nullopt, true, nullptr));
    Move bad_pass{{}, {}};
    assert(!validate_move(bad_pass, hand, "2", std::nullopt, true, nullptr));
    assert(validate_move(bad_pass, hand, "2", lead, false, nullptr));

    json::Value input = json::Value::make_object();
    input["requests"] = json::Value::make_array();
    json::Value request = json::Value::make_object();
    request["stage"] = json::Value("play");
    request["your_id"] = json::Value(0);
    request["global"] = json::Value::make_object();
    request["global"]["level"] = json::Value("2");
    request["history"] = json::Value::make_array();
    request["history"].array.push_back(json::Value::make_array());
    input["requests"].array.push_back(request);
    input["responses"] = json::Value::make_array();
    const json::Value output = BotAdapter().decide(input);
    assert(output.contains("response"));
    assert(output.contains("debug"));
    std::cout << "smoke ok\n";
    return 0;
}
