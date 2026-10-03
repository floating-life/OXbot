#include "oxbot/protocol.hpp"
#include "oxbot/state.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

using namespace oxbot;
using json::Value;

namespace {

std::vector<int> range(int first, int count) {
    std::vector<int> out;
    for (int i = 0; i < count; ++i) out.push_back(first + i);
    return out;
}

Value integers(const std::vector<int>& cards) {
    Value out = Value::make_array();
    for (int card : cards) out.array.emplace_back(card);
    return out;
}

Value action_response(const std::vector<int>& action, const std::vector<int>& claim) {
    Value out = Value::make_array();
    out.array.push_back(integers(action));
    out.array.push_back(integers(claim));
    return out;
}

Value action_response(const std::vector<int>& action) { return action_response(action, action); }
Value pass() { return action_response({}); }

Value event(int player, const Value& response) {
    Value out = Value::make_object();
    out["player"] = Value(player);
    out["response"] = response;
    return out;
}

Value history(const std::vector<Value>& events, bool array_empty = false) {
    assert(events.size() <= 4);
    Value out = Value::make_array();
    for (std::size_t i = events.size(); i < 4; ++i) {
        out.array.push_back(array_empty ? Value::make_array() : Value::make_object());
    }
    out.array.insert(out.array.end(), events.begin(), events.end());
    return out;
}

Value global(int tribute = 0, int first = -1, int last = -1, bool resist = false,
             const std::string& level = "2") {
    Value out = Value::make_object();
    out["level"] = Value(level);
    out["tribute"] = Value(tribute);
    out["first"] = first < 0 ? Value(nullptr) : Value(first);
    out["last"] = last < 0 ? Value(nullptr) : Value(last);
    out["resist"] = Value(resist);
    out["tribute_cards"] = Value::make_object();
    out["return_cards"] = Value::make_object();
    return out;
}

Value request(const std::string& stage, const Value& rules) {
    Value out = Value::make_object();
    out["stage"] = Value(stage);
    out["global"] = rules;
    return out;
}

Value play(const Value& rules, const std::vector<Value>& events = {},
           const std::vector<int>& done = {}, int pass_on = -1, bool array_empty = false) {
    Value out = request("play", rules);
    out["history"] = history(events, array_empty);
    out["done"] = integers(done);
    out["pass_on"] = Value(pass_on);
    return out;
}

Value start(int player, const std::vector<int>& hand, const Value& rules) {
    Value input = Value::make_object();
    input["requests"] = Value::make_array();
    input["responses"] = Value::make_array();
    Value deal = request("deal", rules);
    deal["your_id"] = Value(player);
    deal["deliver"] = integers(hand);
    input["requests"].array.push_back(deal);
    return input;
}

void complete(Value* input, const Value& response, const Value& next) {
    (*input)["responses"].array.push_back(response);
    (*input)["requests"].array.push_back(next);
}

bool has(const std::vector<int>& cards, int card) {
    return std::find(cards.begin(), cards.end(), card) != cards.end();
}

StateMirror rebuild(const Value& input) {
    StateMirror mirror;
    if (!mirror.rebuild(input)) {
        std::cerr << "unexpected state error: " << mirror.error << '\n';
        assert(false);
    }
    assert(mirror.context.hand == mirror.hand);
    assert(mirror.context.public_events.size() == mirror.public_events.size());
    assert(mirror.context.remaining_counts == mirror.remaining_counts);
    return mirror;
}

Value positional_input(Value input) {
    const int self = input["requests"].array.front()["your_id"].as_int();
    for (auto& request : input["requests"].array) {
        if (request["stage"].as_string() != "play") continue;
        Value slots = Value::make_array();
        slots.array.resize(4, Value::make_array());
        const auto& events = request["history"].array;
        std::size_t begin = 0;
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (events[i].contains("player") && events[i]["player"].as_int() == self) {
                slots.array[0] = events[i]["response"];
                begin = i + 1;
            }
        }
        for (std::size_t i = begin; i < events.size(); ++i) {
            if (!events[i].contains("player")) continue;
            const int relative = (events[i]["player"].as_int() - self + 4) % 4;
            slots.array[static_cast<std::size_t>(relative)] = events[i]["response"];
        }
        request["history"] = std::move(slots);
    }
    return input;
}

void same_positional_state(const Value& input) {
    const auto chronological = rebuild(input);
    const auto positional = rebuild(positional_input(input));
    assert(chronological.hand == positional.hand);
    assert(chronological.remaining_counts == positional.remaining_counts);
    assert(chronological.context.leading == positional.context.leading);
    assert(chronological.context.done == positional.context.done);
    assert(chronological.context.pass_on == positional.context.pass_on);
    if (!chronological.context.leading) {
        assert(chronological.context.previous.has_value() && positional.context.previous.has_value());
        assert(chronological.context.previous->action == positional.context.previous->action);
        assert(chronological.context.previous->claim == positional.context.previous->claim);
    }
}

void rejects(const Value& input, const std::string& fragment = {}) {
    StateMirror mirror;
    assert(!mirror.rebuild(input));
    assert(!mirror.error.empty());
    if (!fragment.empty()) {
        if (mirror.error.find(fragment) == std::string::npos) std::cerr << mirror.error << '\n';
        assert(mirror.error.find(fragment) != std::string::npos);
    }
    const Value output = BotAdapter().decide(input);
    assert(output["response"].is_null());
    assert(output["error"].as_string() == "state_rebuild_failed");
    assert(output["debug"].as_string().find("fail_closed") != std::string::npos);
}

void basic_and_leading() {
    Value rules = global(0, -1, -1, false, "10");
    rules.object.erase("resist");
    Value input = start(0, range(0, 27), rules);
    StateMirror mirror = rebuild(input);
    assert(mirror.context.stage == DecisionContext::Stage::Deal);
    assert(mirror.context.level == "0");
    assert(mirror.context.player_id == 0);
    const Value deal_output = BotAdapter().decide(input);
    assert(deal_output["debug"].as_string().find("version=oxbot-model-v1") != std::string::npos);
    assert(deal_output["debug"].as_string().find("policy=stage_rules") != std::string::npos);
    assert(deal_output["debug"].as_string().find("model_status=model_not_configured") != std::string::npos);
    rules["resist"] = Value(false);
    complete(&input, Value::make_array(), play(rules, {}, {}, -1, true));
    mirror = rebuild(input);
    assert(mirror.context.leading && !mirror.context.previous.has_value());
    assert(mirror.hand.size() == 27);
    assert(mirror.public_events.empty());
    assert((mirror.remaining_counts == std::array<int, 4>{{27, 27, 27, 27}}));
    const Value fallback_output = BotAdapter().decide(input);
    assert(fallback_output["debug"].as_string().find("policy=rule_fallback") != std::string::npos);
    assert(fallback_output["debug"].as_string().find("legality_fallback=0") != std::string::npos);

    complete(&input, action_response({0}), play(rules, {event(0, action_response({0})), event(1, pass()), event(2, pass()), event(3, pass())}));
    mirror = rebuild(input);
    assert(mirror.context.leading);
    assert(mirror.context.previous->action == std::vector<int>{0});
    assert(!has(mirror.hand, 0) && mirror.hand.size() == 26);
    assert(mirror.public_events.size() == 4);
    complete(&input, action_response({1}), play(rules, {event(0, action_response({1})), event(1, pass()), event(2, pass()), event(3, pass())}));
    mirror = rebuild(input);
    assert(mirror.context.leading);
    assert(mirror.public_events.size() == 8); // Equal passes in distinct turns survive.
    assert((mirror.remaining_counts == std::array<int, 4>{{25, 27, 27, 27}}));
    same_positional_state(input);

    Value follow = start(2, range(54, 27), global());
    complete(&follow, Value::make_array(), play(global(), {event(0, action_response({0})), event(1, action_response({4}))}));
    mirror = rebuild(follow);
    assert(!mirror.context.leading && mirror.context.previous->action == std::vector<int>{4});
    assert((mirror.remaining_counts == std::array<int, 4>{{26, 26, 27, 27}}));
    same_positional_state(follow);

    Value claims = start(0, range(0, 27), global());
    complete(&claims, Value::make_array(), play(global()));
    complete(&claims, action_response({4, 5}, {5, 5}), play(global(), {event(0, action_response({4, 5}, {5, 5})), event(1, pass()), event(2, pass()), event(3, pass())}));
    mirror = rebuild(claims);
    assert(mirror.context.previous->claim == std::vector<int>({5, 5}));
    assert(mirror.hand.size() == 25); // Claim IDs need not be physical entities.
    same_positional_state(claims);

    Value bad_positional = positional_input(claims);
    bad_positional["requests"].array.back()["history"].array[0] = action_response({6});
    rejects(bad_positional, "own_response_mismatch");
    bad_positional = positional_input(claims);
    bad_positional["requests"].array.back()["history"].array[1] = event(1, pass());
    rejects(bad_positional, "mixed_formats");
}

void single_tribute_replay() {
    Value empty = global(1, 0, 3);
    Value settled = empty;
    settled["tribute_cards"]["3"] = Value(105);
    settled["return_cards"]["0"] = Value(8);
    std::vector<int> original = range(54, 26);
    original.push_back(105);
    Value input = start(3, original, empty);
    complete(&input, Value::make_array(), request("tribute", empty));
    complete(&input, integers({105}), play(settled));
    StateMirror mirror = rebuild(input);
    assert(!has(mirror.hand, 105) && has(mirror.hand, 8));
    assert(mirror.hand.size() == 27);
    assert(mirror.public_events.size() == 2);
    assert(mirror.public_events[0].stage == "tribute" && mirror.public_events[0].target == 0);
    assert(mirror.public_events[1].stage == "return" && mirror.public_events[1].target == 3);
    // A received return is played in a later response.  Exchange must happen
    // before replaying that response, and it must never add the card twice.
    complete(&input, action_response({8}), play(settled, {event(3, action_response({8})), event(0, pass()), event(1, pass()), event(2, pass())}));
    mirror = rebuild(input);
    assert(!has(mirror.hand, 8) && !has(mirror.hand, 105));
    assert(mirror.hand.size() == 26 && mirror.remaining_counts[3] == 26);
    same_positional_state(input);

    Value before_return = empty;
    before_return["tribute_cards"]["3"] = Value(105);
    Value receiver = start(0, range(0, 27), empty);
    complete(&receiver, Value::make_array(), request("return", before_return));
    mirror = rebuild(receiver);
    assert(mirror.hand == range(0, 27)); // The incoming tribute is not a return option.
    Value bad_return = receiver;
    complete(&bad_return, integers({105}), play(settled));
    rejects(bad_return, "original_hand");
    complete(&receiver, integers({8}), play(settled));
    mirror = rebuild(receiver);
    assert(!has(mirror.hand, 8) && has(mirror.hand, 105));
}

Value double_source(int self, int offered, const Value& settled, const Value& empty) {
    std::vector<int> hand = range(54, 26);
    hand.push_back(offered);
    Value input = start(self, hand, empty);
    Value before = empty;
    if (self != empty["last"].as_int()) {
        const std::string first_source = std::to_string(empty["last"].as_int());
        before["tribute_cards"][first_source] = settled["tribute_cards"][first_source];
    }
    complete(&input, Value::make_array(), request("tribute", before));
    complete(&input, integers({offered}), play(settled));
    return input;
}

void double_tribute_and_resist() {
    Value empty = global(2, 0, 1);
    Value settled = empty;
    settled["tribute_cards"]["1"] = Value(40);
    settled["tribute_cards"]["3"] = Value(53);
    settled["return_cards"]["0"] = Value(8);
    settled["return_cards"]["2"] = Value(12);
    StateMirror mirror = rebuild(double_source(3, 53, settled, empty));
    assert(has(mirror.hand, 8) && !has(mirror.hand, 12) && !has(mirror.hand, 53));
    mirror = rebuild(double_source(1, 40, settled, empty));
    assert(has(mirror.hand, 12) && !has(mirror.hand, 8) && !has(mirror.hand, 40));

    settled["tribute_cards"]["1"] = Value(48);
    settled["tribute_cards"]["3"] = Value(49);
    mirror = rebuild(double_source(3, 49, settled, empty));
    assert(has(mirror.hand, 12) && !has(mirror.hand, 8)); // Equal points: clockwise.

    empty = global(2, 0, 1, false, "5");
    settled = empty;
    settled["tribute_cards"]["1"] = Value(17); // Current-level 5 beats A.
    settled["tribute_cards"]["3"] = Value(0);
    settled["return_cards"]["0"] = Value(8);
    settled["return_cards"]["2"] = Value(12);
    mirror = rebuild(double_source(1, 17, settled, empty));
    assert(has(mirror.hand, 8) && !has(mirror.hand, 12));

    empty = global(1, 0, 3, true);
    settled = empty;
    settled["tribute_cards"]["3"] = Value(-1);
    settled["return_cards"]["0"] = Value(-1);
    std::vector<int> hand = range(54, 25);
    hand.push_back(53);
    hand.push_back(107);
    Value resist = start(3, hand, empty);
    complete(&resist, Value::make_array(), request("tribute", empty));
    complete(&resist, Value::make_array(), play(settled));
    mirror = rebuild(resist);
    assert(mirror.hand == hand && mirror.context.resist);
    assert(mirror.public_events.size() == 2);
    assert(mirror.public_events[0].move.pass());
}

void finished_window_overlap() {
    const Value rules = global();
    Value input = start(0, range(0, 27), rules);
    complete(&input, Value::make_array(), play(rules));
    for (int turn = 0; turn < 3; ++turn) {
        const std::vector<Value> window = {event(0, action_response({turn})), event(1, action_response(range(27 + 9 * turn, 9))),
                                          event(2, action_response(range(54 + 9 * turn, 9))), event(3, pass())};
        complete(&input, action_response({turn}), play(rules, window, turn == 2 ? std::vector<int>{1, 2} : std::vector<int>{}, turn == 2 ? 2 : -1));
    }
    // Both finished seats are skipped.  Passing transfers the lead directly
    // to this player, so the next snapshot fully overlaps the previous tail.
    complete(&input, pass(), play(rules, {event(1, action_response(range(45, 9))), event(2, action_response(range(72, 9))), event(3, pass()), event(0, pass())}, {1, 2}));
    StateMirror mirror = rebuild(input);
    assert(mirror.context.leading && mirror.public_events.size() == 13);
    complete(&input, action_response({3}), play(rules, {event(3, pass()), event(0, pass()), event(0, action_response({3})), event(3, pass())}, {1, 2}));
    mirror = rebuild(input);
    assert(mirror.context.leading && mirror.public_events.size() == 15);
    complete(&input, action_response({4}), play(rules, {event(0, action_response({3})), event(3, pass()), event(0, action_response({4})), event(3, pass())}, {1, 2}));
    mirror = rebuild(input);
    assert(mirror.public_events.size() == 17);
    assert((mirror.remaining_counts == std::array<int, 4>{{22, 0, 0, 27}}));
    same_positional_state(input);

    Value jumped = start(0, range(0, 27), rules);
    complete(&jumped, Value::make_array(), play(rules));
    for (int turn = 0; turn < 3; ++turn) {
        complete(&jumped, action_response({turn}), play(rules, {event(0, action_response({turn})), event(1, pass()), event(2, pass()),
                 event(3, action_response(range(81 + 9 * turn, 9)))}, turn == 2 ? std::vector<int>{3} : std::vector<int>{}, turn == 2 ? 3 : -1));
    }
    // After self passes, players 1/2 pass, the finished 3 passes the lead to
    // partner 1, then 1 plays and 2 passes.  Four intervening actions push the
    // own anchor outside the new window; each of the four is a new event.
    complete(&jumped, pass(), play(rules, {event(1, pass()), event(2, pass()), event(1, action_response({27})), event(2, pass())}, {3}));
    mirror = rebuild(jumped);
    assert(mirror.public_events.size() == 17);
    assert((mirror.remaining_counts == std::array<int, 4>{{24, 26, 27, 0}}));
    assert(!mirror.context.leading);
    same_positional_state(jumped);
}

void invalid_input_is_explicit() {
    Value good = start(0, range(0, 27), global());
    complete(&good, Value::make_array(), play(global()));
    Value bad = good;
    bad["requests"].array[0]["your_id"] = Value(0.5);
    rejects(bad, "your_id");
    bad = good;
    bad["requests"].array[0]["deliver"].array[0] = Value(0.5);
    rejects(bad, "deliver");
    bad = good;
    bad["requests"].array[0]["deliver"].array[0] = Value(108);
    rejects(bad, "deliver");
    bad = good;
    bad["requests"].array[0]["deliver"].array[0] = Value(1);
    rejects(bad, "deliver");
    bad = good;
    bad["requests"].array.back()["stage"] = Value("unknown");
    rejects(bad, "unknown_stage");
    bad = good;
    bad["responses"].array.clear();
    rejects(bad, "one_pending");
    bad = good;
    bad["requests"].array.back()["global"]["level"] = Value(2.5);
    rejects(bad, "level");
    bad = good;
    bad["requests"].array.back()["history"].array.clear();
    rejects(bad, "four_slots");
    bad = good;
    bad["requests"].array.erase(bad["requests"].array.begin());
    bad["responses"].array.clear();
    rejects(bad, "deal_missing");
    bad = good;
    complete(&bad, action_response({107}), play(global(), {event(0, action_response({107})), event(1, pass()), event(2, pass()), event(3, pass())}));
    rejects(bad, "replayed_hand");
    bad = good;
    complete(&bad, action_response({0}), play(global(), {event(0, action_response({1})), event(1, pass()), event(2, pass()), event(3, pass())}));
    rejects(bad, "overlap");
    bad = good;
    bad["requests"].array.back()["history"] = history({event(1, action_response({0}))});
    rejects(bad, "private_hand");

    // A complete but pathological return hand may have no validated action.
    // The adapter must not disguise [] as a legal return or invent a card.
    std::vector<int> no_return = range(40, 14);
    const auto other_deck = range(94, 13);
    no_return.insert(no_return.end(), other_deck.begin(), other_deck.end());
    Value before_return = global(1, 0, 3);
    before_return["tribute_cards"]["3"] = Value(8);
    Value no_action = start(0, no_return, global(1, 0, 3));
    complete(&no_action, Value::make_array(), request("return", before_return));
    rebuild(no_action);
    const Value output = BotAdapter().decide(no_action);
    assert(output["response"].is_null());
    assert(output["error"].as_string() == "no_validated_action");
}

}  // namespace

int main() {
    basic_and_leading();
    single_tribute_replay();
    double_tribute_and_resist();
    finished_window_overlap();
    invalid_input_is_explicit();
    std::cout << "state tests ok\n";
    return 0;
}
