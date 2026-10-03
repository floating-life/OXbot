// Offline JSONL diagnostic process. This is never part of the BotZone upload.
#include "oxbot/protocol.hpp"
#include "oxbot/features.hpp"
#include "oxbot/fabledan_features.hpp"
#include "oxbot/fabledan_candidates.hpp"
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>

using namespace oxbot;
using json::Value;

static std::vector<int> read_cards(const Value& value) {
    if (!value.is_array()) throw std::runtime_error("expected card array");
    std::vector<int> out;
    for (const auto& v : value.array) out.push_back(v.as_int(-1000));
    return out;
}
static Value cards_json(const std::vector<int>& cards) {
    Value out = Value::make_array();
    for (int c : cards) out.push_back(Value(c));
    return out;
}
static Move read_move(const Value& value) {
    if (!value.is_array()) throw std::runtime_error("expected move array");
    if (value.array.empty()) return {};
    if (value.array.size() != 2) throw std::runtime_error("invalid move shape");
    return {read_cards(value.array[0]), read_cards(value.array[1])};
}
static Value move_json(const Move& move) {
    Value out = Value::make_array();
    out.push_back(cards_json(move.action));
    out.push_back(cards_json(move.claim));
    return out;
}
static std::string kind_name(PokerKind k) {
    switch (k) {
        case PokerKind::Pass: return "pass";
        case PokerKind::Invalid: return "invalid";
        case PokerKind::Single: return "single";
        case PokerKind::Pair: return "pair";
        case PokerKind::Three: return "three";
        case PokerKind::Straight: return "straight";
        case PokerKind::Set: return "set";
        case PokerKind::ThreeStraight: return "three_straight";
        case PokerKind::TriplePairs: return "triple_pairs";
        case PokerKind::Bomb: return "bomb";
        case PokerKind::StraightFlush: return "straight_flush";
        case PokerKind::Rocket: return "rocket";
    }
    return "invalid";
}
static Value float_json(const std::vector<float>& values) {
    Value out = Value::make_array();
    for (float value : values) out.push_back(Value(static_cast<double>(value)));
    return out;
}
static DecisionContext observation_context(const Value& observation) {
    DecisionContext context;
    context.player_id = observation["player"].as_int(-1);
    context.hand = read_cards(observation["hand"]);
    context.level = observation["level"].as_string();
    context.leading = observation["leading"].as_bool();
    context.tribute = observation["tribute"].as_int(0);
    context.resist = observation["resist"].as_bool();
    const auto counts = read_cards(observation["remaining_counts"]);
    if (counts.size() != 4) throw std::runtime_error("remaining_counts_shape_invalid");
    for (int i = 0; i < 4; ++i) context.remaining_counts[static_cast<std::size_t>(i)] = counts[static_cast<std::size_t>(i)];
    for (const auto& item : observation["history"].array) {
        PublicEvent event;
        event.player = item["player"].as_int(-1);
        event.has_player = event.player >= 0 && event.player < 4;
        event.stage = item["stage"].as_string("play");
        event.move.action = read_cards(item["action"]);
        if (!item["claim"].is_null()) event.move.claim = read_cards(item["claim"]);
        context.public_events.push_back(std::move(event));
    }
    if (observation.contains("done")) {
        for (const auto& item : observation["done"].array) context.done.push_back(item.as_int(-1));
    }
    for (auto it = context.public_events.rbegin(); it != context.public_events.rend(); ++it) {
        if (it->stage == "play" && !it->move.action.empty()) {
            context.previous = it->move;
            break;
        }
    }
    return context;
}
static Value handle(const Value& in) {
    const std::string command = in["command"].as_string();
    if (command == "bot") {
        // Offline reuse is an acceleration only; botzone/main.cpp loads once
        // per process, as required by the ordinary JSON deployment contract.
        static std::map<std::string, std::unique_ptr<BotAdapter>> adapters;
        const std::string path = in["model"].as_string();
        const std::string strategy = in["strategy"].as_string();
        const std::string candidate_version = in["candidate_version"].as_string();
        const std::string cache_key = path + "\x1f" + strategy + "\x1f" + candidate_version;
        auto& adapter = adapters[cache_key];
        if (!adapter) adapter = std::make_unique<BotAdapter>(path, strategy, candidate_version);
        return adapter->decide(in["input"]);
    }
    Value out = Value::make_object();
    if (command == "features") {
        DecisionContext context;
        if (in.contains("input")) {
            StateMirror mirror;
            if (!mirror.rebuild(in["input"])) throw std::runtime_error(mirror.error);
            context = mirror.context;
        } else context = observation_context(in["observation"]);
        out["state"] = float_json(encode_state(context));
        out["tokens"] = cards_json(encode_history(context));
        out["actions"] = Value::make_array();
        out["types"] = Value::make_array();
        for (const auto& item : in["moves"].array) {
            const auto move = read_move(item);
            out["actions"].push_back(float_json(encode_action(move, context)));
            const auto type = classify(move.claim);
            Value meta = Value::make_object();
            meta["kind"] = Value(kind_name(type.kind));
            meta["key"] = Value(type.key);
            meta["secondary"] = Value(type.secondary);
            out["types"].push_back(std::move(meta));
        }
        return out;
    }
    if (command == "fabledan_features") {
        const DecisionContext context = observation_context(in["observation"]);
        out["tokens"] = cards_json(encode_fabledan_history(context));
        out["actions"] = Value::make_array();
        for (const auto& item : in["moves"].array) {
            out["actions"].push_back(float_json(encode_fabledan_action(
                read_move(item), context, in["feature_dim"].as_int(80))));
        }
        return out;
    }
    if (command == "fabledan_candidates") {
        const DecisionContext context = observation_context(in["observation"]);
        const int feature_dim = in["feature_dim"].as_int(80);
        const auto generated = feature_dim == 224 ? fabledan_structure_candidates(context) :
            legal_moves(context.hand, context.level, context.previous, context.leading);
        out["moves"] = Value::make_array();
        out["actions"] = Value::make_array();
        for (const auto& move : generated) {
            if (!fabledan_generator_candidate(move, context)) continue;
            out["moves"].push_back(move_json(move));
            out["actions"].push_back(float_json(encode_fabledan_action(move, context, feature_dim)));
        }
        return out;
    }
    if (command == "classify") {
        const auto t = classify(read_cards(in["claim"]));
        out["kind"] = Value(kind_name(t.kind));
        out["key"] = Value(t.key);
        out["secondary"] = Value(t.secondary);
        return out;
    }
    const std::string level = in["level"].as_string("2");
    if (command == "beats") {
        out["ok"] = Value(beats(read_move(in["previous"]), read_move(in["move"]), level));
        return out;
    }
    if (command == "state") {
        StateMirror state;
        out["ok"] = Value(state.rebuild(in["input"]));
        out["hand"] = cards_json(state.hand);
        out["leading"] = Value(state.context.leading);
        // Expose the exact replayed comparison target to offline diagnostics.
        // Keeping this on the probe (rather than reconstructing it in a
        // Python harness) prevents candidate-count and feature traces from
        // silently treating every follow play as a lead.
        if (state.context.previous.has_value()) {
            out["previous"] = move_json(*state.context.previous);
        } else {
            out["previous"] = Value(nullptr);
        }
        out["player"] = Value(state.player);
        out["remaining_counts"] = cards_json(std::vector<int>(state.remaining_counts.begin(), state.remaining_counts.end()));
        out["public_event_count"] = Value(static_cast<int>(state.public_events.size()));
        out["error"] = Value(state.error);
        return out;
    }
    const auto hand = read_cards(in["hand"]);
    std::optional<Move> previous;
    if (in.contains("previous") && !in["previous"].is_null()) previous = read_move(in["previous"]);
    const bool leading = in["leading"].as_bool(!previous.has_value());
    if (command == "generate") {
        out["moves"] = Value::make_array();
        if (in["metadata"].as_bool()) out["types"] = Value::make_array();
        for (const auto& move : legal_moves(hand, level, previous, leading)) {
            out["moves"].push_back(move_json(move));
            if (in["metadata"].as_bool()) {
                const auto type = classify(move.claim);
                Value meta = Value::make_object();
                meta["kind"] = Value(kind_name(type.kind));
                meta["key"] = Value(type.key);
                meta["secondary"] = Value(type.secondary);
                out["types"].push_back(std::move(meta));
            }
        }
    } else if (command == "validate") {
        std::string reason;
        out["ok"] = Value(validate_move(read_move(in["move"]), hand, level, previous, leading, &reason));
        out["reason"] = Value(reason);
    } else if (command == "tribute" || command == "return") {
        out["cards"] = Value::make_array();
        for (int c : hand) {
            const bool ok = command == "tribute" ? valid_tribute_card(hand, c, level) :
                (in["variant"].as_string() == "botzone" ?
                    valid_return_card(hand, c, level, RuleVariant::BotzoneCompat, nullptr) :
                    valid_return_card(hand, c, level));
            if (ok) out["cards"].push_back(Value(c));
        }
    } else throw std::runtime_error("unknown command");
    return out;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string line;
    while (std::getline(std::cin, line)) {
        try { std::cout << json::dump(handle(json::parse(line))) << '\n' << std::flush; }
        catch (const std::exception& ex) {
            Value out = Value::make_object();
            out["error"] = Value(ex.what());
            std::cout << json::dump(out) << '\n' << std::flush;
        }
    }
}
