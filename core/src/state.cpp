#include "oxbot/state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace oxbot {
namespace {

constexpr int kPlayers = 4;
constexpr int kInitialCards = 27;

bool fail(std::string* error, const std::string& reason) {
    if (error) *error = reason;
    return false;
}

// Do not use Value::as_int for protocol input: a fractional, infinite or very
// large JSON number must not silently turn into a player/card identifier.
bool integer(const json::Value& value, int low, int high, int* out) {
    if (!value.is_number() || !std::isfinite(value.number) ||
        std::floor(value.number) != value.number || value.number < low || value.number > high) return false;
    *out = static_cast<int>(value.number);
    return true;
}

bool card_array(const json::Value& value, std::vector<int>* out,
                bool unique, const std::string& name, std::string* error) {
    if (!value.is_array()) return fail(error, name + "_not_array");
    out->clear();
    std::set<int> seen;
    for (const auto& item : value.array) {
        int card = -1;
        if (!integer(item, 0, kCardCount - 1, &card)) return fail(error, name + "_invalid_card");
        if (unique && !seen.insert(card).second) return fail(error, name + "_duplicate_card");
        out->push_back(card);
    }
    return true;
}

bool remove_card(std::vector<int>* hand, int card) {
    const auto it = std::find(hand->begin(), hand->end(), card);
    if (it == hand->end()) return false;
    hand->erase(it);
    return true;
}

bool play_response(const json::Value& value, Move* out, std::string* error) {
    *out = Move{};
    if (!value.is_array()) return fail(error, "play_response_not_array");
    // The referee accepts the old [] pass form; the adapter always emits the
    // canonical [[],[]] form.  History records are normalized for comparison.
    if (value.array.empty()) return true;
    if (value.array.size() != 2) return fail(error, "play_response_wrong_size");
    if (!card_array(value.array[0], &out->action, true, "action", error) ||
        !card_array(value.array[1], &out->claim, false, "claim", error)) return false;
    if (out->action.size() != out->claim.size()) return fail(error, "action_claim_size_mismatch");
    return true;
}

bool empty_slot(const json::Value& value) {
    return (value.is_array() && value.array.empty()) || (value.is_object() && value.object.empty());
}

bool history_snapshot(const json::Value& request, int self, std::vector<PublicEvent>* out,
                      bool* positional, std::string* error) {
    const auto* history = request.find("history");
    if (!history || !history->is_array() || history->array.size() != 4) {
        return fail(error, "history_missing_or_not_four_slots");
    }
    out->clear();
    // FableDan also accepts relative-seat snapshots: slot 0 is self, then
    // clockwise opponents/partner.  An empty slot means no new event, while
    // [[], []] is an actual pass.  Unlike chronological object snapshots,
    // these slots may have gaps and do not carry older overlapping turns.
    bool has_object = false;
    bool has_array = false;
    for (const auto& item : history->array) {
        if (empty_slot(item)) continue;
        has_object = has_object || item.is_object();
        has_array = has_array || item.is_array();
    }
    if (has_object && has_array) return fail(error, "history_mixed_formats");
    *positional = has_array || (!has_object && std::all_of(
        history->array.begin(), history->array.end(),
        [](const json::Value& item) { return item.is_array(); }));
    if (*positional) {
        for (std::size_t slot = 0; slot < history->array.size(); ++slot) {
            const auto& item = history->array[slot];
            if (empty_slot(item)) continue;
            PublicEvent event;
            event.player = (self + static_cast<int>(slot)) % kPlayers;
            event.has_player = true;
            if (!play_response(item, &event.move, error)) return false;
            out->push_back(std::move(event));
        }
        return true;
    }
    bool saw_event = false;
    for (const auto& item : history->array) {
        if (empty_slot(item)) {
            if (saw_event) return fail(error, "history_empty_slot_after_event");
            continue;
        }
        if (!item.is_object()) return fail(error, "history_entry_not_object");
        const auto* owner = item.find("player");
        const auto* response = item.find("response");
        PublicEvent event;
        if (!owner || !integer(*owner, 0, kPlayers - 1, &event.player)) return fail(error, "history_invalid_player");
        if (!response || !play_response(*response, &event.move, error)) return fail(error, "history_invalid_response");
        event.has_player = true;
        saw_event = true;
        out->push_back(std::move(event));
    }
    return true;
}

bool same_event(const PublicEvent& a, const PublicEvent& b) {
    return a.player == b.player && a.move.action == b.move.action && a.move.claim == b.move.claim;
}

bool merge_snapshot(const std::vector<PublicEvent>& snapshot, int self,
                    bool has_previous_response, bool possible_pass_on_jump, bool positional,
                    std::vector<PublicEvent>* all,
                    std::string* error) {
    if (!has_previous_response) {
        // Before this Bot's first play there can be at most three public
        // actions.  A full window or an own event means history was omitted.
        if (snapshot.size() > 3) return fail(error, "initial_play_history_incomplete");
        for (const auto& event : snapshot) {
            if (event.player == self) return fail(error, "unrecorded_own_play");
        }
        *all = snapshot;
        return true;
    }

    if (positional) {
        // Own responses have already been replayed from the envelope.  Only
        // validate slot 0 when present, then append the other seats' new
        // events.  A wind/pass-on jump can omit our old response entirely.
        std::size_t first_new = 0;
        if (!snapshot.empty() && snapshot.front().player == self) {
            if (all->empty() || !same_event(all->back(), snapshot.front())) {
                return fail(error, "history_own_response_mismatch");
            }
            first_new = 1;
        }
        all->insert(all->end(), snapshot.begin() + static_cast<std::ptrdiff_t>(first_new), snapshot.end());
        return true;
    }

    // The latest completed own response is an unambiguous turn anchor.  Find
    // its rightmost occurrence in the next four-slot window.  This also works
    // when finished players shorten a rotation and an older own pass remains
    // in the same window; equal pass values must not be globally deduplicated.
    int own_index = -1;
    for (std::size_t i = 0; i < snapshot.size(); ++i) {
        if (snapshot[i].player == self) own_index = static_cast<int>(i);
    }
    if (own_index < 0 && possible_pass_on_jump && snapshot.size() == 4) {
        // Passing after somebody finished can transfer the lead to that
        // player's partner before reaching us again.  There are then four
        // intervening actions and the anchor falls just outside the window.
        // All four entries are new, including repeated pass values.
        all->insert(all->end(), snapshot.begin(), snapshot.end());
        return true;
    }
    if (own_index < 0 || all->empty()) return fail(error, "previous_own_response_missing_from_history");
    const std::size_t overlap = static_cast<std::size_t>(own_index + 1);
    if (overlap > all->size()) return fail(error, "history_overlap_too_long");
    const std::size_t begin = all->size() - overlap;
    for (std::size_t i = 0; i < overlap; ++i) {
        if (!same_event((*all)[begin + i], snapshot[i])) return fail(error, "history_overlap_mismatch");
    }
    all->insert(all->end(), snapshot.begin() + static_cast<std::ptrdiff_t>(overlap), snapshot.end());
    return true;
}

bool leading_from(const std::vector<PublicEvent>& history, int self) {
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (it->player == self) break;
        if (!it->move.pass()) return false;
    }
    return true;
}

std::optional<Move> previous_from(const std::vector<PublicEvent>& history) {
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (!it->move.pass()) return it->move;
    }
    return std::nullopt;
}

struct Global {
    std::string level;
    int tribute = 0;
    int first = -1;
    int last = -1;
    bool resist = false;
    bool has_resist = false;
};

bool parse_global(const json::Value& request, bool deal, Global* out, std::string* error) {
    const auto* value = request.find("global");
    if (!value || !value->is_object()) return fail(error, "global_missing_or_not_object");
    const auto* level = value->find("level");
    if (!level) return fail(error, "level_missing");
    if (level->is_string()) out->level = normalize_level(level->string);
    else {
        int numeric = -1;
        if (!integer(*level, 0, 10, &numeric)) return fail(error, "invalid_level");
        out->level = normalize_level(std::to_string(numeric));
    }
    if (normalize_rank_index(out->level) < 0) return fail(error, "invalid_level");
    const auto* tribute = value->find("tribute");
    if (!tribute || !integer(*tribute, 0, 2, &out->tribute)) return fail(error, "invalid_tribute");
    for (const auto& field : {std::make_pair("first", &out->first), std::make_pair("last", &out->last)}) {
        const auto* number = value->find(field.first);
        *field.second = -1;
        if (number && !number->is_null() && !integer(*number, 0, kPlayers - 1, field.second)) {
            return fail(error, std::string("invalid_") + field.first);
        }
    }
    if (out->tribute > 0 && (out->first < 0 || out->last < 0 || out->first == out->last)) {
        return fail(error, "exchange_participants_missing_or_invalid");
    }
    if (out->tribute == 2 && out->first % 2 == out->last % 2) return fail(error, "double_tribute_same_team");
    const auto* resist = value->find("resist");
    out->has_resist = resist != nullptr;
    out->resist = false;
    if (resist) {
        if (!resist->is_bool()) return fail(error, "resist_not_boolean");
        out->resist = resist->boolean;
    } else if (!deal) return fail(error, "resist_missing");
    if (out->tribute == 0 && out->resist) return fail(error, "resist_without_tribute");
    return true;
}

bool same_global(const Global& a, const Global& b) {
    return a.level == b.level && a.tribute == b.tribute && a.first == b.first && a.last == b.last;
}

using CardMap = std::map<int, int>;
struct Exchange {
    CardMap tributes;
    CardMap returns;
    std::vector<std::pair<int, int>> pairs; // tribute source -> recipient
};

bool parse_map(const json::Value* value, bool resist, CardMap* out, std::string* error) {
    out->clear();
    if (!value || !value->is_object()) return fail(error, "exchange_map_missing_or_not_object");
    std::set<int> seen;
    for (const auto& item : value->object) {
        if (item.first.size() != 1 || item.first[0] < '0' || item.first[0] > '3') {
            return fail(error, "exchange_map_invalid_player");
        }
        const int owner = item.first[0] - '0';
        int card = -1;
        if (!integer(item.second, resist ? -1 : 0, resist ? -1 : kCardCount - 1, &card)) {
            return fail(error, "exchange_map_invalid_card");
        }
        if (!resist && !seen.insert(card).second) return fail(error, "exchange_map_duplicate_card");
        (*out)[owner] = card;
    }
    return true;
}

int point_key(int card, const std::string& level) {
    const std::string rank = rank_name(card);
    if (rank == "o") return 14;
    if (rank == "O") return 15;
    if (rank == normalize_level(level)) return 13;
    const int natural = normalize_rank_index(rank);
    return natural == 0 ? 12 : natural - 1;
}

bool parse_exchange(const json::Value& request, const Global& global, bool complete,
                    Exchange* out, std::string* error) {
    *out = Exchange{};
    const auto& value = request["global"];
    const auto* tributes = value.find("tribute_cards");
    const auto* returns = value.find("return_cards");
    if (global.tribute == 0) {
        if ((tributes && (!tributes->is_object() || !tributes->object.empty())) ||
            (returns && (!returns->is_object() || !returns->object.empty()))) {
            return fail(error, "cards_without_tribute");
        }
        return true;
    }
    if (!parse_map(tributes, global.resist, &out->tributes, error) ||
        !parse_map(returns, global.resist, &out->returns, error)) return false;
    const std::size_t wanted = static_cast<std::size_t>(global.tribute);
    if (out->tributes.size() > wanted || out->returns.size() > wanted ||
        (complete && (out->tributes.size() != wanted || out->returns.size() != wanted))) {
        return fail(error, "exchange_maps_incomplete_or_oversized");
    }
    for (const auto& item : out->tributes) {
        if (item.first != global.last && (global.tribute != 2 || item.first != (global.last + 2) % kPlayers)) {
            return fail(error, "unexpected_tribute_source");
        }
    }
    for (const auto& item : out->returns) {
        if (item.first != global.first && (global.tribute != 2 || item.first != (global.first + 2) % kPlayers)) {
            return fail(error, "unexpected_return_source");
        }
    }
    if (!global.resist) {
        std::set<int> all_cards;
        for (const auto& item : out->tributes) all_cards.insert(item.second);
        for (const auto& item : out->returns) {
            if (!all_cards.insert(item.second).second) return fail(error, "same_card_tributed_and_returned");
        }
    }
    if (global.tribute == 1) out->pairs.push_back({global.last, global.first});
    else if (out->tributes.size() == wanted) {
        const int source_a = global.last;
        const int source_b = (global.last + 2) % kPlayers;
        const int first = global.first;
        const int partner = (first + 2) % kPlayers;
        if (global.resist || point_key(out->tributes.at(source_a), global.level) ==
                              point_key(out->tributes.at(source_b), global.level)) {
            out->pairs.push_back({(first + 1) % kPlayers, first});
            out->pairs.push_back({(first + 3) % kPlayers, partner});
        } else if (point_key(out->tributes.at(source_a), global.level) >
                   point_key(out->tributes.at(source_b), global.level)) {
            out->pairs.push_back({source_a, first});
            out->pairs.push_back({source_b, partner});
        } else {
            out->pairs.push_back({source_b, first});
            out->pairs.push_back({source_a, partner});
        }
    }
    return true;
}

bool apply_exchange(const Exchange& exchange, const Global& global, int self,
                    const std::optional<int>& own_tribute, const std::optional<int>& own_return,
                    std::vector<int>* hand, std::string* error) {
    const auto matches = [self](const CardMap& cards, const std::optional<int>& response) {
        const auto it = cards.find(self);
        return it == cards.end() ? !response.has_value() : response.has_value() && it->second == *response;
    };
    if (!matches(exchange.tributes, own_tribute) || !matches(exchange.returns, own_return)) {
        return fail(error, "own_exchange_response_missing_or_mismatched");
    }
    if (global.resist) return true;
    for (const auto& pair : exchange.pairs) {
        if (self == pair.first && !remove_card(hand, exchange.tributes.at(pair.first))) {
            return fail(error, "tribute_not_in_original_hand");
        }
        if (self == pair.second && !remove_card(hand, exchange.returns.at(pair.second))) {
            return fail(error, "return_not_in_original_hand");
        }
    }
    for (const auto& pair : exchange.pairs) {
        if (self == pair.first) hand->push_back(exchange.returns.at(pair.second));
        if (self == pair.second) hand->push_back(exchange.tributes.at(pair.first));
    }
    const std::set<int> unique(hand->begin(), hand->end());
    if (unique.size() != hand->size()) return fail(error, "exchange_incoming_card_already_owned");
    return true;
}

void add_exchange_events(const Exchange& exchange, const Global& global, std::vector<PublicEvent>* events) {
    // The sources' turn order is public and independent of object-key order.
    for (const std::string& stage : {std::string("tribute"), std::string("return")}) {
        const CardMap& cards = stage == "tribute" ? exchange.tributes : exchange.returns;
        const int first = stage == "tribute" ? global.last : global.first;
        if (first < 0) continue;
        for (int n = 0; n < global.tribute; ++n) {
            const int owner = (first + n * 2) % kPlayers;
            const auto found = cards.find(owner);
            if (found == cards.end()) continue;
            PublicEvent event;
            event.player = owner;
            event.has_player = true;
            event.stage = stage;
            if (found->second >= 0) event.move = Move{{found->second}, {found->second}};
            for (const auto& pair : exchange.pairs) {
                if (stage == "tribute" && pair.first == owner) event.target = pair.second;
                if (stage == "return" && pair.second == owner) event.target = pair.first;
            }
            events->push_back(std::move(event));
        }
    }
}

}  // namespace

bool StateMirror::rebuild(const json::Value& full_input) {
    player = -1;
    hand.clear();
    public_events.clear();
    remaining_counts = {{-1, -1, -1, -1}};
    context = DecisionContext{};
    error.clear();
    if (!full_input.is_object()) return fail(&error, "input_not_object");
    const auto* requests = full_input.find("requests");
    const auto* responses = full_input.find("responses");
    if (!requests || !requests->is_array() || requests->array.empty()) return fail(&error, "requests_missing_or_empty");
    if (!responses || !responses->is_array()) return fail(&error, "responses_missing_or_not_array");
    if (requests->array.size() != responses->array.size() + 1) return fail(&error, "requests_responses_not_one_pending");

    Global initial_global;
    Global latest_global;
    Exchange settled_exchange;
    Exchange latest_exchange;
    std::optional<int> own_tribute;
    std::optional<int> own_return;
    bool started_play = false;
    bool has_own_play = false;
    bool possible_pass_on_jump = false;
    bool has_resist = false;
    bool known_resist = false;
    std::vector<PublicEvent> plays;
    std::vector<PublicEvent> latest_snapshot;

    for (std::size_t i = 0; i < requests->array.size(); ++i) {
        const auto& request = requests->array[i];
        if (!request.is_object()) return fail(&error, "request_not_object");
        const auto* stage_value = request.find("stage");
        if (!stage_value || !stage_value->is_string()) return fail(&error, "stage_missing_or_not_string");
        const std::string& stage = stage_value->string;
        if (stage != "deal" && stage != "tribute" && stage != "return" && stage != "play") return fail(&error, "unknown_stage");
        const bool completed = i < responses->array.size();
        Global global;
        if (!parse_global(request, stage == "deal", &global, &error)) return false;
        if (i == 0) {
            if (stage != "deal") return fail(&error, "initial_deal_missing");
            const auto* id = request.find("your_id");
            const auto* deliver = request.find("deliver");
            if (!id || !integer(*id, 0, kPlayers - 1, &player)) return fail(&error, "invalid_or_missing_your_id");
            if (!deliver || !card_array(*deliver, &hand, true, "deliver", &error)) return fail(&error, "invalid_or_missing_deliver");
            if (hand.size() != kInitialCards) return fail(&error, "deliver_not_27_cards");
            initial_global = global;
        } else {
            if (stage == "deal" || request.contains("deliver")) return fail(&error, "unexpected_redeal");
            if (const auto* id = request.find("your_id")) {
                int value = -1;
                if (!integer(*id, 0, kPlayers - 1, &value) || value != player) return fail(&error, "your_id_changed_or_invalid");
            }
            if (!same_global(initial_global, global)) return fail(&error, "global_contract_changed");
        }
        if (global.has_resist) {
            if (has_resist && known_resist != global.resist) return fail(&error, "resist_changed");
            has_resist = true;
            known_resist = global.resist;
        }
        if (stage != "deal" && !parse_exchange(request, global, stage == "play", &latest_exchange, &error)) return false;
        if (started_play && stage != "play") return fail(&error, "exchange_stage_after_play");

        if (stage == "deal") {
            context.stage = DecisionContext::Stage::Deal;
            if (completed && (!responses->array[i].is_array() || !responses->array[i].array.empty())) return fail(&error, "deal_response_not_empty");
        } else if (stage == "tribute" || stage == "return") {
            if (global.tribute == 0) return fail(&error, "exchange_stage_without_tribute");
            const bool expected = stage == "tribute" ?
                (player == global.last || (global.tribute == 2 && player == (global.last + 2) % kPlayers)) :
                (player == global.first || (global.tribute == 2 && player == (global.first + 2) % kPlayers));
            if (!expected) return fail(&error, "exchange_request_wrong_player");
            context.stage = stage == "tribute" ? DecisionContext::Stage::Tribute : DecisionContext::Stage::Return;
            std::optional<int>& remembered = stage == "tribute" ? own_tribute : own_return;
            if (remembered.has_value()) return fail(&error, "duplicate_exchange_request");
            if (completed) {
                const auto& response = responses->array[i];
                if (!response.is_array() || response.array.size() != (global.resist ? 0U : 1U)) return fail(&error, "exchange_response_wrong_shape");
                int card = -1;
                if (!global.resist) {
                    if (!integer(response.array[0], 0, kCardCount - 1, &card)) return fail(&error, "exchange_response_invalid_card");
                    if (std::find(hand.begin(), hand.end(), card) == hand.end()) return fail(&error, "exchange_card_not_in_original_hand");
                }
                remembered = card; // -1 records an actual resistant response.
            }
        } else {
            context.stage = DecisionContext::Stage::Play;
            if (!started_play) {
                if (!apply_exchange(latest_exchange, global, player, own_tribute, own_return, &hand, &error)) return false;
                settled_exchange = latest_exchange;
                started_play = true;
            } else if (settled_exchange.tributes != latest_exchange.tributes || settled_exchange.returns != latest_exchange.returns) {
                return fail(&error, "exchange_cards_changed_after_play");
            }
            bool positional = false;
            if (!history_snapshot(request, player, &latest_snapshot, &positional, &error) ||
                !merge_snapshot(latest_snapshot, player, has_own_play, possible_pass_on_jump,
                                positional, &plays, &error)) return false;
            context.leading = leading_from(latest_snapshot, player);
            context.previous = previous_from(latest_snapshot);
            int request_pass_on = -1;
            const auto* pass_on = request.find("pass_on");
            if (!pass_on || !integer(*pass_on, -1, kPlayers - 1, &request_pass_on)) {
                return fail(&error, "invalid_or_missing_pass_on");
            }
            if (completed) {
                Move move;
                if (!play_response(responses->array[i], &move, &error)) return false;
                possible_pass_on_jump = move.pass() && request_pass_on >= 0;
                for (const int card : move.action) {
                    if (!remove_card(&hand, card)) return fail(&error, "play_card_not_in_replayed_hand");
                }
                PublicEvent event;
                event.player = player;
                event.has_player = true;
                event.move = std::move(move);
                plays.push_back(std::move(event));
                has_own_play = true;
            }
        }
        latest_global = global;
    }

    remaining_counts = {{kInitialCards, kInitialCards, kInitialCards, kInitialCards}};
    std::set<int> played_cards;
    for (const auto& event : plays) {
        int& count = remaining_counts[static_cast<std::size_t>(event.player)];
        count -= static_cast<int>(event.move.action.size());
        if (count < 0) return fail(&error, "remaining_count_underflow");
        for (const int card : event.move.action) {
            if (!played_cards.insert(card).second) return fail(&error, "physical_card_played_twice");
            if (event.player != player && std::find(hand.begin(), hand.end(), card) != hand.end()) {
                return fail(&error, "public_play_conflicts_with_private_hand");
            }
        }
    }
    if (remaining_counts[static_cast<std::size_t>(player)] != static_cast<int>(hand.size())) return fail(&error, "own_count_replay_mismatch");
    context.done.clear();
    context.pass_on = -1;
    const auto& current = requests->array.back();
    if (context.stage == DecisionContext::Stage::Play) {
        const auto* done = current.find("done");
        const auto* pass_on = current.find("pass_on");
        if (!done || !done->is_array()) return fail(&error, "done_missing_or_not_array");
        std::set<int> seen;
        for (const auto& item : done->array) {
            int owner = -1;
            if (!integer(item, 0, kPlayers - 1, &owner) || !seen.insert(owner).second) return fail(&error, "invalid_done_player");
            if (remaining_counts[static_cast<std::size_t>(owner)] != 0) return fail(&error, "done_count_mismatch");
            context.done.push_back(owner);
        }
        for (int owner = 0; owner < kPlayers; ++owner) {
            if (remaining_counts[static_cast<std::size_t>(owner)] == 0 && !seen.count(owner)) {
                return fail(&error, "empty_hand_missing_from_done");
            }
        }
        if (seen.count(player)) return fail(&error, "request_for_finished_player");
        if (!pass_on || !integer(*pass_on, -1, kPlayers - 1, &context.pass_on)) return fail(&error, "invalid_or_missing_pass_on");
        if (context.pass_on >= 0 && !seen.count(context.pass_on)) return fail(&error, "pass_on_player_not_done");
    }
    add_exchange_events(latest_exchange, latest_global, &public_events);
    public_events.insert(public_events.end(), plays.begin(), plays.end());
    context.player_id = player;
    context.level = latest_global.level;
    context.resist = latest_global.resist;
    context.tribute = latest_global.tribute;
    context.variant = RuleVariant::BotzoneCompat;
    context.hand = hand;
    context.public_events = public_events;
    context.remaining_counts = remaining_counts;
    return true;
}

}  // namespace oxbot
