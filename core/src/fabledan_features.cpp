#include "oxbot/fabledan_features.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace oxbot {
namespace {

constexpr int kVocab = 48;
constexpr int kMaxSequence = 512;
constexpr int kFeatureDim = 80;
constexpr int kStructureFeatureDim = 224;
constexpr int kPass = 0;
constexpr int kTypeBase = 19;
constexpr int kPlayerBase = 15;
constexpr int kTribute = 30;
constexpr int kReturn = 31;
constexpr int kRankBase = 32;

int fbd_rank(int card) {
    if (!valid_card_id(card)) throw std::runtime_error("fabledan_card_invalid");
    const int face = face_id(card);
    if (face >= 52) return face == 52 ? 13 : 14;
    return face / 4;
}

int fbd_level(const std::string& value) {
    const int result = normalize_rank_index(normalize_level(value));
    if (result < 0 || result >= 13) throw std::runtime_error("fabledan_level_invalid");
    return result;
}

int fbd_type(PokerKind kind) {
    // FableDan: PASS,SINGLE,PAIR,TRIPLE,FULL,STRAIGHT,PLATE,TUBE,
    // BOMB,SFLUSH,ROCKET.  PokerKind also contains Invalid and orders
    // ordinary sequence variants differently, so never static_cast here.
    switch (kind) {
        case PokerKind::Pass: return 0;
        case PokerKind::Single: return 1;
        case PokerKind::Pair: return 2;
        case PokerKind::Three: return 3;
        case PokerKind::Set: return 4;
        case PokerKind::Straight: return 5;
        case PokerKind::TriplePairs: return 6;
        case PokerKind::ThreeStraight: return 7;
        case PokerKind::Bomb: return 8;
        case PokerKind::StraightFlush: return 9;
        case PokerKind::Rocket: return 10;
        case PokerKind::Invalid: break;
    }
    throw std::runtime_error("fabledan_type_invalid");
}

int fbd_order(int rank, int level) {
    if (rank < 0 || rank > 14) throw std::runtime_error("fabledan_rank_invalid");
    int position = 0;
    // Same table as competition/fabledan/cards.py::_build_order_tables:
    // 2..K,A, current level, small joker, big joker.
    for (int candidate = 1; candidate <= 12; ++candidate) {
        if (candidate == level) continue;
        if (candidate == rank) return position;
        ++position;
    }
    if (level != 0) {
        if (rank == 0) return position;
        ++position;
    }
    if (rank == level) return position;
    ++position;
    if (rank == 13) return position;
    ++position;
    if (rank == 14) return position;
    throw std::runtime_error("fabledan_order_invalid");
}

int fbd_key(const Move& move, const std::string& level) {
    const PokerType type = classify(move.claim);
    const int lv = fbd_level(level);
    if (type.kind == PokerKind::Pass || type.kind == PokerKind::Rocket) return 0;
    if (type.kind == PokerKind::Straight || type.kind == PokerKind::StraightFlush ||
        type.kind == PokerKind::ThreeStraight || type.kind == PokerKind::TriplePairs) {
        // C++ rule classification uses natural rank indices for sequence
        // starts (A2345=0, 2..6=1, ...); FableDan uses sequence values
        // (A2345=1, 2..6=2, ...).
        if (type.key < 0) throw std::runtime_error("fabledan_sequence_key_invalid");
        return type.key + 1;
    }
    return fbd_order(type.key, lv);
}

std::vector<int> claim_ranks(const Move& move) {
    std::vector<int> result;
    const std::vector<int>& source = move.claim.empty() ? move.action : move.claim;
    result.reserve(source.size());
    for (const int card : source) result.push_back(fbd_rank(card));
    std::sort(result.begin(), result.end());
    return result;
}

void append_event_tokens(std::vector<int>* tokens, const PublicEvent& event,
                         int viewer, int level) {
    if (!event.has_player || event.player < 0 || event.player >= 4) {
        throw std::runtime_error("fabledan_event_owner_invalid");
    }
    const int relative = (event.player - viewer + 4) % 4;
    tokens->push_back(kPlayerBase + relative);
    if (event.stage == "tribute" || event.stage == "return") {
        // Resistant exchange records carry no card.  FableDan's engine does
        // not emit a token for a resisted tribute/return, so omit the event.
        if (event.move.action.empty()) {
            tokens->pop_back();
            return;
        }
        tokens->push_back(event.stage == "tribute" ? kTribute : kReturn);
        tokens->push_back(kRankBase + fbd_rank(event.move.action.front()));
        return;
    }
    const PokerType type = classify(event.move.claim);
    const int mapped = event.move.action.empty() ? kPass : fbd_type(type.kind);
    tokens->push_back(kTypeBase + mapped);
    if (!event.move.action.empty()) {
        const auto ranks = claim_ranks(event.move);
        for (const int rank : ranks) tokens->push_back(kRankBase + rank);
    }
    (void)level;
}

std::vector<float> state_base(const DecisionContext& context) {
    if (context.player_id < 0 || context.player_id >= 4) {
        throw std::runtime_error("fabledan_player_invalid");
    }
    const int level = fbd_level(context.level);
    std::vector<float> result(kFeatureDim, 0.f);
    for (const int card : context.hand) result[static_cast<std::size_t>(fbd_rank(card))] += 0.25f;
    int wild = 0;
    for (const int card : context.hand) if (is_level_card(card, context.level)) ++wild;
    result[15] = static_cast<float>(wild) / 2.f;
    result[16] = static_cast<float>(context.hand.size()) / 27.f;
    for (int relative = 0; relative < 4; ++relative) {
        const int count = context.remaining_counts[static_cast<std::size_t>((context.player_id + relative) % 4)];
        if (count < 0 || count > 27) throw std::runtime_error("fabledan_remaining_invalid");
        result[static_cast<std::size_t>(17 + relative)] = static_cast<float>(count) / 27.f;
    }
    for (int relative = 0; relative < 4; ++relative) {
        const int owner = (context.player_id + relative) % 4;
        if (std::find(context.done.begin(), context.done.end(), owner) != context.done.end()) {
            result[static_cast<std::size_t>(21 + relative)] = 1.f;
        }
    }
    result[static_cast<std::size_t>(25 + level)] = 1.f;
    return result;
}

void validate_feature_dim(int feat_dim) {
    if (feat_dim != kFeatureDim && feat_dim != kStructureFeatureDim) {
        throw std::runtime_error("fabledan_feature_dim_invalid");
    }
}

// Counts feasible windows, rather than enumerating physical card combinations.
// Sequence values 1..14 contain both the low and high ace; jokers are excluded.
int structure_windows(const std::array<int, 15>& counts, int length,
                      int multiplicity, int wild) {
    int total = 0;
    for (int low = 1; low <= 15 - length; ++low) {
        int cost = 0;
        for (int offset = 0; offset < length; ++offset) {
            const int rank = (low + offset - 1) % 13;
            cost += std::max(0, multiplicity - counts[static_cast<std::size_t>(rank)]);
        }
        if (cost <= wild) ++total;
    }
    return total;
}

void append_structure_features(std::vector<float>* result, const Move& move,
                               const DecisionContext& context) {
    result->resize(kStructureFeatureDim, 0.f);
    std::array<bool, kCardCount> present{};
    for (const int card : context.hand) {
        if (!valid_card_id(card) || present[static_cast<std::size_t>(card)]) {
            throw std::runtime_error("fabledan_hand_invalid");
        }
        present[static_cast<std::size_t>(card)] = true;
        (*result)[static_cast<std::size_t>(80 + face_id(card))] += 0.5f;
    }
    // Wildcard claims can name a different face.  Remove the actual card IDs
    // played, never their declared replacements.
    for (const int card : move.action) {
        if (!valid_card_id(card) || !present[static_cast<std::size_t>(card)]) {
            throw std::runtime_error("fabledan_action_not_in_hand");
        }
        present[static_cast<std::size_t>(card)] = false;
    }
    std::array<int, 15> natural{};
    std::array<std::array<int, 15>, 4> suited{};
    int wild = 0;
    int remaining = 0;
    for (const int card : context.hand) {
        if (!present[static_cast<std::size_t>(card)]) continue;
        ++remaining;
        const int face = face_id(card);
        (*result)[static_cast<std::size_t>(134 + face)] += 0.5f;
        if (is_level_card(card, context.level)) {
            ++wild;
            continue;
        }
        const int rank = fbd_rank(card);
        ++natural[static_cast<std::size_t>(rank)];
        if (face < 52) {
            ++suited[static_cast<std::size_t>(face % 4)][static_cast<std::size_t>(rank)];
        }
    }
    for (std::size_t rank = 0; rank < natural.size(); ++rank) {
        (*result)[188U + rank] = static_cast<float>(natural[rank]) / 4.f;
    }
    (*result)[203] = static_cast<float>(wild) / 2.f;
    (*result)[204] = static_cast<float>(remaining) / 27.f;
    for (int count = 1; count <= 8; ++count) {
        int ranks = 0;
        for (const int available : natural) if (available >= count) ++ranks;
        (*result)[static_cast<std::size_t>(204 + count)] = static_cast<float>(ranks) / 15.f;
    }
    (*result)[213] = static_cast<float>(structure_windows(natural, 5, 1, 0)) / 10.f;
    (*result)[214] = static_cast<float>(structure_windows(natural, 5, 1, wild)) / 10.f;
    int flushes = 0;
    int wild_flushes = 0;
    for (const auto& counts : suited) {
        flushes += structure_windows(counts, 5, 1, 0);
        wild_flushes += structure_windows(counts, 5, 1, wild);
    }
    (*result)[215] = static_cast<float>(flushes) / 40.f;
    (*result)[216] = static_cast<float>(wild_flushes) / 40.f;
    (*result)[217] = static_cast<float>(structure_windows(natural, 3, 2, 0)) / 12.f;
    (*result)[218] = static_cast<float>(structure_windows(natural, 3, 2, wild)) / 12.f;
    (*result)[219] = static_cast<float>(structure_windows(natural, 2, 3, 0)) / 13.f;
    (*result)[220] = static_cast<float>(structure_windows(natural, 2, 3, wild)) / 13.f;
    int full_houses = 0;
    int max_bomb = 0;
    for (std::size_t triple = 0; triple < 13U; ++triple) {
        if (natural[triple] == 0) continue;
        for (std::size_t pair = 0; pair < natural.size(); ++pair) {
            if (pair == triple || natural[pair] == 0) continue;
            if (pair >= 13U && natural[pair] < 2) continue;
            const int cost = std::max(0, 3 - natural[triple]) +
                             std::max(0, 2 - natural[pair]);
            if (cost <= wild) ++full_houses;
        }
        const int bomb = natural[triple] + wild;
        if (bomb >= 4) max_bomb = std::max(max_bomb, std::min(bomb, 10));
    }
    (*result)[221] = static_cast<float>(full_houses) / 182.f;
    (*result)[222] = static_cast<float>(max_bomb) / 10.f;
    (*result)[223] = natural[13] >= 2 && natural[14] >= 2 ? 1.f : 0.f;
}

}  // namespace

std::vector<int> encode_fabledan_history(const DecisionContext& context) {
    if (context.player_id < 0 || context.player_id >= 4) {
        throw std::runtime_error("fabledan_player_invalid");
    }
    const int level = fbd_level(context.level);
    std::vector<int> tokens{1, 2 + level};
    for (const auto& event : context.public_events) {
        // Resistant exchange records have no rank and are absent from the
        // Python engine's event stream.
        if ((event.stage == "tribute" || event.stage == "return") && event.move.action.empty()) continue;
        append_event_tokens(&tokens, event, context.player_id, level);
    }
    if (tokens.size() > static_cast<std::size_t>(kMaxSequence)) {
        std::vector<int> trimmed;
        trimmed.reserve(kMaxSequence);
        trimmed.insert(trimmed.end(), tokens.begin(), tokens.begin() + 2);
        trimmed.insert(trimmed.end(), tokens.end() - (kMaxSequence - 2), tokens.end());
        tokens.swap(trimmed);
    }
    for (const int token : tokens) {
        if (token < 0 || token >= kVocab) throw std::runtime_error("fabledan_token_invalid");
    }
    return tokens;
}

std::vector<float> encode_fabledan_action(const Move& move,
                                          const DecisionContext& context,
                                          int feat_dim) {
    validate_feature_dim(feat_dim);
    (void)fbd_level(context.level);
    auto result = state_base(context);
    const PokerType type = classify(move.claim);
    const int mapped = move.action.empty() ? kPass : fbd_type(type.kind);
    result[static_cast<std::size_t>(38 + mapped)] = 1.f;
    const auto ranks = claim_ranks(move);
    for (const int rank : ranks) result[static_cast<std::size_t>(49 + rank)] += 0.25f;
    result[64] = static_cast<float>(move.action.size()) / 27.f;
    int wild = 0;
    for (const int card : move.action) if (is_level_card(card, context.level)) ++wild;
    result[65] = static_cast<float>(wild) / 2.f;
    result[66] = move.action.empty() ? 0.f : static_cast<float>(fbd_key(move, context.level)) / 15.f;
    if (!context.leading && context.previous.has_value() && !context.previous->action.empty()) {
        const PokerType lead = classify(context.previous->claim);
        const int lead_type = fbd_type(lead.kind);
        result[static_cast<std::size_t>(67 + lead_type)] = 1.f;
        result[78] = static_cast<float>(fbd_key(*context.previous, context.level)) / 15.f;
    } else {
        result[79] = 1.f;
    }
    if (feat_dim == kStructureFeatureDim) append_structure_features(&result, move, context);
    return result;
}

bool fabledan_generator_candidate(const Move& move,
                                  const DecisionContext& context) {
    if (move.action.empty()) return true;
    const PokerType type = classify(move.claim);
    const int level = fbd_level(context.level);
    std::array<int, 15> natural{};
    for (const int card : context.hand) {
        const int rank = fbd_rank(card);
        if (!is_level_card(card, context.level)) ++natural[static_cast<std::size_t>(rank)];
    }
    int action_wild = 0;
    for (const int card : move.action) if (is_level_card(card, context.level)) ++action_wild;
    const int rank = type.key;
    const auto required_repeated = [&](int target, int amount) {
        if (target < 0 || target >= 13 || natural[static_cast<std::size_t>(target)] == 0) return -1;
        return std::max(0, amount - natural[static_cast<std::size_t>(target)]);
    };
    switch (type.kind) {
        case PokerKind::Single:
            return action_wild == 0 || (rank == level && natural[static_cast<std::size_t>(level)] == 0 && action_wild == 1);
        case PokerKind::Pair:
            if (rank >= 13) return action_wild == 0 && natural[static_cast<std::size_t>(rank)] >= 2;
            if (natural[static_cast<std::size_t>(rank)] == 0) {
                return rank == level && action_wild == 2;
            }
            return action_wild == required_repeated(rank, 2);
        case PokerKind::Three: {
            const int required = required_repeated(rank, 3);
            return required >= 0 && action_wild == required;
        }
        case PokerKind::Set: {
            const int triple = required_repeated(type.key, 3);
            if (triple < 0 || type.secondary < 0 || type.secondary == type.key) return false;
            int pair = 0;
            if (type.secondary >= 13) {
                if (natural[static_cast<std::size_t>(type.secondary)] < 2) return false;
            } else {
                // FableDan's full-house generator requires at least one
                // natural card in the pair rank; it never makes an all-wild
                // pair attachment.
                if (natural[static_cast<std::size_t>(type.secondary)] == 0) return false;
                pair = required_repeated(type.secondary, 2);
            }
            return pair >= 0 && action_wild == triple + pair;
        }
        case PokerKind::Bomb: {
            const int required = required_repeated(rank, type.length);
            return required >= 0 && action_wild == required;
        }
        case PokerKind::Straight:
        case PokerKind::StraightFlush:
        case PokerKind::ThreeStraight:
        case PokerKind::TriplePairs:
        case PokerKind::Rocket:
            // These generators deliberately use wildcards to fill missing
            // ranks; the later semantic pass keeps the minimum-wild version.
            return true;
        case PokerKind::Pass:
        case PokerKind::Invalid:
            return false;
    }
    return false;
}

std::vector<std::vector<float>> encode_fabledan_actions(
    const std::vector<Move>& moves, const DecisionContext& context, int feat_dim) {
    validate_feature_dim(feat_dim);
    std::vector<std::vector<float>> result;
    result.reserve(moves.size());
    for (const auto& move : moves) result.push_back(encode_fabledan_action(move, context, feat_dim));
    return result;
}

}  // namespace oxbot
