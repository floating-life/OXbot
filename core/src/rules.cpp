#include "oxbot/rules.hpp"

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <utility>

namespace oxbot {
namespace {

struct Target {
    int rank = -1;  // cardscale/natural-rank index: A=0, 2=1, ..., K=12
    int suit = -1;  // -1 means any suit; 0..3 means an exact suit
};

void set_reason(std::string* reason, const std::string& value) {
    if (reason) *reason = value;
}

int rank_code(const std::string& rank) {
    if (rank == "o") return 13;
    if (rank == "O") return 14;
    return normalize_rank_index(rank);
}

int rank_code_of(int card) {
    return valid_card_id(card) ? rank_code(rank_name(card)) : -1;
}

bool natural_rank(int rank) { return rank >= 0 && rank < 13; }

bool is_normal_kind(PokerKind kind) {
    return kind == PokerKind::Single || kind == PokerKind::Pair ||
           kind == PokerKind::Three || kind == PokerKind::Straight ||
           kind == PokerKind::Set || kind == PokerKind::ThreeStraight ||
           kind == PokerKind::TriplePairs;
}

bool is_scaled_kind(PokerKind kind) {
    return kind == PokerKind::Straight || kind == PokerKind::ThreeStraight ||
           kind == PokerKind::TriplePairs;
}

// The ordinary point order is 2,3,...,A with the current level moved to the
// top. Sequence types intentionally use cardscale order instead.
int ordinary_key(int rank, const std::string& level) {
    if (rank < 0 || rank > 14) return -1;
    const int current = normalize_rank_index(normalize_level(level));
    std::vector<int> order;
    order.reserve(15U);
    for (int r = 1; r < 13; ++r) {
        if (r != current) order.push_back(r);
    }
    if (current != 0) order.push_back(0);  // A remains below the level.
    if (current >= 0) order.push_back(current);
    order.push_back(13);  // small joker
    order.push_back(14);  // big joker
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i] == rank) return static_cast<int>(i);
    }
    return -1;
}

bool same_rank_counts(const std::map<int, int>& counts, int rank, int count) {
    const auto it = counts.find(rank);
    return it != counts.end() && it->second == count;
}

bool all_count(const std::map<int, int>& counts, int count, int number_of_ranks) {
    if (static_cast<int>(counts.size()) != number_of_ranks) return false;
    for (const auto& item : counts) {
        if (item.second != count) return false;
    }
    return true;
}

int straight_start(const std::vector<int>& ranks) {
    if (ranks.size() != 5U) return -1;
    if (!std::all_of(ranks.begin(), ranks.end(), natural_rank)) return -1;
    std::vector<int> sorted = ranks;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return -1;
    if (sorted == std::vector<int>{0, 1, 2, 3, 4}) return 0;       // A2345
    if (sorted == std::vector<int>{0, 9, 10, 11, 12}) return 9;    // 10JQKA
    if (sorted.back() - sorted.front() == 4) return sorted.front();
    return -1;
}

int three_straight_start(const std::vector<int>& ranks) {
    if (ranks.size() != 2U) return -1;
    if (!std::all_of(ranks.begin(), ranks.end(), natural_rank)) return -1;
    std::vector<int> sorted = ranks;
    std::sort(sorted.begin(), sorted.end());
    if (sorted == std::vector<int>{0, 1}) return 0;    // AAA222
    if (sorted == std::vector<int>{0, 12}) return 12;  // KKKAAA
    if (sorted[1] == sorted[0] + 1) return sorted[0];
    return -1;
}

int triple_pairs_start(const std::vector<int>& ranks) {
    if (ranks.size() != 3U) return -1;
    if (!std::all_of(ranks.begin(), ranks.end(), natural_rank)) return -1;
    std::vector<int> sorted = ranks;
    std::sort(sorted.begin(), sorted.end());
    if (sorted == std::vector<int>{0, 1, 2}) return 0;    // AA2233
    if (sorted == std::vector<int>{0, 11, 12}) return 11; // QQKKAA
    if (sorted[1] == sorted[0] + 1 && sorted[2] == sorted[1] + 1) {
        return sorted[0];
    }
    return -1;
}

bool flush(const std::vector<int>& cards) {
    if (cards.empty()) return false;
    const int first = suit_index(cards.front());
    if (first < 0) return false;
    return std::all_of(cards.begin(), cards.end(), [&](int card) {
        return suit_index(card) == first;
    });
}

bool contains_id(const std::vector<int>& cards, int id) {
    return std::find(cards.begin(), cards.end(), id) != cards.end();
}

bool hand_has_duplicate_id(const std::vector<int>& hand) {
    std::set<int> seen;
    for (const int id : hand) {
        if (!valid_card_id(id) || !seen.insert(id).second) return true;
    }
    return false;
}

int canonical_face(const Target& target) {
    if (target.rank == 13) return 52;
    if (target.rank == 14) return 53;
    if (!natural_rank(target.rank)) return -1;
    // A non-heart suit keeps a wildcard claim from being confused with the
    // physical covering card. Exact-suit templates keep their requested suit.
    const int suit = target.suit >= 0 ? target.suit : 1;
    return target.rank * 4 + suit;
}

// Canonical action faces preserve all suit and wildcard decisions. The
// trailing type/keys canonicalize only claim descriptions and physical deck
// copies that have the same game meaning.
using MoveKey = std::vector<int>;

void enumerate_targets(const std::vector<Target>& targets,
                       std::size_t index,
                       const std::vector<int>& hand,
                       const std::string& level,
                       const std::vector<int>& wildcards,
                       std::set<int>& used_cards,
                       std::set<int>& used_wildcards,
                       std::vector<int>& action,
                       std::vector<int>& claim,
                       const std::function<void(std::vector<int>, std::vector<int>)>& add) {
    if (index == targets.size()) {
        add(action, claim);
        return;
    }
    const Target& target = targets[index];
    if (!natural_rank(target.rank) && target.rank != 13 && target.rank != 14) return;
    const bool same_target_as_previous = index > 0U &&
        targets[index - 1U].rank == target.rank &&
        targets[index - 1U].suit == target.suit;
    const int previous_action = action.empty() ? -1 : action.back();
    const int previous_choice = previous_action < 0 ? -1 :
        (is_level_card(previous_action, level) ? kCardCount + previous_action : previous_action);

    // First use matching physical non-wild cards. Keeping their actual ids in
    // the claim preserves suit information and makes modulo-54 matching
    // deterministic across the two physical decks.
    for (const int card : hand) {
        if (!valid_card_id(card) || is_level_card(card, level) || used_cards.count(card) != 0U) {
            continue;
        }
        // The two copies of one face have identical future possibilities.
        // Use the lower available physical id first while retaining the
        // ability to use both copies in the same action.
        const bool lower_copy_available = std::any_of(hand.begin(), hand.end(), [&](int other) {
            return other < card && same_face(other, card) && used_cards.count(other) == 0U;
        });
        if (lower_copy_available) continue;
        if (same_target_as_previous && card <= previous_choice) {
            continue;
        }
        if (rank_code_of(card) != target.rank) continue;
        if (target.suit >= 0 && suit_index(card) != target.suit) continue;
        used_cards.insert(card);
        action.push_back(card);
        claim.push_back(card);
        enumerate_targets(targets, index + 1U, hand, level, wildcards,
                          used_cards, used_wildcards, action, claim, add);
        claim.pop_back();
        action.pop_back();
        used_cards.erase(card);
    }

    // A heart current-level card is the only wildcard. It can cover any
    // natural rank (including its own rank), but never a joker.
    if (!natural_rank(target.rank)) return;  // jokers are never wildcards
    for (const int wildcard : wildcards) {
        if (used_wildcards.count(wildcard) != 0U) continue;
        const bool lower_copy_available = std::any_of(wildcards.begin(), wildcards.end(), [&](int other) {
            return other < wildcard && used_wildcards.count(other) == 0U;
        });
        if (lower_copy_available) continue;
        if (same_target_as_previous && kCardCount + wildcard <= previous_choice) {
            continue;
        }
        const int claimed = canonical_face(target);
        if (!valid_card_id(claimed)) continue;
        used_wildcards.insert(wildcard);
        action.push_back(wildcard);
        claim.push_back(claimed);
        enumerate_targets(targets, index + 1U, hand, level, wildcards,
                          used_cards, used_wildcards, action, claim, add);
        claim.pop_back();
        action.pop_back();
        used_wildcards.erase(wildcard);
    }
}

std::vector<Target> repeated_targets(int rank, int count, int suit = -1) {
    std::vector<Target> targets;
    targets.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) targets.push_back(Target{rank, suit});
    return targets;
}

std::vector<int> straight_ranks_from(int start) {
    if (start == 9) return {9, 10, 11, 12, 0};
    std::vector<int> result;
    for (int i = 0; i < 5; ++i) result.push_back(start + i);
    return result;
}

std::vector<int> three_straight_ranks_from(int start) {
    if (start == 12) return {12, 0};
    return {start, start + 1};
}

std::vector<int> triple_pairs_ranks_from(int start) {
    if (start == 11) return {11, 12, 0};
    return {start, start + 1, start + 2};
}

}  // namespace

std::string stage_name(DecisionContext::Stage stage) {
    switch (stage) {
        case DecisionContext::Stage::Deal: return "deal";
        case DecisionContext::Stage::Tribute: return "tribute";
        case DecisionContext::Stage::Return: return "return";
        case DecisionContext::Stage::Play: return "play";
    }
    return "play";
}

PokerType classify(const std::vector<int>& claim) {
    PokerType invalid;
    invalid.length = static_cast<int>(claim.size());
    if (claim.empty()) return {PokerKind::Pass, 0, -1, -1};
    for (const int id : claim) {
        if (!valid_card_id(id)) return invalid;
    }

    std::map<int, int> counts;
    std::vector<int> ranks;
    ranks.reserve(claim.size());
    for (const int id : claim) {
        const int rank = rank_code_of(id);
        if (rank < 0) return invalid;
        ++counts[rank];
        ranks.push_back(rank);
    }
    const int n = static_cast<int>(claim.size());
    if (n == 1) return {PokerKind::Single, 1, ranks.front(), -1};

    if (n == 2) {
        if (ranks[0] == ranks[1]) return {PokerKind::Pair, 2, ranks[0], -1};
        return invalid;
    }

    if (n == 3) {
        if (counts.size() == 1U && natural_rank(ranks.front())) {
            return {PokerKind::Three, 3, ranks.front(), -1};
        }
        return invalid;
    }

    if (n == 4) {
        if (same_rank_counts(counts, 13, 2) && same_rank_counts(counts, 14, 2)) {
            return {PokerKind::Rocket, 4, 0, -1};
        }
        if (counts.size() == 1U && natural_rank(ranks.front())) {
            return {PokerKind::Bomb, 4, ranks.front(), -1};
        }
        return invalid;
    }

    if (n == 5) {
        if (counts.size() == 1U && natural_rank(ranks.front())) {
            return {PokerKind::Bomb, 5, ranks.front(), -1};
        }
        int triple = -1;
        int pair = -1;
        for (const auto& item : counts) {
            if (item.second == 3) triple = item.first;
            if (item.second == 2) pair = item.first;
        }
        if (triple >= 0 && pair >= 0 && counts.size() == 2U) {
            return {PokerKind::Set, 5, triple, pair};
        }
        if (counts.size() == 5U) {
            const int key = straight_start(ranks);
            if (key >= 0) {
                return {flush(claim) ? PokerKind::StraightFlush : PokerKind::Straight,
                        5, key, -1};
            }
        }
        return invalid;
    }

    if (n == 6) {
        if (counts.size() == 1U && natural_rank(ranks.front())) {
            return {PokerKind::Bomb, 6, ranks.front(), -1};
        }
        if (all_count(counts, 3, 2)) {
            std::vector<int> distinct;
            for (const auto& item : counts) distinct.push_back(item.first);
            const int key = three_straight_start(distinct);
            if (key >= 0) return {PokerKind::ThreeStraight, 6, key, -1};
        }
        if (all_count(counts, 2, 3)) {
            std::vector<int> distinct;
            for (const auto& item : counts) distinct.push_back(item.first);
            const int key = triple_pairs_start(distinct);
            if (key >= 0) return {PokerKind::TriplePairs, 6, key, -1};
        }
        return invalid;
    }

    if (n > 6 && n <= 10 && counts.size() == 1U && natural_rank(ranks.front())) {
        return {PokerKind::Bomb, n, ranks.front(), -1};
    }
    return invalid;
}

bool is_legal_claim(const std::vector<int>& action,
                    const std::vector<int>& claim,
                    const std::string& level,
                    std::string* reason) {
    if (action.empty() || claim.empty()) {
        if (action.empty() && claim.empty()) return true;
        set_reason(reason, "pass_claim_mismatch");
        return false;
    }
    if (action.size() != claim.size()) {
        set_reason(reason, "claim_length_mismatch");
        return false;
    }

    // Claims describe faces. Two physical decks therefore compare modulo 54;
    // exact physical ids are reserved for ownership validation of action.
    std::map<int, int> remaining;
    for (const int id : claim) {
        if (!valid_card_id(id)) {
            set_reason(reason, "invalid_claim_card_id");
            return false;
        }
        ++remaining[face_id(id)];
    }
    for (const int id : action) {
        if (!valid_card_id(id)) {
            set_reason(reason, "invalid_card_id");
            return false;
        }
        if (is_level_card(id, level)) continue;
        const int face = face_id(id);
        const auto it = remaining.find(face);
        if (it == remaining.end() || it->second <= 0) {
            set_reason(reason, "claim_does_not_cover_action");
            return false;
        }
        --it->second;
    }
    for (const auto& item : remaining) {
        if (item.second <= 0) continue;
        if (item.first >= 52) {
            set_reason(reason, "joker_in_claim");
            return false;
        }
    }
    return true;
}

bool beats(const Move& previous, const Move& current, const std::string& level) {
    const PokerType before = classify(previous.claim);
    const PokerType after = classify(current.claim);
    if (!before.valid() || !after.valid() || before.pass() || after.pass()) return false;
    if (after.kind == PokerKind::Rocket) return true;
    if (before.kind == PokerKind::Rocket) return false;

    if (is_normal_kind(before.kind)) {
        if (is_normal_kind(after.kind)) {
            if (before.kind != after.kind) return false;
            if (is_scaled_kind(before.kind)) return after.key > before.key;
            return ordinary_key(after.key, level) > ordinary_key(before.key, level);
        }
        // Any bomb or five-card straight flush beats an ordinary type.
        return after.kind == PokerKind::Bomb || after.kind == PokerKind::StraightFlush;
    }

    if (before.kind == PokerKind::Bomb) {
        if (after.kind == PokerKind::Bomb) {
            if (after.length != before.length) return after.length > before.length;
            return ordinary_key(after.key, level) > ordinary_key(before.key, level);
        }
        if (after.kind == PokerKind::StraightFlush) return before.length < 6;
        return false;
    }

    if (before.kind == PokerKind::StraightFlush) {
        if (after.kind == PokerKind::Bomb) return after.length >= 6;
        if (after.kind == PokerKind::StraightFlush) return after.key > before.key;
    }
    return false;
}

bool validate_move(const Move& move,
                   const std::vector<int>& hand,
                   const std::string& level,
                   const std::optional<Move>& previous,
                   bool leading,
                   std::string* reason) {
    if (hand_has_duplicate_id(hand)) {
        set_reason(reason, "invalid_or_duplicate_hand");
        return false;
    }
    if (move.action.empty()) {
        if (!move.claim.empty()) {
            set_reason(reason, "pass_claim_nonempty");
            return false;
        }
        if (leading || !previous.has_value() || previous->action.empty()) {
            set_reason(reason, "invalid_pass");
            return false;
        }
        return true;
    }
    if (move.claim.empty()) {
        set_reason(reason, "nonpass_claim_empty");
        return false;
    }
    std::vector<int> remaining = hand;
    std::set<int> seen_action;
    for (const int id : move.action) {
        if (!valid_card_id(id)) {
            set_reason(reason, "invalid_card_id");
            return false;
        }
        if (!seen_action.insert(id).second) {
            set_reason(reason, "duplicate_action_card");
            return false;
        }
        const auto it = std::find(remaining.begin(), remaining.end(), id);
        if (it == remaining.end()) {
            set_reason(reason, "card_not_in_hand");
            return false;
        }
        remaining.erase(it);
    }
    if (!is_legal_claim(move.action, move.claim, level, reason)) return false;
    const PokerType type = classify(move.claim);
    if (!type.valid() || type.pass()) {
        set_reason(reason, "invalid_poker_type");
        return false;
    }
    if (!leading) {
        if (!previous.has_value() || previous->action.empty()) {
            set_reason(reason, "missing_previous_move");
            return false;
        }
        if (!beats(*previous, move, level)) {
            set_reason(reason, "move_not_bigger");
            return false;
        }
    }
    return true;
}

std::vector<Move> legal_moves(const std::vector<int>& hand,
                              const std::string& level,
                              const std::optional<Move>& previous,
                              bool leading) {
    std::vector<Move> result;
    std::set<MoveKey> seen;
    const std::string normalized_level = normalize_level(level);
    std::vector<int> wildcards;
    for (const int id : hand) {
        if (is_level_card(id, normalized_level)) wildcards.push_back(id);
    }

    auto add = [&](std::vector<int> action, std::vector<int> claim) {
        const auto add_one = [&](std::vector<int> raw_action, std::vector<int> raw_claim) {
            std::sort(raw_action.begin(), raw_action.end());
            std::sort(raw_claim.begin(), raw_claim.end());
            Move candidate{raw_action, raw_claim};
            if (validate_move(candidate, hand, normalized_level, previous, leading, nullptr)) {
                MoveKey key;
                for (const int id : candidate.action) key.push_back(face_id(id));
                std::sort(key.begin(), key.end());
                const PokerType type = classify(candidate.claim);
                key.push_back(kCardCount);
                key.push_back(static_cast<int>(type.kind));
                key.push_back(type.length);
                key.push_back(type.key);
                key.push_back(type.secondary);
                if (seen.insert(key).second) result.push_back(std::move(candidate));
            }
        };
        add_one(action, claim);

        // A wildcard's canonical suit is arbitrary.  If that choice happens
        // to make a mixed-suit straight appear flush, also emit a different
        // suit for the wildcard so ordinary-straight semantics are retained.
        if (std::any_of(action.begin(), action.end(), [&](int id) {
                return is_level_card(id, normalized_level);
            }) && classify(claim).kind == PokerKind::StraightFlush) {
            std::vector<int> alternate = claim;
            for (std::size_t i = 0; i < action.size() && i < alternate.size(); ++i) {
                if (!is_level_card(action[i], normalized_level)) continue;
                const int rank = rank_index(alternate[i]);
                const int suit = suit_index(alternate[i]);
                if (rank >= 0 && suit >= 0) {
                    alternate[i] = rank * 4 + ((suit + 1) % 4);
                }
                break;
            }
            add_one(action, alternate);
        }
    };

    auto add_targets = [&](const std::vector<Target>& targets) {
        std::set<int> used_cards;
        std::set<int> used_wildcards;
        std::vector<int> action;
        std::vector<int> claim;
        enumerate_targets(targets, 0U, hand, normalized_level, wildcards,
                          used_cards, used_wildcards, action, claim, add);
    };

    // Singles, pairs, threes and bombs (4..10 of one natural rank).
    for (int rank = 0; rank < 13; ++rank) {
        add_targets(repeated_targets(rank, 1));
        add_targets(repeated_targets(rank, 2));
        add_targets(repeated_targets(rank, 3));
        for (int length = 4; length <= 10; ++length) {
            add_targets(repeated_targets(rank, length));
        }
    }

    // Three with two. The pair may be a pair of jokers when those jokers are
    // physically present; wildcards themselves never claim a joker.
    for (int triple = 0; triple < 13; ++triple) {
        for (int pair = 0; pair < 15; ++pair) {
            if (triple == pair) continue;
            std::vector<Target> targets = repeated_targets(triple, 3);
            const std::vector<Target> pair_targets = repeated_targets(pair, 2);
            targets.insert(targets.end(), pair_targets.begin(), pair_targets.end());
            add_targets(targets);
        }
    }

    // Ordinary straights and same-suit straight flushes.
    for (int start = 0; start <= 9; ++start) {
        const std::vector<int> ranks = straight_ranks_from(start);
        std::vector<Target> any_suit;
        for (const int rank : ranks) any_suit.push_back(Target{rank, -1});
        add_targets(any_suit);
        for (int suit = 0; suit < 4; ++suit) {
            std::vector<Target> same_suit;
            for (const int rank : ranks) same_suit.push_back(Target{rank, suit});
            add_targets(same_suit);
        }
    }

    // Two consecutive triples: A-2, 2-K, and K-A.
    for (int start = 0; start < 13; ++start) {
        const std::vector<int> ranks = three_straight_ranks_from(start);
        if (ranks.size() != 2U) continue;
        std::vector<Target> targets;
        for (const int rank : ranks) {
            const std::vector<Target> part = repeated_targets(rank, 3);
            targets.insert(targets.end(), part.begin(), part.end());
        }
        add_targets(targets);
    }

    // Three consecutive pairs: A-2-3, 2-...-K, and Q-K-A.
    for (int start = 0; start < 12; ++start) {
        const std::vector<int> ranks = triple_pairs_ranks_from(start);
        if (ranks.size() != 3U) continue;
        std::vector<Target> targets;
        for (const int rank : ranks) {
            const std::vector<Target> part = repeated_targets(rank, 2);
            targets.insert(targets.end(), part.begin(), part.end());
        }
        add_targets(targets);
    }

    // Joker pairs and the four-joker rocket are not reachable through natural
    // rank templates.
    std::vector<int> small_jokers;
    std::vector<int> big_jokers;
    for (const int id : hand) {
        if (is_small_joker_id(id)) small_jokers.push_back(id);
        if (is_big_joker_id(id)) big_jokers.push_back(id);
    }
    for (const int id : small_jokers) add({id}, {id});
    for (const int id : big_jokers) add({id}, {id});
    if (small_jokers.size() >= 2U) {
        add({small_jokers[0], small_jokers[1]},
            {small_jokers[0], small_jokers[1]});
    }
    if (big_jokers.size() >= 2U) {
        add({big_jokers[0], big_jokers[1]},
            {big_jokers[0], big_jokers[1]});
    }
    if (small_jokers.size() >= 2U && big_jokers.size() >= 2U) {
        add({small_jokers[0], small_jokers[1], big_jokers[0], big_jokers[1]},
            {small_jokers[0], small_jokers[1], big_jokers[0], big_jokers[1]});
    }

    if (!leading && previous.has_value() && !previous->action.empty()) {
        result.push_back(Move{{}, {}});
    }
    std::sort(result.begin(), result.end(), [&](const Move& lhs, const Move& rhs) {
        const PokerType a = classify(lhs.claim);
        const PokerType b = classify(rhs.claim);
        if (a.kind != b.kind) return static_cast<int>(a.kind) < static_cast<int>(b.kind);
        if (a.length != b.length) return a.length < b.length;
        if (a.key != b.key) return a.key < b.key;
        return lhs.action < rhs.action;
    });
    return result;
}

bool valid_tribute_card(const std::vector<int>& hand,
                        int card,
                        const std::string& level,
                        std::string* reason) {
    if (hand_has_duplicate_id(hand)) {
        set_reason(reason, "invalid_or_duplicate_hand");
        return false;
    }
    if (!contains_id(hand, card)) {
        set_reason(reason, "card_not_in_hand");
        return false;
    }
    const std::string current = normalize_level(level);
    if (std::find_if(hand.begin(), hand.end(), is_big_joker_id) != hand.end()) {
        if (!is_big_joker_id(card)) set_reason(reason, "must_tribute_big_joker");
        return is_big_joker_id(card);
    }
    if (std::find_if(hand.begin(), hand.end(), is_small_joker_id) != hand.end()) {
        if (!is_small_joker_id(card)) set_reason(reason, "must_tribute_small_joker");
        return is_small_joker_id(card);
    }

    bool has_level = false;
    bool has_nonheart_level = false;
    for (const int id : hand) {
        if (rank_name(id) == current) {
            has_level = true;
            if (suit_index(id) != 0) has_nonheart_level = true;
        }
    }
    if (has_level && has_nonheart_level) {
        if (rank_name(card) != current) set_reason(reason, "must_tribute_level");
        return rank_name(card) == current;
    }

    int best = -1;
    for (const int id : hand) {
        if (is_joker_id(id) || (has_level && rank_name(id) == current)) continue;
        best = std::max(best, ordinary_key(rank_code_of(id), current));
    }
    const bool valid = best >= 0 && ordinary_key(rank_code_of(card), current) == best;
    if (!valid) set_reason(reason, "not_highest_tribute_card");
    return valid;
}

bool valid_return_card(const std::vector<int>& hand,
                       int card,
                       const std::string& level,
                       std::string* reason) {
    if (!contains_id(hand, card)) {
        set_reason(reason, "card_not_in_hand");
        return false;
    }
    const int rank = rank_code_of(card);
    const int current = normalize_rank_index(normalize_level(level));
    const bool valid = natural_rank(rank) && rank >= 1 && rank <= 9 && rank != current;
    if (!valid) set_reason(reason, "return_must_be_natural_2_to_10_and_not_level");
    return valid;
}

bool valid_return_card(const std::vector<int>& hand,
                       int card,
                       const std::string& level,
                       RuleVariant variant,
                       std::string* reason) {
    if (!contains_id(hand, card)) {
        set_reason(reason, "card_not_in_hand");
        return false;
    }
    const int rank = rank_code_of(card);
    const int current = normalize_rank_index(normalize_level(level));
    bool valid = natural_rank(rank) && rank >= 1 && rank <= 9 && rank != current;
    if (variant == RuleVariant::BotzoneCompat) {
        // The retrieved official judge uses pointorder.index('8') when the
        // level is 9 and pointorder.index('9') otherwise.  Since pointorder
        // starts at natural 2, this is exactly 2..8 for level 9 and 2..9 for
        // every other level, excluding the current level.
        const int official_max = current == normalize_rank_index("9") ?
            normalize_rank_index("8") : normalize_rank_index("9");
        valid = natural_rank(rank) && rank >= 1 && rank <= official_max && rank != current;
    }
    if (!valid) set_reason(reason, "return_not_accepted_by_variant");
    return valid;
}

}  // namespace oxbot
