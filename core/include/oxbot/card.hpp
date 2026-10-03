#pragma once

#include <array>
#include <string>

namespace oxbot {

constexpr int kCardsPerDeck = 54;
constexpr int kDeckCount = 2;
constexpr int kCardCount = kCardsPerDeck * kDeckCount;

inline bool valid_card_id(int id) { return id >= 0 && id < kCardCount; }
inline int deck_offset(int id) { return id / kCardsPerDeck; }
inline int deck_index(int id) { return id % kCardsPerDeck; }
// The referee numbers the two physical decks consecutively.  A claim is a
// description of a card face rather than an owned physical card, so callers
// that compare action/claim multisets should use this canonical face id.
inline int face_id(int id) { return valid_card_id(id) ? id % kCardsPerDeck : -1; }
inline bool same_face(int lhs, int rhs) {
    return valid_card_id(lhs) && valid_card_id(rhs) && face_id(lhs) == face_id(rhs);
}
inline bool is_joker_id(int id) { return valid_card_id(id) && deck_index(id) >= 52; }
inline bool is_big_joker_id(int id) { return valid_card_id(id) && deck_index(id) == 53; }
inline bool is_small_joker_id(int id) { return valid_card_id(id) && deck_index(id) == 52; }
inline int suit_index(int id) { return valid_card_id(id) && deck_index(id) < 52 ? deck_index(id) % 4 : -1; }
inline int rank_index(int id) { return valid_card_id(id) && deck_index(id) < 52 ? deck_index(id) / 4 : -1; }

inline const std::array<const char*, 13>& natural_ranks() {
    static const std::array<const char*, 13> ranks =
        {{"A", "2", "3", "4", "5", "6", "7", "8", "9", "0", "J", "Q", "K"}};
    return ranks;
}
inline const std::array<const char*, 4>& suits() {
    static const std::array<const char*, 4> names = {{"h", "d", "s", "c"}};
    return names;
}

inline std::string rank_name(int id) {
    if (is_small_joker_id(id)) return "o";
    if (is_big_joker_id(id)) return "O";
    const int r = rank_index(id);
    return r >= 0 ? natural_ranks()[static_cast<std::size_t>(r)] : "?";
}
inline std::string card_name(int id) {
    if (is_small_joker_id(id)) return "jo";
    if (is_big_joker_id(id)) return "jO";
    const int s = suit_index(id);
    const int r = rank_index(id);
    if (s < 0 || r < 0) return "?";
    return std::string(suits()[static_cast<std::size_t>(s)]) + natural_ranks()[static_cast<std::size_t>(r)];
}

inline int normalize_rank_index(const std::string& rank) {
    if (rank == "10") return 9;
    for (int i = 0; i < 13; ++i) {
        if (rank == natural_ranks()[static_cast<std::size_t>(i)]) return i;
    }
    return -1;
}

inline std::string normalize_level(const std::string& level) {
    // BotZone writes ten as `0`; the Python FableDan tools also accept the
    // human-facing aliases `10` and `T`.  Normalize all three while keeping
    // the canonical rank table unchanged.
    return (level == "10" || level == "T" || level == "t") ? "0" : level;
}

inline bool is_level_card(int id, const std::string& level) {
    // The heart card is the wildcard in the BotZone referee's claim check.
    return valid_card_id(id) && suit_index(id) == 0 && rank_name(id) == normalize_level(level);
}

inline bool is_natural_id(int id) {
    return valid_card_id(id) && deck_index(id) < 52;
}

}  // namespace oxbot
