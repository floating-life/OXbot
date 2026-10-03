#pragma once

#include "oxbot/card.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace oxbot {

enum class RuleVariant { BotzoneCompat, Standard };

enum class PokerKind {
    Pass,
    Invalid,
    Single,
    Pair,
    Three,
    Straight,
    Set,
    ThreeStraight,
    TriplePairs,
    Bomb,
    StraightFlush,
    Rocket,
};

struct PokerType {
    PokerKind kind = PokerKind::Invalid;
    int length = 0;
    // Natural cardscale rank (A=0, 2=1, ..., K=12, jo=13, jO=14).
    // Sequence keys are their lowest rank, including the A-low and A-high
    // forms; a bomb stores its rank here and its size in length.
    int key = -1;
    int secondary = -1; // set's pair rank, otherwise unused

    bool valid() const { return kind != PokerKind::Invalid; }
    bool pass() const { return kind == PokerKind::Pass; }
};

struct Move {
    std::vector<int> action;
    std::vector<int> claim;

    bool pass() const { return action.empty(); }
};

// A public play reconstructed from BotZone history.  The owner is unknown
// for an empty history slot; retaining that distinction lets the state layer
// keep protocol evidence without treating it as a hand-ownership fact.
struct PublicEvent {
    int player = -1;
    Move move;
    bool has_player = false;
    std::string stage = "play";
    int target = -1;
};

struct DecisionContext {
    enum class Stage { Deal, Tribute, Return, Play };

    Stage stage = Stage::Play;
    std::vector<int> hand;
    std::string level = "2";
    RuleVariant variant = RuleVariant::BotzoneCompat;
    bool resist = false;
    bool leading = true;
    std::optional<Move> previous;

    // Optional public information populated by StateMirror.  These fields
    // are deliberately additive so existing policies and tests can continue
    // using the small P0 context.
    int player_id = -1;
    std::array<int, 4> remaining_counts{{-1, -1, -1, -1}};
    std::vector<PublicEvent> public_events;
    std::vector<int> done;
    int pass_on = -1;
    int tribute = 0;
};

std::string stage_name(DecisionContext::Stage stage);
PokerType classify(const std::vector<int>& claim);

// Mirrors the supplied BotZone comparison rules (including its unconditional
// acceptance of a current rocket). A false result is also returned for
// different ordinary types; callers can inspect validate_move for a reason.
bool beats(const Move& previous, const Move& current, const std::string& level);

bool is_legal_claim(const std::vector<int>& action,
                    const std::vector<int>& claim,
                    const std::string& level,
                    std::string* reason = nullptr);

bool validate_move(const Move& move,
                   const std::vector<int>& hand,
                   const std::string& level,
                   const std::optional<Move>& previous,
                   bool leading,
                   std::string* reason = nullptr);

std::vector<Move> legal_moves(const std::vector<int>& hand,
                              const std::string& level,
                              const std::optional<Move>& previous,
                              bool leading);

bool valid_tribute_card(const std::vector<int>& hand,
                        int card,
                        const std::string& level,
                        std::string* reason = nullptr);
bool valid_return_card(const std::vector<int>& hand,
                       int card,
                       const std::string& level,
                       std::string* reason = nullptr);

// Explicit variant overload.  The four-argument legacy overload remains a
// conservative natural 2..10 predicate for offline callers.  BotzoneCompat
// mirrors the currently retrieved official BotZone referee: natural 2..9,
// with level 9 using the stricter 2..8 boundary, and always excluding the
// current level.  A strategy must use this overload for online compatibility.
bool valid_return_card(const std::vector<int>& hand,
                       int card,
                       const std::string& level,
                       RuleVariant variant,
                       std::string* reason = nullptr);

}  // namespace oxbot
