#include "oxbot/policy.hpp"
#include "oxbot/features.hpp"
#include "oxbot/fabledan_features.hpp"
#include "oxbot/fabledan_candidates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace oxbot {

SelectionStrategy parse_selection_strategy(const std::string& name) {
    if (name.empty() || name == "raw") return SelectionStrategy::Raw;
    if (name == "raw-pass-bias") return SelectionStrategy::RawPassBias;
    if (name == "group-logmeanexp") return SelectionStrategy::GroupLogMeanExp;
    throw std::invalid_argument("unknown_selection_strategy:" + name);
}

const char* selection_strategy_name(SelectionStrategy strategy) {
    if (strategy == SelectionStrategy::RawPassBias) return "raw-pass-bias";
    return strategy == SelectionStrategy::GroupLogMeanExp ? "group-logmeanexp" : "raw";
}

const char* selection_strategy_version(SelectionStrategy strategy) {
    if (strategy == SelectionStrategy::RawPassBias) return "raw-pass-bias-v1";
    return strategy == SelectionStrategy::GroupLogMeanExp ? "group-logmeanexp-v1" : "raw-v1";
}

namespace {

// This descriptor mirrors tools/analyze_candidate_groups.py.  It collapses
// equivalent concrete deck assignments while retaining straight-flush suits.
struct CandidateGroupKey {
    int kind = -1;
    int length = 0;
    int key = -1;
    int secondary = -1;
    std::array<int, 15> rank_counts{};
    int wild_count = 0;
    bool straight_flush = false;
    std::array<int, 54> suit_signature{};

    bool operator<(const CandidateGroupKey& other) const {
        if (kind != other.kind) return kind < other.kind;
        if (length != other.length) return length < other.length;
        if (key != other.key) return key < other.key;
        if (secondary != other.secondary) return secondary < other.secondary;
        if (rank_counts != other.rank_counts) return rank_counts < other.rank_counts;
        if (wild_count != other.wild_count) return wild_count < other.wild_count;
        if (straight_flush != other.straight_flush) return straight_flush < other.straight_flush;
        return suit_signature < other.suit_signature;
    }
};

CandidateGroupKey group_key(const Move& move, const PokerType& type,
                            const std::string& level) {
    CandidateGroupKey result;
    result.kind = static_cast<int>(type.kind);
    result.length = static_cast<int>(move.action.size());
    result.key = type.key;
    result.secondary = type.secondary;
    result.straight_flush = type.kind == PokerKind::StraightFlush;
    for (const int card : move.action) {
        const int face = face_id(card);
        if (face < 0 || face >= kCardsPerDeck) throw std::runtime_error("group_card_invalid");
        ++result.suit_signature[static_cast<std::size_t>(face)];
        const int rank = face < 52 ? face / 4 : 13 + face - 52;
        ++result.rank_counts[static_cast<std::size_t>(rank)];
    }
    const int level_index = normalize_rank_index(normalize_level(level));
    if (level_index >= 0) result.wild_count = result.suit_signature[static_cast<std::size_t>(level_index * 4)];
    if (!result.straight_flush) result.suit_signature.fill(0);
    return result;
}

double log_mean_exp(const std::vector<double>& scores) {
    if (scores.empty()) throw std::runtime_error("empty_selection_group");
    const double largest = *std::max_element(scores.begin(), scores.end());
    double total = 0.0;
    for (const double score : scores) total += std::exp(score - largest);
    return largest + std::log(total / static_cast<double>(scores.size()));
}

std::size_t select_candidate(const std::vector<Move>& moves, const std::vector<float>& scores,
                             const DecisionContext& context, SelectionStrategy strategy) {
    if (moves.empty() || moves.size() != scores.size()) throw std::runtime_error("selection_shape_invalid");
    if (strategy == SelectionStrategy::Raw || strategy == SelectionStrategy::RawPassBias) {
        // Validation-only calibration experiment frozen as v1: following a
        // trick, reduce the unique pass candidate by 0.5 logit.  Leading
        // states have no legal pass, so this branch is a no-op there.  The
        // default release strategy remains Raw.
        constexpr double pass_bias = -0.5;
        std::size_t best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < scores.size(); ++index) {
            double score = static_cast<double>(scores[index]);
            if (strategy == SelectionStrategy::RawPassBias && moves[index].pass()) score += pass_bias;
            if (score > best_score) {
                best_score = score;
                best = index;
            }
        }
        return best;
    }
    struct Group { std::vector<std::size_t> members; double score = 0.0; };
    // Keep first-seen order to match Python's defaultdict/np.argmax tie rule.
    std::map<CandidateGroupKey, std::size_t> indexes;
    std::vector<Group> groups;
    for (std::size_t index = 0; index < moves.size(); ++index) {
        const CandidateGroupKey key = group_key(moves[index], classify(moves[index].claim), context.level);
        const auto found = indexes.find(key);
        std::size_t group = 0;
        if (found == indexes.end()) {
            group = groups.size();
            indexes.emplace(key, group);
            groups.emplace_back();
        } else {
            group = found->second;
        }
        groups[group].members.push_back(index);
    }
    std::size_t best_group = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t group = 0; group < groups.size(); ++group) {
        std::vector<double> values;
        values.reserve(groups[group].members.size());
        for (const std::size_t member : groups[group].members) values.push_back(static_cast<double>(scores[member]));
        groups[group].score = log_mean_exp(values);
        if (groups[group].score > best_score) {
            best_score = groups[group].score;
            best_group = group;
        }
    }
    std::size_t best_member = groups[best_group].members.front();
    for (const std::size_t member : groups[best_group].members) {
        if (scores[member] > scores[best_member]) best_member = member;
    }
    return best_member;
}

bool has_magic(const std::string& path, const char* magic) {
    if (path.empty() || path == ":embedded:") return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    char bytes[8]{};
    input.read(bytes, sizeof(bytes));
    return input.gcount() == static_cast<std::streamsize>(sizeof(bytes)) &&
           std::equal(bytes, bytes + sizeof(bytes), magic);
}

}  // namespace

Move RulePolicy::choose(const DecisionContext& context) {
    if (context.stage == DecisionContext::Stage::Deal) return Move{{}, {}};
    if (context.stage == DecisionContext::Stage::Tribute) {
        if (context.resist) return Move{{}, {}};
        for (const int card : context.hand) {
            if (!is_level_card(card, context.level) && valid_tribute_card(context.hand, card, context.level, nullptr)) {
                return Move{{card}, {card}};
            }
        }
        return Move{{}, {}};
    }
    if (context.stage == DecisionContext::Stage::Return) {
        if (context.resist) return Move{{}, {}};
        // The online referee's return boundary is stricter than the attached
        // corrected profile: use the explicit official compatibility variant
        // so a level-9/10 response cannot be rejected on BotZone.
        for (const int card : context.hand) {
            if (valid_return_card(context.hand, card, context.level,
                                  RuleVariant::BotzoneCompat, nullptr)) {
                return Move{{card}, {card}};
            }
        }
        return Move{{}, {}};
    }

    const auto moves = legal_moves(context.hand, context.level, context.previous, context.leading);
    std::array<int, 13> hand_ranks{};
    for (int card : context.hand) if (rank_index(card) >= 0) ++hand_ranks[static_cast<std::size_t>(rank_index(card))];
    const Move* best = nullptr;
    double best_cost = std::numeric_limits<double>::infinity();
    for (const auto& move : moves) {
        if (move.pass()) continue;
        if (move.action.size() == context.hand.size()) return move;
        const PokerType type = classify(move.claim);
        double cost = -3.0 * static_cast<double>(move.action.size());
        const bool bomb = type.kind == PokerKind::Bomb || type.kind == PokerKind::Rocket || type.kind == PokerKind::StraightFlush;
        if (bomb) cost += 16;
        for (int card : move.action) {
            const int rank = rank_index(card);
            if (is_level_card(card, context.level)) cost += 2.5;
            if (is_joker_id(card)) cost += 3.0;
            if (rank >= 0) {
                cost += .04 * (rank == 0 ? 13 : rank);
                if (!bomb && hand_ranks[static_cast<std::size_t>(rank)] >= 4) cost += 2.0;
            }
        }
        if (cost < best_cost) { best = &move; best_cost = cost; }
    }
    if (best) return *best;
    if (!context.leading && context.previous.has_value()) return Move{{}, {}};
    if (!context.hand.empty()) return Move{{context.hand.front()}, {context.hand.front()}};
    return Move{{}, {}};
}

ModelPolicy::ModelPolicy(std::string model_path, std::string strategy) {
    if (model_path.empty()) {
        status_ = "model_not_configured";
        if (!strategy.empty()) {
            selection_strategy_ = parse_selection_strategy(strategy);
            selection_source_ = "command";
        }
        return;
    }
    if (has_magic(model_path, "FBDN001") || has_magic(model_path, "FBDN002")) {
        fabledan_backend_ = true;
        ready_ = fabledan_network_.load(model_path);
        status_ = fabledan_network_.status();
        if (ready_ && fabledan_network_.feature_dim() == 224) {
            fabledan_version_ = "fabledan-cpp-v2-structure";
        }
    } else {
        ready_ = network_.load(model_path);
        status_ = network_.status();
    }
    if (!strategy.empty()) {
        selection_strategy_ = parse_selection_strategy(strategy);
        selection_source_ = "command";
    } else if (ready_ && !fabledan_backend_) {
        selection_strategy_ = parse_selection_strategy(network_.selection_strategy());
        selection_source_ = network_.has_selection_manifest() ? "manifest" : "default";
    }
}

Move ModelPolicy::choose(const DecisionContext& context) {
    used_model_ = false;
    if (context.stage != DecisionContext::Stage::Play) return fallback_.choose(context);
    if (ready_) {
        try {
            const auto moves = fabledan_backend_ && fabledan_network_.feature_dim() == 224
                ? fabledan_structure_candidates(context)
                : legal_moves(context.hand, context.level, context.previous, context.leading);
            if (moves.empty()) throw std::runtime_error("no_legal_candidates");
            std::vector<Move> candidates;
            std::vector<std::vector<float>> actions;
            candidates.reserve(moves.size());
            actions.reserve(moves.size());
            if (fabledan_backend_ && fabledan_network_.feature_dim() == 224) {
                candidates = moves;
                actions = encode_fabledan_actions(candidates, context, 224);
            } else if (fabledan_backend_) {
                // FableDan training collapses realization variants with
                // identical 80-dim rows before scoring.  C++ legal_moves
                // deliberately retains more physical realizations for the
                // referee, so perform the same feature-level canonicalization
                // while retaining the first legal physical move.
                // First collapse all physical realizations that have the
                // same FableDan semantic row *except* wildcard usage.  The
                // Python generator always consumes natural cards before
                // wildcards, whereas rules::legal_moves intentionally emits
                // both realizations for referee validation.  Retain the
                // least-wildcard realization, then the first deterministic
                // suit/deck representative.
                std::map<std::vector<float>, std::size_t> semantic_rows;
                for (const auto& move : moves) {
                    if (!fabledan_generator_candidate(move, context)) continue;
                    auto row = encode_fabledan_action(move, context);
                    auto semantic = row;
                    semantic[65] = 0.f;  // action wildcard count
                    const auto found = semantic_rows.find(semantic);
                    if (found == semantic_rows.end()) {
                        semantic_rows.emplace(std::move(semantic), candidates.size());
                        candidates.push_back(move);
                        actions.push_back(std::move(row));
                    } else {
                        const std::size_t index = found->second;
                        if (row[65] < actions[index][65]) {
                            candidates[index] = move;
                            actions[index] = std::move(row);
                        }
                    }
                }
            } else {
                candidates = moves;
                for (const auto& move : moves) actions.push_back(encode_action(move, context));
            }
            const auto scores = fabledan_backend_
                ? fabledan_network_.score(encode_fabledan_history(context), actions)
                : network_.score(encode_history(context), encode_state(context), actions);
            if (scores.size() != candidates.size()) throw std::runtime_error("model_score_count_mismatch");
            for (float score : scores) if (!std::isfinite(score)) throw std::runtime_error("model_score_nonfinite");
            const auto best = select_candidate(candidates, scores, context, selection_strategy_);
            std::string reason;
            if (!validate_move(candidates[best], context.hand, context.level, context.previous, context.leading, &reason)) {
                throw std::runtime_error("model_action_invalid:" + reason);
            }
            used_model_ = true;
            status_ = "model_selected";
            return candidates[best];
        } catch (const std::exception& error) {
            status_ = std::string("model_fallback:") + error.what();
        }
    }
    return fallback_.choose(context);
}

}  // namespace oxbot
