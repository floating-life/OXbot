#include "oxbot/protocol.hpp"

#include <algorithm>
#include <cctype>

namespace oxbot {
namespace {
constexpr const char* kDefaultDiagnosticVersion = "oxbot-model-v1";

bool valid_diagnostic_version(const std::string& value) {
    if (value.empty() || value.size() > 96) return false;
    for (const unsigned char c : value) {
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}

std::string safe_candidate_version(const std::string& candidate) {
    return valid_diagnostic_version(candidate) ? candidate : kDefaultDiagnosticVersion;
}

std::string model_diagnostic(const ModelPolicy& policy) {
    std::string result = ";model_status=" + policy.status();
    if (!policy.payload_sha().empty()) result += ";model_sha=" + policy.payload_sha().substr(0, 12);
    result += ";selection_strategy=" + std::string(policy.selection_version());
    result += ";selection_source=" + policy.selection_source();
    return result;
}

}  // namespace

BotAdapter::BotAdapter(std::string model_path, std::string strategy, std::string candidate_version)
    : policy_(std::move(model_path), std::move(strategy)) {
    // A package-level candidate identity wins when explicitly supplied.  An
    // unversioned/legacy model otherwise keeps the historical v1 label, while
    // a newer model header can carry its own diagnostic identity.
    version_ = candidate_version.empty() ? policy_.diagnostic_version() : safe_candidate_version(candidate_version);
    if (!valid_diagnostic_version(version_)) version_ = kDefaultDiagnosticVersion;
}

json::Value BotAdapter::cards(const std::vector<int>& values) {
    json::Value result = json::Value::make_array();
    for (const int value : values) result.array.emplace_back(value);
    return result;
}

json::Value BotAdapter::response_for(const DecisionContext& context, const Move& move) {
    if (context.stage == DecisionContext::Stage::Deal) return json::Value::make_array();
    if (context.stage == DecisionContext::Stage::Tribute || context.stage == DecisionContext::Stage::Return) {
        if (move.action.empty()) return json::Value::make_array();
        json::Value result = json::Value::make_array();
        result.array.push_back(json::Value(move.action.front()));
        return result;
    }
    json::Value result = json::Value::make_array();
    result.array.push_back(cards(move.action));
    result.array.push_back(cards(move.claim));
    return result;
}

bool BotAdapter::legal_for_stage(const DecisionContext& context, const Move& move, std::string* reason) {
    if (context.stage == DecisionContext::Stage::Deal) return move.action.empty() && move.claim.empty();
    if (context.stage == DecisionContext::Stage::Tribute) {
        if (context.resist) return move.action.empty() && move.claim.empty();
        return move.action.size() == 1 && valid_tribute_card(context.hand, move.action.front(), context.level, reason);
    }
    if (context.stage == DecisionContext::Stage::Return) {
        if (context.resist) return move.action.empty() && move.claim.empty();
        return move.action.size() == 1 && valid_return_card(context.hand, move.action.front(), context.level, context.variant, reason);
    }
    return validate_move(move, context.hand, context.level, context.previous, context.leading, reason);
}

json::Value BotAdapter::decide(const json::Value& full_input) {
    StateMirror mirror;
    json::Value output = json::Value::make_object();
    std::string debug;
    if (!mirror.rebuild(full_input)) {
        // There is no generally legal move for an unknown hand or stage.
        // A null response deliberately fails closed instead of inventing a
        // pass/card and calling it a safe fallback.
        output["response"] = json::Value(nullptr);
        output["error"] = json::Value("state_rebuild_failed");
        debug = "version=" + version_ + ";status=fail_closed;state=" + mirror.error + model_diagnostic(policy_);
        output["debug"] = json::Value(debug.substr(0, 900));
        return output;
    }

    DecisionContext context = mirror.context;
    Move move = policy_.choose(context);
    std::string reason;
    std::string primary_rejection;
    bool legality_fallback = false;
    if (!legal_for_stage(context, move, &reason)) {
        primary_rejection = reason.empty() ? "stage_action_invalid" : reason;
        RulePolicy fallback;
        move = fallback.choose(context);
        legality_fallback = true;
        reason.clear();
        if (!legal_for_stage(context, move, &reason)) {
            output["response"] = json::Value(nullptr);
            output["error"] = json::Value("no_validated_action");
            debug = "version=" + version_ + ";status=fail_closed;policy=none;legality_fallback=1;stage=" + stage_name(context.stage) +
                    model_diagnostic(policy_) + ";primary_check=" + primary_rejection;
            if (!reason.empty()) debug += ";check=" + reason;
            output["debug"] = json::Value(debug.substr(0, 900));
            return output;
        }
    }
    output["response"] = response_for(context, move);
    const char* selected_policy = context.stage != DecisionContext::Stage::Play ? "stage_rules" :
        (policy_.used_model() && !legality_fallback ? "model" : "rule_fallback");
    debug = "version=" + version_ + ";policy=" + selected_policy +
            ";stage=" + stage_name(context.stage) + model_diagnostic(policy_) +
            ";legality_fallback=" + (legality_fallback ? "1" : "0");
    if (!primary_rejection.empty()) debug += ";primary_check=" + primary_rejection;
    if (!reason.empty()) debug += ";check=" + reason;
    output["debug"] = json::Value(debug.substr(0, 900));
    return output;
}

}  // namespace oxbot
