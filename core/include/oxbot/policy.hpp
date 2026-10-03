#pragma once

#include "oxbot/mini_json.hpp"
#include "oxbot/rules.hpp"
#include "oxbot/network.hpp"
#include "oxbot/fabledan_network.hpp"

#include <string>

namespace oxbot {

enum class SelectionStrategy { Raw, RawPassBias, GroupLogMeanExp };

// Canonical command-line/manifest names.  The parser throws on an unknown
// name so a release cannot silently run an unreviewed selection policy.
SelectionStrategy parse_selection_strategy(const std::string& name);
const char* selection_strategy_name(SelectionStrategy strategy);
const char* selection_strategy_version(SelectionStrategy strategy);

class Policy {
public:
    virtual ~Policy() = default;
    virtual Move choose(const DecisionContext& context) = 0;
};

class RulePolicy final : public Policy {
public:
    Move choose(const DecisionContext& context) override;
};

class ModelPolicy final : public Policy {
public:
    // An empty strategy accepts the model-header manifest default.  A
    // non-empty strategy is an explicit command-line/package override.
    explicit ModelPolicy(std::string model_path = {}, std::string strategy = {});
    Move choose(const DecisionContext& context) override;
    bool ready() const { return ready_; }
    bool used_model() const { return used_model_; }
    const std::string& status() const { return status_; }
    const std::string& payload_sha() const {
        return fabledan_backend_ ? fabledan_network_.payload_sha() : network_.payload_sha();
    }
    SelectionStrategy selection_strategy() const { return selection_strategy_; }
    const std::string& selection_source() const { return selection_source_; }
    const std::string& diagnostic_version() const {
        return fabledan_backend_ ? fabledan_version_ : network_.model_version();
    }
    const char* selection_name() const { return selection_strategy_name(selection_strategy_); }
    const char* selection_version() const { return selection_strategy_version(selection_strategy_); }

private:
    bool ready_ = false;
    bool used_model_ = false;
    std::string status_;
    CandidateNetwork network_;
    FableDanNetwork fabledan_network_;
    bool fabledan_backend_ = false;
    std::string fabledan_version_ = "fabledan-cpp-v1";
    RulePolicy fallback_;
    SelectionStrategy selection_strategy_ = SelectionStrategy::Raw;
    std::string selection_source_ = "default";
};

}  // namespace oxbot
