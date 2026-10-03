#pragma once

#include "oxbot/mini_json.hpp"
#include "oxbot/policy.hpp"
#include "oxbot/state.hpp"

#include <string>

namespace oxbot {

class BotAdapter {
public:
    // Empty strategy accepts the model header manifest default.  A supplied
    // name is an explicit command-line/package experiment override.
    // An optional candidate version is an explicit package identity override;
    // otherwise the version is derived from the loaded model header.
    explicit BotAdapter(std::string model_path = {}, std::string strategy = {},
                        std::string candidate_version = {});
    json::Value decide(const json::Value& full_input);
    const std::string& diagnostic_version() const { return version_; }

private:
    ModelPolicy policy_;
    // Set by the constructor from the model header or explicit package
    // candidate identity; an empty value is never emitted.
    std::string version_;

    static json::Value cards(const std::vector<int>& values);
    static json::Value response_for(const DecisionContext& context, const Move& move);
    static bool legal_for_stage(const DecisionContext& context, const Move& move, std::string* reason);
};

}  // namespace oxbot
