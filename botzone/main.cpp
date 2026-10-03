#include "oxbot/mini_json.hpp"
#include "oxbot/protocol.hpp"

#include <iostream>
#include <string>

namespace {

// BotZone starts a keep-running process before sending the first deal request.
// Touching the embedded model in the constructor can consume that first-turn
// deadline on a slower platform even though deal itself never needs inference.
// Identify only a well-formed request envelope here; BotAdapter still performs
// the authoritative state/legality validation below.
bool is_deal_request(const oxbot::json::Value& input) {
    const oxbot::json::Value* requests = input.find("requests");
    if (requests == nullptr || !requests->is_array() || requests->array.empty()) return false;
    const oxbot::json::Value& request = requests->array.back();
    return request.is_object() && request["stage"].as_string() == "deal";
}

void write_output(const oxbot::json::Value& output) {
    // BotZone's keep-running mode uses an out-of-band line after each JSON
    // response to distinguish a live stdin session from a one-shot process.
    // Without this marker the referee treats the child as finished after the
    // first deal, then the next request can be reported as a 2-second TLE
    // even though the JSON response itself was produced.
    std::cout << oxbot::json::dump(output) << '\n';
    std::cout << ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n" << std::flush;
}

struct SessionHistory {
    bool initialized = false;
    oxbot::json::Value requests = oxbot::json::Value::make_array();
    oxbot::json::Value responses = oxbot::json::Value::make_array();
};

// In keep-running mode BotZone sends the first round as the usual envelope,
// then sends only the next request object.  StateMirror deliberately accepts
// the complete replay envelope, so rebuild that envelope inside the process
// instead of treating the incremental request as malformed JSON.
bool capture_envelope(const oxbot::json::Value& input, SessionHistory* history) {
    if (!input.is_object()) return false;
    const auto* requests = input.find("requests");
    const auto* responses = input.find("responses");
    if (!requests || !responses || !requests->is_array() || !responses->is_array() ||
        requests->array.empty() || requests->array.size() != responses->array.size() + 1) {
        return false;
    }
    history->requests = *requests;
    history->responses = *responses;
    history->initialized = true;
    return true;
}

bool append_incremental_request(const oxbot::json::Value& input, SessionHistory* history,
                                oxbot::json::Value* full_input) {
    if (!history->initialized || !input.is_object() || input.contains("requests") ||
        !input.contains("stage")) return false;
    history->requests.array.push_back(input);
    *full_input = oxbot::json::Value::make_object();
    (*full_input)["requests"] = history->requests;
    (*full_input)["responses"] = history->responses;
    return true;
}

void remember_response(SessionHistory* history, const oxbot::json::Value& output) {
    if (!history->initialized) return;
    const auto* response = output.find("response");
    if (response) history->responses.array.push_back(*response);
}

}  // namespace

#ifndef OXBOT_MODEL_PATH
#define OXBOT_MODEL_PATH "models/oxbot-bc-v1.bin"
#endif
#ifndef OXBOT_POLICY_STRATEGY
#define OXBOT_POLICY_STRATEGY ""
#endif
#ifndef OXBOT_CANDIDATE_VERSION
// Empty means derive the diagnostic identity from the model header.  The
// historical v1 fallback is applied inside BotAdapter and in this process'
// parse-error path below.
#define OXBOT_CANDIDATE_VERSION ""
#endif

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string model_path = OXBOT_MODEL_PATH;
    std::string strategy = OXBOT_POLICY_STRATEGY;
    // Offline/package harness overrides. BotZone invokes the program
    // without args, so the compiled package default remains authoritative.
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string option = argv[index];
            if ((option == "--model" || option == "--strategy") && index + 1 < argc) {
                const std::string value = argv[++index];
                if (option == "--model") model_path = value;
                else strategy = value;
            } else {
                throw std::runtime_error("invalid_arguments");
            }
        }
    } catch (const std::exception& error) {
        // Invalid command-line arguments are a process-level failure.  Keep
        // the historical one-line diagnostic so offline harnesses can report
        // the cause without attempting to construct a policy.
        oxbot::json::Value output = oxbot::json::Value::make_object();
        output["response"] = oxbot::json::Value(nullptr);
        output["error"] = oxbot::json::Value("parse_or_decision_error");
        std::string version = OXBOT_CANDIDATE_VERSION;
        if (version.empty()) version = "oxbot-model-v1";
        std::string message = "version=" + version + ";status=fail_closed;error=" + std::string(error.what());
        if (message.size() > 900) message.resize(900);
        output["debug"] = oxbot::json::Value(message);
        std::cout << oxbot::json::dump(output) << '\n' << std::flush;
        return 0;
    }

    std::string line;
    if (!std::getline(std::cin, line)) return 0;

    // The deal response is deterministic and does not inspect model weights.
    // Answer it before constructing the embedded ModelPolicy, then load the
    // model while the referee is preparing the first play request.  This
    // removes model I/O/checksum/tensor setup from BotZone's doubled 2-second
    // first-turn budget without changing any play/tribute/return behavior.
    SessionHistory session;
    bool first_line_was_deal = false;
    try {
        const oxbot::json::Value first_input = oxbot::json::parse(line);
        capture_envelope(first_input, &session);
        if (is_deal_request(first_input)) {
            oxbot::BotAdapter preflight({}, strategy, OXBOT_CANDIDATE_VERSION);
            oxbot::json::Value output = preflight.decide(first_input);
            std::string debug = output["debug"].as_string();
            const std::string unloaded = ";model_status=model_not_configured";
            const std::size_t marker = debug.find(unloaded);
            if (marker != std::string::npos) {
                debug.replace(marker, unloaded.size(), ";model_status=deferred");
                output["debug"] = oxbot::json::Value(debug);
            }
            write_output(output);
            remember_response(&session, output);
            first_line_was_deal = true;
        }
    } catch (const std::exception&) {
        // Let the normal per-line handler below report parse/state errors.
    }

    // BotZone's long-running mode keeps stdin open and sends one JSON request
    // per line.  Construct the policy once (including model loading), then
    // answer every line while preserving the same model/strategy identity.
    oxbot::BotAdapter adapter(model_path, strategy, OXBOT_CANDIDATE_VERSION);
    auto process_line = [&](const std::string& request_line) {
        oxbot::json::Value output = oxbot::json::Value::make_object();
        try {
            const oxbot::json::Value input = oxbot::json::parse(request_line);
            oxbot::json::Value full_input = input;
            bool replayed_session = false;
            if (capture_envelope(input, &session)) {
                replayed_session = true;
            } else if (append_incremental_request(input, &session, &full_input)) {
                replayed_session = true;
            }
            output = adapter.decide(full_input);
            if (replayed_session) remember_response(&session, output);
        } catch (const std::exception& error) {
            output["response"] = oxbot::json::Value(nullptr);
            output["error"] = oxbot::json::Value("parse_or_decision_error");
            std::string version = adapter.diagnostic_version();
            if (version.empty()) version = "oxbot-model-v1";
            std::string message = "version=" + version + ";status=fail_closed;error=" + std::string(error.what());
            if (message.size() > 900) message.resize(900);
            output["debug"] = oxbot::json::Value(message);
        }
        write_output(output);
    };
    if (!first_line_was_deal) process_line(line);
    while (std::getline(std::cin, line)) {
        process_line(line);
    }
    return 0;
}
