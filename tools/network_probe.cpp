#include "oxbot/mini_json.hpp"
#include "oxbot/network.hpp"
#include <chrono>
#include <iostream>

int main() {
    std::string line;
    while (std::getline(std::cin,line)) {
        oxbot::json::Value out = oxbot::json::Value::make_object();
        try {
            const auto in = oxbot::json::parse(line);
            oxbot::CandidateNetwork network;
            out["ok"] = oxbot::json::Value(network.load(in["path"].as_string()));
            out["status"] = oxbot::json::Value(network.status());
            out["sha"] = oxbot::json::Value(network.payload_sha());
            if (network.ready() && in.contains("tokens")) {
                std::vector<int> tokens;
                std::vector<float> state;
                std::vector<std::vector<float>> actions;
                for (const auto& t : in["tokens"].array) tokens.push_back(t.as_int(-1));
                for (const auto& v : in["state"].array) state.push_back(static_cast<float>(v.number));
                for (const auto& row : in["actions"].array) {
                    std::vector<float> action;
                    for (const auto& v : row.array) action.push_back(static_cast<float>(v.number));
                    actions.push_back(std::move(action));
                }
                const auto start = std::chrono::steady_clock::now();
                std::map<std::string, std::vector<float>> trace;
                const auto scores = network.score(tokens,state,actions,in["trace"].as_bool() ? &trace : nullptr);
                out["milliseconds"] = oxbot::json::Value(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                out["scores"] = oxbot::json::Value::make_array();
                for (float v : scores) out["scores"].push_back(oxbot::json::Value(static_cast<double>(v)));
                if (!trace.empty()) {
                    out["trace"] = oxbot::json::Value::make_object();
                    for (const auto& item : trace) {
                        out["trace"][item.first] = oxbot::json::Value::make_array();
                        for (float v : item.second) out["trace"][item.first].push_back(oxbot::json::Value(static_cast<double>(v)));
                    }
                }
            }
        } catch (const std::exception& ex) {
            out["error"] = oxbot::json::Value(ex.what());
        }
        std::cout << oxbot::json::dump(out) << '\n' << std::flush;
    }
}
