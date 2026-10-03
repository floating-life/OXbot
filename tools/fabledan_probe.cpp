#include "oxbot/fabledan_network.hpp"
#include "oxbot/mini_json.hpp"

#include <chrono>
#include <iostream>
#include <vector>

// Small line-oriented probe used by the exporter/parity checks.  It keeps the
// FBDN backend independent from the Botzone policy and mirrors tools/
// network_probe.cpp's JSON shape:
//   {"path":"...","tokens":[...],"features":[[...],...]}
int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        oxbot::json::Value out = oxbot::json::Value::make_object();
        try {
            const auto input = oxbot::json::parse(line);
            oxbot::FableDanNetwork network;
            out["ok"] = oxbot::json::Value(network.load(input["path"].as_string()));
            out["status"] = oxbot::json::Value(network.status());
            out["sha"] = oxbot::json::Value(network.payload_sha());
            if (network.ready()) out["feature_dim"] = oxbot::json::Value(network.feature_dim());
            if (network.ready() && input.contains("tokens")) {
                if (!input["tokens"].is_array() || !input["features"].is_array()) {
                    throw std::runtime_error("probe_input_shape_invalid");
                }
                std::vector<int> tokens;
                for (const auto& value : input["tokens"].array) {
                    const int token = value.as_int(-1);
                    if (token < 0) throw std::runtime_error("probe_token_invalid");
                    tokens.push_back(token);
                }
                std::vector<std::vector<float>> features;
                for (const auto& row : input["features"].array) {
                    if (!row.is_array()) throw std::runtime_error("probe_feature_row_invalid");
                    std::vector<float> values;
                    values.reserve(row.array.size());
                    for (const auto& value : row.array) {
                        if (!value.is_number()) throw std::runtime_error("probe_feature_invalid");
                        values.push_back(static_cast<float>(value.number));
                    }
                    features.push_back(std::move(values));
                }
                const auto start = std::chrono::steady_clock::now();
                const auto scores = network.score(tokens, features);
                out["milliseconds"] = oxbot::json::Value(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count());
                out["scores"] = oxbot::json::Value::make_array();
                for (const float value : scores) {
                    out["scores"].push_back(
                        oxbot::json::Value(static_cast<double>(value)));
                }
            }
        } catch (const std::exception& error) {
            out["error"] = oxbot::json::Value(error.what());
        }
        std::cout << oxbot::json::dump(out) << '\n' << std::flush;
    }
    return 0;
}
