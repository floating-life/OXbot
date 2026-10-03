#include "oxbot/network.hpp"
#include "oxbot/mini_json.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace oxbot {
namespace {
constexpr const char* kDefaultModelVersion = "oxbot-model-v1";
std::uint32_t rotate(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
std::uint32_t little_u32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

// Versions are emitted in a semicolon-delimited diagnostic field.  Keep the
// accepted alphabet deliberately narrow so a model header cannot inject a
// second diagnostic field (or an unexpected control character).
bool valid_model_version(const std::string& value) {
    if (value.empty() || value.size() > 96) return false;
    for (const unsigned char c : value) {
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}
}

std::string sha256_hex(const std::vector<unsigned char>& input) {
    static constexpr std::uint32_t constants[] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t, 8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::vector<unsigned char> data = input;
    const auto bits = static_cast<std::uint64_t>(data.size()) * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) data.push_back(0);
    for (int i = 7; i >= 0; --i) data.push_back(static_cast<unsigned char>(bits >> (i * 8)));
    for (std::size_t offset = 0; offset < data.size(); offset += 64) {
        std::array<std::uint32_t, 64> w{};
        for (int t = 0; t < 16; ++t) {
            for (int j = 0; j < 4; ++j) w[static_cast<std::size_t>(t)] =
                (w[static_cast<std::size_t>(t)] << 8) | data[offset + static_cast<std::size_t>(4*t+j)];
        }
        for (int t = 16; t < 64; ++t) {
            const auto a = w[static_cast<std::size_t>(t-15)], b = w[static_cast<std::size_t>(t-2)];
            w[static_cast<std::size_t>(t)] = w[static_cast<std::size_t>(t-16)] +
                (rotate(a,7)^rotate(a,18)^(a>>3)) + w[static_cast<std::size_t>(t-7)] + (rotate(b,17)^rotate(b,19)^(b>>10));
        }
        auto v = h;
        for (int t = 0; t < 64; ++t) {
            const auto t1 = v[7] + (rotate(v[4],6)^rotate(v[4],11)^rotate(v[4],25)) +
                ((v[4]&v[5])^(~v[4]&v[6])) + constants[t] + w[static_cast<std::size_t>(t)];
            const auto t2 = (rotate(v[0],2)^rotate(v[0],13)^rotate(v[0],22)) +
                ((v[0]&v[1])^(v[0]&v[2])^(v[1]&v[2]));
            for (int j = 7; j > 0; --j) v[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j-1)];
            v[4] += t1; v[0] = t1+t2;
        }
        for (std::size_t j = 0; j < h.size(); ++j) h[j] += v[j];
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (auto word : h) out << std::setw(8) << word;
    return out.str();
}

const CandidateNetwork::Tensor& CandidateNetwork::get(const std::string& name, const std::vector<int>& shape) const {
    const auto it = tensors_.find(name);
    require(it != tensors_.end(), "missing_tensor");
    require(it->second.shape == shape, "tensor_shape_mismatch");
    return it->second;
}

bool CandidateNetwork::load(const std::string& path) {
    ready_ = false; tensors_.clear(); sha_.clear(); model_version_ = kDefaultModelVersion;
    selection_strategy_ = "raw";
    selection_manifest_present_ = false;
    try {
        std::vector<unsigned char> bytes;
#ifdef OXBOT_EMBEDDED_MODEL
        // tools/amalgamate.py may include the exact versioned binary bytes
        // before this source. The same parser and checksum still validate it.
        if (path == ":embedded:") {
            bytes.assign(oxbot_embedded_model, oxbot_embedded_model + sizeof(oxbot_embedded_model));
        } else
#endif
        {
            std::ifstream in(path, std::ios::binary | std::ios::ate);
            require(static_cast<bool>(in), "model_missing");
            const auto size = in.tellg();
            require(size >= 12 && size <= 8 * 1024 * 1024, "model_size_invalid");
            in.seekg(0);
            bytes.resize(static_cast<std::size_t>(size));
            in.read(reinterpret_cast<char*>(bytes.data()), size);
            require(static_cast<bool>(in), "model_read_failed");
        }
        require(bytes.size() >= 12 && bytes.size() <= 8 * 1024 * 1024, "model_size_invalid");
        require(std::memcmp(bytes.data(), "OXGDQ001", 8) == 0, "model_magic_mismatch");
        const auto header_length = little_u32(bytes.data()+8);
        require(header_length <= 65536 && 12 + header_length <= bytes.size(), "model_header_invalid");
        const auto header = json::parse(std::string(bytes.begin()+12, bytes.begin()+12+header_length));
        require(header["architecture"].as_string() == "oxbot-causal-candidate-v1", "architecture_mismatch");
        require(header["feature_version"].as_string() == "oxbot-observation-v1", "feature_version_mismatch");
        require(header["rules_contract"].as_string() == "botzone-corrected-fa63589d-v1", "rules_contract_mismatch");
        require(header["dtype"].as_string() == "float32-le", "dtype_mismatch");
        // Additive metadata: old/frozen binaries have no model_version and
        // intentionally retain the v1 diagnostic identity.
        if (header.contains("model_version")) {
            const auto configured = header["model_version"].as_string();
            require(valid_model_version(configured), "model_version_invalid");
            model_version_ = configured;
        }
        // The optional selection manifest is additive.  The frozen v1
        // binary has no such field and therefore remains raw by default.
        if (header.contains("selection")) {
            const auto& selection = header["selection"];
            require(selection.is_object(), "selection_manifest_invalid");
            const auto configured = selection["default"].as_string();
            require(configured == "raw" || configured == "raw-pass-bias" || configured == "group-logmeanexp",
                    "selection_strategy_invalid");
            selection_strategy_ = configured;
            selection_manifest_present_ = true;
        }
        const auto& config = header["config"];
        require(config["width"].as_int() == 64 && config["layers"].as_int() == 2 &&
                config["heads"].as_int() == 4 && config["feedforward"].as_int() == 256 &&
                config["vocabulary"].as_int() == 128 && config["history_length"].as_int() == 256 &&
                config["state_features"].as_int() == 128 && config["action_features"].as_int() == 128 &&
                config["layer_norm_epsilon"].number == 1e-5, "model_config_mismatch");
        std::vector<unsigned char> payload(bytes.begin()+12+header_length, bytes.end());
        sha_ = sha256_hex(payload);
        require(sha_ == header["payload_sha256"].as_string(), "model_sha_mismatch");
        const std::uint16_t endian = 1;
        require(*reinterpret_cast<const unsigned char*>(&endian) == 1 && sizeof(float) == 4, "unsupported_float_layout");
        const auto& entries = header["tensors"];
        require(entries.is_array(), "tensor_table_invalid");
        std::size_t total = 0;
        for (const auto& entry : entries.array) {
            const auto name = entry["name"].as_string();
            require(!name.empty() && !tensors_.count(name), "tensor_name_invalid");
            const int offset = entry["offset"].as_int(-1), count = entry["count"].as_int(-1);
            require(offset >= 0 && count > 0, "tensor_bounds_invalid");
            require(static_cast<std::size_t>(offset) == total && total+4*static_cast<std::size_t>(count) <= payload.size(), "tensor_bounds_invalid");
            Tensor tensor;
            require(entry["shape"].is_array(), "tensor_shape_invalid");
            std::size_t product = 1;
            for (const auto& value : entry["shape"].array) {
                const int dimension = value.as_int(-1);
                require(dimension > 0 && dimension <= 1024, "tensor_dimension_invalid");
                product *= static_cast<std::size_t>(dimension);
                require(product <= payload.size()/4, "tensor_shape_overflow");
                tensor.shape.push_back(dimension);
            }
            require(product == static_cast<std::size_t>(count), "tensor_count_mismatch");
            tensor.data.resize(product);
            std::memcpy(tensor.data.data(), payload.data()+offset, product*4);
            for (float value : tensor.data) require(std::isfinite(value), "nonfinite_weight");
            total += product*4;
            tensors_.emplace(name, std::move(tensor));
        }
        require(total == payload.size() && tensors_.size() == 36, "tensor_table_size_mismatch");
        get("embedding.weight", {128,64}); get("position.weight", {256,64});
        auto dense = [&](const std::string& name, int rows, int cols) { get(name+".weight", {rows,cols}); get(name+".bias", {rows}); };
        auto layer_norm = [&](const std::string& name) { get(name+".weight", {64}); get(name+".bias", {64}); };
        for (int i = 0; i < 2; ++i) {
            const auto prefix = "blocks."+std::to_string(i)+".";
            layer_norm(prefix+"norm1"); layer_norm(prefix+"norm2");
            dense(prefix+"qkv",192,64); dense(prefix+"projection",64,64);
            dense(prefix+"ff1",256,64); dense(prefix+"ff2",64,256);
        }
        layer_norm("final_norm"); dense("state_projection",64,128); dense("action_projection",64,128);
        dense("head1",64,192); dense("head2",1,64);
        ready_ = true; status_ = "ready"; return true;
    } catch (const std::exception& ex) {
        tensors_.clear(); status_ = ex.what(); return false;
    }
}

std::vector<float> CandidateNetwork::linear(const std::vector<float>& x, int rows, int in, int out, const std::string& name) const {
    const auto& w = get(name+".weight",{out,in}).data;
    const auto& b = get(name+".bias",{out}).data;
    std::vector<float> y(static_cast<std::size_t>(rows*out));
    for (int r = 0; r < rows; ++r) for (int o = 0; o < out; ++o) {
        float sum = b[static_cast<std::size_t>(o)];
        for (int i = 0; i < in; ++i) sum += x[static_cast<std::size_t>(r*in+i)] * w[static_cast<std::size_t>(o*in+i)];
        y[static_cast<std::size_t>(r*out+o)] = sum;
    }
    return y;
}

void CandidateNetwork::norm(std::vector<float>& x, int rows, const std::string& name) const {
    const auto& w = get(name+".weight",{64}).data;
    const auto& b = get(name+".bias",{64}).data;
    for (int r = 0; r < rows; ++r) {
        float mean = 0, variance = 0;
        for (int i = 0; i < 64; ++i) mean += x[static_cast<std::size_t>(r*64+i)];
        mean /= 64;
        for (int i = 0; i < 64; ++i) { const float d = x[static_cast<std::size_t>(r*64+i)]-mean; variance += d*d; }
        const float inv = 1.0f / std::sqrt(variance/64+1e-5f);
        for (int i = 0; i < 64; ++i) x[static_cast<std::size_t>(r*64+i)] = (x[static_cast<std::size_t>(r*64+i)]-mean)*inv*w[static_cast<std::size_t>(i)]+b[static_cast<std::size_t>(i)];
    }
}

std::vector<float> CandidateNetwork::score(const std::vector<int>& tokens, const std::vector<float>& state,
                                          const std::vector<std::vector<float>>& actions,
                                          std::map<std::string, std::vector<float>>* trace) const {
    require(ready_, "model_not_ready");
    require(!tokens.empty() && tokens.size() <= 256 && state.size() == 128, "feature_shape_invalid");
    const int length = static_cast<int>(tokens.size());
    const auto& embedding = get("embedding.weight",{128,64}).data;
    const auto& position = get("position.weight",{256,64}).data;
    std::vector<float> x(tokens.size()*64);
    for (int t = 0; t < length; ++t) {
        const int id = tokens[static_cast<std::size_t>(t)];
        require(id > 0 && id < 128, "token_invalid");
        for (int d = 0; d < 64; ++d) x[static_cast<std::size_t>(t*64+d)] = embedding[static_cast<std::size_t>(id*64+d)] + position[static_cast<std::size_t>(t*64+d)];
    }
    if (trace) (*trace)["embedding"] = x;
    int output_rows = length;
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "blocks."+std::to_string(layer)+".";
        auto normalized = x; norm(normalized,length,prefix+"norm1");
        const auto qkv = linear(normalized,length,64,192,prefix+"qkv");
        // Only the final token of the final block feeds the scorer. Its keys
        // and values still see every prefix token; other final-block output
        // rows have no consumers. Preserve the full path for layer tracing.
        const int first_query = layer == 1 && !trace ? length - 1 : 0;
        const int rows = length - first_query;
        std::vector<float> attention(static_cast<std::size_t>(rows * 64),0);
        for (int t = first_query; t < length; ++t) for (int h = 0; h < 4; ++h) {
            std::vector<float> probabilities(static_cast<std::size_t>(t+1));
            float largest = -std::numeric_limits<float>::infinity();
            for (int s = 0; s <= t; ++s) {
                float dot = 0;
                for (int d = 0; d < 16; ++d) dot += qkv[static_cast<std::size_t>(t*192+h*16+d)]*qkv[static_cast<std::size_t>(s*192+64+h*16+d)];
                probabilities[static_cast<std::size_t>(s)] = dot/4;
                largest = std::max(largest,dot/4);
            }
            float sum = 0;
            for (float& p : probabilities) { p = std::exp(p-largest); sum += p; }
            for (int s = 0; s <= t; ++s) for (int d = 0; d < 16; ++d) attention[static_cast<std::size_t>((t-first_query)*64+h*16+d)] +=
                probabilities[static_cast<std::size_t>(s)]/sum*qkv[static_cast<std::size_t>(s*192+128+h*16+d)];
        }
        if (first_query) x = std::vector<float>(x.end()-64,x.end());
        output_rows = rows;
        const auto projected = linear(attention,rows,64,64,prefix+"projection");
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += projected[i];
        normalized = x; norm(normalized,rows,prefix+"norm2");
        auto hidden = linear(normalized,rows,64,256,prefix+"ff1");
        for (float& v : hidden) v = std::max(0.0f,v);
        const auto residual = linear(hidden,rows,256,64,prefix+"ff2");
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += residual[i];
        if (trace) (*trace)["blocks."+std::to_string(layer)] = x;
    }
    norm(x,output_rows,"final_norm");
    if (trace) (*trace)["final_norm"] = x;
    auto state_hidden = linear(state,1,128,64,"state_projection");
    for (float& v : state_hidden) v = std::max(0.0f,v);
    if (trace) (*trace)["state_projection"] = state_hidden;
    const auto& action_weights = get("action_projection.weight", {64,128}).data;
    const auto& action_bias = get("action_projection.bias", {64}).data;
    const auto& head_weights = get("head1.weight", {64,192}).data;
    auto fixed_head = get("head1.bias", {64}).data;
    // The context prefix is identical for every candidate. Accumulate it in
    // exactly the original order, then continue each dot product with that
    // candidate's action values, avoiding any floating-point reassociation.
    for (int o = 0; o < 64; ++o) {
        float sum = fixed_head[static_cast<std::size_t>(o)];
        for (int i = 0; i < 64; ++i) sum += x[x.size()-64+static_cast<std::size_t>(i)] * head_weights[static_cast<std::size_t>(o*192+i)];
        for (int i = 0; i < 64; ++i) sum += state_hidden[static_cast<std::size_t>(i)] * head_weights[static_cast<std::size_t>(o*192+64+i)];
        fixed_head[static_cast<std::size_t>(o)] = sum;
    }
    std::vector<float> result;
    result.reserve(actions.size());
    for (const auto& action : actions) {
        require(action.size() == 128, "action_shape_invalid");
        std::vector<float> hidden;
        if (trace) {
            const auto action_hidden = linear(action,1,128,64,"action_projection");
            std::vector<float> joined;
            joined.reserve(192);
            joined.insert(joined.end(),x.end()-64,x.end());
            joined.insert(joined.end(),state_hidden.begin(),state_hidden.end());
            for (float v : action_hidden) joined.push_back(std::max(0.0f,v));
            hidden = linear(joined,1,192,64,"head1");
        } else {
            std::vector<int> nonzero;
            for (int i = 0; i < 128; ++i) if (action[static_cast<std::size_t>(i)] != 0.f) nonzero.push_back(i);
            auto action_hidden = action_bias;
            for (int o = 0; o < 64; ++o) {
                float sum = action_hidden[static_cast<std::size_t>(o)];
                for (int i : nonzero) sum += action[static_cast<std::size_t>(i)] * action_weights[static_cast<std::size_t>(o*128+i)];
                action_hidden[static_cast<std::size_t>(o)] = std::max(0.f,sum);
            }
            hidden = fixed_head;
            for (int o = 0; o < 64; ++o) {
                float sum = hidden[static_cast<std::size_t>(o)];
                for (int i = 0; i < 64; ++i) sum += action_hidden[static_cast<std::size_t>(i)] * head_weights[static_cast<std::size_t>(o*192+128+i)];
                hidden[static_cast<std::size_t>(o)] = sum;
            }
        }
        for (float& v : hidden) v = std::max(0.0f,v);
        const float value = linear(hidden,1,64,1,"head2")[0];
        require(std::isfinite(value), "nonfinite_prediction");
        result.push_back(value);
    }
    return result;
}

} // namespace oxbot
