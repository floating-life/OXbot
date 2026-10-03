#include "oxbot/fabledan_network.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

// The amalgamated source also contains CandidateNetwork's private helper
// named `require`; keep this backend's calls distinct even though the detail
// namespace is imported for the member-function implementation below.
#define require fbd_require

namespace oxbot {
namespace fbdn_detail {

constexpr char kMagicV1[8] = {'F', 'B', 'D', 'N', '0', '0', '1', '\0'};
constexpr char kMagicV2[8] = {'F', 'B', 'D', 'N', '0', '0', '2', '\0'};
constexpr std::uint32_t kDtypeFp16 = 1;
constexpr std::uint32_t kDtypeFp32 = 2;
constexpr std::size_t kMaxFileBytes = 64U * 1024U * 1024U;
constexpr std::size_t kExpectedTensors = 64;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::uint16_t read_u16_at(const std::vector<unsigned char>& bytes,
                          std::size_t offset) {
    require(offset <= bytes.size() && bytes.size() - offset >= 2,
            "binary_bounds");
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(bytes[offset]) |
        (static_cast<std::uint16_t>(bytes[offset + 1]) << 8));
}

std::uint32_t read_u32_at(const std::vector<unsigned char>& bytes,
                          std::size_t offset) {
    require(offset <= bytes.size() && bytes.size() - offset >= 4,
            "binary_bounds");
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t read_u64_at(const std::vector<unsigned char>& bytes,
                          std::size_t offset) {
    require(offset <= bytes.size() && bytes.size() - offset >= 8,
            "binary_bounds");
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8U * i);
    }
    return value;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits >> 15);
    const std::uint32_t exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fU);
    const std::uint32_t fraction = static_cast<std::uint32_t>(bits & 0x03ffU);
    float value;
    if (exponent == 0) {
        if (fraction == 0) {
            value = 0.0f;
        } else {
            value = std::ldexp(static_cast<float>(fraction) / 1024.0f, -14);
        }
    } else if (exponent == 31) {
        if (fraction == 0) {
            value = std::numeric_limits<float>::infinity();
        } else {
            value = std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        value = std::ldexp(1.0f + static_cast<float>(fraction) / 1024.0f,
                           static_cast<int>(exponent) - 15);
    }
    return sign == 0 ? value : -value;
}

std::uint32_t rotr(std::uint32_t value, unsigned int amount) {
    return (value >> amount) | (value << (32U - amount));
}

std::array<unsigned char, 32> sha256(const unsigned char* input,
                                      std::size_t input_size) {
    static constexpr std::uint32_t constants[] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 8> state{{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U}};

    std::vector<unsigned char> data(input, input + input_size);
    const std::uint64_t bit_count = static_cast<std::uint64_t>(input_size) * 8U;
    data.push_back(0x80U);
    while (data.size() % 64U != 56U) data.push_back(0U);
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<unsigned char>(bit_count >> shift));
    }

    for (std::size_t offset = 0; offset < data.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (unsigned int i = 0; i < 16; ++i) {
            const std::size_t at = offset + static_cast<std::size_t>(i) * 4U;
            words[i] = (static_cast<std::uint32_t>(data[at]) << 24) |
                       (static_cast<std::uint32_t>(data[at + 1]) << 16) |
                       (static_cast<std::uint32_t>(data[at + 2]) << 8) |
                       static_cast<std::uint32_t>(data[at + 3]);
        }
        for (unsigned int i = 16; i < 64; ++i) {
            const std::uint32_t a = words[i - 15];
            const std::uint32_t b = words[i - 2];
            const std::uint32_t small_a = rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3);
            const std::uint32_t small_b = rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10);
            words[i] = words[i - 16] + small_a + words[i - 7] + small_b;
        }

        std::array<std::uint32_t, 8> v = state;
        for (unsigned int i = 0; i < 64; ++i) {
            const std::uint32_t choose = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const std::uint32_t majority = (v[0] & v[1]) ^ (v[0] & v[2]) ^
                                            (v[1] & v[2]);
            const std::uint32_t big_a = rotr(v[4], 6) ^ rotr(v[4], 11) ^
                                        rotr(v[4], 25);
            const std::uint32_t big_b = rotr(v[0], 2) ^ rotr(v[0], 13) ^
                                        rotr(v[0], 22);
            const std::uint32_t first = v[7] + big_a + choose + constants[i] + words[i];
            const std::uint32_t second = big_b + majority;
            for (int j = 7; j > 0; --j) v[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j - 1)];
            v[4] += first;
            v[0] = first + second;
        }
        for (std::size_t i = 0; i < state.size(); ++i) state[i] += v[i];
    }

    std::array<unsigned char, 32> digest{};
    for (std::size_t i = 0; i < state.size(); ++i) {
        digest[i * 4] = static_cast<unsigned char>(state[i] >> 24);
        digest[i * 4 + 1] = static_cast<unsigned char>(state[i] >> 16);
        digest[i * 4 + 2] = static_cast<unsigned char>(state[i] >> 8);
        digest[i * 4 + 3] = static_cast<unsigned char>(state[i]);
    }
    return digest;
}

std::string hex(const std::array<unsigned char, 32>& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const unsigned char value : bytes) out << std::setw(2) << static_cast<int>(value);
    return out.str();
}

struct TensorRecord {
    std::string name;
    std::vector<int> shape;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

const std::vector<std::string>& expected_tensor_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> out{"rope_cos", "rope_sin", "token_emb.weight"};
        for (int block = 0; block < 4; ++block) {
            const std::string prefix = "blocks." + std::to_string(block) + ".";
            const std::vector<std::string> suffixes{
                "attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
                "attn.v_proj.weight", "attn.out_proj.weight", "attn.q_norm.weight",
                "attn.k_norm.weight", "ffn_norm.weight", "ffn.gate_proj.weight",
                "ffn.up_proj.weight", "ffn.down_proj.weight"};
            for (const auto& suffix : suffixes) out.push_back(prefix + suffix);
        }
        const std::vector<std::string> tail{
            "final_norm.weight", "hand_mlp.0.weight", "hand_mlp.0.bias",
            "hand_mlp.2.weight", "hand_mlp.2.bias", "hand_mlp.4.weight",
            "hand_mlp.4.bias", "hand_mlp.6.weight", "hand_mlp.6.bias",
            "q_head.0.weight", "q_head.0.bias", "q_head.2.weight",
            "q_head.2.bias", "q_head.4.weight", "q_head.4.bias",
            "q_head.6.weight", "q_head.6.bias"};
        out.insert(out.end(), tail.begin(), tail.end());
        return out;
    }();
    return names;
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "model_missing");
    const std::streamoff size = input.tellg();
    require(size >= 0 && static_cast<std::uint64_t>(size) <= kMaxFileBytes,
            "model_size_invalid");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    if (!bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    require(static_cast<bool>(input), "model_read_failed");
    return bytes;
}

}  // namespace fbdn_detail

using namespace fbdn_detail;

const FableDanNetwork::Tensor& FableDanNetwork::get(
    const std::string& name, const std::vector<int>& shape) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("missing_tensor:" + name);
    if (it->second.shape != shape) throw std::runtime_error("tensor_shape_mismatch:" + name);
    return it->second;
}

bool FableDanNetwork::load(const std::string& path) {
    ready_ = false;
    tensors_.clear();
    status_ = "model_not_loaded";
    payload_sha_.clear();
    feat_dim_ = 80;
    try {
        const std::vector<unsigned char> bytes = read_file(path);
        require(bytes.size() >= 116U, "model_header_invalid");
        const bool format_v1 = std::memcmp(bytes.data(), kMagicV1, sizeof(kMagicV1)) == 0;
        const bool format_v2 = std::memcmp(bytes.data(), kMagicV2, sizeof(kMagicV2)) == 0;
        require(format_v1 || format_v2, "model_magic_mismatch");
        const std::uint32_t version = read_u32_at(bytes, 8);
        const std::uint32_t dtype = read_u32_at(bytes, 12);
        const std::uint32_t header_bytes = read_u32_at(bytes, 16);
        const std::uint32_t tensor_count = read_u32_at(bytes, 20);
        require(version == (format_v1 ? 1U : 2U), "model_version_mismatch");
        require(dtype == kDtypeFp16 || dtype == kDtypeFp32, "dtype_mismatch");
        require(header_bytes >= 116U && header_bytes <= bytes.size(),
                "model_header_invalid");
        require(tensor_count == kExpectedTensors, "tensor_count_mismatch");

        const std::uint32_t config_values[] = {
            read_u32_at(bytes, 24),  read_u32_at(bytes, 28),  read_u32_at(bytes, 32),
            read_u32_at(bytes, 36),  read_u32_at(bytes, 40),  read_u32_at(bytes, 44),
            read_u32_at(bytes, 48),  read_u32_at(bytes, 52),  read_u32_at(bytes, 56),
            read_u32_at(bytes, 60),  read_u32_at(bytes, 64),  read_u32_at(bytes, 68),
            read_u32_at(bytes, 72)};
        const std::uint32_t expected_config[] = {
            128U, 4U, 4U, 64U, 64U, 512U, 512U, 3U, 1024U, 3U, 512U, 48U,
            format_v1 ? 80U : 224U};
        for (std::size_t i = 0; i < 13U; ++i) {
            require(config_values[i] == expected_config[i], "model_config_mismatch");
        }
        std::uint32_t rms_bits = read_u32_at(bytes, 76);
        float rms_eps = 0.0f;
        std::memcpy(&rms_eps, &rms_bits, sizeof(rms_eps));
        require(std::isfinite(rms_eps) && rms_eps > 0.0f && rms_eps < 1e-3f,
                "model_config_mismatch");
        const std::uint32_t payload_bytes = read_u32_at(bytes, 80);
        require(static_cast<std::uint64_t>(header_bytes) + payload_bytes == bytes.size(),
                "payload_size_mismatch");

        std::array<unsigned char, 32> expected_sha{};
        std::memcpy(expected_sha.data(), bytes.data() + 84, expected_sha.size());
        const auto actual_sha = sha256(bytes.data() + header_bytes, payload_bytes);
        require(actual_sha == expected_sha, "payload_sha_mismatch");
        payload_sha_ = hex(actual_sha);

        std::size_t cursor = 116U;
        std::vector<TensorRecord> records;
        records.reserve(tensor_count);
        std::map<std::string, bool> names_seen;
        for (std::uint32_t index = 0; index < tensor_count; ++index) {
            require(cursor <= header_bytes && header_bytes - cursor >= 4U,
                    "tensor_table_invalid");
            const std::uint16_t name_bytes = read_u16_at(bytes, cursor);
            const unsigned int rank = bytes[cursor + 2];
            cursor += 4U;
            require(rank > 0U && rank <= 4U, "tensor_shape_invalid");
            std::vector<int> shape;
            shape.reserve(rank);
            std::uint64_t count = 1;
            for (unsigned int axis = 0; axis < rank; ++axis) {
                require(cursor <= header_bytes && header_bytes - cursor >= 4U,
                        "tensor_table_invalid");
                const std::uint32_t dimension = read_u32_at(bytes, cursor);
                cursor += 4U;
                require(dimension > 0U && dimension <= 8192U, "tensor_dimension_invalid");
                require(count <= std::numeric_limits<std::uint64_t>::max() / dimension,
                        "tensor_shape_overflow");
                count *= dimension;
                require(count <= 16U * 1024U * 1024U, "tensor_shape_overflow");
                shape.push_back(static_cast<int>(dimension));
            }
            require(cursor <= header_bytes && header_bytes - cursor >= 16U,
                    "tensor_table_invalid");
            const std::uint64_t offset = read_u64_at(bytes, cursor);
            const std::uint64_t byte_count = read_u64_at(bytes, cursor + 8U);
            cursor += 16U;
            require(name_bytes > 0U && name_bytes <= 512U,
                    "tensor_name_invalid");
            require(cursor <= header_bytes &&
                        static_cast<std::uint64_t>(header_bytes - cursor) >= name_bytes,
                    "tensor_table_invalid");
            const std::string name(reinterpret_cast<const char*>(bytes.data() + cursor),
                                   name_bytes);
            cursor += name_bytes;
            require(names_seen.emplace(name, true).second, "tensor_name_invalid");
            const std::uint64_t item_bytes = dtype == kDtypeFp16 ? 2U : 4U;
            require(byte_count == count * item_bytes, "tensor_byte_count_mismatch");
            require(offset <= payload_bytes && byte_count <= payload_bytes - offset,
                    "tensor_bounds_invalid");
            records.push_back(TensorRecord{name, std::move(shape), offset, byte_count});
        }
        require(cursor == header_bytes, "tensor_table_size_mismatch");

        std::vector<std::size_t> order(records.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return records[a].offset < records[b].offset;
        });
        std::uint64_t expected_offset = 0;
        for (const std::size_t index : order) {
            require(records[index].offset == expected_offset,
                    "tensor_payload_gap");
            expected_offset += records[index].bytes;
        }
        require(expected_offset == payload_bytes, "tensor_payload_size_mismatch");

        const auto& expected_names = expected_tensor_names();
        require(expected_names.size() == kExpectedTensors, "internal_tensor_contract");
        for (const auto& name : expected_names) require(names_seen.count(name) == 1U, "missing_tensor");
        const unsigned char* payload = bytes.data() + header_bytes;
        for (const auto& record : records) {
            Tensor tensor;
            tensor.shape = record.shape;
            std::uint64_t count = 1;
            for (const int dimension : tensor.shape) count *= static_cast<std::uint64_t>(dimension);
            require(count <= std::numeric_limits<std::size_t>::max(), "tensor_shape_overflow");
            tensor.data.resize(static_cast<std::size_t>(count));
            for (std::size_t i = 0; i < tensor.data.size(); ++i) {
                float value;
                const std::size_t at = static_cast<std::size_t>(record.offset) +
                                        i * (dtype == kDtypeFp16 ? 2U : 4U);
                if (dtype == kDtypeFp16) {
                    const std::uint16_t bits = static_cast<std::uint16_t>(
                        static_cast<std::uint16_t>(payload[at]) |
                        (static_cast<std::uint16_t>(payload[at + 1]) << 8));
                    value = half_to_float(bits);
                } else {
                    const std::uint32_t bits = static_cast<std::uint32_t>(payload[at]) |
                                               (static_cast<std::uint32_t>(payload[at + 1]) << 8) |
                                               (static_cast<std::uint32_t>(payload[at + 2]) << 16) |
                                               (static_cast<std::uint32_t>(payload[at + 3]) << 24);
                    std::memcpy(&value, &bits, sizeof(value));
                }
                require(std::isfinite(value), "nonfinite_weight");
                tensor.data[i] = value;
            }
            tensors_.emplace(record.name, std::move(tensor));
        }

        d_model_ = 128;
        n_blocks_ = 4;
        n_heads_ = 4;
        qk_dim_ = 64;
        v_dim_ = 64;
        ffn_hidden_ = 512;
        hand_hidden_ = 512;
        n_hand_layers_ = 3;
        q_hidden_ = 1024;
        n_q_layers_ = 3;
        max_seq_ = 512;
        vocab_ = 48;
        feat_dim_ = static_cast<int>(config_values[12]);
        rms_eps_ = rms_eps;

        get("rope_cos", {512, 32});
        get("rope_sin", {512, 32});
        get("token_emb.weight", {48, 128});
        for (int block = 0; block < 4; ++block) {
            const std::string p = "blocks." + std::to_string(block) + ".";
            get(p + "attn_norm.weight", {128});
            get(p + "attn.q_proj.weight", {256, 128});
            get(p + "attn.k_proj.weight", {256, 128});
            get(p + "attn.v_proj.weight", {256, 128});
            get(p + "attn.out_proj.weight", {128, 256});
            get(p + "attn.q_norm.weight", {64});
            get(p + "attn.k_norm.weight", {64});
            get(p + "ffn_norm.weight", {128});
            get(p + "ffn.gate_proj.weight", {512, 128});
            get(p + "ffn.up_proj.weight", {512, 128});
            get(p + "ffn.down_proj.weight", {128, 512});
        }
        get("final_norm.weight", {128});
        get("hand_mlp.0.weight", {512, feat_dim_});
        get("hand_mlp.0.bias", {512});
        get("hand_mlp.2.weight", {512, 512});
        get("hand_mlp.2.bias", {512});
        get("hand_mlp.4.weight", {512, 512});
        get("hand_mlp.4.bias", {512});
        get("hand_mlp.6.weight", {128, 512});
        get("hand_mlp.6.bias", {128});
        get("q_head.0.weight", {1024, 256});
        get("q_head.0.bias", {1024});
        get("q_head.2.weight", {1024, 1024});
        get("q_head.2.bias", {1024});
        get("q_head.4.weight", {1024, 1024});
        get("q_head.4.bias", {1024});
        get("q_head.6.weight", {1, 1024});
        get("q_head.6.bias", {1});
        require(tensors_.size() == kExpectedTensors, "tensor_table_size_mismatch");
        ready_ = true;
        status_ = "ready";
        return true;
    } catch (const std::exception& error) {
        ready_ = false;
        tensors_.clear();
        payload_sha_.clear();
        status_ = error.what();
        return false;
    }
}

std::vector<float> FableDanNetwork::linear(const std::vector<float>& x, int rows,
                                           int in, int out,
                                           const std::string& name) const {
    require(rows >= 0 && static_cast<std::size_t>(rows) * static_cast<std::size_t>(in) == x.size(),
            "linear_shape_invalid");
    const auto& weight = get(name + ".weight", {out, in}).data;
    const auto bias_it = tensors_.find(name + ".bias");
    const bool has_bias = bias_it != tensors_.end();
    if (has_bias) require(bias_it->second.shape == std::vector<int>{out},
                          "tensor_shape_mismatch");
    std::vector<float> y(static_cast<std::size_t>(rows) * static_cast<std::size_t>(out));
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < out; ++column) {
            float sum = has_bias ? bias_it->second.data[static_cast<std::size_t>(column)] : 0.0f;
            for (int input = 0; input < in; ++input) {
                sum += x[static_cast<std::size_t>(row * in + input)] *
                       weight[static_cast<std::size_t>(column * in + input)];
            }
            y[static_cast<std::size_t>(row * out + column)] = sum;
        }
    }
    return y;
}

std::vector<float> FableDanNetwork::rms(const std::vector<float>& x, int rows,
                                        int dim,
                                        const std::string& weight_name) const {
    require(rows >= 0 && static_cast<std::size_t>(rows) * static_cast<std::size_t>(dim) == x.size(),
            "norm_shape_invalid");
    const auto& weight = get(weight_name, {dim}).data;
    std::vector<float> y(x.size());
    for (int row = 0; row < rows; ++row) {
        float mean_square = 0.0f;
        for (int column = 0; column < dim; ++column) {
            const float value = x[static_cast<std::size_t>(row * dim + column)];
            mean_square += value * value;
        }
        mean_square /= static_cast<float>(dim);
        const float scale = 1.0f / std::sqrt(mean_square + rms_eps_);
        for (int column = 0; column < dim; ++column) {
            y[static_cast<std::size_t>(row * dim + column)] =
                x[static_cast<std::size_t>(row * dim + column)] * scale *
                weight[static_cast<std::size_t>(column)];
        }
    }
    return y;
}

std::vector<float> FableDanNetwork::mlp(
    const std::vector<float>& x, const std::string& prefix,
    const std::vector<int>& linear_indices) const {
    require(!linear_indices.empty(), "mlp_contract_invalid");
    std::vector<float> current = x;
    for (std::size_t layer = 0; layer < linear_indices.size(); ++layer) {
        const std::string name = prefix + std::to_string(linear_indices[layer]);
        const auto weight_it = tensors_.find(name + ".weight");
        if (weight_it == tensors_.end()) {
            throw std::runtime_error("missing_tensor:" + name + ".weight");
        }
        const auto& weight = weight_it->second;
        require(weight.shape.size() == 2U, "mlp_shape_invalid");
        current = linear(current, 1, weight.shape[1], weight.shape[0], name);
        if (layer + 1U != linear_indices.size()) {
            for (float& value : current) value = std::max(0.0f, value);
        }
    }
    return current;
}

std::vector<float> FableDanNetwork::context(const std::vector<int>& tokens) const {
    require(ready_, "model_not_ready");
    require(!tokens.empty() && tokens.size() <= static_cast<std::size_t>(max_seq_),
            "token_shape_invalid");
    const int length = static_cast<int>(tokens.size());
    const auto& embedding = get("token_emb.weight", {vocab_, d_model_}).data;
    const auto& rope_cos = get("rope_cos", {max_seq_, qk_dim_ / 2}).data;
    const auto& rope_sin = get("rope_sin", {max_seq_, qk_dim_ / 2}).data;
    std::vector<float> x(static_cast<std::size_t>(length * d_model_));
    for (int time = 0; time < length; ++time) {
        const int token = tokens[static_cast<std::size_t>(time)];
        require(token >= 0 && token < vocab_, "token_invalid");
        for (int dim = 0; dim < d_model_; ++dim) {
            x[static_cast<std::size_t>(time * d_model_ + dim)] =
                embedding[static_cast<std::size_t>(token * d_model_ + dim)];
        }
    }

    const int q_width = n_heads_ * qk_dim_;
    const int v_width = n_heads_ * v_dim_;
    for (int block = 0; block < n_blocks_; ++block) {
        const std::string prefix = "blocks." + std::to_string(block) + ".";
        const auto normalized = rms(x, length, d_model_, prefix + "attn_norm.weight");
        auto query = linear(normalized, length, d_model_, q_width,
                            prefix + "attn.q_proj");
        auto key = linear(normalized, length, d_model_, q_width,
                          prefix + "attn.k_proj");
        const auto value = linear(normalized, length, d_model_, v_width,
                                  prefix + "attn.v_proj");
        const auto& query_weight = get(prefix + "attn.q_norm.weight", {qk_dim_}).data;
        const auto& key_weight = get(prefix + "attn.k_norm.weight", {qk_dim_}).data;
        for (int time = 0; time < length; ++time) {
            for (int head = 0; head < n_heads_; ++head) {
                float q_mean_square = 0.0f;
                float k_mean_square = 0.0f;
                for (int dim = 0; dim < qk_dim_; ++dim) {
                    const std::size_t at = static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim);
                    q_mean_square += query[at] * query[at];
                    k_mean_square += key[at] * key[at];
                }
                const float q_scale = 1.0f / std::sqrt(q_mean_square / static_cast<float>(qk_dim_) + rms_eps_);
                const float k_scale = 1.0f / std::sqrt(k_mean_square / static_cast<float>(qk_dim_) + rms_eps_);
                for (int dim = 0; dim < qk_dim_; ++dim) {
                    const std::size_t at = static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim);
                    query[at] *= q_scale * query_weight[static_cast<std::size_t>(dim)];
                    key[at] *= k_scale * key_weight[static_cast<std::size_t>(dim)];
                }
                for (int dim = 0; dim < qk_dim_ / 2; ++dim) {
                    const std::size_t q_at = static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim);
                    const std::size_t q_other = q_at + static_cast<std::size_t>(qk_dim_ / 2);
                    const float q_first = query[q_at];
                    const float q_second = query[q_other];
                    const float q_cos = rope_cos[static_cast<std::size_t>(time * (qk_dim_ / 2) + dim)];
                    const float q_sin = rope_sin[static_cast<std::size_t>(time * (qk_dim_ / 2) + dim)];
                    query[q_at] = q_first * q_cos - q_second * q_sin;
                    query[q_other] = q_first * q_sin + q_second * q_cos;
                    const float k_first = key[q_at];
                    const float k_second = key[q_other];
                    key[q_at] = k_first * q_cos - k_second * q_sin;
                    key[q_other] = k_first * q_sin + k_second * q_cos;
                }
            }
        }

        std::vector<float> attention(static_cast<std::size_t>(length * v_width), 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(qk_dim_));
        for (int time = 0; time < length; ++time) {
            for (int head = 0; head < n_heads_; ++head) {
                float largest = -std::numeric_limits<float>::infinity();
                for (int source = 0; source <= time; ++source) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < qk_dim_; ++dim) {
                        dot += query[static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim)] *
                               key[static_cast<std::size_t>(source * q_width + head * qk_dim_ + dim)];
                    }
                    largest = std::max(largest, dot * scale);
                }
                float denominator = 0.0f;
                for (int source = 0; source <= time; ++source) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < qk_dim_; ++dim) {
                        dot += query[static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim)] *
                               key[static_cast<std::size_t>(source * q_width + head * qk_dim_ + dim)];
                    }
                    denominator += std::exp(dot * scale - largest);
                }
                for (int source = 0; source <= time; ++source) {
                    float dot = 0.0f;
                    for (int dim = 0; dim < qk_dim_; ++dim) {
                        dot += query[static_cast<std::size_t>(time * q_width + head * qk_dim_ + dim)] *
                               key[static_cast<std::size_t>(source * q_width + head * qk_dim_ + dim)];
                    }
                    const float probability = std::exp(dot * scale - largest) / denominator;
                    for (int dim = 0; dim < v_dim_; ++dim) {
                        attention[static_cast<std::size_t>(time * v_width + head * v_dim_ + dim)] +=
                            probability * value[static_cast<std::size_t>(source * v_width + head * v_dim_ + dim)];
                    }
                }
            }
        }

        const auto projected = linear(attention, length, v_width, d_model_,
                                      prefix + "attn.out_proj");
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += projected[i];
        const auto ffn_input = rms(x, length, d_model_, prefix + "ffn_norm.weight");
        const auto gate = linear(ffn_input, length, d_model_, ffn_hidden_,
                                 prefix + "ffn.gate_proj");
        const auto up = linear(ffn_input, length, d_model_, ffn_hidden_,
                               prefix + "ffn.up_proj");
        std::vector<float> gated(gate.size());
        for (std::size_t i = 0; i < gated.size(); ++i) {
            const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
            gated[i] = gate[i] * sigmoid * up[i];
        }
        const auto down = linear(gated, length, ffn_hidden_, d_model_,
                                 prefix + "ffn.down_proj");
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += down[i];
    }
    x = rms(x, length, d_model_, "final_norm.weight");
    return std::vector<float>(x.end() - d_model_, x.end());
}

std::vector<float> FableDanNetwork::score(
    const std::vector<int>& tokens,
    const std::vector<std::vector<float>>& features) const {
    std::vector<float> flat;
    flat.reserve(features.size() * static_cast<std::size_t>(feat_dim_));
    for (const auto& row : features) {
        require(row.size() == static_cast<std::size_t>(feat_dim_),
                "feature_shape_invalid");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return forward(tokens, flat, features.size());
}

std::vector<float> FableDanNetwork::forward(const std::vector<int>& tokens,
                                            const std::vector<float>& features,
                                            std::size_t rows) const {
    require(ready_, "model_not_ready");
    require(rows <= std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(feat_dim_) &&
                features.size() == rows * static_cast<std::size_t>(feat_dim_),
            "feature_shape_invalid");
    const auto ctx = context(tokens);
    std::vector<float> result;
    result.reserve(rows);
    static const std::vector<int> hand_layers{0, 2, 4, 6};
    static const std::vector<int> q_layers{0, 2, 4, 6};
    for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t start = row * static_cast<std::size_t>(feat_dim_);
        const std::vector<float> feature(features.begin() + static_cast<std::ptrdiff_t>(start),
                                         features.begin() + static_cast<std::ptrdiff_t>(start + feat_dim_));
        for (const float value : feature) require(std::isfinite(value), "nonfinite_feature");
        const auto hand = mlp(feature, "hand_mlp.", hand_layers);
        std::vector<float> joined;
        joined.reserve(static_cast<std::size_t>(d_model_ * 2));
        joined.insert(joined.end(), ctx.begin(), ctx.end());
        joined.insert(joined.end(), hand.begin(), hand.end());
        const auto q = mlp(joined, "q_head.", q_layers);
        require(q.size() == 1U && std::isfinite(q[0]), "nonfinite_prediction");
        result.push_back(q[0]);
    }
    return result;
}

}  // namespace oxbot

#undef require
