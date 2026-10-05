#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace oxbot {

// Stand-alone inference backend for the four-block FableDan transformer.
//
// This intentionally does not share the CandidateNetwork (OXGDQ001) model
// contract.  FBDN001 (80 features) and FBDN002 (224 features) are external
// binary weight files produced by
// competition/tools/export_fabledan_cpp.py.  The weights are normally
// stored as little-endian float16 and decoded to float32 on load, so the
// backend does not depend on NumPy, a zip reader, or a C++ ML framework.
class FableDanNetwork {
public:
    FableDanNetwork() = default;

    bool load(const std::string& path);
    bool ready() const { return ready_; }
    const std::string& status() const { return status_; }
    const std::string& payload_sha() const { return payload_sha_; }

    // Encode a token history and return the final context vector.  Token IDs
    // use FableDan's vocabulary (0..47); padding is not required because the
    // deployment model consumes one unpadded sequence at a time.
    std::vector<float> context(const std::vector<int>& tokens) const;

    // Score one Q value for each candidate feature row.  Each row must have
    // exactly feature_dim() values (80 for FBDN001, 224 for FBDN002).
    std::vector<float> score(
        const std::vector<int>& tokens,
        const std::vector<std::vector<float>>& features) const;

    // Flat-row forward interface useful to C++ callers that already keep a
    // contiguous feature matrix.  `rows` is the number of candidate rows and
    // the input size must be rows * feat_dim.
    std::vector<float> forward(const std::vector<int>& tokens,
                               const std::vector<float>& features,
                               std::size_t rows) const;

    // Alias with the same nested-row shape as score(), for callers that use
    // the PyTorch/Numpy terminology.
    std::vector<float> forward(
        const std::vector<int>& tokens,
        const std::vector<std::vector<float>>& features) const {
        return score(tokens, features);
    }

    int feature_dim() const { return feat_dim_; }
    int vocabulary() const { return vocab_; }
    int max_sequence() const { return max_seq_; }

private:
    struct Tensor {
        std::vector<int> shape;
        std::vector<float> data;
    };

    std::map<std::string, Tensor> tensors_;

    // Per-block keys/values of the last encoded token sequence; see context().
    struct EncoderCache {
        std::vector<int> tokens;
        std::vector<std::vector<float>> keys;
        std::vector<std::vector<float>> values;
    };
    mutable EncoderCache cache_;
    bool ready_ = false;
    std::string status_ = "model_not_loaded";
    std::string payload_sha_;

    int d_model_ = 128;
    int n_blocks_ = 4;
    int n_heads_ = 4;
    int qk_dim_ = 64;
    int v_dim_ = 64;
    int ffn_hidden_ = 512;
    int hand_hidden_ = 512;
    int n_hand_layers_ = 3;
    int q_hidden_ = 1024;
    int n_q_layers_ = 3;
    int max_seq_ = 512;
    int vocab_ = 48;
    int feat_dim_ = 80;
    float rms_eps_ = 1e-6f;

    const Tensor& get(const std::string& name,
                      const std::vector<int>& shape) const;
    std::vector<float> linear(const std::vector<float>& x, int rows,
                              int in, int out,
                              const std::string& name) const;
    std::vector<float> rms(const std::vector<float>& x, int rows, int dim,
                           const std::string& weight_name) const;
    // Applies the MLP to `rows` stacked input rows at once.
    std::vector<float> mlp(const std::vector<float>& x,
                           const std::string& prefix,
                           const std::vector<int>& linear_indices,
                           int rows) const;
};

}  // namespace oxbot
