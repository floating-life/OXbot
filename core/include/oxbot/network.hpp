#pragma once
#include <map>
#include <string>
#include <vector>

namespace oxbot {

std::string sha256_hex(const std::vector<unsigned char>& bytes);

// Fixed 64x2 transformer contract. The model file contains float32 tensors;
// loading validates the exact architecture, feature/rule versions and SHA.
class CandidateNetwork {
public:
    bool load(const std::string& path);
    bool ready() const { return ready_; }
    const std::string& status() const { return status_; }
    const std::string& payload_sha() const { return sha_; }
    // The model header may identify the candidate build.  Binaries exported
    // before this field existed (including frozen v1) deliberately resolve
    // to the historical value so their diagnostics remain reproducible.
    const std::string& model_version() const { return model_version_; }
    // Optional policy defaults are carried in newly exported model headers.
    // Frozen v1 binaries predate this field and intentionally resolve to raw.
    const std::string& selection_strategy() const { return selection_strategy_; }
    bool has_selection_manifest() const { return selection_manifest_present_; }
    std::vector<float> score(const std::vector<int>& tokens,
                             const std::vector<float>& state,
                             const std::vector<std::vector<float>>& actions,
                             std::map<std::string, std::vector<float>>* trace = nullptr) const;
private:
    struct Tensor { std::vector<int> shape; std::vector<float> data; };
    std::map<std::string, Tensor> tensors_;
    bool ready_ = false;
    std::string status_ = "model_not_loaded";
    std::string sha_;
    std::string model_version_ = "oxbot-model-v1";
    std::string selection_strategy_ = "raw";
    bool selection_manifest_present_ = false;
    const Tensor& get(const std::string& name, const std::vector<int>& shape) const;
    std::vector<float> linear(const std::vector<float>& x, int rows, int in, int out,
                              const std::string& name) const;
    void norm(std::vector<float>& x, int rows, const std::string& name) const;
};

} // namespace oxbot
