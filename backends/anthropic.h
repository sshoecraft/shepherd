#pragma once

#include "api_backend.h"

#ifdef ENABLE_API_BACKENDS
#include <curl/curl.h>
#include <memory>
#include "../nlohmann/json.hpp"
#endif

/// @brief Model information structure
struct AnthropicModelInfo {
    std::string model_name;
    size_t context_window;
    int max_output_tokens;
};

/// @brief Anthropic-specific tokenizer using custom implementation
class AnthropicTokenizer : public Tokenizer {
public:
    explicit AnthropicTokenizer(const std::string& model_name = "claude-3-sonnet");

    int count_tokens(const std::string& text) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;
    std::string get_tokenizer_name() const override;

private:
    std::string model_name_;
    // Custom Anthropic tokenization implementation will be added
};

/// @brief Backend manager for Anthropic Claude API with integrated context management
class AnthropicBackend : public ApiBackend {
public:
    explicit AnthropicBackend(size_t max_context_tokens);
    ~AnthropicBackend() override;

    bool initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    std::string generate_from_session(const SessionContext& session, int max_tokens = 0) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;

private:
    std::string make_api_request(const std::string& json_payload);
    std::string make_get_request(const std::string& endpoint);
    size_t query_model_context_size(const std::string& model_name);
    int query_max_output_tokens(const std::string& model_name);
    std::string parse_anthropic_response(const std::string& response_json);
    void discover_api_metadata();
    static const AnthropicModelInfo* get_model_info(const std::string& model_name);

#ifdef ENABLE_API_BACKENDS
    CURL* curl_ = nullptr;
    std::string api_endpoint_ = "https://api.anthropic.com/v1/messages";
    std::string api_version_;
    nlohmann::json tools_json_;  // Cached tools array for API requests
    bool tools_built_ = false;   // Flag to track if tools have been built
#endif
};