#pragma once

#include "api_backend.h"

#ifdef ENABLE_API_BACKENDS
#include <curl/curl.h>
#include <memory>
#endif

/// @brief Grok-specific tokenizer using tiktoken library (OpenAI-compatible)
class GrokTokenizer : public Tokenizer {
public:
    explicit GrokTokenizer(const std::string& model_name = "grok-1");

    int count_tokens(const std::string& text) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;
    std::string get_tokenizer_name() const override;

private:
    std::string model_name_;
    // tiktoken implementation will be added when library is integrated (same as OpenAI)
};

/// @brief Backend manager for xAI Grok API (OpenAI-compatible)
class GrokBackend : public ApiBackend {
public:
    explicit GrokBackend(size_t max_context_tokens);
    ~GrokBackend() override;

    bool initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;

private:
    std::string make_api_request(const std::string& json_payload);
    std::string make_get_request(const std::string& endpoint);
    size_t query_model_context_size(const std::string& model_name);
    std::string parse_grok_response(const std::string& response_json);

#ifdef ENABLE_API_BACKENDS
    CURL* curl_ = nullptr;
    std::string api_endpoint_ = "https://api.x.ai/v1/chat/completions";
    size_t max_context_size_ = 128000; // Grok context size
#endif
};