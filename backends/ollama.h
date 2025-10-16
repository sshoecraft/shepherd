#pragma once

#include "api_backend.h"
#include "../nlohmann/json.hpp"

#ifdef ENABLE_API_BACKENDS
#include <memory>
#endif

/// @brief Ollama-specific tokenizer (uses same approximation as OpenAI)
class OllamaTokenizer : public Tokenizer {
public:
    explicit OllamaTokenizer(const std::string& model_name);

    int count_tokens(const std::string& text) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;
    std::string get_tokenizer_name() const override;

private:
    std::string model_name_;
};

/// @brief Backend manager for Ollama API (OpenAI-compatible)
class OllamaBackend : public ApiBackend {
public:
    explicit OllamaBackend(size_t max_context_tokens);
    ~OllamaBackend() override;

    bool initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;

    /// @brief Set custom API base URL (must be called before initialize)
    void set_api_base(const std::string& api_base);

private:
    /// @brief Make HTTP POST request to Ollama API
    std::string make_api_request(const std::string& json_payload);

    /// @brief Make HTTP GET request to Ollama API
    std::string make_get_request(const std::string& endpoint);

    /// @brief Query model info to get context size
    size_t query_model_context_size(const std::string& model_name);

    /// @brief Parse Ollama API response (OpenAI-compatible format)
    std::string parse_ollama_response(const std::string& response_json);

#ifdef ENABLE_API_BACKENDS
    std::string api_endpoint_ = "http://localhost:11434/v1/chat/completions";
    // Note: max_context_size_ is inherited from BackendManager base class

    // Tools support
    nlohmann::json tools_json_;
    bool tools_built_ = false;
#endif
};
