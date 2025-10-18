#pragma once

#include "api_backend.h"

#ifdef ENABLE_API_BACKENDS
#include <memory>
#include "../nlohmann/json.hpp"
#endif

/// @brief Backend manager for OpenAI API with integrated context management
class OpenAIBackend : public ApiBackend {
public:
    explicit OpenAIBackend(size_t max_context_tokens);
    ~OpenAIBackend() override;

    bool initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    std::string generate_from_session(const SessionContext& session, int max_tokens = 0) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;

    /// @brief Set custom API base URL (must be called before initialize)
    void set_api_base(const std::string& api_base);

private:
    /// @brief Make HTTP POST request to OpenAI API
    /// @param json_payload Request payload
    /// @return API response
    std::string make_api_request(const std::string& json_payload);

    /// @brief Make HTTP GET request to OpenAI API
    /// @param endpoint API endpoint (e.g., "/models/gpt-4")
    /// @return API response
    std::string make_get_request(const std::string& endpoint);

    /// @brief Query model info from OpenAI API to get context size
    /// @param model_name Model name to query
    /// @return Context size in tokens, or 0 if failed
    size_t query_model_context_size(const std::string& model_name);

    /// @brief Parse OpenAI API response
    /// @param response_json API response JSON
    /// @return Generated text content
    std::string parse_openai_response(const std::string& response_json);

#ifdef ENABLE_API_BACKENDS
    std::string api_endpoint_ = "https://api.openai.com/v1/chat/completions";
    nlohmann::json tools_json_;  // Cached tools array for API requests (OpenAI format)
    // Note: max_context_size_ is inherited from BackendManager base class
#endif
};