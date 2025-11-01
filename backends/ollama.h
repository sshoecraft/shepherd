#pragma once

#include "api.h"
#include "nlohmann/json.hpp"
#include <memory>

/// @brief Backend manager for Ollama API (Native endpoint)
class OllamaBackend : public ApiBackend {
public:
    std::string api_endpoint = "http://localhost:11434/api/chat";

    explicit OllamaBackend(size_t context_size);
    ~OllamaBackend() override;

    // Implement pure virtual methods from ApiBackend
    Response parse_http_response(const HttpResponse& http_response) override;

    nlohmann::json build_request_from_session(const Session& session, int max_tokens) override;

    nlohmann::json build_request(const Session& session,
                                  Message::Type type,
                                  const std::string& content,
                                  const std::string& tool_name,
                                  const std::string& tool_id,
                                  int max_tokens = 0) override;

    std::string parse_response(const nlohmann::json& response) override;
    int extract_tokens_to_evict(const HttpResponse& response) override;
    std::map<std::string, std::string> get_api_headers() override;
    std::string get_api_endpoint() override;

    /// @brief Query model info from Ollama's /api/show endpoint
    /// @param model_name Model to query
    /// @return Context size in tokens, or 0 if query failed
    size_t query_model_context_size(const std::string& model_name) override;

    /// @brief Initialize backend - query model if not specified, then detect context size
    void initialize(Session& session) override;

private:
    /// @brief Query available models from Ollama's /api/tags endpoint
    /// @return First available model name, or empty string if query failed
    std::string query_available_model();
};
