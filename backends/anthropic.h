#pragma once

#include "api.h"
#include "models.h"
#include <memory>
#include "nlohmann/json.hpp"

/// @brief Backend manager for Anthropic Claude API with integrated context management
class AnthropicBackend : public ApiBackend {
public:
    std::string api_endpoint = "https://api.anthropic.com/v1/messages";
    std::string api_key;
    ModelConfig model_config;
    // model_name is inherited from Backend base class

    AnthropicBackend(size_t context_size, Session& session, EventCallback callback);
    ~AnthropicBackend() override;

    // Override generate_from_session to provide true streaming for API server mode
    void generate_from_session(Session& session, int max_tokens = 0) override;

    // Implement pure virtual methods from ApiBackend
    Response parse_http_response(const HttpResponse& http_response) override;

    nlohmann::json build_request_from_session(const Session& session, int max_tokens) override;

    nlohmann::json build_request(const Session& session,
                                  Message::Role role,
                                  const std::string& content,
                                  const std::string& tool_name,
                                  const std::string& tool_id,
                                  int max_tokens = 0) override;

    std::string parse_response(const nlohmann::json& response) override;
    int extract_tokens_to_evict(const HttpResponse& response) override;
    std::map<std::string, std::string> get_api_headers() override;
    std::string get_api_endpoint() override;

    // Override set_model to update model_config when model changes
    void set_model(const std::string& model) override;

    // Override query_model_context_size to use Models database
    size_t query_model_context_size(const std::string& model_name) override;

protected:
    // Query available models from Anthropic API
    std::vector<std::string> fetch_models() override;

private:
    /// @brief Make HTTP GET request to Anthropic docs/API
    /// @param url Full URL to fetch
    /// @return Response body
    std::string make_get_request(const std::string& url);

    std::string api_version;
};