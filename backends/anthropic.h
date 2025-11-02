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

    explicit AnthropicBackend(size_t context_size);
    ~AnthropicBackend() override;

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

    // Override initialize to add Anthropic-specific setup
    void initialize(Session& session) override;

    // Override query_model_context_size to use Models database
    size_t query_model_context_size(const std::string& model_name) override;

private:
    /// @brief Make HTTP GET request to Anthropic docs/API
    /// @param url Full URL to fetch
    /// @return Response body
    std::string make_get_request(const std::string& url);

    std::string api_version;
};