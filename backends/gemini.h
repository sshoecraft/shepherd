#pragma once

#include "api.h"
#include "models.h"
#include <memory>
#include "nlohmann/json.hpp"

/// @brief Backend manager for Google Gemini API
class GeminiBackend : public ApiBackend {
public:
    std::string api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/";
    std::string api_key;
    ModelConfig model_config;
    // model_name is inherited from Backend base class

    explicit GeminiBackend(size_t context_size);
    ~GeminiBackend() override;

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
    std::string get_streaming_endpoint();

    // Override to provide streaming support
    Response add_message_stream(Session& session, Message::Type type, const std::string& content,
                               StreamCallback callback,
                               const std::string& tool_name = "", const std::string& tool_id = "",
                               int prompt_tokens = 0, int max_tokens = 0) override;

    // Override initialize to add Gemini-specific setup
    void initialize(Session& session) override;

private:
    size_t query_model_context_size(const std::string& model_name) override;
    std::string make_get_request(const std::string& url);
};