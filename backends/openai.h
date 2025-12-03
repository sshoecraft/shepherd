
#pragma once

#include "api.h"
#include "models.h"
#include <memory>
#include "nlohmann/json.hpp"
#include "http_client.h"

/// @brief Backend manager for OpenAI API with integrated context management
class OpenAIBackend : public ApiBackend {
public:
    std::string api_endpoint = "https://api.openai.com/v1/chat/completions";
    std::string api_key;
    ModelConfig model_config;
    // model_name is inherited from Backend base class

    // Azure OpenAI specific
    std::string deployment_name;  // Azure deployment name (replaces model in URL)
    std::string api_version;      // Azure API version (e.g., "2024-06-01")

    explicit OpenAIBackend(size_t context_size);
    ~OpenAIBackend() override;

	// Streaming support
	Response add_message_stream(Session& session,
	                          Message::Type type,
	                          const std::string& content,
	                          StreamCallback callback,
	                          const std::string& tool_name = "",
	                          const std::string& tool_id = "",
	                          int prompt_tokens = 0,
	                          int max_tokens = 0) override;

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

	// Override initialize to add OpenAI-specific setup
	void initialize(Session& session) override;

	// Override query_model_context_size to use Models database
	size_t query_model_context_size(const std::string& model_name) override;

private:
	/// @brief Query available model from OpenAI API server
	/// @return Model name from server, or empty string if failed
	std::string query_available_model();

	/// @brief Make HTTP GET request to OpenAI API
	/// @param endpoint API endpoint (e.g., "/models")
	/// @return API response body
	std::string make_get_request(const std::string& endpoint);

	/// @brief Test if the server supports SSE streaming
	/// Sets streaming_enabled based on test result
	void test_streaming_support();
};
