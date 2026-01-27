
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

    // OpenAI strict mode - skip non-standard params like top_k, repetition_penalty
    bool openai_strict = false;

    OpenAIBackend(size_t context_size, Session& session, EventCallback callback);
    ~OpenAIBackend() override;

	// Override generate_from_session to provide streaming for API server mode
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

	// Override query_model_context_size to use Models database
	size_t query_model_context_size(const std::string& model_name) override;

protected:
	// Override fetch_models to query /v1/models endpoint
	std::vector<std::string> fetch_models() override;

private:
	/// @brief Make HTTP GET request to OpenAI API
	/// @param endpoint API endpoint (e.g., "/models")
	/// @return API response body
	std::string make_get_request(const std::string& endpoint);

	/// @brief Test if the server supports SSE streaming
	/// Sets streaming_enabled based on test result
	void test_streaming_support();
};
