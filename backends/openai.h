
#pragma once

#include "api.h"
#include <memory>
#include "nlohmann/json.hpp"
#include "http_client.h"

/// @brief Backend manager for OpenAI API with integrated context management
class OpenAIBackend : public ApiBackend {
public:
    std::string api_endpoint = "https://api.openai.com/v1/chat/completions";
    std::string api_key;
    std::string model_name;

    explicit OpenAIBackend(size_t context_size);
    ~OpenAIBackend() override;

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

private:
	/// @brief Query available model from OpenAI API server
	/// @return Model name from server, or empty string if failed
	std::string query_available_model();

	/// @brief Query model info from OpenAI API to get context size
	/// @param model_name Model name to query
	/// @return Context size in tokens, or 0 if failed
	size_t query_model_context_size(const std::string& model_name);

	/// @brief Make HTTP GET request to OpenAI API
	/// @param endpoint API endpoint (e.g., "/models")
	/// @return API response body
	std::string make_get_request(const std::string& endpoint);
};
