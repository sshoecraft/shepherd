#pragma once

#include "api_backend.h"

#ifdef ENABLE_API_BACKENDS
#include <curl/curl.h>
#include <memory>
#endif

/// @brief Backend manager for Google Gemini API
class GeminiBackend : public ApiBackend {
public:
    std::string api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/";

    explicit GeminiBackend(size_t max_context_tokens);
    ~GeminiBackend() override;

    bool initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    // generate_from_session now uses base class implementation that calls our format/parse methods
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;

protected:
    // New architecture: Required virtual methods from ApiBackend
    std::string format_api_request(const SessionContext& session, int max_tokens) override;
    int extract_tokens_to_evict(const std::string& error_message) override;
    ApiResponse parse_api_response(const HttpResponse& http_response) override;
    std::map<std::string, std::string> get_api_headers() override;
    std::string get_api_endpoint() override;
    void parse_specific_config(const std::string& json) override;

private:
    std::string make_api_request(const std::string& json_payload);
    std::string make_get_request(const std::string& endpoint);
    size_t query_model_context_size(const std::string& model_name);
    std::string parse_gemini_response(const std::string& response_json);

#ifdef ENABLE_API_BACKENDS
    CURL* curl_ = nullptr;
    size_t context_size_ = 128000; // Gemini Pro context size
#endif
};