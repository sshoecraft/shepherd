
#pragma once

#include "shepherd.h"
#include "backend.h"
#include "nlohmann/json.hpp"
#include "http_client.h"
#include <vector>
#include <string>
#include <deque>
#include <chrono>
#include <mutex>
#include <thread>
#include <memory>

// Forward declaration
class SharedOAuthCache;

class ApiBackend : public Backend {
public:
    // Common API backend configuration (all API backends can use these)
    long timeout_seconds = 300;  // 5 minutes default
    int max_retries = 3;
    long connect_timeout_seconds = 30;

    // Sampling parameters (common across API backends)
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 0;  // 0 = disabled (backend default)
    float frequency_penalty = 0.5f;  // OpenAI, TensorRT-LLM
    float presence_penalty = 0.0f;   // OpenAI
    float repeat_penalty = 1.2f;     // Ollama, TensorRT-LLM (as repetition_penalty)
    std::vector<std::string> stop_sequences;  // Stop generation at these sequences

    // Sampling mode: when false, don't send sampling parameters in requests
    bool sampling = true;

    ApiBackend(size_t max_context_tokens, Session& session, EventCallback callback);
    virtual ~ApiBackend() = default;

    // Parse backend configuration (sampling parameters)
    void parse_backend_config() override;

    // Generation from Session
    // All output flows through callback, session token counts are updated
    void generate_from_session(Session& session, int max_tokens = 0) override;

    // Count tokens for a message (formats JSON and estimates using EMA)
    int count_message_tokens(Message::Role role,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;

    /// @brief Update the EMA-tracked chars/token ratio
    /// Used to correct estimator based on actual backend measurements
    /// @param ratio New chars/token ratio to set
    void set_chars_per_token(float ratio) { chars_per_token = ratio; }

    /// @brief Set shared OAuth cache for per-request backend mode
    /// When set, OAuth tokens are cached globally instead of per-backend
    /// @param cache Shared OAuth cache instance
    void set_shared_oauth_cache(std::shared_ptr<SharedOAuthCache> cache);

    // Override to provide default tool call filtering for API backends
    // Include common formats used by various models:
    // - XML-style tags (Claude, etc.)
    // - JSON format (when model outputs raw JSON tool calls)
    std::vector<std::string> get_tool_call_markers() const override {
        return {
            // XML-style markers
            "<tool_call", "<function_call", "<tools",
            "<execute_command", "<.execute_command",
            "<read", "<.read", "<write", "<.write",
            "<bash", "<.bash", "<edit", "<.edit",
            "<glob", "<.glob", "<grep", "<.grep",
            // JSON-style markers (for models that output raw JSON)
            "{\"name\"", "{ \"name\""
        };
    }

    std::vector<std::string> get_tool_call_end_markers() const override {
        return {
            // XML-style end markers
            "</tool_call>", "</function_call>", "</tools>",
            "</execute_command>", "</.execute_command>",
            "</read>", "</.read>", "</write>", "</.write>",
            "</bash>", "</.bash>", "</edit>", "</.edit>",
            "</glob>", "</.glob>", "</grep>", "</.grep>",
            // JSON ends with closing brace (but need to match balanced braces)
            "}\n", "} ", "}$"
        };
    }

protected:
    /// @brief Flag to track if we've tested streaming capability
    bool streaming_tested = false;

    /// @brief Output content through common filter (backticks, buffering)
    /// API backends call this for streaming output
    /// @param text Text to output
    /// @param len Length of text
    /// @return true to continue, false if callback requested cancellation
    bool output(const char* text, size_t len);
    bool output(const std::string& text) { return output(text.c_str(), text.length()); }

    /// @brief Calibrate token counts by sending probe messages
    /// Sends system+"." and system+tools+"." to get exact token counts
    /// Backend-agnostic, uses build_request() to format requests
    void calibrate_token_counts(Session& session);

    // Concrete backends must implement these:

    /// @brief Parse HTTP response into unified Response structure
    /// @param http_response Raw HTTP response from API
    /// @return Response struct with content, tool_calls, tokens, errors
    virtual Response parse_http_response(const HttpResponse& http_response) = 0;

    /// @brief Build API request JSON from complete session (for generate_from_session)
    /// @param session Session with all messages already included
    /// @param max_tokens Maximum tokens to generate (0 = unlimited)
    /// @return JSON request body for API
    virtual nlohmann::json build_request_from_session(const Session& session, int max_tokens) = 0;

    /// @brief Build API request JSON from session and new message (for add_message)
    /// @param session Current session state
    /// @param role Role of new message (USER, ASSISTANT, TOOL_RESPONSE, etc.)
    /// @param content Content of new message
    /// @param tool_name Tool name if this is a tool result
    /// @param tool_id Tool call ID if required by API
    /// @param max_tokens Maximum tokens for assistant response (0 = no limit)
    /// @return JSON object ready to send
    virtual nlohmann::json build_request(const Session& session,
                                         Message::Role role,
                                         const std::string& content,
                                         const std::string& tool_name,
                                         const std::string& tool_id,
                                         int max_tokens = 0) = 0;

    /// @brief Parse API response to extract generated text
    /// @param response JSON response from API
    /// @return Generated text content
    virtual std::string parse_response(const nlohmann::json& response) = 0;

    /// @brief Check if HTTP response indicates context length error
    /// @param response HTTP response from API
    /// @return Number of tokens to evict, or -1 if not a context error
    virtual int extract_tokens_to_evict(const HttpResponse& response) = 0;

    /// @brief Get API headers (including auth)
    /// @return Map of header name to value
    virtual std::map<std::string, std::string> get_api_headers() = 0;

    /// @brief Get API endpoint URL
    /// @return Full endpoint URL
    virtual std::string get_api_endpoint() = 0;

    /// @brief Query the model's context size from the API
    /// @param model_name Name of the model to query
    /// @return Context size in tokens, or 0 if not available
    virtual size_t query_model_context_size(const std::string& model_name) = 0;

    // HTTP client for making requests
    std::unique_ptr<HttpClient> http_client;

    // OAuth 2.0 token management
    struct OAuthToken {
        std::string access_token;
        std::string token_type = "Bearer";
        time_t expires_at = 0;  // Unix timestamp when token expires
        bool is_valid() const { return !access_token.empty() && time(nullptr) < expires_at; }
    };

    /// @brief Acquire OAuth 2.0 access token using client credentials
    /// @param client_id OAuth client ID
    /// @param client_secret OAuth client secret
    /// @param token_url OAuth token endpoint URL
    /// @param scope OAuth scope (optional)
    /// @return OAuth token structure with access_token and expiry
    OAuthToken acquire_oauth_token(const std::string& client_id,
                                    const std::string& client_secret,
                                    const std::string& token_url,
                                    const std::string& scope = "");

    /// @brief Ensure we have a valid OAuth token, refreshing if needed
    /// @return true if valid token available, false otherwise
    bool ensure_valid_oauth_token();

protected:
    /// @brief Adaptive token estimation using EMA
    /// Starts at 1.75 chars/token (conservative for dense content like hex), refines based on API responses
    float chars_per_token = 1.75f;

    /// @brief Baseline token count from first API call (System + warmup User message)
    /// This is the fixed baseline we use for all subsequent delta calculations
    int baseline_tokens_ = 0;

    /// @brief Flag to track if this is the first API call (warmup turn)
    /// First call establishes the baseline, subsequent calls use delta calculation
    bool first_call_ = true;

    // OAuth 2.0 configuration and token storage
    std::string oauth_client_id_;
    std::string oauth_client_secret_;
    std::string oauth_token_url_;
    std::string oauth_scope_;
    OAuthToken oauth_token_;

    // Shared OAuth cache for per-request backend mode
    std::shared_ptr<SharedOAuthCache> shared_oauth_cache_;

    // Rate limiting tracking
    std::deque<std::chrono::steady_clock::time_point> request_timestamps_;
    std::mutex rate_limit_mutex_;

    /// @brief Enforce rate limits by sleeping if necessary
    void enforce_rate_limits();

    /// @brief Estimate tokens for a single message using EMA ratio
    /// @param content Message content
    /// @return Estimated token count
    int estimate_message_tokens(const std::string& content) const;

    /// @brief Update session token counts from API response using delta tracking
    /// Computes delta from prompt_tokens to derive actual per-message token counts,
    /// corrects Message.tokens for recently added messages, and refines EMA ratio.
    /// Called from all streaming and non-streaming generate_from_session() paths.
    /// @param session Session to update
    /// @param prompt_tokens Total prompt tokens from API response
    /// @param completion_tokens Completion tokens from API response
    void update_session_tokens(Session& session, int prompt_tokens, int completion_tokens);

    /// @brief Add a tool response message to the session
    /// @param session Session to add to
    /// @param content Tool result content
    /// @param tool_name Name of the tool
    /// @param tool_id Tool call ID for correlation
    void add_tool_response(Session& session,
                           const std::string& content,
                           const std::string& tool_name,
                           const std::string& tool_id);
};


