
#pragma once

#include "shepherd.h"
#include "backends/backend.h"
#include "nlohmann/json.hpp"
#include "http_client.h"
#include <vector>
#include <string>
#include <deque>
#include <chrono>
#include <mutex>
#include <thread>

class ApiBackend : public Backend {
public:
    // Common API backend configuration (all API backends can use these)
    long timeout_seconds = 120;
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

    explicit ApiBackend(size_t max_context_tokens);
    virtual ~ApiBackend() = default;

    // Parse backend configuration (sampling parameters)
    void parse_backend_config() override;

    // Initialize backend (query context size if needed, calibrate tokens)
    void initialize(Session& session) override;

    // Main transactional message interface with eviction
    Response add_message(Session& session,
                        Message::Type type,
                        const std::string& content,
                        const std::string& tool_name = "",
                        const std::string& tool_id = "",
                        int prompt_tokens = 0,
                        int max_tokens = 0) override;

    // Streaming version - base implementation (backends override for real streaming)
    Response add_message_stream(Session& session,
                              Message::Type type,
                              const std::string& content,
                              StreamCallback callback,
                              const std::string& tool_name = "",
                              const std::string& tool_id = "",
                              int prompt_tokens = 0,
                              int max_tokens = 0) override;

    // Stateless generation from Session (for server with prefix caching)
    Response generate_from_session(const Session& session, int max_tokens = 0, StreamCallback callback = nullptr) override;

    // Count tokens for a message (formats JSON and estimates using EMA)
    int count_message_tokens(Message::Type type,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;

    /// @brief Update the EMA-tracked chars/token ratio
    /// Used to correct estimator based on actual backend measurements
    /// @param ratio New chars/token ratio to set
    void set_chars_per_token(float ratio) { chars_per_token = ratio; }

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
    /// @param type Type of new message (USER, ASSISTANT, TOOL, etc.)
    /// @param content Content of new message
    /// @param tool_name Tool name if this is a tool result
    /// @param tool_id Tool call ID if required by API
    /// @param max_tokens Maximum tokens for assistant response (0 = no limit)
    /// @return JSON object ready to send
    virtual nlohmann::json build_request(const Session& session,
                                         Message::Type type,
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
    /// Starts at 2.5 chars/token (conservative for code-heavy content), refines based on API responses
    float chars_per_token = 2.5f;

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

    // Rate limiting tracking
    std::deque<std::chrono::steady_clock::time_point> request_timestamps_;
    std::mutex rate_limit_mutex_;

    /// @brief Enforce rate limits by sleeping if necessary
    void enforce_rate_limits();

    /// @brief Estimate tokens for a single message using EMA ratio
    /// @param content Message content
    /// @return Estimated token count
    int estimate_message_tokens(const std::string& content) const;

    /// @brief Update chars_per_token ratio based on API response
    /// @param total_chars Total characters in messages sent
    /// @param actual_tokens Actual token count from API response
    void update_token_ratio(int total_chars, int actual_tokens);

    /// @brief Update message token counts from API response
    /// Updates the last user message with prompt tokens and adds assistant message with completion tokens
    /// Also updates EMA ratio for better future estimates
    /// @param prompt_tokens Total prompt tokens from API (all messages sent)
    /// @param completion_tokens Assistant response tokens from API
    void update_message_tokens_from_api(int prompt_tokens, int completion_tokens);

    /// @brief Proactively evict messages using token estimation
    /// Updates message token counts based on estimation, then calls eviction
    /// @param estimated_tokens Total estimated tokens in context
    void evict_with_estimation(int estimated_tokens);
};

#if 0
#include "http_client.h"
#include "nlohmann/json.hpp"
#include <vector>
#include <functional>

/// @brief Simple context manager for API backends
/// Just stores messages in memory - no special formatting
/// Each API backend's generate() reads messages and formats for their specific API
class ApiContextManager : public ContextManager {
public:
    explicit ApiContextManager(size_t max_context_tokens);

    /// @brief Not used by API backends - they read messages directly
    std::string get_context_for_inference() override;

    /// @brief Use EMA-tracked chars/token ratio for estimation
    int count_tokens(const std::string& text) override;

    /// @brief Calculate JSON overhead
    int calculate_json_overhead() const override;

    /// @brief Update the EMA-tracked chars/token ratio from backend
    void set_chars_per_token(float ratio) { chars_per_token = ratio; }

private:
    float chars_per_token = 4.0f;  ///< EMA-tracked chars/token ratio
};

/// @brief Generic response structure for API backends
/// Backend implementations parse HTTP responses into this common format
struct ApiResponse {
    // Success fields
    std::string content;                ///< The actual generated text
    int prompt_tokens = 0;              ///< Input token count from API
    int completion_tokens = 0;          ///< Output token count from API

    // Error fields
    bool is_error = false;              ///< True if this is an error response
    int error_code = 0;                 ///< HTTP status code if error
    std::string error_message;          ///< Parsed error message from API
    std::string error_type;             ///< Error classification: "context_overflow", "rate_limit", "auth", etc.

    // Raw data for debugging
    std::string raw_response;           ///< Original JSON response body

    // Tool calls (if any)
    std::string tool_calls_json;        ///< JSON string of tool calls if present
};

/// @brief Base class for all API-based backends (Anthropic, OpenAI, Gemini, Grok, Ollama)
/// Provides common functionality:
/// - Context management via ContextManager member (inherited from Backend)
/// - Default implementations of add_*_message() methods
/// - Common HTTP utilities
/// Each derived API backend needs to:
/// - Implement generate() - format context to API-specific JSON, send HTTP, parse response
/// - Build tools_json_ in initialize() - API-specific tools format
/// - Implement API-specific endpoints/auth
class ApiBackend : public Backend {
public:
    // Common API backend configuration (all API backends can use these)
    long timeout_seconds = 120;
    int max_retries = 3;
    long connect_timeout_seconds = 30;

    explicit ApiBackend(size_t max_context_tokens);
    virtual ~ApiBackend() = default;

    /// @brief Add a system message to the context
    /// Default implementation adds to context manager
    /// @param content System message content
    void add_system_message(const std::string& content) override;

    /// @brief Add a user message to the context
    /// Default implementation adds to context manager
    /// @param content User message content
    void add_user_message(const std::string& content) override;

    /// @brief Add an assistant message to the context
    /// Default implementation adds to context manager
    /// @param content Assistant message content
    void add_assistant_message(const std::string& content) override;

    /// @brief Add a tool result message to the context
    /// Default implementation adds to context manager
    /// @param tool_name Name of tool that was executed
    /// @param content Tool execution result
    /// @param tool_call_id Optional tool call ID (required by some APIs like OpenAI, Anthropic)
    void add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id = "") override;

    /// @brief API backends are stateless - no KV cache eviction needed
    /// @param tokens_needed Number of tokens that need to be freed
    /// @return Always false (no eviction needed for stateless APIs)
    uint32_t evict_to_free_space(uint32_t tokens_needed) override;

    /// @brief Set tools from external source (for server mode)
    /// @param tools_json JSON array of tools in OpenAI format
    void set_tools_from_json(const std::string& tools_json);

    /// @brief Generate from session context (NEW unified interface)
    /// Base class implements HTTP and retry logic, calling backend-specific format/parse methods
    /// @param session Session context containing system, messages, tools
    /// @param max_tokens Maximum tokens to generate (0 = model default)
    /// @return Generated response text (may contain tool call JSON)
    std::string generate_from_session(const SessionContext& session, int max_tokens = 0) override;

    /// @brief Build SessionContext from current context_manager_ state
    /// Helper for old generate() method to call new generate_from_session()
    /// @param session Output parameter to populate (avoids copy)
    void build_session_from_context(SessionContext& session);

protected:
    void parse_backend_config() override;
    virtual void parse_specific_config(const std::string& json) {
        // Override in derived classes for backend-specific config
    }
#ifdef ENABLE_API_BACKENDS
    /// @brief Shared HTTP client for making API requests
    /// Available for backends that want to use it (OpenAI, Gemini, etc.)
    std::unique_ptr<HttpClient> http_client_;
#endif

    /// @brief Parsed tool information from session.tools
    std::vector<ToolInfo> tools_data_;

    /// @brief Flag to track if tools have been built
    bool tools_built_ = false;

    /// @brief Adaptive token estimation using EMA
    /// Starts at 2.5 chars/token (conservative for code-heavy content), refines based on API responses
    float chars_per_token = 2.5f;

    /// @brief Baseline token count from first API call (System + warmup User message)
    /// This is the fixed baseline we use for all subsequent delta calculations
    int baseline_tokens_ = 0;

    /// @brief Flag to track if this is the first API call (warmup turn)
    /// First call establishes the baseline, subsequent calls use delta calculation
    bool first_call_ = true;

    /// @brief Estimate total tokens in current context using adaptive ratio
    /// @return Estimated token count
    int estimate_context_tokens() const;

    /// @brief Update chars_per_token ratio based on API response
    /// @param total_chars Total characters in messages sent
    /// @param actual_tokens Actual token count from API response
    void update_token_ratio(int total_chars, int actual_tokens);

    /// @brief Proactively evict messages using token estimation
    /// Updates message token counts based on estimation, then calls eviction
    /// Fixes bug where eviction aborted because messages had token_count=0
    /// @param estimated_tokens Total estimated tokens in context
    void evict_with_estimation(int estimated_tokens);

    /// @brief Estimate tokens for a single message using EMA ratio
    /// @param content Message content
    /// @return Estimated token count
    int estimate_message_tokens(const std::string& content) const;

    /// @brief Update message token counts from API response
    /// Updates the last user message with prompt tokens and adds assistant message with completion tokens
    /// Also updates EMA ratio for better future estimates
    /// @param prompt_tokens Total prompt tokens from API (all messages sent)
    /// @param completion_tokens Assistant response tokens from API
    void update_message_tokens_from_api(int prompt_tokens, int completion_tokens);

    // ========== New Architecture: Pure Virtual Methods for Backend Implementations ==========

    /// @brief Format API request from SessionContext
    /// Each backend implements this to convert SessionContext to their API's JSON format
    /// @param session The session context containing messages, system prompt, etc
    /// @param max_tokens Maximum tokens to generate (0 = use model default)
    /// @return JSON string formatted for the specific API
    virtual std::string format_api_request(const SessionContext& session, int max_tokens) = 0;

    /// @brief Extract tokens needed from context overflow error message
    /// Each backend parses its own API's error format to determine how many tokens to evict
    /// @param error_message The error message from the API
    /// @return Number of tokens that need to be freed, or -1 if can't parse
    virtual int extract_tokens_to_evict(const std::string& error_message) = 0;

    /// @brief Parse HTTP response into generic ApiResponse
    /// Each backend implements this to extract content, tokens, errors from their API format
    /// @param http_response The raw HTTP response from the API server
    /// @return Parsed response in generic format
    virtual ApiResponse parse_api_response(const HttpResponse& http_response) = 0;

    /// @brief Get API-specific HTTP headers
    /// Each backend provides their authentication and other required headers
    /// @return Map of header name to header value
    virtual std::map<std::string, std::string> get_api_headers() = 0;

    /// @brief Get API endpoint URL
    /// Each backend provides their specific endpoint
    /// @return Full URL for API requests
    virtual std::string get_api_endpoint() = 0;
};
#endif
