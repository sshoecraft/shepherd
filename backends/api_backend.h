#pragma once

#include "../backend_manager.h"
#include "../http_client.h"
#include "../nlohmann/json.hpp"
#include <memory>
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

    /// @brief Not used - token counts come from API responses
    /// @return Always returns 0 (unused)
    int count_tokens(const std::string& text) override;

    /// @brief Calculate JSON overhead
    int calculate_json_overhead() const override;
};

/// @brief Base class for all API-based backends (Anthropic, OpenAI, Gemini, Grok, Ollama)
/// Provides common functionality:
/// - Context management via ContextManager member (inherited from BackendManager)
/// - Default implementations of add_*_message() methods
/// - Common HTTP utilities
/// Each derived API backend needs to:
/// - Implement generate() - format context to API-specific JSON, send HTTP, parse response
/// - Build tools_json_ in initialize() - API-specific tools format
/// - Implement API-specific endpoints/auth
class ApiBackend : public BackendManager {
public:
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
    /// Each API backend implements this to format SessionContext to their specific API format
    /// @param session Session context containing system, messages, tools
    /// @param max_tokens Maximum tokens to generate (0 = model default)
    /// @return Generated response text (may contain tool call JSON)
    std::string generate_from_session(const SessionContext& session, int max_tokens = 0) override = 0;

protected:
#ifdef ENABLE_API_BACKENDS
    /// @brief Shared HTTP client for making API requests
    /// Available for backends that want to use it (OpenAI, Gemini, etc.)
    std::unique_ptr<HttpClient> http_client_;
#endif

    /// @brief Parsed tool information from ToolRegistry
    std::vector<ToolInfo> tools_data_;

    /// @brief Flag to track if tools have been built from registry
    bool tools_built_ = false;

    /// @brief Build tools data from ToolRegistry
    /// Populates tools_data_ with tool info from registry
    /// Should be called once during first generate() call
    void build_tools_from_registry();

    /// @brief Adaptive token estimation using EMA
    /// Starts at 4.0 chars/token, refines based on API responses
    float chars_per_token_ = 4.0f;

    /// @brief Baseline token count from first API call (System + warmup User message)
    /// This is the fixed baseline we use for all subsequent delta calculations
    int baseline_tokens_ = 0;

    /// @brief Flag to track if this is the first API call (warmup turn)
    /// First call establishes the baseline, subsequent calls use delta calculation
    bool first_call_ = true;

    /// @brief Estimate total tokens in current context using adaptive ratio
    /// @return Estimated token count
    int estimate_context_tokens() const;

    /// @brief Update chars_per_token_ ratio based on API response
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

    /// @brief Execute API request with automatic retry on context overflow
    /// Implements reactive eviction: when API returns context overflow error,
    /// evicts oldest messages and retries (only in CLI mode, not server mode)
    /// @param build_request_func Lambda that builds the JSON request from current context
    /// @param execute_request_func Lambda that makes the API call and returns response
    /// @param max_retries Maximum number of retry attempts (default 3)
    /// @return Final response string after successful request or all retries exhausted
    std::string generate_with_retry(
        std::function<nlohmann::json()> build_request_func,
        std::function<std::string(const std::string&)> execute_request_func,
        int max_retries = 3
    );
};
