#pragma once

#include "../backend_manager.h"
#include "../http_client.h"
#include <memory>
#include <vector>

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
};
