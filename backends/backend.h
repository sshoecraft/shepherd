
#pragma once

#include "session.h"
#include "message.h"
#include "tools/tool_parser.h"
#include <vector>
#include <string>

// Unified response structure returned by all backends
struct Response {
    // Response codes for different error conditions
    enum Code {
        SUCCESS = 0,              // Request succeeded
        ERROR = 1,                // Generic error
        CONTEXT_FULL = 2,         // Context window full (need eviction)
        MAX_TOKENS_TOO_HIGH = 3   // max_tokens parameter exceeds available space
    };

    Code code = SUCCESS;                             // Response code (SUCCESS, ERROR, etc.)
    bool success = true;                             // Backward compat: false if error occurred
    std::string content;                             // Assistant's text response
    std::vector<ToolParser::ToolCall> tool_calls;   // Parsed tool calls (if any)
    int prompt_tokens = 0;                           // Tokens used in prompt
    int completion_tokens = 0;                       // Tokens generated
    std::string finish_reason;                       // "stop", "tool_calls", "length", "error"
    std::string error;                               // Error message (empty if success)

    // For CONTEXT_FULL or MAX_TOKENS_TOO_HIGH errors
    int overflow_tokens = 0;                         // How many tokens over the limit
    int actual_prompt_tokens = 0;                    // Actual prompt size measured by backend (if known)
};

// Forward decl
class Session;

class Backend {
public:
    explicit Backend(size_t context_size);
    virtual ~Backend() = default;

    // Initialize backend (validate setup, calibrate, etc.)
    // Called once after construction before first use
    virtual void initialize(Session& session) {}

    // Main transactional message interface
    // Adds message and generates response, handling eviction if needed
    // Only adds to session on success
    // Returns Response with success flag, content, tool_calls, tokens
    // prompt_tokens: estimated tokens for this message (0 = backend calculates)
    // max_tokens: max tokens for assistant response (0 = auto-calculate from available space)
    virtual Response add_message(Session& session,
                                Message::Type type,
                                const std::string& content,
                                const std::string& tool_name = "",
                                const std::string& tool_id = "",
                                int prompt_tokens = 0,
                                int max_tokens = 0) = 0;

    // Stateless generation from Session (for server with prefix caching)
    virtual Response generate_from_session(const Session& session, int max_tokens = 0) = 0;

    // Tool support
    virtual std::vector<std::string> get_tool_call_markers() const { return {}; }

    /// @brief Count tokens for a message (without adding to context)
    /// Formats exactly as add_message() would, but only returns token count
    /// Used for proactive eviction to determine if message will fit
    /// API backends: estimates using EMA on formatted JSON
    /// GPU backends: tokenizes formatted message
    /// @param type Message type (USER, ASSISTANT, TOOL)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL messages)
    /// @param tool_id Tool call ID (for TOOL messages)
    /// @return Token count for the formatted message
    virtual int count_message_tokens(Message::Type type,
                                     const std::string& content,
                                     const std::string& tool_name = "",
                                     const std::string& tool_id = "") = 0;

    // Public member variables (no getters/setters per RULES.md)
    std::string name;
    std::string backend_name;
    std::string model_name;
    size_t context_size;
    int last_prompt_tokens;
    int last_completion_tokens;
    int context_token_count;
    bool is_local = false;  // true for GPU/local backends (llamacpp, tensorrt), false for API backends

protected:
    virtual void parse_backend_config(const std::string& json) {
        // Default: no-op. Backends override if they have specific config
    }
};

#if 0
#include "context_manager.h"
#include "tools_manager.h"
#include "tokenizer.h"
#include "models.h"
#include "session_context.h"
#include <string>
#include <vector>
#include <memory>

/// @brief Abstract base class for all backend managers
/// Provides unified interface for local (llama.cpp, TensorRT) and API-based (OpenAI, Anthropic, etc.) backends
/// Each backend maintains its own context in native format for efficiency
class BackendManager {
public:
    explicit BackendManager(size_t max_context_tokens);
    virtual ~BackendManager() = default;

    /// @brief Initialize the backend with model/connection
    /// @param model_path_or_name Model file path (local) or model name (API)
    /// @param api_key API key for cloud providers (empty for local)
    /// @param template_path Path to custom chat template file (optional, llamacpp only)
    /// @return True if initialization successful
    virtual bool initialize(const std::string& model_path_or_name, const std::string& api_key = "", const std::string& template_path = "") = 0;

    /// @brief Generate response based on current context
    /// Main orchestrates tool execution - backend just generates once
    /// @param max_tokens Maximum tokens to generate (0 = model default)
    /// @return Generated response text (may contain tool call JSON)
    virtual std::string generate(int max_tokens = 0) = 0;

    /// @brief Add a user message to the context
    /// Backend updates context manager and KV cache if applicable
    /// @param content User message content
    virtual void add_user_message(const std::string& content) = 0;

    /// @brief Add a tool result message to the context
    /// Backend formats appropriately for its protocol and updates KV cache if applicable
    /// @param tool_name Name of tool that was executed
    /// @param content Tool execution result
    /// @param tool_call_id Optional tool call ID (required by some API backends like OpenAI)
    virtual void add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id = "") = 0;

    /// @brief Add an assistant message to the context
    /// Backend updates context manager and KV cache if applicable
    /// @param content Assistant message content
    virtual void add_assistant_message(const std::string& content) = 0;

    /// @brief Add a system message to the context
    /// @param content System message content
    virtual void add_system_message(const std::string& content) = 0;

    /// @brief Process a session context and generate response
    /// Stateless generation from Session (used by server)
    /// Backend reads Session, formats request appropriately, and generates
    /// @param session Session containing system, messages, tools
    /// @param max_tokens Maximum tokens to generate (0 = model default)
    /// @return Response struct with content, tool_calls, tokens, errors
    virtual Response generate_from_session(const Session& session, int max_tokens = 0) = 0;

    /// @brief Get backend name/identifier
    /// @return Backend name (e.g., "llamacpp", "openai", "anthropic")
    virtual std::string get_backend_name() const = 0;

    /// @brief Get model name/path currently loaded
    /// @return Model identifier
    virtual std::string get_model_name() const = 0;

    /// @brief Check if backend is ready for inference
    /// @return True if ready
    virtual bool is_ready() const = 0;

    /// @brief Shutdown and cleanup resources
    virtual void shutdown() = 0;

    /// @brief Get context manager for direct access
    /// @return Reference to context manager
    ContextManager& get_context_manager() { return *context_manager_; }

    /// @brief Get tools manager for direct access
    /// @return Reference to tools manager
    ToolsManager& get_tools_manager() { return *tools_manager_; }

    /// @brief Get context utilization (0.0 to 1.0)
    double get_context_utilization() const { return context_manager_->get_context_utilization(); }

    /// @brief Get model configuration (family, version, tool format settings)
    /// @return ModelConfig for this backend (default: generic)
    virtual ModelConfig get_model_config() const { return ModelConfig::create_generic(); }

    /// @brief Get tool call markers for this model (e.g., "<|python_tag|>")
    /// @return Vector of marker strings that indicate tool calls
    virtual std::vector<std::string> get_tool_call_markers() const { return {}; }

    /// @brief Get number of messages in hot context
    size_t get_message_count() const { return context_manager_->get_message_count(); }

    /// @brief Get last prompt token count from API response (for API backends)
    /// Returns 0 for local backends or if no API call has been made yet
    int get_last_prompt_tokens() const { return last_prompt_tokens_; }

    /// @brief Get last completion token count from API response (for API backends)
    /// Returns 0 for local backends or if no API call has been made yet
    int get_last_completion_tokens() const { return last_completion_tokens_; }

    /// @brief Get current token count in context
    /// Default implementation uses ContextManager's tracked count (for API backends)
    /// GPU backends (llama.cpp, TensorRT) should override to query actual state
    /// @return Number of tokens currently in context/KV cache
    virtual int get_context_token_count() const {
        if (!context_manager_) return 0;
        return context_manager_->get_total_tokens();
    }

    /// @brief Clear all context
    void clear_context() { context_manager_->clear(); }

    /// @brief Evict messages from context to free space
    /// Called by KV cache callback when space is needed
    /// @param tokens_needed Number of tokens that need to be freed
    /// @return New head position (where freed space begins), or UINT32_MAX on failure
    virtual uint32_t evict_to_free_space(uint32_t tokens_needed) = 0;

    /// @brief Set sampling parameters (for backends that support it)
    /// @param temperature Sampling temperature (0.0-2.0)
    /// @param top_p Nucleus sampling probability (0.0-1.0)
    /// @param top_k Top-K sampling
    /// @param min_keep Minimum tokens to keep
    /// Default implementation does nothing (API backends handle this via their own parameters)
    virtual void set_sampling_params(float temperature, float top_p, int top_k, int min_keep) {
        // Default: no-op for backends that don't support configurable sampling
    }

    /// @brief Set repetition penalty parameters (for backends that support it)
    /// @param penalty_repeat Repetition penalty (1.0 = disabled)
    /// @param penalty_freq Frequency penalty (0.0 = disabled)
    /// @param penalty_present Presence penalty (0.0 = disabled)
    /// @param penalty_last_n Last n tokens to penalize (0 = disabled)
    /// Default implementation does nothing (API backends handle this via their own parameters)
    virtual void set_penalty_params(float penalty_repeat, float penalty_freq, float penalty_present, int penalty_last_n) {
        // Default: no-op for backends that don't support configurable penalties
    }

    /// @brief Set tensor parallelism (for backends that support it)
    /// @param tp Number of GPUs to use (0 = auto/all, 1 = single GPU, >1 = specific count)
    /// Default implementation does nothing (API backends don't support this)
    virtual void set_tensor_parallel(int tp) {
        // Default: no-op for backends that don't support multi-GPU
    }

    /// @brief Set pipeline parallelism (for backends that support it)
    /// @param pp Number of pipeline stages (0 = auto, 1 = disabled, >1 = stages)
    /// Default implementation does nothing (API backends don't support this)
    virtual void set_pipeline_parallel(int pp) {
        // Default: no-op for backends that don't support pipeline parallelism
    }

    /// @brief Set number of GPU layers to offload (for backends that support it)
    /// @param gpu_layers Number of model layers to offload to GPU (-1 = auto/all, 0 = CPU only, >0 = specific count)
    /// Default implementation does nothing (API backends don't support this)
    virtual void set_gpu_layers(int gpu_layers) {
        // Default: no-op for backends that don't support GPU layer offloading
    }

protected:
    /// @brief Each backend implements its own context manager
    std::unique_ptr<ContextManager> context_manager_;

    /// @brief Each backend implements its own tools manager
    std::unique_ptr<ToolsManager> tools_manager_;

    /// @brief Each backend implements its own tokenizer
    std::unique_ptr<Tokenizer> tokenizer_;

    /// @brief Update token counts from API response
    /// @param prompt_tokens Actual prompt tokens from API
    /// @param completion_tokens Actual completion tokens from API
    /// @param estimated_prompt_tokens Original estimated prompt tokens
    void update_token_counts_from_api(int prompt_tokens, int completion_tokens, int estimated_prompt_tokens);

    std::string model_name_;
    std::string api_key_;
    size_t context_size_ = 0;  // Requested max context (0 = use model default, or override for testing)
    bool initialized_ = false;

    /// @brief Token counts from last API call (for API backends)
    /// Local backends should leave these at 0
    int last_prompt_tokens_ = 0;
    int last_completion_tokens_ = 0;
};

/// @brief Factory for creating backend managers
class BackendFactory {
public:
    /// @brief Create backend manager by backend name
    /// @param backend Backend name ("llamacpp", "tensorrt", "openai", "anthropic", "gemini", "grok")
    /// @param model_path_or_name Model file path or name
    /// @param max_context_tokens Maximum context window size
    /// @param api_key API key for cloud providers
    /// @return Unique pointer to backend instance
    static std::unique_ptr<Backend> create_backend(Config &config, size_t context_size);

    /// @brief Get list of available backends
    /// @return Vector of backend names
    static std::vector<std::string> get_available_backends();

    /// @brief Check if backend is available (compiled in)
    /// @param backend Backend name to check
    /// @return True if available
    static bool is_backend_available(const std::string& backend);
};
#endif

/// @brief Exception thrown by backend managers
class BackendError : public std::runtime_error {
public:
    explicit BackendError(const std::string& message)
        : std::runtime_error("Backend: " + message) {}
};

/// @brief Exception thrown when context is full and automatic eviction is disabled
class ContextFullException : public std::runtime_error {
public:
    explicit ContextFullException(const std::string& message)
        : std::runtime_error(message) {}
};
