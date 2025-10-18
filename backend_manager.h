#pragma once

#include "context_manager.h"
#include "tools_manager.h"
#include "tokenizer.h"
#include "model_config.h"
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
    /// This is the NEW unified interface that backends should implement
    /// Backend reads SessionContext, formats request appropriately, and generates
    /// @param session Session context containing system, messages, tools
    /// @param max_tokens Maximum tokens to generate (0 = model default)
    /// @return Generated response text (may contain tool call JSON)
    virtual std::string generate_from_session(const SessionContext& session, int max_tokens = 0) = 0;

    /// @brief Get backend name/identifier
    /// @return Backend name (e.g., "llamacpp", "openai", "anthropic")
    virtual std::string get_backend_name() const = 0;

    /// @brief Get model name/path currently loaded
    /// @return Model identifier
    virtual std::string get_model_name() const = 0;

    /// @brief Get maximum context size supported
    /// @return Max context window in tokens
    virtual size_t get_max_context_size() const = 0;

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
    size_t max_context_size_ = 0;  // Requested max context (0 = use model default, or override for testing)
    bool initialized_ = false;
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
    static std::unique_ptr<BackendManager> create_backend(
        const std::string& backend,
        const std::string& model_path_or_name,
        size_t max_context_tokens,
        const std::string& api_key = ""
    );

    /// @brief Get list of available backends
    /// @return Vector of backend names
    static std::vector<std::string> get_available_backends();

    /// @brief Check if backend is available (compiled in)
    /// @param backend Backend name to check
    /// @return True if available
    static bool is_backend_available(const std::string& backend);
};

/// @brief Exception thrown by backend managers
class BackendManagerError : public std::runtime_error {
public:
    explicit BackendManagerError(const std::string& message)
        : std::runtime_error("BackendManager: " + message) {}
};