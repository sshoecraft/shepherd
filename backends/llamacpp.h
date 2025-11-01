#pragma once

#include "shepherd.h"
#include "backend.h"
#include "models.h"
#include "minja.hpp"
#include <regex>

#ifdef ENABLE_LLAMACPP
#include <vector>
#include <string>
#include <memory>

// Forward declarations for llama.cpp types
typedef int32_t llama_token;
struct common_chat_templates;
#endif

/// @brief Backend manager for llama.cpp local models
class LlamaCppBackend : public Backend {
public:
    explicit LlamaCppBackend(size_t max_context_tokens);
    ~LlamaCppBackend() override;

    // New Backend interface
    void initialize(Session& session) override;
    Response add_message(Session& session, Message::Type type, const std::string& content,
                        const std::string& tool_name = "", const std::string& tool_id = "",
                        int prompt_tokens = 0, int max_tokens = 0) override;
    Response generate_from_session(const Session& session, int max_tokens = 0) override;

    std::vector<std::string> get_tool_call_markers() const override;

    // Helper methods
    bool is_ready() const;
    void shutdown();
    int get_context_token_count() const;
    uint32_t evict_to_free_space(uint32_t tokens_needed);
    ModelConfig get_model_config() const;

    // Sampling parameters (public per RULES.md)
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    int min_keep = 1;

    // Repetition penalty parameters
    float penalty_repeat = 1.1f;
    float penalty_freq = 0.1f;
    float penalty_present = 0.0f;
    int penalty_last_n = 64;

    // GPU offload configuration
    int gpu_layers = -1;
    int tensor_parallel = 0;
    int pipeline_parallel = 0;

protected:
    void parse_backend_config(const std::string& json) override;

private:
    // Internal methods (not part of public interface)
    bool initialize_old(const std::string& model_path, const std::string& api_key = "", const std::string& template_path = "");
    std::string generate(int max_tokens = 0);

    /// @brief Run inference using llama.cpp
    /// @param prompt_text The text to generate from
    /// @param max_tokens Maximum tokens to generate
    /// @param suppress_streaming Don't stream output (for tool calls)
    /// @return Generated text response
    std::string run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming = false);

    /// @brief Parse JSON arguments from tool call
    std::map<std::string, std::any> parse_json_to_args(const std::string& json);

    /// @brief Format system message with tools directly from Session::Tool vector
    /// @param system_content The base system message content
    /// @param tools Vector of tools from the session
    /// @return Formatted system message with tools included
    std::string format_system_message_with_tools(const std::string& system_content,
                                                  const std::vector<Session::Tool>& tools);

    /// @brief Render a single message through the chat template
    /// @param msg Message to render
    /// @param add_generation_prompt Whether to add generation prompt
    /// @return Rendered message text ready for tokenization
    std::string render_message(const Message& msg, bool add_generation_prompt = false);

    /// @brief Format a single message and decode it into KV cache
    /// @param msg Message to format and decode
    /// @return True if successful, false otherwise
    bool format_and_decode_message(Message& msg);

    /// @brief Log token state comparison (messages vs KV cache) at debug level 3+
    /// @param context Description of where this is being called from
    void log_token_state(const std::string& context) const;

    /// @brief Count tokens for a message (formats through chat template and tokenizes)
    /// @param type Message type (USER, ASSISTANT, TOOL)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL messages)
    /// @param tool_id Tool call ID (for TOOL messages)
    /// @return Token count for the formatted message
    int count_message_tokens(Message::Type type,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;

    /// @brief Count tokens in text using llama.cpp tokenizer
    /// @param text Text to tokenize
    /// @return Number of tokens
    int count_tokens_in_text(const std::string& text) const;

#ifdef ENABLE_LLAMACPP
    void* model_ctx = nullptr; // llama_context*
    void* model = nullptr;     // llama_model*
    std::string model_path;
    std::string chat_template_text; // Cached chat template from model
    int n_batch = 512; // Batch size for prompt processing (set during init)

    // State tracking
    // For server mode: backend maintains its own session tracking what's in KV cache
    Session backend_session;

    // For CLI mode: pointer to current session being processed (for eviction callbacks)
    const Session* current_session = nullptr;

    bool initialized = false;
    int last_assistant_kv_tokens = 0;

    // Chat templates for tool handling - use void* to avoid complex forward declarations
    void* chat_templates = nullptr;

    // Parsed minja template node for message rendering
    void* template_node = nullptr; // std::shared_ptr<minja::TemplateNode>*

    // Tool call markers extracted from chat template (e.g., "<|python_tag|>")
    std::vector<std::string> tool_call_markers;
    bool have_tool_markers = false;

    // Detected model configuration
    ModelConfig model_config;

    // Pre-formatted system message (with tools embedded for Qwen)
    std::string formatted_system_message;

    // Multi-GPU configuration (must persist for lifetime of model_params)
    std::vector<float> tensor_split;  // Proportion for each GPU (e.g., [1.0, 1.0] for even split across 2 GPUs)
    std::vector<void*> gpu_devices;  // Device pointers (ggml_backend_dev_t*) for multi-GPU (NULL-terminated)

    // Formatted token counts (including all template overhead)
    int system_formatted_tokens = 0;
    int current_user_formatted_tokens = 0;

#endif
};