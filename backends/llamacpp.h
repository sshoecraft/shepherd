#pragma once

#include "shepherd.h"
#include "gpu.h"
#include "models.h"
#include "chat_template.h"
#include "llama.cpp/vendor/minja/minja.hpp"
#include <regex>

#ifdef ENABLE_LLAMACPP
#include <vector>
#include <string>
#include <memory>

// Note: harmony.h no longer needed - parser is now in GpuBackend (parser.h)

// Forward declarations for llama.cpp types
typedef int32_t llama_token;
struct common_chat_templates;
#endif

/// @brief Backend manager for llama.cpp local models
class LlamaCppBackend : public GpuBackend {
public:
    LlamaCppBackend(size_t max_context_tokens, Session& session, EventCallback callback);
    ~LlamaCppBackend() override;

    void add_message(Session& session, Message::Role role, const std::string& content,
                    const std::string& tool_name = "", const std::string& tool_id = "",
                    int max_tokens = 0) override;
    void generate_from_session(Session& session, int max_tokens = 0) override;
    void prefill_session(Session& session) override;
    void generate_from_prefilled(Session& session, int max_tokens = 0) override;

    std::vector<std::string> get_tool_call_markers() const override;
    std::vector<std::string> get_tool_call_end_markers() const override;
    std::vector<std::string> get_thinking_start_markers() const override;
    std::vector<std::string> get_thinking_end_markers() const override;
    const ChatTemplates::ChatTemplateCaps* get_chat_template_caps() const override;

    // Helper methods
    bool is_ready() const;
    void shutdown() override;
    int get_context_token_count() const;
    uint32_t evict_to_free_space(uint32_t tokens_needed);
    ModelConfig get_model_config() const;

    // Sampling parameters (public per RULES.md)
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    int min_keep = 1;

    // Track which sampling parameters were explicitly set in config
    bool temperature_from_config = false;
    bool top_p_from_config = false;
    bool top_k_from_config = false;

    // Repetition penalty parameters
    float penalty_repeat = 1.1f;
    float penalty_freq = 0.1f;
    float penalty_present = 0.0f;
    int penalty_last_n = 64;

    // GPU offload configuration
    int gpu_layers = -1;
    int tensor_parallel = 0;
    int pipeline_parallel = 0;

    // KV cache type (f16, f32, q8_0, q4_0)
    std::string cache_type = "f16";

    // Flash Attention
    bool flash_attn = false;

    // Speculative Decoding
    std::string model_draft;
    int draft_max = 16;
    float draft_p_min = 0.75f;

protected:
    void parse_backend_config() override;

private:
    // Internal methods (not part of public interface)
    bool initialize_old(const std::string& model_path, const std::string& api_key = "", const std::string& template_path = "");
    std::string generate(const Session& session, int max_tokens = 0, EventCallback callback = nullptr);

    /// @brief Run inference using llama.cpp
    /// @param prompt_text The text to generate from
    /// @param max_tokens Maximum tokens to generate
    /// @param suppress_streaming Don't stream output (for tool calls)
    /// @param callback Optional streaming callback for token-by-token output
    /// @param generation_prompt_tokens Tokens to pre-feed to harmony parser (already decoded into KV cache)
    /// @return Generated text response
    std::string run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming = false, EventCallback callback = nullptr, const std::vector<llama_token>& generation_prompt_tokens = {});

    /// @brief Parse JSON arguments from tool call
    std::map<std::string, std::any> parse_json_to_args(const std::string& json);

    /// @brief Render a message using full conversation context
    /// Uses format_message_incremental which renders full conversation and extracts diff
    /// @param all_messages All messages in conversation so far
    /// @param target_index Index of message to render
    /// @param tools Available tools for template
    /// @param add_generation_prompt Whether to add generation prompt
    /// @return Rendered message text for the target message only
    std::string render_message(
        const std::vector<Message>& all_messages,
        size_t target_index,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt = false);

    /// @brief Format a message with full conversation context and decode into KV cache
    /// @param all_messages All messages in conversation (including the one to decode)
    /// @param target_index Index of message to decode
    /// @param tools Available tools
    /// @param add_generation_prompt Whether to add generation prompt
    /// @return True if successful, false otherwise
    bool format_and_decode_message(
        std::vector<Message>& all_messages,
        size_t target_index,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt = false);

    /// @brief Log token state comparison (messages vs KV cache) at debug level 3+
    /// @param context Description of where this is being called from
    void log_token_state(const std::string& context) const;

    /// @brief Count tokens for a message (formats through chat template and tokenizes)
    /// @param role Message role (USER, ASSISTANT, TOOL_RESPONSE)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL_RESPONSE messages)
    /// @param tool_id Tool call ID (for TOOL_RESPONSE messages)
    /// @return Token count for the formatted message
    int count_message_tokens(Message::Role role,
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

    // Speculative decoding (draft model)
    void* draft_model = nullptr;      // llama_model* for draft
    void* draft_model_ctx = nullptr;  // llama_context* for draft
    void* spec_state = nullptr;       // common_speculative*
    std::string chat_template_text; // Cached chat template from model
    int n_batch = 512;  // Logical batch size for prompt processing
    int n_ubatch = 512; // Physical micro-batch size (must be <= n_batch)

    // Token-level KV cache tracking (for accurate prefix caching)
    // This mirrors the actual tokens in the KV cache, enabling
    // precise prefix matching regardless of how content was extracted/stored
    std::vector<llama_token> kv_cache_mirror;

    // For CLI mode: pointer to current session being processed (for eviction callbacks)
    const Session* current_session = nullptr;

    bool initialized = false;
    int last_assistant_kv_tokens = 0;
    int last_completion_tokens = 0;  // Tokens generated in last response (for KV tracking)
    std::vector<llama_token> last_generated_tokens;  // Generated tokens for kv_cache_mirror sync

    // Server mode context-full signaling (can't throw from C callback)
    bool context_full_in_server_mode = false;
    int context_full_tokens_needed = 0;

    // Chat templates for tool handling - use void* to avoid complex forward declarations
    void* chat_templates = nullptr;

    // Parsed minja template node for message rendering (kept for MinjaTemplate)
    void* template_node = nullptr; // std::shared_ptr<minja::TemplateNode>*

    // Chat template instance
    std::unique_ptr<ChatTemplates::ChatTemplate> chat_template;

    // Tool call markers extracted from chat template (e.g., "<|python_tag|>")
    std::vector<std::string> tool_call_markers;
    bool have_tool_markers = false;

    // Harmony stop token IDs (for GPT-OSS models with channels)
    // Used for debug logging of stop token detection
    std::vector<int32_t> harmony_stop_tokens;

    // Harmony mode flag - true if model uses GPT-OSS channel format
    // Parser is now managed by GpuBackend (HarmonyParser or GenericParser)
    bool harmony_enabled = false;

    // UTF-8 buffering for incomplete multi-byte sequences across tokens
    std::string parser_utf8_buffer;

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

    // Track if last generation hit length limit (for auto-continuation)
    bool last_generation_hit_length_limit = false;

#endif
};