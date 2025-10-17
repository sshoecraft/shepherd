#pragma once

#include "../backend_manager.h"
#include "../context_manager.h"
#include "../tokenizer.h"
#include "../model_config.h"
#include "../minja.hpp"
#include <regex>

#ifdef ENABLE_LLAMACPP
#include <vector>
#include <string>
#include <memory>

// Forward declarations for llama.cpp types
typedef int32_t llama_token;
struct common_chat_templates;
#endif

/// @brief LlamaCpp-specific tokenizer using built-in llama.cpp tokenization
class LlamaCppTokenizer : public Tokenizer {
public:
    explicit LlamaCppTokenizer(void* model = nullptr);

    int count_tokens(const std::string& text) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;
    std::string get_tokenizer_name() const override;

#ifdef ENABLE_LLAMACPP
    void set_model(void* model) { model_ = model; }
#else
    void set_model(void* model) { /* No-op when llama.cpp disabled */ }
#endif

private:
#ifdef ENABLE_LLAMACPP
    void* model_ = nullptr; // llama_model*
#endif
};

/// @brief LlamaCpp-specific context manager that maintains context efficiently
class LlamaCppContextManager : public ContextManager {
public:
    explicit LlamaCppContextManager(size_t max_context_tokens);

    std::string get_context_for_inference() override;
    std::string get_context_for_inference(bool add_generation_prompt);
    std::string render_single_message(const Message& msg, bool add_generation_prompt = false);
    int count_tokens(const std::string& text) override;
    int calculate_json_overhead() const override;
    void clear() override;

    /// @brief Set the model pointer for chat template access
#ifdef ENABLE_LLAMACPP
    void set_model(void* model) { model_ = model; }

    /// @brief Set the minja template node for message rendering
    void set_template_node(void* template_node) { template_node_ = template_node; }

    /// @brief Get token position range for messages
    /// @param start_msg_index First message index
    /// @param end_msg_index Last message index (inclusive)
    /// @return Pair of (start_pos, end_pos) token positions
    std::pair<int, int> get_token_range_for_messages(int start_msg_index, int end_msg_index) const;

    /// @brief Mark messages as evicted from cache
    /// @param start_msg_index First message evicted
    /// @param end_msg_index Last message evicted (inclusive)
    void mark_messages_evicted(int start_msg_index, int end_msg_index);

    /// @brief Extract message format patterns from chat template
    void extract_message_patterns();
#else
    void set_model(void* model) { /* No-op when llama.cpp disabled */ }
    void set_template_node(void* template_node) { /* No-op */ }
    std::pair<int, int> get_token_range_for_messages(int, int) const { return {0, 0}; }
    void mark_messages_evicted(int, int) { /* No-op */ }
#endif

private:
    void evict_old_messages();

    /// @brief Simple string context for llama.cpp
    std::string context_text_;
    std::vector<size_t> message_boundaries_; // Track message starts for eviction

#ifdef ENABLE_LLAMACPP
    void* model_ = nullptr; // llama_model* for chat template access
    void* template_node_ = nullptr; // std::shared_ptr<minja::TemplateNode>* for rendering

    // Pre-rendered message format patterns for efficient formatting
    std::string system_pattern_;
    std::string user_pattern_;
    std::string assistant_pattern_;
    std::string tool_pattern_;
    bool patterns_extracted_ = false;
#endif
};

/// @brief Backend manager for llama.cpp local models
class LlamaCppBackend : public BackendManager {
public:
    explicit LlamaCppBackend(size_t max_context_tokens);
    ~LlamaCppBackend() override;

    bool initialize(const std::string& model_path, const std::string& api_key = "", const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    std::string generate_from_session(const SessionContext& session, int max_tokens = 0) override;
    void add_user_message(const std::string& content) override;
    void add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id = "") override;
    void add_assistant_message(const std::string& content) override;
    void add_system_message(const std::string& content) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    ModelConfig get_model_config() const override;
    std::vector<std::string> get_tool_call_markers() const override;
    bool is_ready() const override;
    void shutdown() override;

    // Evict messages from KV cache to free space
    // Returns the new head position (where freed space begins), or UINT32_MAX on failure
    uint32_t evict_to_free_space(uint32_t tokens_needed) override;

    // Override base class sampling parameter method
    void set_sampling_params(float temperature, float top_p, int top_k, int min_keep) override {
        temperature_ = temperature;
        top_p_ = top_p;
        top_k_ = top_k;
        min_keep_ = min_keep;
    }

    // Override base class penalty parameter method
    void set_penalty_params(float penalty_repeat, float penalty_freq, float penalty_present, int penalty_last_n) override {
        penalty_repeat_ = penalty_repeat;
        penalty_freq_ = penalty_freq;
        penalty_present_ = penalty_present;
        penalty_last_n_ = penalty_last_n;
    }

    // Set GPU layers before initialization (-1 = auto/all, 0 = CPU only, >0 = specific count)
    void set_gpu_layers(int gpu_layers) {
        gpu_layers_ = gpu_layers;
    }

private:
    /// @brief Run inference using llama.cpp
    /// @param prompt_text The text to generate from
    /// @param max_tokens Maximum tokens to generate
    /// @param suppress_streaming Don't stream output (for tool calls)
    /// @return Generated text response
    std::string run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming = false);

    /// @brief Parse JSON arguments from tool call
    std::map<std::string, std::any> parse_json_to_args(const std::string& json);

    /// @brief Get formatted context with tool definitions for inference
    std::string get_context_with_tools();

    /// @brief Format a single message and decode it into KV cache
    /// @param msg Message to format and decode
    /// @return True if successful, false otherwise
    bool format_and_decode_message(Message& msg);

#ifdef ENABLE_LLAMACPP
    void* model_ctx_ = nullptr; // llama_context*
    void* model_ = nullptr;     // llama_model*
    std::string model_path_;
    std::string db_path_;       // Store db_path for delayed context manager creation
    size_t max_context_size_ = 0;
    std::string chat_template_text_; // Cached chat template from model
    int n_batch_ = 512; // Batch size for prompt processing (set during init)

    // Chat templates for tool handling - use void* to avoid complex forward declarations
    void* chat_templates_ = nullptr;

    // Parsed minja template node for message rendering
    void* template_node_ = nullptr; // std::shared_ptr<minja::TemplateNode>*

    // Tool call markers extracted from chat template (e.g., "<|python_tag|>")
    std::vector<std::string> tool_call_markers_;
    bool have_tool_markers_ = false;

    // Detected model configuration
    ModelConfig model_config_;

    // Pre-formatted system message (with tools embedded for Qwen)
    std::string formatted_system_message_;

    // Sampling parameters
    float temperature_ = 0.7f;
    float top_p_ = 0.95f;
    int top_k_ = 40;
    int min_keep_ = 1;

    // Repetition penalty parameters
    float penalty_repeat_ = 1.1f;
    float penalty_freq_ = 0.1f;
    float penalty_present_ = 0.0f;
    int penalty_last_n_ = 64;

    // GPU offload configuration
    int gpu_layers_ = -1;  // -1 = auto/all, 0 = CPU only, >0 = specific layer count

    // Server mode flag - suppresses all streaming output
    bool server_mode_ = false;

    // Formatted token counts (including all template overhead)
    int system_formatted_tokens_ = 0;
    int current_user_formatted_tokens_ = 0;

#endif
};