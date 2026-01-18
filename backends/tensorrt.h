#pragma once

#include "gpu.h"
#include "models.h"
#include "chat_template.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <optional>

#ifdef ENABLE_TENSORRT
#include <vector>
#include <memory>
#include <cstdint>

// Forward declarations for TensorRT-LLM types
namespace tensorrt_llm::executor {
    class Executor;
    class Request;
    class Response;
    enum class ModelType;
}
#endif

/// @brief TensorRT-specific tokenizer using tokenizers-cpp
class TensorRTTokenizer {
public:
    explicit TensorRTTokenizer(const std::string& tokenizer_path = "");
    ~TensorRTTokenizer();

    int count_tokens(const std::string& text);
    std::vector<int> encode(const std::string& text, bool add_special_tokens = false);
    std::string decode(const std::vector<int>& tokens);
    std::string get_tokenizer_name() const;

    bool load_tokenizer(const std::string& tokenizer_path);

private:
    void* tokenizer_ = nullptr; // TokenizerHandle from C API
    std::string tokenizer_path_;
};

/// @brief Backend for TensorRT-LLM acceleration
class TensorRTBackend : public GpuBackend {
public:
    TensorRTBackend(size_t context_size, Session& session, EventCallback callback);
    ~TensorRTBackend() override;

    void parse_backend_config() override;

    void add_message(Session& session, Message::Role role, const std::string& content,
                    const std::string& tool_name = "", const std::string& tool_id = "",
                    int max_tokens = 0) override;
    void generate_from_session(Session& session, int max_tokens = 0) override;
    void prefill_session(Session& session) override;
    void generate_from_prefilled(Session& session, int max_tokens = 0) override;
    int count_message_tokens(Message::Role role, const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;
    const ChatTemplates::ChatTemplateCaps* get_chat_template_caps() const override;

    // Backend state
    bool is_ready() const;
    void shutdown();

private:
#ifdef ENABLE_TENSORRT
    /// @brief Detect model family from chat template and config
    ModelConfig detect_model_family();

    /// @brief Monitor KV cache events in background thread
    void monitor_kv_events();

    /// @brief Handle KV cache block removal event
    void handle_kv_cache_removed(const std::vector<uint64_t>& block_hashes);

    /// @brief Render a single message through the chat template
    std::string render_message(const Message& msg, bool add_generation_prompt = false);

    /// @brief Tokenize and accumulate message tokens (does not enqueue to TensorRT yet)
    bool tokenize_and_accumulate_message(Message& msg, bool add_generation_prompt = false);

    /// @brief Internal generation logic called by add_message and generate_from_session
    std::string generate(const Session& session, int max_tokens, EventCallback callback = nullptr);

    // TensorRT executor and event management
    void* executor_ = nullptr;    // tensorrt_llm::executor::Executor*
    void* event_manager_ = nullptr;  // std::shared_ptr<KVCacheEventManager>*

    // Backend state tracking (what's in TensorRT KV cache)
    Session backend_session;
    const Session* current_session = nullptr;  // For eviction callbacks

    // Token accumulation for TensorRT's all-at-once request model
    std::vector<int32_t> accumulated_tokens;

    // Model and path info
    std::string model_path_;
    uint64_t current_request_id_ = 0;  // Track active request
    bool request_active_ = false;

    // Engine build config parameters (read from config.json)
    int max_seq_len = 2048;
    int max_batch_size = 1;
    int max_beam_width = 1;

    // KV cache event monitoring
    std::thread kv_event_monitor_thread_;
    std::atomic<bool> monitoring_events_{false};

    // Track which blocks correspond to which token ranges
    // block_hash -> (start_token, end_token)
    std::map<uint64_t, std::pair<size_t, size_t>> block_to_tokens_;
    std::mutex block_map_mutex_;

    // Chat template support
    void* template_node_ = nullptr; // std::shared_ptr<minja::TemplateNode>* (kept for MinjaTemplate fallback)
    std::string chat_template_text_;
    std::unique_ptr<ChatTemplates::ChatTemplate> chat_template_;  // ChatTemplate abstraction
    std::vector<std::string> stop_tokens;  // Loaded from tokenizer_config.json (string form)
    std::vector<std::vector<int32_t>> stop_token_ids;  // Encoded stop token IDs for TensorRT
    std::optional<int32_t> eos_token_id;  // EOS token ID for TensorRT

    // BOS token configuration (loaded from tokenizer_config.json)
    std::string bos_token;  // String form for chat template
    bool add_bos_token = false;
    int bos_token_id = -1;

    // Tokenizer
    std::unique_ptr<TensorRTTokenizer> tokenizer_;

    // Model configuration for prompt format
    ModelConfig model_config_;

    // Sampling parameters
    // Sampling parameters
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;

    // Penalty parameters
    float repetition_penalty = 1.2f;  // Match API backend default
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.5f;   // Match API backend default
    float length_penalty = 0.0f;
    int no_repeat_ngram_size = 0;

    // Track orphaned user questions during eviction
    std::optional<Message> open_user_question_;

    // Track if last generation hit length limit (for auto-continuation)
    bool last_generation_hit_length_limit = false;

    // Initialization state
    bool initialized = false;
#endif
};