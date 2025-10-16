#pragma once

#include "../backend_manager.h"
#include "../model_config.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <map>

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
class TensorRTTokenizer : public Tokenizer {
public:
    explicit TensorRTTokenizer(const std::string& tokenizer_path = "");
    ~TensorRTTokenizer();

    int count_tokens(const std::string& text) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& tokens) override;
    std::string get_tokenizer_name() const override;

    bool load_tokenizer(const std::string& tokenizer_path);

private:
    void* tokenizer_ = nullptr; // tokenizers::Tokenizer pointer
    std::string tokenizer_path_;
};

/// @brief TensorRT-specific context manager for KV cache management
class TensorRTContextManager : public ContextManager {
public:
    explicit TensorRTContextManager(size_t max_context_tokens);

    std::string get_context_for_inference() override;
    int count_tokens(const std::string& text) override;
    int calculate_json_overhead() const override;

    /// @brief Track token positions for messages to map KV cache blocks
    void add_message(const Message& message) override;

    /// @brief Get message indices that overlap with given token range
    std::vector<int> get_messages_in_token_range(size_t start_token, size_t end_token) const;

    /// @brief Get total tokens before a given message index
    size_t get_tokens_before_message(int msg_index) const;

    /// @brief Set the minja template node for message rendering
    void set_template_node(void* template_node) { template_node_ = template_node; }

    /// @brief Set model configuration for role mapping
    void set_model_config(const ModelConfig& config) { model_config_ = config; }

private:
    void evict_old_messages();

    /// @brief Track cumulative token positions for each message
    /// message_token_positions_[i] = total tokens BEFORE message i
    std::vector<size_t> message_token_positions_;

    /// @brief Parsed minja template node for rendering
    void* template_node_ = nullptr; // std::shared_ptr<minja::TemplateNode>*

    /// @brief Model configuration for role mapping
    ModelConfig model_config_;
};

/// @brief Backend manager for TensorRT-LLM acceleration
class TensorRTBackend : public BackendManager {
public:
    explicit TensorRTBackend(size_t max_context_tokens);
    ~TensorRTBackend() override;

    bool initialize(const std::string& model_path, const std::string& api_key = "", const std::string& template_path = "") override;
    std::string generate(int max_tokens = 0) override;
    void add_user_message(const std::string& content) override;
    void add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id = "") override;
    void add_assistant_message(const std::string& content) override;
    void add_system_message(const std::string& content) override;
    std::string get_backend_name() const override;
    std::string get_model_name() const override;
    size_t get_max_context_size() const override;
    bool is_ready() const override;
    void shutdown() override;
    uint32_t evict_to_free_space(uint32_t tokens_needed) override;
    ModelConfig get_model_config() const override;

private:
#ifdef ENABLE_TENSORRT
    /// @brief Detect model family from chat template and config
    ModelConfig detect_model_family();

    /// @brief Monitor KV cache events in background thread
    void monitor_kv_events();

    /// @brief Handle KV cache block removal event
    void handle_kv_cache_removed(const std::vector<uint64_t>& block_hashes);

    void* executor_ = nullptr;    // tensorrt_llm::executor::Executor*
    void* event_manager_ = nullptr;  // std::shared_ptr<KVCacheEventManager>*
    std::string model_path_;
    size_t max_context_size_ = 8192;
    uint64_t current_request_id_ = 0;  // Track active request
    bool request_active_ = false;

    // KV cache event monitoring
    std::thread kv_event_monitor_thread_;
    std::atomic<bool> monitoring_events_{false};

    // Track which blocks correspond to which token ranges
    // block_hash -> (start_token, end_token)
    std::map<uint64_t, std::pair<size_t, size_t>> block_to_tokens_;
    std::mutex block_map_mutex_;

    // Chat template support
    void* template_node_ = nullptr; // std::shared_ptr<minja::TemplateNode>*
    std::string chat_template_text_;

    // Model configuration for prompt format
    ModelConfig model_config_;
#endif
};