#pragma once

#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <stdexcept>
#include "model_config.h"

/// @brief Represents a single message in the conversation
struct Message {
    enum Type {
        SYSTEM,
        USER,
        ASSISTANT,
        TOOL,        // Tool response message
        FUNCTION     // Function call message
    };

    Type type;
    std::string content;
    int token_count;     // Track tokens per message
    int64_t timestamp;
    bool in_kv_cache;    // Whether this message has been confirmed in KV cache

    // Optional fields for tool/function messages
    std::string tool_name;
    std::string tool_call_id;

    Message(Type t, const std::string& c, int tokens = 0)
        : type(t), content(c), token_count(tokens), timestamp(get_current_timestamp()), in_kv_cache(false) {}

    // Get standardized role string for RAG storage
    // Backends will translate these to their specific format
    // (e.g., Gemini translates "assistant" -> "model")
    std::string get_role() const {
        switch (type) {
            case SYSTEM: return "system";
            case USER: return "user";
            case ASSISTANT: return "assistant";  // Standardized, not backend-specific
            case TOOL: return "tool";
            case FUNCTION: return "function";
            default: return "user";
        }
    }

    // Get model-specific role string based on model family
    std::string get_role_for_model(const ModelConfig& config) const {
        // Non-tool messages use standard roles
        if (type != TOOL) {
            return get_role();
        }

        // Tool result role varies by model family
        return config.tool_result_role;
    }

private:
    static int64_t get_current_timestamp();
};

/// @brief Represents a complete conversation turn (user + assistant)


/// @brief Abstract interface for backend-specific context management
/// Each backend implements this to manage context in its native format
class ContextManager {
public:
    explicit ContextManager(size_t max_context_tokens);
    virtual ~ContextManager() = default;

    /// @brief Add a message to the context window
    /// Automatically handles eviction to RAG when window is full
    /// @param message Message to add (with token count already set)
    virtual void add_message(const Message& message);

    /// @brief Get the current context serialized for backend API
    /// Each backend implements this to format messages appropriately
    /// @return Context ready for backend's generate() method (JSON string)
    virtual std::string get_context_for_inference() = 0;

    /// @brief Count tokens in text (backend-specific tokenization)
    /// @param text Text to tokenize
    /// @return Number of tokens
    virtual int count_tokens(const std::string& text) = 0;

    /// @brief Calculate JSON overhead tokens for context serialization
    /// Each backend implements this to account for its JSON format overhead
    /// @return Number of additional tokens used by JSON structure
    virtual int calculate_json_overhead() const = 0;

    /// @brief Get current context utilization (0.0 to 1.0)
    double get_context_utilization() const;

    /// @brief Get total number of messages in hot context
    size_t get_message_count() const;

    /// @brief Clear all messages from hot context
    virtual void clear();

    /// @brief Remove last N messages from context (for warmup cleanup)
    /// @param count Number of messages to remove from end
    void remove_last_messages(size_t count);

    /// @brief Set maximum context size
    void set_max_context_tokens(size_t max_tokens);

    /// @brief Get maximum context size
    size_t get_max_context_tokens() const;

    /// @brief Get access to messages for token count updates
    /// Used by backends to update token counts from API responses
    std::deque<Message>& get_messages() { return messages_; }

    /// @brief Recalculate total token count from all messages
    /// Used after updating individual message token counts
    void recalculate_total_tokens();

    /// @brief Get total tokens including JSON overhead
    /// @return Message tokens + JSON structure overhead
    int get_total_tokens() const;

    /// @brief Calculate which messages to evict to free required tokens
    /// @param tokens_needed Number of tokens that need to be freed
    /// @param max_evict_index Maximum message index to consider for eviction (exclusive).
    ///        For KV cache backends, this should be cached_message_count to only evict cached messages.
    ///        Default SIZE_MAX means consider all messages.
    /// @return Pair of (start_message_index, end_message_index) to evict, or (-1, -1) if can't free enough
    std::pair<int, int> calculate_messages_to_evict(int tokens_needed, size_t max_evict_index = SIZE_MAX) const;

    /// @brief Evict specific messages from context (public interface for backends)
    /// Archives messages to RAG and removes them from context
    /// @param tokens_needed Number of tokens that need to be freed
    /// @return Pair of (start_message_index, end_message_index) that were evicted, or (-1, -1) if eviction failed
    std::pair<int, int> evict_messages(int tokens_needed);

    /// @brief Evict specific message range from context (for backends with KV cache)
    /// Archives messages to RAG and removes them from context
    /// @param start_msg Starting message index (inclusive)
    /// @param end_msg Ending message index (inclusive)
    /// @return True if eviction successful, false otherwise
    bool evict_messages_by_index(int start_msg, int end_msg);

    /// @brief Evict oldest messages when context is full
    /// Archives to RAG before removing from context
    /// Public for API backends to call when doing proactive eviction
    void evict_oldest_messages();

protected:

    /// @brief Check if context needs eviction
    bool needs_eviction(int additional_tokens) const;

    /// @brief Get available tokens (reserves space for response)
    size_t get_available_tokens() const;

protected:
    /// @brief The actual message storage - efficient deque
    std::deque<Message> messages_;

    /// @brief System messages count (preserved during eviction)
    size_t system_message_count_ = 0;

    size_t max_context_tokens_;
    int current_token_count_ = 0;

    /// @brief Whether to automatically evict messages when add_message() exceeds token limits
    /// - true: ContextManager manages eviction (for stateless backends like Gemini/Claude)
    /// - false: Backend manages eviction via KV cache callbacks (for llama.cpp/TensorRT)
    bool auto_evict_on_add_ = true;
};

/// @brief Exception thrown by context managers
class ContextManagerError : public std::runtime_error {
public:
    explicit ContextManagerError(const std::string& message)
        : std::runtime_error("ContextManager: " + message) {}
};