
#pragma once

#include "session.h"
#include "message.h"
#include "tools/tool_parser.h"
#include <vector>
#include <string>
#include <functional>
#include <set>
#include <memory>

// Forward declaration for chat template capabilities
namespace ChatTemplates {
    struct ChatTemplateCaps;
}

// Forward declaration for channel parsing
namespace ChannelParsing {
    class ChannelParser;
}

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
    std::string tool_calls_json;                     // Raw tool_calls JSON for persistence
    int prompt_tokens = 0;                           // Tokens used in prompt
    int completion_tokens = 0;                       // Tokens generated
    std::string finish_reason;                       // "stop", "tool_calls", "length", "error"
    std::string error;                               // Error message (empty if success)
    bool was_streamed = false;                       // True if response was streamed (already output to user)

    // For CONTEXT_FULL or MAX_TOKENS_TOO_HIGH errors
    int overflow_tokens = 0;                         // How many tokens over the limit
    int actual_prompt_tokens = 0;                    // Actual prompt size measured by backend (if known)
};

// Forward decl
class Session;

/// @brief Event types for backend-to-frontend callback communication
/// These are streaming events, NOT stored messages
enum class CallbackEvent {
    CONTENT,      // Assistant text chunk
    THINKING,     // Reasoning/thinking chunk (if show_thinking enabled)
    TOOL_CALL,    // Model requesting a tool call
    TOOL_RESULT,  // Result of tool execution (summary in content)
    USER_PROMPT,  // Echo user's prompt
    SYSTEM,       // System info/status messages
    ERROR,        // Error occurred (message in content, type in name)
    STOP,         // Generation complete (finish_reason in content)
    CODEBLOCK,    // Code block content (inside ```)
    STATS         // Performance stats (prefill/decode speed, KV cache info)
};

class Backend {
public:
    // Callback for backend-to-frontend communication (pure callback architecture)
    // event: Type of callback event (CONTENT, THINKING, TOOL_CALL, ERROR, STOP, etc.)
    // content: Event content (text chunk, error message, finish_reason, etc.)
    // name: Tool name (for TOOL_CALL), error type (for ERROR)
    // id: Tool call ID (for TOOL_CALL)
    // Returns true to continue, false to cancel
    using EventCallback = std::function<bool(CallbackEvent event,
                                             const std::string& content,
                                             const std::string& name,
                                             const std::string& id)>;

    Backend(size_t context_size, Session& session, EventCallback callback);
    virtual ~Backend();  // Defined in .cpp where ChannelParser is complete

    // Main transactional message interface
    // Adds message and generates response, handling eviction if needed
    // All output flows through callback (CONTENT, TOOL_CALL, ERROR, STOP)
    // Session is updated with messages and token counts
    virtual void add_message(Session& session,
                            Message::Role role,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "",
                            int max_tokens = 0) = 0;

    // Generation from Session (for server with prefix caching)
    // All output flows through callback, session token counts are updated
    virtual void generate_from_session(Session& session, int max_tokens = 0) = 0;

    // Tool and thinking tag markers (model-specific, extracted from chat template)
    virtual std::vector<std::string> get_tool_call_markers() const { return {}; }
    virtual std::vector<std::string> get_tool_call_end_markers() const { return {}; }
    virtual std::vector<std::string> get_thinking_start_markers() const { return {}; }
    virtual std::vector<std::string> get_thinking_end_markers() const { return {}; }

    // Get chat template capabilities (for channel-based models)
    // Returns nullptr if no template or not available
    virtual const ChatTemplates::ChatTemplateCaps* get_chat_template_caps() const { return nullptr; }

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
    virtual int count_message_tokens(Message::Role role,
                                     const std::string& content,
                                     const std::string& tool_name = "",
                                     const std::string& tool_id = "") = 0;

    // Shutdown and cleanup resources (called before switching providers)
    virtual void shutdown() {}

    /// @brief Set the model and update model-specific configuration
    /// Override in API backends to update model_config, max_output_tokens, etc.
    /// @param model New model name/ID
    virtual void set_model(const std::string& model) {
        model_name = model;
    }

    /// @brief Get available models (lazy-loaded and cached)
    /// @return Vector of model IDs (empty for local/GPU backends)
    std::vector<std::string> get_models() {
        if (cached_models.empty()) {
            cached_models = fetch_models();
        }
        return cached_models;
    }

    // Public member variables (no getters/setters per RULES.md)
    std::string name;
    std::string backend_name;
    std::string model_name;
    size_t context_size;
    int max_output_tokens = 0;  // Maximum tokens for completion (0 = no limit)
    std::set<std::string> valid_tool_names;  // Tool names for filtering (set by frontend)
    bool is_local = false;  // true for GPU/local backends (llamacpp, tensorrt), false for API backends
    bool streaming_enabled = false;  // true if backend supports and has enabled streaming
    bool sse_handles_output = false;  // true if SSE handles all output (CLI client backend)
    EventCallback callback;  // Frontend callback for streaming output events

protected:
    virtual void parse_backend_config() {
        // Default: no-op. Backends override if they have specific config
    }

    /// @brief Fetch models from API (override in API backends)
    /// @return Vector of model IDs
    virtual std::vector<std::string> fetch_models() { return {}; }

    // Unified output function - all backends call this instead of callback directly
    // Filters tool calls and thinking blocks, emits structured callbacks
    // Returns true to continue, false if callback requested cancellation
    bool output(const char* text, size_t len);
    bool output(const std::string& text) { return output(text.c_str(), text.length()); }

    // Process output through channel parser (for GPT-OSS harmony format)
    // If model has channels: routes through ChannelParser first, then output()
    // If no channels: falls back to output() directly
    // Returns true to continue, false if cancelled or stop requested
    bool process_output(const char* text, size_t len);
    bool process_output(const std::string& text) { return process_output(text.c_str(), text.length()); }

    // Reset filter state between requests
    void reset_output_state();

    // Flush any pending output at end of response
    void flush_output();

    // Control flags
    bool show_thinking = false;

public:
    // Tool calls detected during streaming via emit_tool_call()
    // Used by derived backends and API server to access tool calls captured by channel parser
    std::vector<ToolParser::ToolCall> pending_tool_calls;

private:
    std::vector<std::string> cached_models;  // Lazy-loaded model list

    // Filtering state machine
    enum FilterState {
        FILTER_NORMAL,
        FILTER_DETECTING_TAG,
        FILTER_IN_THINKING,
        FILTER_IN_TOOL_CALL,
        FILTER_CHECKING_CLOSE
    };

    FilterState filter_state = FILTER_NORMAL;
    bool in_tool_call = false;
    bool in_thinking = false;
    bool in_code_block = false;
    bool skip_to_newline = false;  // Skip rest of line (e.g., language after ```)
    int json_brace_depth = 0;

    std::string tag_buffer;
    std::string current_tag;
    std::string backtick_buffer;
    std::string buffered_tool_call;
    std::string buffered_thinking;
    std::string output_buffer;  // Buffer for batching small outputs

    // Marker vectors (cached from virtual methods)
    std::vector<std::string> tool_call_start_markers;
    std::vector<std::string> tool_call_end_markers;
    std::vector<std::string> thinking_start_markers;
    std::vector<std::string> thinking_end_markers;
    bool markers_initialized = false;

    // Channel parser for GPT-OSS harmony format (integrated for all modes)
    std::unique_ptr<ChannelParsing::ChannelParser> channel_parser_;
    bool channel_parsing_enabled_ = false;

    // Filter helpers
    void ensure_markers_initialized();
    bool matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched = nullptr) const;
    bool could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const;
    void emit_tool_call();
    bool flush_output_buffer();  // Returns true to continue, false if cancelled
};

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
