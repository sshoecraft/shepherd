#pragma once

#include <string>
#include <vector>
#include <map>
#include <any>
#include <functional>

// Forward declaration
class Tool;

/// @brief Message roles for context management
enum class MessageRole {
    SYSTEM,
    USER,
    ASSISTANT,
    TOOL
};

/// @brief Represents a tool/function call from the model
struct ToolCall {
    std::string id;          // Unique ID for this tool call (needed by some APIs)
    std::string name;         // Name of the tool to call
    std::map<std::string, std::any> arguments;  // Tool arguments
};


/// @brief Response from model generation
struct ModelResponse {
    std::string text;                    // Text response from model (may be empty if only tool calls)
    std::vector<ToolCall> tool_calls;    // Tool calls to execute (may be empty)
    bool needs_tool_use;                 // Whether model is waiting for tool results
};

/// @brief Configuration for a backend
struct BackendConfig {
    std::string model_path_or_name;      // Model file path or API model name
    std::string api_key;                  // API key (for cloud backends)
    size_t max_context_tokens;            // Maximum context window size
    std::string system_prompt;            // System prompt to use
    std::vector<Tool*> available_tools;  // Tools available to the model
    bool debug_mode;                      // Enable debug output
    std::string log_file;                 // Optional log file path
};

/// @brief New unified backend interface
/// Each backend handles ALL model-specific interaction including:
/// - Message formatting for its specific API/model
/// - Tool registration in the model's expected format
/// - Response parsing including tool call extraction
/// - Context management in the model's native format
/// - Tool result integration back into context
class IBackendManager {
public:
    virtual ~IBackendManager() = default;

    /// @brief Initialize the backend with configuration
    /// @param config Backend configuration including model, tools, system prompt
    /// @return True if initialization successful
    virtual bool initialize(const BackendConfig& config) = 0;

    /// @brief Process a user turn - handles everything internally
    /// This method:
    /// 1. Formats the user message appropriately
    /// 2. Sends to model with proper context
    /// 3. Parses response including any tool calls
    /// 4. Returns structured response for main to handle
    /// @param user_message The user's input message
    /// @return Structured response with text and/or tool calls
    virtual ModelResponse process_turn(const std::string& user_message) = 0;

    /// @brief Add tool execution results back to context
    /// The backend formats these appropriately for its model
    /// @param results Tool execution results to add
    virtual void add_tool_results(const std::vector<ToolResult>& results) = 0;

    /// @brief Get backend name/identifier
    /// @return Backend name (e.g., "llamacpp", "openai", "anthropic")
    virtual std::string get_backend_name() const = 0;

    /// @brief Get model name currently loaded
    /// @return Model identifier
    virtual std::string get_model_name() const = 0;

    /// @brief Get current context utilization
    /// @return Utilization from 0.0 to 1.0
    virtual double get_context_utilization() const = 0;

    /// @brief Clear conversation context
    virtual void clear_context() = 0;

    /// @brief Check if backend is ready
    /// @return True if ready for inference
    virtual bool is_ready() const = 0;

    /// @brief Shutdown and cleanup
    virtual void shutdown() = 0;

    /// @brief Set callback for displaying output
    /// Allows backend to display text as needed
    void set_display_callback(std::function<void(const std::string&)> callback) {
        display_callback_ = callback;
    }

protected:
    /// @brief Display text to user (if callback is set)
    void display(const std::string& text) {
        if (display_callback_) {
            display_callback_(text);
        }
    }

    std::function<void(const std::string&)> display_callback_;
};