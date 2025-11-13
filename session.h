
#pragma once

#include <deque>
#include <vector>
#include <functional>
#include "message.h"
#include "nlohmann/json.hpp"

// Forward declarations to avoid circular dependency
class Backend;
struct Response;

class Session {
public:
	Backend *backend;

	// System message ("You are an unhelpful idiot", etc)
	std::string system_message;
	int system_message_tokens = 0;

	// List of messages (user/tool/model)
    std::deque<Message> messages;

    // Last prompt_tokens from API (for delta calculation)
    int last_prompt_tokens = 0;

    // Total tokens from API (authoritative source of truth)
    int total_tokens = 0;

    // Track last user and assistant messages to preserve critical context
    // These must be protected from eviction and reserved in space calculations
    int last_user_message_index = -1;
    int last_user_message_tokens = 0;
    int last_assistant_message_index = -1;
    int last_assistant_message_tokens = 0;

    // Auto-eviction flag: when true, proactively evict messages before exceeding context limit
    // Set to true when user's context_size < backend's API context size
    bool auto_evict = false;

    // Desired completion tokens - calculated once during initialization
    // Scales from ~1,374 tokens (8K context) to 4,096 (32K+ context)
    // Used for max_tokens calculation and eviction decisions
    int desired_completion_tokens = 0;

    /// Tool definitions available in this session
    /// This is the single source of truth for tools in a session
    /// - CLI mode: populated from ToolRegistry at startup
    /// - Server mode: populated from client request
    struct Tool {
        std::string name;           // Tool name (sanitized)
        std::string description;    // Tool description
        nlohmann::json parameters;  // Full JSON schema object (not string)

        // Helper to get parameters as string for legacy code
        std::string parameters_json() const {
            return parameters.dump();
        }

        // Helper to get parameter string for simple text format
        std::string parameters_text() const {
            // Extract required parameters for simple format like "query: string"
            std::string result;
            if (parameters.contains("properties")) {
                for (auto& [key, value] : parameters["properties"].items()) {
                    if (!result.empty()) result += ", ";
                    result += key + ": ";
                    if (value.contains("type")) {
                        result += value["type"].get<std::string>();
                    } else {
                        result += "any";
                    }
                }
            }
            return result.empty() ? "none" : result;
        }
    };
    std::vector<Tool> tools;

    // Main message interface - handles eviction and delegates to backend
    /// @brief Add a message to the session with automatic eviction if needed
    /// @param type Message type (USER, ASSISTANT, TOOL)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL messages)
    /// @param tool_id Tool call ID (for TOOL messages)
    /// @param prompt_tokens Estimated tokens for message (0 = auto-calculate)
    /// @param max_tokens Max tokens for assistant response (0 = auto-calculate)
    /// @return Response from backend (success, content, tool_calls, tokens)
    Response add_message(Message::Type type, const std::string& content, const std::string& tool_name = "", const std::string& tool_id = "", int prompt_tokens = 0, int max_tokens = 0);

    // Forward declaration for streaming callback
    using StreamCallback = std::function<bool(const std::string& delta,
                                              const std::string& accumulated,
                                              const Response& partial_response)>;

    /// Streaming version of add_message
    /// @param type Message type (USER, ASSISTANT, TOOL, etc.)
    /// @param content Message content
    /// @param callback Streaming callback for incremental output
    /// @param tool_name Tool name (for TOOL messages)
    /// @param tool_id Tool call ID (for TOOL messages)
    /// @param prompt_tokens Pre-calculated prompt tokens (0 = backend calculates)
    /// @param max_tokens Max tokens for assistant response (0 = auto-calculate)
    /// @return Response from backend (success, content, tool_calls, tokens)
    Response add_message_stream(Message::Type type, const std::string& content,
                               StreamCallback callback,
                               const std::string& tool_name = "",
                               const std::string& tool_id = "",
                               int prompt_tokens = 0,
                               int max_tokens = 0);

    // Eviction methods - used by both API and GPU backends
    // Two-pass strategy: Pass 1 evicts complete turns, Pass 2 evicts mini-turns

    /// @brief Calculate which messages to evict using two-pass strategy
    /// @param tokens_needed Number of tokens to free
    /// @return Vector of (start_index, end_index) ranges to evict, empty if cannot evict enough
    ///         Each range is inclusive and non-contiguous to preserve protected messages
    std::vector<std::pair<int, int>> calculate_messages_to_evict(int tokens_needed);

    /// @brief Actually remove messages from session
    /// @param ranges Vector of (start_idx, end_idx) ranges to evict (inclusive)
    /// @return true if successful
    bool evict_messages(const std::vector<std::pair<int, int>>& ranges);

private:
    // Helper methods for auto-eviction
    /// @brief Check if adding message would exceed context limit
    bool needs_eviction(int additional_tokens) const;

    /// @brief Get available tokens (may reserve space for response)
    int get_available_tokens() const;
};

/// @brief Calculate truncation scale factor based on context size
/// @param context_size Total context window size in tokens
/// @return Scale factor between 0.33 and 0.6 for truncating user input and tool results
/// @details Larger contexts get larger percentages to allow more generous limits
///          while still preserving space for responses:
///          - 8k context  → 35.7% (allows ~1,250 tokens)
///          - 16k context → 38.4% (allows ~4,416 tokens)
///          - 32k context → 43.8% (allows ~12,045 tokens)
///          - 64k context → 54.6% (allows ~32,487 tokens)
inline double calculate_truncation_scale(int context_size) {
    // Linear scaling: larger contexts get more generous limits
    double scale = 0.33 + (context_size / 80000.0) * 0.27;
    // Clamp between 33% and 60%
    return std::max(0.33, std::min(0.6, scale));
}

/// @brief Calculate desired completion tokens to reserve for assistant responses
/// @param context_size Total context window size in tokens
/// @param max_output_tokens Model's maximum output token limit (0 = no limit)
/// @return Number of tokens to reserve for completion, capped by model's max_output_tokens
/// @details Uses calculate_truncation_scale() on half the context size to determine
///          a reasonable completion size that scales with context. Result is capped
///          at the model's max_output_tokens limit.
///          - 8K context  → 1,374 tokens
///          - 16K context → 2,856 tokens
///          - 32K+ context → 4,096 tokens (capped by model max_output_tokens)
inline int calculate_desired_completion_tokens(int context_size, int max_output_tokens) {
    if (context_size == 0) return 0;

    // Use half the context size to calculate a more conservative scale
    double scale = calculate_truncation_scale(context_size / 2);
    int desired = static_cast<int>((context_size / 2) * scale);

    // Cap at model's max_output_tokens if specified
    if (max_output_tokens > 0 && desired > max_output_tokens) {
        return max_output_tokens;
    }

    return desired;
}
