
#pragma once

#include <deque>
#include <vector>
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

    // Eviction methods - used by both API and GPU backends
    // Two-pass strategy: Pass 1 evicts complete turns, Pass 2 evicts mini-turns

    /// @brief Calculate which messages to evict using two-pass strategy
    /// @param tokens_needed Number of tokens to free
    /// @return Pair of (start_index, end_index) to evict, or (-1, -1) if cannot evict enough
    std::pair<int, int> calculate_messages_to_evict(int tokens_needed);

    /// @brief Actually remove messages from session
    /// @param start_idx Starting index to evict (inclusive)
    /// @param end_idx Ending index to evict (inclusive)
    /// @return true if successful
    bool evict_messages(int start_idx, int end_idx);

private:
    // Helper methods for auto-eviction
    /// @brief Check if adding message would exceed context limit
    bool needs_eviction(int additional_tokens) const;

    /// @brief Get available tokens (may reserve space for response)
    int get_available_tokens() const;
};
