#pragma once

#include <string>
#include <vector>
#include <deque>

/// @brief Structure to hold all context for a conversation session
/// Used in both interactive mode and server mode to organize system prompt,
/// messages, and tools in a unified way.
struct SessionContext {
    /// System prompt/context
    std::string system_prompt;

    /// Conversation messages (role + content pairs)
    struct Message {
        std::string role;      // "system", "user", "assistant", "tool"
        std::string content;
        std::string name;      // For tool messages
        std::string tool_call_id;  // For tool messages
    };
    std::deque<Message> messages;

    /// Tool definitions available in this session
    struct ToolDefinition {
        std::string name;
        std::string description;
        std::string parameters_json;  // JSON schema for parameters
    };
    std::vector<ToolDefinition> tools;

    // Server mode tracking state
    // (These fields are only used in server mode for prefix caching optimization)

    /// Track whether system prompt has been sent to backend
    bool system_sent = false;

    /// Track whether tools have been sent to backend
    bool tools_sent = false;

    /// Track how many messages have been sent to backend
    /// (Used to detect which messages are new and need to be added)
    size_t messages_sent_count = 0;

    /// Client identifier for this session (server mode only)
    std::string client_id;
};
