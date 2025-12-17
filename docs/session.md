# Session Module

## Overview

The Session class (`session.cpp/h`) manages conversation state including message history, token counting, context window management, and tool definitions.

## Key Responsibilities

1. **Message History**: Stores conversation messages
2. **Token Management**: Tracks token counts per message and total
3. **Context Eviction**: Removes old messages when context fills
4. **Tool Definitions**: Holds tools for backend formatting
5. **Backend Delegation**: Routes `add_message()` calls to backend

## Session Class

```cpp
class Session {
public:
    // Event callback type (same as Backend::EventCallback)
    using EventCallback = std::function<bool(
        Message::Type type,
        const std::string& content,
        const std::string& tool_name,
        const std::string& tool_call_id
    )>;

    // Main message interface
    Response add_message(
        Message::Type type,
        const std::string& content,
        EventCallback callback = nullptr,  // Streaming callback
        const std::string& tool_name = "",
        const std::string& tool_id = "",
        int prompt_tokens = 0,
        int max_tokens = 0
    );

    // Message storage
    std::vector<Message> messages;
    int total_tokens{0};

    // Tool definitions (for backend)
    std::vector<Tool> tools;

    // Backend reference (set by Frontend::connect_provider)
    Backend* backend{nullptr};

    // Tracking indices
    int last_user_message_index{-1};
    int last_user_message_tokens{0};
    int last_assistant_message_index{-1};
    int last_assistant_message_tokens{0};
};
```

## Message Structure

```cpp
struct Message {
    enum Type {
        SYSTEM,
        USER,
        ASSISTANT,
        TOOL_REQUEST,    // Model requesting tool execution
        TOOL_RESPONSE,   // Result of tool execution
        FUNCTION         // Legacy (maps to TOOL_RESPONSE)
    };

    Type type;
    std::string content;
    int tokens{0};

    // Tool-related fields
    std::string tool_name;
    std::string tool_call_id;
    std::vector<ToolParser::ToolCall> tool_calls;

    // Helper methods
    std::string get_role() const;  // "system", "user", "assistant", "tool"
};
```

## add_message() Flow

```cpp
Response Session::add_message(Message::Type type,
                             const std::string& content,
                             EventCallback callback,
                             const std::string& tool_name,
                             const std::string& tool_id,
                             int prompt_tokens,
                             int max_tokens) {
    // 1. Calculate tokens if not provided
    if (prompt_tokens == 0) {
        prompt_tokens = backend->count_message_tokens(type, content, tool_name, tool_id);
    }

    // 2. Calculate max_tokens if not provided
    if (max_tokens == 0) {
        max_tokens = calculate_desired_completion_tokens(backend->context_size, ...);
    }

    // 3. Delegate to backend (with callback for streaming)
    Response resp = backend->add_message(*this, type, content, callback,
                                         tool_name, tool_id, prompt_tokens, max_tokens);

    // 4. Handle auto-continuation if hit length limit
    while (resp.finish_reason == "length") {
        Response continuation = backend->add_message(*this, type, "", callback, ...);
        resp.content += continuation.content;
        resp.finish_reason = continuation.finish_reason;
    }

    return resp;
}
```

## Context Window Management

### Token Tracking (Source of Truth)

**Session is the authoritative source of truth for token counts.** Backends update session token fields during generation:

```cpp
// Token fields in Session (updated by backend after generation)
int total_tokens;                    // Total tokens from API response
int last_prompt_tokens;              // Prompt tokens from last API call
int last_assistant_message_tokens;   // Completion tokens from last response
```

In API backends, `generate_from_session()` updates session after receiving API response:
```cpp
void ApiBackend::generate_from_session(Session& session, int max_tokens) {
    // ... make API call ...
    Response resp = parse_http_response(http_response);

    // Update session token counts (session is source of truth)
    session.total_tokens = resp.prompt_tokens + resp.completion_tokens;
    session.last_prompt_tokens = resp.prompt_tokens;
    session.last_assistant_message_tokens = resp.completion_tokens;
}
```

Each message also tracks its token count:
```cpp
Message msg(Message::USER, "Hello world", 15);  // 15 tokens
session.messages.push_back(msg);
session.total_tokens += 15;
```

### Eviction Strategy

Two-pass eviction when context fills:

**Pass 1: Complete Turn Eviction**
- Remove complete user+assistant turn pairs
- Skip system message (always kept)
- Clear associated tool calls

**Pass 2: Mini-Turn Eviction**
- Remove individual messages if needed
- Last resort before failing

```cpp
bool evict_for_tokens(int tokens_needed) {
    // Pass 1: Remove complete turns
    while (total_tokens + tokens_needed > context_size) {
        if (!evict_oldest_turn()) break;
    }

    // Pass 2: Remove individual messages
    while (total_tokens + tokens_needed > context_size) {
        if (!evict_oldest_message()) return false;
    }

    return true;
}
```

## Tool Definitions

Tools are stored for backend formatting:

```cpp
struct Tool {
    std::string name;
    std::string description;
    std::string parameters;  // JSON Schema
};

// Populated by Frontend::init_tools()
tools.populate_session_tools(session);
```

Backend uses `session.tools` when formatting requests:
```cpp
// In OpenAI backend:
for (const auto& tool : session.tools) {
    json tool_obj = {
        {"type", "function"},
        {"function", {
            {"name", tool.name},
            {"description", tool.description},
            {"parameters", json::parse(tool.parameters)}
        }}
    };
    request["tools"].push_back(tool_obj);
}
```

## Message History Access

```cpp
// Iterate messages
for (const auto& msg : session.messages) {
    std::cout << msg.get_role() << ": " << msg.content << "\n";
}

// Find last user message
for (int i = session.messages.size() - 1; i >= 0; i--) {
    if (session.messages[i].type == Message::USER) {
        // Found it
        break;
    }
}

// Clear history (keep system)
void clear_history() {
    messages.erase(
        std::remove_if(messages.begin(), messages.end(),
            [](const Message& m) { return m.type != Message::SYSTEM; }),
        messages.end()
    );
    recalculate_total_tokens();
}
```

## Thread Safety

Session is NOT thread-safe. The GenerationThread pattern ensures only one thread accesses Session at a time:
- Main thread submits requests to GenerationThread
- GenerationThread calls `session.add_message()`
- Main thread waits until complete before accessing session

## Backend Integration

Session holds a raw pointer to the active backend:
```cpp
// Set by Frontend::connect_provider()
session.backend = frontend->backend.get();

// Used internally by add_message()
Response resp = backend->add_message(*this, ...);
```

The backend is owned by Frontend; Session just references it.

## Serialization

Messages can be serialized for:
- Saving conversation history
- Sending to API backends
- Logging/debugging

```cpp
nlohmann::json to_json() const {
    json j;
    j["messages"] = json::array();
    for (const auto& msg : messages) {
        j["messages"].push_back({
            {"role", msg.get_role()},
            {"content", msg.content}
        });
    }
    return j;
}
```
