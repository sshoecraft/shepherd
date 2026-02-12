#pragma once

#include "backend.h"
#include "provider.h"
#include "session.h"
#include "tools/tools.h"
#include "memory_extraction.h"
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <any>

// Forward declaration
class Tools;

/// @brief Logical colors for frontend output
/// Maps to ANSI codes (CLI) or ncurses pairs (TUI)
enum class FrontendColor {
    DEFAULT,  // White/default terminal color
    GREEN,    // User input, tool results
    YELLOW,   // Tool calls
    RED,      // Errors, system warnings
    CYAN,     // Code blocks
    GRAY      // Thinking, dim text
};

/// @brief Get logical color for a callback event type
/// Centralizes color decisions so CLI and TUI are consistent
inline FrontendColor get_color_for_event(CallbackEvent event) {
    switch (event) {
        case CallbackEvent::USER_PROMPT:  return FrontendColor::GREEN;
        case CallbackEvent::TOOL_CALL:    return FrontendColor::YELLOW;
        case CallbackEvent::TOOL_RESULT:  return FrontendColor::GREEN;
        case CallbackEvent::TOOL_DISP:    return FrontendColor::YELLOW;  // Same as TOOL_CALL
        case CallbackEvent::RESULT_DISP:  return FrontendColor::GREEN;   // Same as TOOL_RESULT
        case CallbackEvent::ERROR:        return FrontendColor::RED;
        case CallbackEvent::SYSTEM:       return FrontendColor::RED;
        case CallbackEvent::THINKING:     return FrontendColor::GRAY;
        case CallbackEvent::CODEBLOCK:    return FrontendColor::CYAN;
        case CallbackEvent::CONTENT:      return FrontendColor::DEFAULT;
        case CallbackEvent::STOP:         return FrontendColor::DEFAULT;
        case CallbackEvent::STATS:        return FrontendColor::GRAY;
        default:                          return FrontendColor::DEFAULT;
    }
}

/// @brief Get indentation (number of spaces) for a callback event type
/// Centralizes formatting so CLI and TUI are consistent
inline int get_indent_for_event(CallbackEvent event) {
    switch (event) {
        case CallbackEvent::USER_PROMPT:  return 0;  // "> " prefix handled separately
        case CallbackEvent::SYSTEM:       return 0;
        case CallbackEvent::ERROR:        return 0;
        case CallbackEvent::TOOL_CALL:    return 2;  // "  read(...)"
        case CallbackEvent::TOOL_RESULT:  return 4;  // "    Output: ..."
        case CallbackEvent::TOOL_DISP:    return 2;  // Same as TOOL_CALL
        case CallbackEvent::RESULT_DISP:  return 4;  // Same as TOOL_RESULT
        case CallbackEvent::CONTENT:      return 2;  // Assistant response
        case CallbackEvent::THINKING:     return 2;
        case CallbackEvent::CODEBLOCK:    return 4;
        case CallbackEvent::STATS:        return 2;
        case CallbackEvent::STOP:         return 0;
        default:                          return 0;
    }
}

/// @brief Base class for all frontend presentation layers (CLI, Server)
/// Manages backend, providers, and session lifecycle
class Frontend {
public:
    Frontend();
    virtual ~Frontend();

    /// @brief Event callback for streaming output from backend
    /// Derived classes override to handle output (display, SSE, etc.)
    /// @param event Event type (CONTENT, TOOL_CALL, ERROR, STOP, etc.)
    /// @param content Event content (text delta, error message, finish_reason, etc.)
    /// @param name Tool name (for TOOL_CALL), error type (for ERROR)
    /// @param id Tool call ID for correlation
    /// @return true to continue generation, false to cancel
    virtual bool on_event(CallbackEvent event,
                          const std::string& content,
                          const std::string& name,
                          const std::string& id) { return true; }

    /// @brief Factory method to create and initialize appropriate frontend
    /// @param mode Frontend mode: "cli", "api-server", "cli-server"
    /// @param host Server host (for server modes)
    /// @param port Server port (for server modes)
    /// @param cmdline_provider Optional provider from command-line override
    /// @param no_mcp If true, skip MCP initialization
    /// @param no_tools If true, skip all tool initialization
    static std::unique_ptr<Frontend> create(const std::string& mode, const std::string& host, int port,
                                            Provider* cmdline_provider = nullptr,
                                            bool no_mcp = false, bool no_tools = false,
                                            const std::string& target_provider = "",
                                            bool no_rag = false,
                                            bool mem_tools = false);

    /// @brief Initialize the frontend (register tools, etc) - called by create()
    virtual void init(bool no_mcp = false, bool no_tools = false, bool no_rag = false, bool mem_tools = false) {}

protected:
    /// @brief Common tool initialization - initializes RAG and registers all tools
    /// @param no_mcp If true, skip MCP initialization
    /// @param no_tools If true, skip all tool initialization
    /// @param force_local If true, force local tool init even if server_tools is set (for fallback)
    /// @param no_rag If true, skip RAG initialization and memory extraction
    void init_tools(bool no_mcp, bool no_tools, bool force_local = false, bool no_rag = false, bool mem_tools = false);

    /// @brief Initialize tools from remote server (when --server-tools and API provider)
    /// Called after provider connection when config->server_tools is true
    /// @param server_url Base URL with /v1 (e.g., "http://localhost:8000/v1")
    /// @param api_key API key for Bearer token authentication
    void init_remote_tools(const std::string& server_url, const std::string& api_key);

public:

    /// @brief Start the frontend main loop
    /// Pure virtual - subclasses implement their specific behavior
    /// Connects to provider internally before running
    /// @param cmdline_provider Optional provider from command-line override (takes priority)
    virtual int run(Provider* cmdline_provider = nullptr) = 0;

    /// @brief Handle slash commands (e.g., /provider, /model, /clear)
    /// @param input The full input string starting with /
    /// @param tools Tools reference for /tools command
    /// @return true if command was handled, false if not recognized
    bool handle_slash_commands(const std::string& input, Tools& tools);

    /// @brief Get provider by name (returns nullptr if not found)
    Provider* get_provider(const std::string& name);

    /// @brief List all provider names (sorted by priority)
    std::vector<std::string> list_providers() const;

    /// @brief Connect to next available provider
    /// @return true if connected, false if all providers fail
    bool connect_next_provider();

    /// @brief Connect to a specific provider by name
    /// @param name Provider name
    /// @return true if connected, false if connection fails
    bool connect_provider(const std::string& name);

    /// @brief Execute a tool, truncate result based on context, and add to session
    /// This is the common implementation for CLI, TUI, and CLI-server.
    /// API server does NOT use this - it returns tool calls to the client.
    /// @param tools The Tools instance
    /// @param tool_name Name of the tool to execute
    /// @param params_json Tool parameters as JSON string
    /// @param tool_call_id Tool call ID for correlation
    /// @return ToolResult with success/summary/error for display
    ToolResult execute_tool(Tools& tools,
                            const std::string& tool_name,
                            const std::string& params_json,
                            const std::string& tool_call_id,
                            const std::string& user_id = "");

    /// @brief Format output for terminal display
    /// Converts LaTeX math notation to Unicode and aligns markdown tables.
    /// Called by CLI/TUI before displaying assistant content.
    /// @param text Raw text from model (may contain LaTeX/markdown)
    /// @return Formatted text suitable for terminal display
    static std::string format_output(const std::string& text);

    /// @brief Add a message directly to session without triggering generation
    /// Frontend owns the session - messages are added directly here.
    /// Use generate_response() afterwards to generate assistant response.
    /// @param role Message role (USER, ASSISTANT, TOOL_RESPONSE)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL_RESPONSE messages)
    /// @param tool_id Tool call ID (for TOOL_RESPONSE messages)
    void add_message_to_session(Message::Role role,
                                const std::string& content,
                                const std::string& tool_name = "",
                                const std::string& tool_id = "");

    /// @brief Enrich the last user message with RAG context injection
    /// Searches RAG backend using keywords from the message, appends
    /// relevant results as [context: ...] suffix. No-op if RAG not
    /// initialized, not enabled in config, or no results found.
    void enrich_with_rag_context(Session& session);

    /// @brief Extract search keywords from a user message
    static std::string extract_keywords(const std::string& message);

    /// @brief Generate response from current session state
    /// This is the unified generation path used by all frontends.
    /// Handles proactive eviction (if auto_evict enabled) and reactive eviction
    /// (on ContextFullException) for stateful frontends.
    /// @param max_tokens Max tokens for response (0 = auto-calculate)
    /// @return true if generation completed, false if error or eviction failed
    bool generate_response(int max_tokens = 0);

    // Queue conversation for background memory extraction (persistent sessions)
    void queue_memory_extraction();

    // Queue only the last exchange for extraction (ephemeral API sessions)
    void queue_memory_extraction_last(const Session& sess, const std::string& req_user_id);

    // Session owned by frontend (source of truth for conversation state)
    Session session;

    // Provider list owned by frontend
    std::vector<Provider> providers;
    std::string current_provider;

    // Backend owned by frontend after connect
    std::unique_ptr<Backend> backend;

    // Tools owned by frontend
    Tools tools;

    // Callback for streaming output - must be set by subclass before connecting
    Backend::EventCallback callback;

    // Background memory extraction thread (started if memory_extraction enabled)
    std::unique_ptr<MemoryExtractionThread> extraction_thread;

    // User identifier for multi-tenant memory isolation
    // CLI/TUI/CLI-Server: hostname:username, API Server: from OpenAI "user" field
    std::string user_id;
};
