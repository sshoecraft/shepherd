#include "frontend.h"
#include "shepherd.h"
#include "cli.h"
#include "api_server.h"
#include "cli_server.h"
#include "tui.h"
#include "tools/tools.h"
#include "tools/remote_tools.h"
#include "tools/utf8_sanitizer.h"
#include "tools/scheduler_tools.h"
#include "rag.h"
#include "mcp/mcp.h"
#include "mcp/mcp_config.h"
#include "scheduler.h"
#include <algorithm>
#include <sstream>
#include <regex>

// Use config->thinking instead of g_show_thinking

// External handler functions
extern int handle_provider_args(const std::vector<std::string>& args,
                                std::function<void(const std::string&)> out,
                                std::unique_ptr<Backend>* backend,
                                Session* session,
                                std::vector<Provider>* providers,
                                std::string* current_provider,
                                Tools* tools);
extern int handle_model_args(const std::vector<std::string>& args,
                             std::function<void(const std::string&)> out,
                             std::unique_ptr<Backend>* backend,
                             std::vector<Provider>* providers,
                             std::string* current_provider);
extern int handle_config_args(const std::vector<std::string>& args,
                              std::function<void(const std::string&)> out);
extern int handle_sched_args(const std::vector<std::string>& args,
                             std::function<void(const std::string&)> out);

Frontend::Frontend() {
}

Frontend::~Frontend() {
}

std::unique_ptr<Frontend> Frontend::create(const std::string& mode, const std::string& host, int port, Provider* cmdline_provider, bool no_mcp, bool no_tools) {
    std::unique_ptr<Frontend> frontend;

    if (mode == "cli") {
        frontend = std::make_unique<CLI>();
    }
    else if (mode == "tui") {
        frontend = std::make_unique<TUI>();
    }
    else if (mode == "api-server") {
        frontend = std::make_unique<APIServer>(host, port, no_mcp, no_tools);
    }
    else if (mode == "cli-server") {
        frontend = std::make_unique<CLIServer>(host, port);
    }
    else {
        throw std::runtime_error("Invalid frontend mode: " + mode);
    }

    // Initialize session with system message from config
    frontend->session.system_message = config->system_message;

    // Load providers from disk
    frontend->providers = Provider::load_providers();

    // Add command-line provider at the front (highest priority)
    if (cmdline_provider) {
        frontend->providers.insert(frontend->providers.begin(), *cmdline_provider);
    }

    // Log mode
    std::string writer_name = (mode == "cli") ? "console" : "server";
    dout(1) << "Frontend initialized in " + mode + " mode" << std::endl;

    // Initialize the frontend (RAG, tools, MCP)
    frontend->init(no_mcp, no_tools);

    return frontend;
}

Provider* Frontend::get_provider(const std::string& name) {
    for (auto& p : providers) {
        if (p.name == name) {
            return &p;
        }
    }
    return nullptr;
}

std::vector<std::string> Frontend::list_providers() const {
    std::vector<std::string> names;
    for (const auto& p : providers) {
        names.push_back(p.name);
    }
    return names;
}

bool Frontend::connect_next_provider() {
    if (providers.empty()) {
        std::cerr << "No providers configured" << std::endl;
        return false;
    }

    if (!callback) {
        std::cerr << "Internal error: callback not set before connecting" << std::endl;
        return false;
    }

    for (auto& p : providers) {
        try {
            backend = p.connect(session, callback);
            if (backend) {
                current_provider = p.name;
                session.backend = backend.get();
                return true;
            }
        } catch (const std::exception& e) {
            std::cerr << "Provider '" + p.name + "' failed: " + std::string(e.what()) << std::endl;
            if (!config->auto_provider) {
                dout(1) << "auto_provider disabled, not trying other providers" << std::endl;
                return false;
            }
        }
    }

    std::cerr << "All providers failed to connect" << std::endl;
    return false;
}

bool Frontend::connect_provider(const std::string& name) {
    Provider* p = get_provider(name);
    if (!p) {
        std::cerr << "Provider '" + name + "' not found" << std::endl;
        return false;
    }

    if (!callback) {
        std::cerr << "Internal error: callback not set before connecting" << std::endl;
        return false;
    }

    try {
        backend = p->connect(session, callback);
        if (backend) {
            current_provider = name;
            session.backend = backend.get();
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Provider '" + name + "' failed: " + std::string(e.what()) << std::endl;
        return false;
    }
}

void Frontend::init_tools(bool no_mcp, bool no_tools, bool force_local) {
    // Initialize RAG system using global config
    std::string db_path = config->memory_database;
    if (db_path.empty()) {
        try {
            db_path = Config::get_default_memory_db_path();
        } catch (const ConfigError& e) {
            std::cerr << "Failed to determine memory database path: " + std::string(e.what()) << std::endl;
            throw;
        }
    } else if (db_path[0] == '~') {
        // Expand ~ if present
        db_path = Config::get_home_directory() + db_path.substr(1);
    }

    if (!RAGManager::initialize(db_path, config->max_db_size)) {
        throw std::runtime_error("Failed to initialize RAG system");
    }
    dout(1) << "RAG initialized with database: " + db_path << std::endl;

    if (no_tools) {
        dout(1) << "Tools disabled" << std::endl;
        return;
    }

    // If CLI backend, skip local tool init - CLI backend gets tools from remote server
    if (config->backend == "cli") {
        dout(1) << "CLI backend: skipping local tool init (tools provided by server)" << std::endl;
        return;
    }

    // If server_tools mode, defer tool init until after provider connection
    // (unless force_local is set, which is used for fallback when provider has no base_url)
    if (config->server_tools && !force_local) {
        dout(1) << "Server tools mode: deferring tool init until provider connected" << std::endl;
        return;
    }

    dout(1) << "Initializing local tools system..." << std::endl;

    // Register all native tools
    register_filesystem_tools(tools);
    register_command_tools(tools);
    register_json_tools(tools);
    register_http_tools(tools);
    register_memory_tools(tools);
    register_mcp_resource_tools(tools);
    register_core_tools(tools);
    register_scheduler_tools(tools);

    // Initialize MCP servers (registers additional tools)
    if (!no_mcp) {
        auto& mcp = MCP::instance();
        mcp.initialize(tools);
        dout(1) << "MCP initialized with " + std::to_string(mcp.get_tool_count()) + " tools" << std::endl;
    } else {
        dout(1) << "MCP system disabled" << std::endl;
    }

    // Build the combined tool list
    tools.build_all_tools();

    dout(1) << "Tools initialized: " + std::to_string(tools.all_tools.size()) + " total" << std::endl;

    // Populate session.tools from Tools instance
    tools.populate_session_tools(session);
    dout(1) << "Session initialized with " + std::to_string(session.tools.size()) + " tools" << std::endl;
}

void Frontend::init_remote_tools(const std::string& server_url, const std::string& api_key) {
    dout(1) << "Fetching remote tools from " + server_url << std::endl;

    int count = register_remote_tools(tools, server_url, api_key);
    if (count < 0) {
        std::cerr << "Warning: Failed to fetch remote tools from " + server_url << std::endl;
    } else {
        dout(1) << "Registered " + std::to_string(count) + " remote tools" << std::endl;
    }

    tools.build_all_tools();
    tools.populate_session_tools(session);

    dout(1) << "Remote tools initialized: " + std::to_string(tools.all_tools.size()) + " total" << std::endl;
}

ToolResult Frontend::execute_tool(Tools& tools,
                                   const std::string& tool_name,
                                   const std::string& params_json,
                                   const std::string& tool_call_id) {
    // Execute the tool
    ToolResult tool_result = tools.execute(tool_name, params_json);

    std::string result_content;
    if (tool_result.success) {
        result_content = utf8_sanitizer::sanitize_utf8(tool_result.content);
    } else {
        result_content = tool_result.error.empty()
            ? "Error: Tool execution failed"
            : "Error: " + tool_result.error;
    }

    // Truncate tool result if needed
    int reserved = session.system_message_tokens;
    if (session.last_user_message_index >= 0) {
        reserved += session.last_user_message_tokens;
    }
    if (session.last_assistant_message_index >= 0) {
        reserved += session.last_assistant_message_tokens;
    }

    double scale = calculate_truncation_scale(backend->context_size);
    int max_tool_result_tokens = (backend->context_size - reserved) * scale;

    // Check if this is a read tool requesting full file (limit=-1)
    bool bypass_truncate_limit = false;
    if (tool_name == "read") {
        try {
            auto params = nlohmann::json::parse(params_json);
            if (params.contains("limit") && params["limit"].is_number() && params["limit"].get<int>() == -1) {
                bypass_truncate_limit = true;
            }
        } catch (...) {}
    }

    if (!bypass_truncate_limit && config->truncate_limit > 0 && config->truncate_limit < max_tool_result_tokens) {
        max_tool_result_tokens = config->truncate_limit;
    }

    int tool_result_tokens = backend->count_message_tokens(Message::TOOL_RESPONSE, result_content, tool_name, tool_call_id);

    if (tool_result_tokens >= max_tool_result_tokens) {
        size_t original_line_count = std::count(result_content.begin(), result_content.end(), '\n');
        std::string truncation_notice = "\n\n[TRUNCATED: Output too large for context window]";
        truncation_notice += "\nOriginal length: " + std::to_string(original_line_count) + " lines";
        truncation_notice += "\nIf you need more: use Read(offset=X, limit=Y), Grep(pattern=...), or Glob with specific patterns";

        dout(1) << "Truncating tool result: " << original_line_count << " lines, "
                << result_content.length() << " bytes -> max " << max_tool_result_tokens << " tokens" << std::endl;

        while (tool_result_tokens >= max_tool_result_tokens && result_content.length() > 100) {
            double ratio = (static_cast<double>(max_tool_result_tokens) / static_cast<double>(tool_result_tokens)) * 0.95;
            int new_len = static_cast<int>(result_content.length() * ratio) - truncation_notice.length();
            if (new_len < 0) new_len = 0;
            result_content = result_content.substr(0, new_len) + truncation_notice;
            tool_result_tokens = backend->count_message_tokens(Message::TOOL_RESPONSE, result_content, tool_name, tool_call_id);
        }

        // Update summary to reflect truncation
        size_t truncated_lines = std::count(result_content.begin(), result_content.end(), '\n');
        tool_result.summary = "Truncated: " + std::to_string(truncated_lines) + "/" +
                              std::to_string(original_line_count) + " lines";
    }

    // Update content with truncated version
    tool_result.content = result_content;

    return tool_result;
}

void Frontend::add_message_to_session(Message::Role role,
                                       const std::string& content,
                                       const std::string& tool_name,
                                       const std::string& tool_id) {
    // Count tokens for eviction decisions
    int message_tokens = 0;
    if (backend) {
        message_tokens = backend->count_message_tokens(role, content, tool_name, tool_id);
    }

    // Create message and add directly to session
    Message msg(role, content, message_tokens);
    msg.tool_name = tool_name;
    msg.tool_call_id = tool_id;
    session.messages.push_back(msg);
    session.total_tokens += message_tokens;

    // Update tracking indices
    int new_index = static_cast<int>(session.messages.size()) - 1;
    if (role == Message::USER) {
        session.last_user_message_index = new_index;
        session.last_user_message_tokens = message_tokens;
    } else if (role == Message::ASSISTANT) {
        session.last_assistant_message_index = new_index;
        session.last_assistant_message_tokens = message_tokens;
    }

    dout(1) << "Frontend::add_message_to_session: role=" + std::to_string(static_cast<int>(role)) +
             ", tokens=" + std::to_string(message_tokens) +
             ", total_tokens=" + std::to_string(session.total_tokens) << std::endl;
}

bool Frontend::generate_response(int max_tokens) {
    if (!backend) {
        callback(CallbackEvent::ERROR, "No backend connected", "backend_error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return false;
    }

    // Calculate max_tokens if not provided
    if (max_tokens == 0 && backend->context_size > 0) {
        // Reserve space for critical context
        int reserved = session.system_message_tokens;
        if (session.last_user_message_index >= 0) {
            reserved += session.last_user_message_tokens;
        }
        if (session.last_assistant_message_index >= 0) {
            reserved += session.last_assistant_message_tokens;
        }

        // Calculate available space
        int available = backend->context_size - reserved - session.total_tokens;
        max_tokens = (available > 0) ? available : 0;

        // Cap at desired completion size
        if (max_tokens > session.desired_completion_tokens) {
            max_tokens = session.desired_completion_tokens;
        }

        dout(1) << "Frontend::generate_response: calculated max_tokens=" + std::to_string(max_tokens) +
                 " (context=" + std::to_string(backend->context_size) +
                 ", reserved=" + std::to_string(reserved) +
                 ", total=" + std::to_string(session.total_tokens) +
                 ", desired_completion=" + std::to_string(session.desired_completion_tokens) + ")" << std::endl;
    }

    // Proactive eviction: if auto_evict is enabled and we would exceed context
    if (session.auto_evict && backend->context_size > 0) {
        // Use calculated max_tokens for eviction check (handles INT_MAX desired_completion_tokens)
        int completion_reserve = (max_tokens > 0) ? max_tokens :
            std::min(session.desired_completion_tokens, static_cast<int>(backend->context_size / 2));
        size_t required = static_cast<size_t>(session.total_tokens) + completion_reserve;
        if (required > backend->context_size) {
            int tokens_over = static_cast<int>(required - backend->context_size);
            dout(1) << "Frontend::generate_response: proactive eviction needed, " +
                     std::to_string(tokens_over) + " tokens over limit" << std::endl;

            auto ranges = session.calculate_messages_to_evict(tokens_over);
            if (ranges.empty()) {
                callback(CallbackEvent::ERROR,
                         "Cannot generate: context full and no messages available for eviction",
                         "context_full", "");
                callback(CallbackEvent::STOP, "error", "", "");
                return false;
            }

            if (!session.evict_messages(ranges)) {
                callback(CallbackEvent::ERROR, "Eviction failed unexpectedly", "eviction_error", "");
                callback(CallbackEvent::STOP, "error", "", "");
                return false;
            }

            // Recalculate max_tokens after eviction
            int reserved = session.system_message_tokens;
            if (session.last_user_message_index >= 0) {
                reserved += session.last_user_message_tokens;
            }
            if (session.last_assistant_message_index >= 0) {
                reserved += session.last_assistant_message_tokens;
            }
            int available = backend->context_size - reserved - session.total_tokens;
            max_tokens = (available > 0) ? available : 0;
            if (max_tokens > session.desired_completion_tokens) {
                max_tokens = session.desired_completion_tokens;
            }
        }
    }

    // Set up content accumulation wrapper around the frontend's callback
    // On STOP event, we capture the assistant message BEFORE TOOL_CALL callbacks fire,
    // because TOOL_CALL handlers may recursively call generate_response() which clears state.
    std::string accumulated_content;
    bool assistant_message_added = false;
    auto original_callback = backend->callback;
    backend->callback = [this, &accumulated_content, &assistant_message_added, &original_callback](
        CallbackEvent event, const std::string& content,
        const std::string& name, const std::string& id) -> bool {
        // Accumulate content for assistant message
        if (event == CallbackEvent::CONTENT ||
            event == CallbackEvent::THINKING ||
            event == CallbackEvent::CODEBLOCK) {
            accumulated_content += content;
        }

        // On STOP, add assistant message BEFORE passing to frontend callback
        // This must happen before TOOL_CALL callbacks, which may trigger recursive generate_response()
        if (event == CallbackEvent::STOP && !assistant_message_added) {
            // Backend has recorded tool calls by now (record_tool_call happens before STOP)
            if (!accumulated_content.empty() || !backend->accumulated_tool_calls.empty()) {
                Message assistant_msg(Message::ASSISTANT, accumulated_content,
                                      session.last_assistant_message_tokens);
                if (!backend->accumulated_tool_calls.empty()) {
                    assistant_msg.tool_calls_json = backend->accumulated_tool_calls.dump();
                }
                session.messages.push_back(assistant_msg);
                session.last_assistant_message_index = static_cast<int>(session.messages.size()) - 1;
                dout(1) << "Frontend::generate_response: added assistant message on STOP, content_len=" +
                         std::to_string(accumulated_content.length()) +
                         ", tool_calls=" + std::to_string(backend->accumulated_tool_calls.size()) << std::endl;
            }
            assistant_message_added = true;
        }

        // Call the frontend's callback for display/handling
        return original_callback(event, content, name, id);
    };

    // Call backend to generate
    try {
        backend->generate_from_session(session, max_tokens);

        // Restore original callback
        backend->callback = original_callback;

        return true;
    } catch (const ContextFullException& e) {
        // Reactive eviction: backend threw because context is full
        dout(1) << "Frontend::generate_response: ContextFullException caught, attempting reactive eviction" << std::endl;

        // Calculate how many tokens to free (cap desired_completion_tokens to avoid overflow with INT_MAX)
        int completion_reserve = std::min(session.desired_completion_tokens, static_cast<int>(backend->context_size / 2));
        int tokens_over = session.total_tokens + completion_reserve - backend->context_size;
        if (tokens_over <= 0) tokens_over = completion_reserve; // Free at least completion space

        auto ranges = session.calculate_messages_to_evict(tokens_over);
        if (ranges.empty()) {
            backend->callback = original_callback;
            callback(CallbackEvent::ERROR, e.what(), "context_full", "");
            callback(CallbackEvent::STOP, "error", "", "");
            return false;
        }

        if (!session.evict_messages(ranges)) {
            backend->callback = original_callback;
            callback(CallbackEvent::ERROR, "Eviction failed after context overflow", "eviction_error", "");
            callback(CallbackEvent::STOP, "error", "", "");
            return false;
        }

        // Retry generation after eviction
        try {
            // Recalculate max_tokens
            int reserved = session.system_message_tokens;
            if (session.last_user_message_index >= 0) {
                reserved += session.last_user_message_tokens;
            }
            if (session.last_assistant_message_index >= 0) {
                reserved += session.last_assistant_message_tokens;
            }
            int available = backend->context_size - reserved - session.total_tokens;
            max_tokens = (available > 0) ? available : 0;
            if (max_tokens > session.desired_completion_tokens) {
                max_tokens = session.desired_completion_tokens;
            }

            // Clear accumulated state for retry
            accumulated_content.clear();
            assistant_message_added = false;

            backend->generate_from_session(session, max_tokens);

            // Restore original callback
            backend->callback = original_callback;

            // Assistant message is added by the callback wrapper on STOP
            return true;
        } catch (const ContextFullException& e2) {
            backend->callback = original_callback;
            callback(CallbackEvent::ERROR, e2.what(), "context_full", "");
            callback(CallbackEvent::STOP, "error", "", "");
            return false;
        } catch (const std::exception& e2) {
            backend->callback = original_callback;
            callback(CallbackEvent::ERROR, e2.what(), "generation_error", "");
            callback(CallbackEvent::STOP, "error", "", "");
            return false;
        }
    } catch (const std::exception& e) {
        backend->callback = original_callback;
        callback(CallbackEvent::ERROR, e.what(), "generation_error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return false;
    }
}

bool Frontend::handle_slash_commands(const std::string& input, Tools& tools) {
    // Tokenize the input
    std::istringstream iss(input);
    std::string cmd;
    iss >> cmd;

    // Parse remaining arguments - handle quoted strings
    std::vector<std::string> args;
    std::string rest;
    std::getline(iss, rest);

    std::string token;
    bool in_quotes = false;
    std::string quoted_arg;

    for (size_t i = 0; i < rest.size(); ++i) {
        char c = rest[i];
        if (c == '"') {
            if (in_quotes) {
                args.push_back(quoted_arg);
                quoted_arg.clear();
                in_quotes = false;
            } else {
                in_quotes = true;
            }
        } else if (std::isspace(c) && !in_quotes) {
            if (!token.empty()) {
                args.push_back(token);
                token.clear();
            }
        } else {
            if (in_quotes) {
                quoted_arg += c;
            } else {
                token += c;
            }
        }
    }
    if (!token.empty()) args.push_back(token);
    if (!quoted_arg.empty()) args.push_back(quoted_arg);

    // /clear - clear the session context
    if (cmd == "/clear") {
        // Clear remote server session (if using CLI backend)
        if (backend) {
            backend->clear_session();
        }
        session.clear();
        callback(CallbackEvent::SYSTEM, "Session cleared.\n", "", "");
        return true;
    }

    // /provider commands
    if (cmd == "/provider") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        handle_provider_args(args, out, &backend, &session, &providers, &current_provider, &tools);
        return true;
    }

    // /model commands
    if (cmd == "/model") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        handle_model_args(args, out, &backend, &providers, &current_provider);
        return true;
    }

    // /config command
    if (cmd == "/config") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        handle_config_args(args, out);
        return true;
    }

    // /sched commands
    if (cmd == "/sched") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        handle_sched_args(args, out);
        return true;
    }

    // /tools command
    if (cmd == "/tools") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        tools.handle_tools_args(args, out);
        return true;
    }

    // /mcp command
    if (cmd == "/mcp") {
        auto out = [this](const std::string& msg) {
            callback(CallbackEvent::CONTENT, msg, "", "");
        };
        handle_mcp_args(args, out);
        return true;
    }

    // /help - show available commands
    if (cmd == "/help") {
        std::string help =
            "Available commands:\n"
            "  /clear              - Clear session context\n"
            "  /provider [name]    - List or switch providers\n"
            "  /model [name]       - List or switch models\n"
            "  /config [key=value] - View or change config\n"
            "  /sched              - Scheduler management\n"
            "  /tools              - Tool management\n"
            "  /mcp                - MCP server management\n"
            "  /help               - Show this help\n";
        callback(CallbackEvent::CONTENT, help, "", "");
        return true;
    }

    // Command not recognized
    return false;
}

// ============================================================================
// Output formatting (markdown table alignment)
// LaTeX → Unicode conversion is now handled in Backend::filter()
// ============================================================================

namespace {

// Get display width of a UTF-8 string (accounting for wide chars)
size_t display_width(const std::string& s) {
    size_t width = 0;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        if ((c & 0x80) == 0) {
            // ASCII
            width++;
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte UTF-8
            width++;
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte UTF-8 (includes CJK which are typically double-width)
            // For simplicity, assume single width; could check ranges for CJK
            width++;
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte UTF-8
            width++;
            i += 4;
        } else {
            width++;
            i++;
        }
    }
    return width;
}

// Trim whitespace from both ends
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t");
    return s.substr(start, end - start + 1);
}

// Format markdown tables with aligned columns
std::string format_tables(const std::string& text) {
    std::istringstream iss(text);
    std::ostringstream oss;
    std::string line;
    std::vector<std::string> table_lines;
    bool in_table = false;

    auto flush_table = [&]() {
        if (table_lines.empty()) return;

        // Parse table into cells
        std::vector<std::vector<std::string>> rows;
        size_t max_cols = 0;

        for (const auto& tl : table_lines) {
            std::vector<std::string> cells;
            std::istringstream row_stream(tl);
            std::string cell;

            // Skip leading |
            if (!tl.empty() && tl[0] == '|') {
                row_stream.get();
            }

            while (std::getline(row_stream, cell, '|')) {
                cells.push_back(trim(cell));
            }

            // Remove empty last cell from trailing |
            if (!cells.empty() && cells.back().empty()) {
                cells.pop_back();
            }

            if (!cells.empty()) {
                rows.push_back(cells);
                max_cols = std::max(max_cols, cells.size());
            }
        }

        if (rows.empty()) {
            for (const auto& tl : table_lines) oss << tl << "\n";
            table_lines.clear();
            return;
        }

        // Calculate column widths
        std::vector<size_t> col_widths(max_cols, 0);
        for (const auto& row : rows) {
            for (size_t i = 0; i < row.size(); i++) {
                col_widths[i] = std::max(col_widths[i], display_width(row[i]));
            }
        }

        // Output formatted table
        for (size_t r = 0; r < rows.size(); r++) {
            const auto& row = rows[r];
            oss << "|";
            for (size_t c = 0; c < max_cols; c++) {
                std::string cell = (c < row.size()) ? row[c] : "";
                size_t cell_width = display_width(cell);
                size_t padding = col_widths[c] - cell_width;

                // Check if this is separator row (all dashes)
                bool is_sep = !cell.empty() && cell.find_first_not_of("-:") == std::string::npos;

                if (is_sep) {
                    oss << " " << std::string(col_widths[c], '-') << " |";
                } else {
                    oss << " " << cell << std::string(padding, ' ') << " |";
                }
            }
            oss << "\n";
        }

        table_lines.clear();
    };

    while (std::getline(iss, line)) {
        // Check if line looks like a table row (starts with | or contains | ... |)
        bool is_table_line = false;
        std::string trimmed = trim(line);
        if (!trimmed.empty()) {
            if (trimmed[0] == '|') {
                is_table_line = true;
            } else if (trimmed.find('|') != std::string::npos &&
                       trimmed.find('|') != trimmed.rfind('|')) {
                // Has at least 2 pipes
                is_table_line = true;
            }
        }

        if (is_table_line) {
            in_table = true;
            table_lines.push_back(line);
        } else {
            if (in_table) {
                flush_table();
                in_table = false;
            }
            oss << line << "\n";
        }
    }

    // Flush any remaining table
    if (in_table) {
        flush_table();
    }

    std::string result = oss.str();
    // Remove trailing newline if original didn't have one
    if (!text.empty() && text.back() != '\n' && !result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    return result;
}

} // anonymous namespace

std::string Frontend::format_output(const std::string& text) {
    // Format markdown tables (LaTeX is handled in Backend::filter())
    return format_tables(text);
}
