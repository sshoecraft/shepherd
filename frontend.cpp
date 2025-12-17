#include "frontend.h"
#include "shepherd.h"
#include "cli.h"
#include "api_server.h"
#include "cli_server.h"
#include "tui.h"
#include "tools/tools.h"
#include "tools/utf8_sanitizer.h"
#include "rag.h"
#include "mcp/mcp.h"
#include "mcp/mcp_config.h"
#include "scheduler.h"
#include <algorithm>
#include <sstream>

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
        frontend = std::make_unique<APIServer>(host, port);
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

void Frontend::init_tools(Session& session, Tools& tools, bool no_mcp, bool no_tools) {
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

    dout(1) << "Initializing tools system..." << std::endl;

    // Register all native tools
    register_filesystem_tools(tools);
    register_command_tools(tools);
    register_json_tools(tools);
    register_http_tools(tools);
    register_memory_tools(tools);
    register_mcp_resource_tools(tools);
    register_core_tools(tools);

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

ToolResult Frontend::execute_tool(Tools& tools,
                                   const std::string& tool_name,
                                   const std::map<std::string, std::any>& parameters,
                                   const std::string& tool_call_id) {
    // Execute the tool
    ToolResult tool_result = tools.execute(tool_name, parameters);

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

    if (config->truncate_limit > 0 && config->truncate_limit < max_tool_result_tokens) {
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
