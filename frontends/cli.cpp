
#include "cli.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "tools/utf8_sanitizer.h"
#include "tools/filesystem_tools.h"
#include "tools/command_tools.h"
#include "tools/json_tools.h"
#include "tools/http_tools.h"
#include "tools/memory_tools.h"
#include "tools/mcp_resource_tools.h"
#include "tools/core_tools.h"
#include "message.h"
#include "config.h"
#include "provider.h"
#include "scheduler.h"
#include "mcp/mcp.h"
#include "backends/api.h"
#include "backends/factory.h"
#include "rag.h"
#include "ansi.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <chrono>

#include "replxx.h"

// External globals from main.cpp
extern int g_debug_level;
extern bool g_show_thinking;
extern std::unique_ptr<Config> config;

// Debug output helper
static void cli_debug(int level, const std::string& text) {
    if (g_debug_level >= level) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time));
        fprintf(stderr, "[%s.%03d] %s\n", timestamp, (int)ms.count(), text.c_str());
    }
}

// Helper: Extract tool call from Response (handles both structured and text-based)
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    if (!resp.tool_calls.empty()) {
        return resp.tool_calls[0];
    }
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

// CLI Implementation
CLI::CLI() : Frontend() {
}

CLI::~CLI() {
    save_history();
    if (replxx) {
        replxx_end(replxx);
        replxx = nullptr;
    }
}

void CLI::init(bool no_mcp, bool no_tools) {
    // Detect interactive mode
    interactive_mode = isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
    colors_enabled = interactive_mode;

    // Check TERM for color support
    const char* term = getenv("TERM");
    if (term && (strcmp(term, "dumb") == 0 || strcmp(term, "") == 0)) {
        colors_enabled = false;
    }

    // Check NO_COLOR environment variable
    if (getenv("NO_COLOR")) {
        colors_enabled = false;
    }

    // Initialize replxx for line editing
    if (interactive_mode) {
        replxx = replxx_init();
        replxx_set_max_history_size(replxx, 1000);

        // Set history file path
        const char* home = getenv("HOME");
        if (home) {
            history_file = std::string(home) + "/.shepherd_history";
            load_history();
        }
    }

    // Set up the event callback for streaming output
    callback = [this](CallbackEvent type, const std::string& content,
                            const std::string& tool_name, const std::string& tool_call_id) -> bool {
        switch (type) {
            case CallbackEvent::CONTENT:
            case CallbackEvent::THINKING:
            case CallbackEvent::CODEBLOCK:
                write_colored(content, type);
                break;
            case CallbackEvent::TOOL_CALL:
                // Queue tool for execution
                pending_tool_calls.push({tool_name, content, tool_call_id});
                break;
            case CallbackEvent::ERROR:
                show_error(content);
                break;
            case CallbackEvent::SYSTEM:
                write_colored(content, type);
                break;
            case CallbackEvent::USER_PROMPT:
                // CLI doesn't need to echo - replxx handles display
                break;
            case CallbackEvent::STOP:
                // Only add newline if no pending tool calls (they handle their own formatting)
                if (pending_tool_calls.empty()) {
                    write_raw("\n");
                }
                break;
        }
        return !generation_cancelled;  // Return false to cancel if escape pressed
    };

    // Use common tool initialization from Frontend base class
    Frontend::init_tools(session, tools, no_mcp, no_tools);

    cli_debug(1, "CLI initialized (interactive: " + std::string(interactive_mode ? "yes" : "no") +
              ", colors: " + std::string(colors_enabled ? "yes" : "no") + ")");
}

int CLI::run() {
    // Populate session.tools from our tools instance
    tools.populate_session_tools(session);

    // Copy tool names to backend for output filtering
    for (const auto& tool : session.tools) {
        backend->valid_tool_names.insert(tool.name);
    }

    cli_debug(1, "Starting CLI mode (interactive: " + std::string(interactive_mode ? "yes" : "no") + ")");

    // Initialize scheduler (unless disabled)
    Scheduler scheduler;
    if (!g_disable_scheduler) {
        scheduler.load();
        scheduler.start();
        cli_debug(1, "Scheduler initialized with " + std::to_string(scheduler.list().size()) + " schedules");
    }

    // Handle warmup if configured
    if (config->warmup && !config->warmup_message.empty()) {
        cli_debug(1, "Running warmup message...");
        session.add_message(Message::USER, config->warmup_message);
    }

    // State for tool loop
    int tool_loop_iteration = 0;
    const int max_consecutive_identical_calls = 10;
    std::vector<std::string> recent_tool_calls;

    // Main synchronous loop
    while (true) {
        // Poll scheduler
        if (!g_disable_scheduler) {
            scheduler.poll();
        }

        // Read input (blocking)
        std::string user_input = read_input("> ");

        // Check for EOF
        if (eof_received) {
            cli_debug(1, "EOF received, exiting");
            break;
        }

        // Skip empty input
        if (user_input.empty()) {
            continue;
        }

        // Handle exit commands
        if (user_input == "exit" || user_input == "quit") {
            cli_debug(1, "User requested exit");
            break;
        }

        // Handle slash commands
        if (!user_input.empty() && user_input[0] == '/') {
            if (Frontend::handle_slash_commands(user_input, tools)) {
                continue;
            }
        }

        cli_debug(1, "User input: " + user_input);

        // Reset state
        generation_cancelled = false;
        tool_loop_iteration = 0;
        recent_tool_calls.clear();

        // Sanitize user input
        user_input = utf8_sanitizer::strip_control_characters(user_input);

        // Truncate if needed
        double scale = calculate_truncation_scale(backend->context_size);
        int available = backend->context_size - session.system_message_tokens;
        int max_user_input_tokens = available * scale;
        int input_tokens = backend->count_message_tokens(Message::USER, user_input, "", "");

        if (input_tokens > max_user_input_tokens) {
            cli_debug(1, "User input too large, truncating");
            std::string truncation_notice = "\n\n[INPUT TRUNCATED: Too large for context window]";
            while (input_tokens >= max_user_input_tokens && user_input.length() > 100) {
                size_t new_len = user_input.length() * 0.9;
                user_input = user_input.substr(0, new_len);
                input_tokens = backend->count_message_tokens(Message::USER, user_input + truncation_notice, "", "");
            }
            user_input += truncation_notice;
        }

        // Send user message and generate response (blocking, streams via callback)
        cli_debug(1, "Submitting user message");
        session.add_message(Message::USER, user_input);

        // Tool loop - process any tool calls
        while (!pending_tool_calls.empty() || tool_loop_iteration == 0) {
            tool_loop_iteration++;
            cli_debug(1, "Tool loop iteration: " + std::to_string(tool_loop_iteration));

            // Check for tool calls from pending queue
            if (pending_tool_calls.empty()) {
                // No more tool calls - done
                break;
            }

            auto tc = pending_tool_calls.front();
            pending_tool_calls.pop();

            // Parse the tool call
            auto tool_call_opt = ToolParser::parse_tool_call(
                tc.args,
                backend->get_tool_call_markers()
            );

            if (!tool_call_opt) {
                // Try wrapping in tags
                tool_call_opt = ToolParser::parse_tool_call(
                    "<tool_call>" + tc.args + "</tool_call>",
                    backend->get_tool_call_markers()
                );
            }

            if (!tool_call_opt) {
                cli_debug(1, "Failed to parse tool call: " + tc.name);
                // Show error to user and send back to model
                std::string error_msg = "Failed to parse tool call arguments for " + tc.name;
                callback(CallbackEvent::ERROR, error_msg + "\n", "tool_parse", "");
                // Send error back to model as tool result
                backend->add_message(session, Message::Role::TOOL_RESPONSE,
                    "Error: " + error_msg, tc.name, tc.tool_call_id);
                continue;
            }

            auto tool_call = tool_call_opt.value();
            tool_call.name = tc.name;
            tool_call.tool_call_id = tc.tool_call_id;

            std::string tool_name = tool_call.name;
            std::string tool_call_id = tool_call.tool_call_id;

            cli_debug(1, "Tool call detected: " + tool_name);

            // Build call signature for loop detection
            std::string call_signature = tool_name + "(";
            bool first_sig_param = true;
            for (const auto& param : tool_call.parameters) {
                if (!first_sig_param) call_signature += ", ";
                first_sig_param = false;
                call_signature += param.first + "=";
                try {
                    if (param.second.type() == typeid(std::string)) {
                        call_signature += "\"" + std::any_cast<std::string>(param.second) + "\"";
                    } else if (param.second.type() == typeid(int)) {
                        call_signature += std::to_string(std::any_cast<int>(param.second));
                    } else if (param.second.type() == typeid(double)) {
                        call_signature += std::to_string(std::any_cast<double>(param.second));
                    } else if (param.second.type() == typeid(bool)) {
                        call_signature += std::any_cast<bool>(param.second) ? "true" : "false";
                    } else {
                        call_signature += "?";
                    }
                } catch (...) {
                    call_signature += "?";
                }
            }
            call_signature += ")";

            // Check for infinite loop
            int consecutive_count = 1;
            for (int i = (int)recent_tool_calls.size() - 1;
                 i >= 0 && i >= (int)recent_tool_calls.size() - 10; i--) {
                if (recent_tool_calls[i] == call_signature) {
                    consecutive_count++;
                    if (consecutive_count >= max_consecutive_identical_calls) {
                        show_error("Detected infinite loop: " + call_signature +
                                   " called " + std::to_string(consecutive_count) +
                                   " times consecutively. Stopping.");
                        // Clear pending tool calls
                        while (!pending_tool_calls.empty()) pending_tool_calls.pop();
                        break;
                    }
                } else {
                    break;
                }
            }
            recent_tool_calls.push_back(call_signature);

            // Build params string for display
            std::string params_str;
            bool first_param = true;
            for (const auto& param : tool_call.parameters) {
                if (!first_param) params_str += ", ";
                first_param = false;
                std::string value_str;
                try {
                    if (param.second.type() == typeid(std::string)) {
                        value_str = std::any_cast<std::string>(param.second);
                    } else if (param.second.type() == typeid(int)) {
                        value_str = std::to_string(std::any_cast<int>(param.second));
                    } else if (param.second.type() == typeid(double)) {
                        value_str = std::to_string(std::any_cast<double>(param.second));
                    } else if (param.second.type() == typeid(bool)) {
                        value_str = std::any_cast<bool>(param.second) ? "true" : "false";
                    } else {
                        value_str = "<unknown>";
                    }
                } catch (...) {
                    value_str = "<error>";
                }
                if (value_str.length() > 50) {
                    value_str = value_str.substr(0, 47) + "...";
                }
                params_str += param.first + "=" + value_str;
            }

            show_tool_call(tool_name, params_str);

            // Execute tool (handles truncation)
            cli_debug(1, "Executing tool: " + tool_name);
            ToolResult tool_result = execute_tool(tools, tool_name, tool_call.parameters, tool_call_id);

            // Display summary (or fallback to error/content snippet)
            std::string display_summary = tool_result.summary;
            if (display_summary.empty()) {
                if (!tool_result.success && !tool_result.error.empty()) {
                    display_summary = tool_result.error;
                } else if (!tool_result.content.empty()) {
                    // Fallback: first line of content
                    size_t newline = tool_result.content.find('\n');
                    display_summary = (newline != std::string::npos)
                        ? tool_result.content.substr(0, newline)
                        : tool_result.content;
                    if (display_summary.length() > 80) {
                        display_summary = display_summary.substr(0, 77) + "...";
                    }
                } else {
                    display_summary = tool_result.success ? "Done" : "Failed";
                }
            }
            show_tool_result(display_summary, tool_result.success);

            // Submit tool result to session (triggers model response)
            cli_debug(1, "Submitting tool result");
            session.add_message(Message::TOOL_RESPONSE, tool_result.content, tool_name, tool_call_id);
        }

        cli_debug(1, "tokens: " + std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

        // Always show token count to stderr
        fprintf(stderr, "tokens: %d/%zu\n", session.total_tokens, backend->context_size);
    }

    // Cleanup
    if (!g_disable_scheduler) {
        scheduler.stop();
    }

    return 0;
}

// ANSI color codes for callback event types
const char* CLI::ansi_color(CallbackEvent event) {
    switch (event) {
        case CallbackEvent::SYSTEM: return ANSI_FG_RED;
        case CallbackEvent::USER_PROMPT: return ANSI_FG_GREEN;
        case CallbackEvent::TOOL_CALL: return ANSI_FG_YELLOW;
        case CallbackEvent::THINKING: return ANSI_FG_BRIGHT_BLACK;
        case CallbackEvent::ERROR: return ANSI_FG_RED;
        case CallbackEvent::CODEBLOCK: return ANSI_FG_CYAN;
        case CallbackEvent::CONTENT: return "";
        case CallbackEvent::STOP: return "";
    }
    return "";
}

void CLI::write_colored(const std::string& text, CallbackEvent type) {
    if (text.empty()) return;

    // Determine indent based on type
    // No indent: USER_PROMPT, SYSTEM, ERROR
    // 4 spaces: CODEBLOCK
    // 2 spaces: everything else (CONTENT, THINKING, TOOL_CALL, etc)
    const char* indent = "";
    if (type == CallbackEvent::CODEBLOCK) {
        indent = "    ";
    } else if (type != CallbackEvent::USER_PROMPT &&
               type != CallbackEvent::SYSTEM &&
               type != CallbackEvent::ERROR) {
        indent = "  ";
    }

    const char* color = colors_enabled ? ansi_color(type) : "";
    const char* reset = colors_enabled ? ANSI_RESET : "";

    // Output with indentation at line starts
    for (char c : text) {
        if (at_line_start && c != '\n' && indent[0] != '\0') {
            printf("%s", indent);
            at_line_start = false;
        }
        if (c != '\n') at_line_start = false;
        printf("%s%c%s", color, c, reset);
        if (c == '\n') {
            at_line_start = true;
        }
    }
    fflush(stdout);
}

void CLI::write_raw(const std::string& text) {
    printf("%s", text.c_str());
    fflush(stdout);
    if (!text.empty()) {
        at_line_start = (text.back() == '\n');
    }
}

void CLI::show_tool_call(const std::string& name, const std::string& params) {
    std::string msg = name + "(" + params + ")\n";
    write_colored(msg, CallbackEvent::TOOL_CALL);
}

void CLI::show_tool_result(const std::string& summary, bool success) {
    // Display one-line summary with 4-space indent
    // Green for success, red for error
    std::string msg = "    " + summary + "\n";
    if (colors_enabled) {
        const char* color = success ? ANSI_FG_GREEN : ANSI_FG_RED;
        write_raw(std::string(color) + msg + ANSI_RESET);
    } else {
        write_raw(msg);
    }
    at_line_start = true;
}

void CLI::show_error(const std::string& error) {
    std::string msg = "Error: " + error + "\n";
    write_colored(msg, CallbackEvent::SYSTEM);
}

void CLI::show_cancelled() {
    if (interactive_mode) {
        std::string msg = "\n[Cancelled]\n";
        write_colored(msg, CallbackEvent::SYSTEM);
    }
}

void CLI::send_message(const std::string& message) {
    write_raw(message);
}

void CLI::add_input(const std::string& input, bool needs_echo) {
    piped_input_queue.push_back(input);
    (void)needs_echo;  // Not used in synchronous mode
}

std::string CLI::read_input(const std::string& prompt) {
    // Check piped input queue first
    if (!piped_input_queue.empty()) {
        std::string input = piped_input_queue.front();
        piped_input_queue.pop_front();
        // Echo piped input
        write_colored(prompt + input + "\n", CallbackEvent::USER_PROMPT);
        return input;
    }

    // Check for piped EOF
    if (piped_eof && piped_input_queue.empty()) {
        eof_received = true;
        return "";
    }

    if (interactive_mode && replxx) {
        const char* line = replxx_input(replxx, prompt.c_str());
        if (line == nullptr) {
            eof_received = true;
            return "";
        }
        std::string input = line;

        // Add to history if non-empty
        if (!input.empty()) {
            replxx_history_add(replxx, input.c_str());
            history.push_back(input);
        }

        return input;
    } else {
        // Non-interactive mode
        std::string input;
        if (!std::getline(std::cin, input)) {
            eof_received = true;
            return "";
        }
        return input;
    }
}

void CLI::load_history() {
    if (history_file.empty() || !replxx) return;

    std::ifstream file(history_file);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                replxx_history_add(replxx, line.c_str());
                history.push_back(line);
            }
        }
    }
}

void CLI::save_history() {
    if (history_file.empty() || !replxx) return;
    replxx_history_save(replxx, history_file.c_str());
}
