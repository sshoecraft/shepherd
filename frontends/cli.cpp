
#include "cli.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/utf8_sanitizer.h"
#include "tools/filesystem_tools.h"
#include "tools/command_tools.h"
#include "tools/json_tools.h"
#include "tools/http_tools.h"
#include "tools/memory_tools.h"
#include "tools/mcp_resource_tools.h"
#include "tools/core_tools.h"
#include "tools/api_tools.h"
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
#include <sys/select.h>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <chrono>

#include "replxx.h"

// External globals from main.cpp
extern std::unique_ptr<Config> config;

#ifdef _DEBUG
extern int g_debug_level;
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
#else
static void cli_debug(int, const std::string&) {}
#endif

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

void CLI::init(bool no_mcp, bool no_tools_flag) {
    // Store tools flag
    no_tools = no_tools_flag;

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
        replxx_enable_bracketed_paste(replxx);  // Enable paste support

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
        // Already cancelled - just keep returning false
        if (generation_cancelled || g_generation_cancelled) {
            return false;
        }

        // Check for escape key to cancel generation
        if (check_escape_key()) {
            generation_cancelled = true;
            g_generation_cancelled = true;  // Set global so backend stops
            show_cancelled();
            return false;
        }

        switch (type) {
            case CallbackEvent::CONTENT:
            case CallbackEvent::THINKING:
            case CallbackEvent::CODEBLOCK:
                write_colored(content, type);
                break;
            case CallbackEvent::TOOL_CALL: {
                // TOOL_CALL fires after STOP - execute immediately
                std::string params_str;
                try {
                    auto params = nlohmann::json::parse(content);
                    bool first = true;
                    for (auto& [key, value] : params.items()) {
                        if (!first) params_str += ", ";
                        first = false;
                        std::string val = value.is_string() ? value.get<std::string>() : value.dump();
                        if (val.length() > 50) val = val.substr(0, 47) + "...";
                        params_str += key + "=" + val;
                    }
                } catch (...) {
                    params_str = content;
                }

                show_tool_call(tool_name, params_str);

                // Execute tool
                ToolResult result = execute_tool(tools, tool_name, content, tool_call_id);

                // Show result
                std::string summary = result.summary.empty() ?
                    (result.success ? result.content.substr(0, 100) : result.error) : result.summary;
                show_tool_result(summary, result.success);

                // Add tool result to session - triggers next generation cycle
                session.add_message(Message::TOOL_RESPONSE, result.content, tool_name, tool_call_id);
                break;
            }
            case CallbackEvent::TOOL_RESULT:
                // Tool results are displayed via show_tool_result(), not callback
                break;
            case CallbackEvent::TOOL_DISP:
                // Display-only tool call (from remote server)
                show_tool_call(tool_name, content);
                break;
            case CallbackEvent::RESULT_DISP:
                // Display-only tool result (from remote server)
                show_tool_result(content, tool_name != "error");
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
            case CallbackEvent::STATS:
                // Only show stats if enabled via --stats flag
                if (config->stats) {
                    write_colored(content, type);
                }
                break;
            case CallbackEvent::STOP:
                write_raw("\n");
                break;
        }
        return !generation_cancelled;  // Return false to cancel if escape pressed
    };

    // Use common tool initialization from Frontend base class
    Frontend::init_tools(session, tools, no_mcp, no_tools);

    cli_debug(1, "CLI initialized (interactive: " + std::string(interactive_mode ? "yes" : "no") +
              ", colors: " + std::string(colors_enabled ? "yes" : "no") + ")");
}

int CLI::run(Provider* cmdline_provider) {
    cli_debug(1, "Starting CLI mode (interactive: " + std::string(interactive_mode ? "yes" : "no") + ")");

    // Determine which provider to connect
    Provider* provider_to_use = nullptr;
    if (cmdline_provider) {
        provider_to_use = cmdline_provider;
    } else if (!providers.empty()) {
        provider_to_use = &providers[0];  // Highest priority
    }

    if (!provider_to_use) {
        callback(CallbackEvent::SYSTEM, "No providers configured. Use 'shepherd provider add' to configure.\n", "", "");
        return 1;
    }

    // Output loading message (no newline - we'll add completion indicator)
    callback(CallbackEvent::SYSTEM, "Loading Provider: " + provider_to_use->name, "", "");

    // Connect to provider
    if (!connect_provider(provider_to_use->name)) {
        callback(CallbackEvent::SYSTEM, " - FAILED\n", "", "");
        return 1;
    }
    // Register other providers as tools (unless tools disabled)
    if (!no_tools) {
        register_provider_tools(tools, current_provider);
    }

    // Populate session.tools from our tools instance
    tools.populate_session_tools(session);

    // Copy tool names to backend for output filtering
    for (const auto& tool : session.tools) {
        backend->valid_tool_names.insert(tool.name);
    }

    // Configure session based on backend capabilities
    session.desired_completion_tokens = calculate_desired_completion_tokens(
        backend->context_size, backend->max_output_tokens);
    session.auto_evict = (backend->context_size > 0 && !backend->is_gpu);

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
        // Tool calls are handled in the callback - they fire AFTER STOP, execute
        // immediately, and trigger recursive generation via session.add_message(TOOL_RESULT)
        cli_debug(1, "Submitting user message");

        // Enter raw mode for escape key detection during generation
        generation_cancelled = false;
        g_generation_cancelled = false;
        enter_generation_mode();
        session.add_message(Message::USER, user_input);
        exit_generation_mode();

        cli_debug(1, "tokens: " + std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

        // Show token count to stderr (only for GPU backends - API backends have their own display)
        if (backend->is_gpu) {
            fprintf(stderr, "tokens: %d/%zu\n", session.total_tokens, backend->context_size);
        }
    }

    // Cleanup
    if (!g_disable_scheduler) {
        scheduler.stop();
    }

    return 0;
}

// ANSI color codes for FrontendColor (centralized in frontend.h)
static const char* ansi_from_color(FrontendColor color) {
    switch (color) {
        case FrontendColor::GREEN:   return ANSI_FG_GREEN;
        case FrontendColor::YELLOW:  return ANSI_FG_YELLOW;
        case FrontendColor::RED:     return ANSI_FG_RED;
        case FrontendColor::CYAN:    return ANSI_FG_CYAN;
        case FrontendColor::GRAY:    return ANSI_FG_BRIGHT_BLACK;
        case FrontendColor::DEFAULT: return "";
    }
    return "";
}

// ANSI color codes for callback event types
const char* CLI::ansi_color(CallbackEvent event) {
    return ansi_from_color(get_color_for_event(event));
}

void CLI::write_colored(const std::string& text, CallbackEvent type) {
    if (text.empty()) return;

    // Get indent from centralized config (frontend.h)
    int indent_spaces = get_indent_for_event(type);
    std::string indent(indent_spaces, ' ');

    const char* color = colors_enabled ? ansi_color(type) : "";
    const char* reset = colors_enabled ? ANSI_RESET : "";

    // Output with indentation at line starts
    // Process line-by-line to handle indentation while preserving UTF-8
    size_t pos = 0;
    while (pos < text.length()) {
        // Add indentation at line starts
        if (at_line_start && text[pos] != '\n' && !indent.empty()) {
            printf("%s", indent.c_str());
            at_line_start = false;
        }

        // Find end of current line (or end of text)
        size_t line_end = text.find('\n', pos);
        if (line_end == std::string::npos) {
            // No newline - print rest of text
            printf("%s%s%s", color, text.substr(pos).c_str(), reset);
            at_line_start = false;
            break;
        } else {
            // Print up to and including the newline
            printf("%s%s%s", color, text.substr(pos, line_end - pos + 1).c_str(), reset);
            at_line_start = true;
            pos = line_end + 1;
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
    // Ensure tool call starts on its own line (model text may not end with newline)
    if (!at_line_start) {
        std::cout << std::endl;
        at_line_start = true;
    }
    // Indentation handled by write_colored via get_indent_for_event()
    std::string msg = name + "(" + params + ")\n";
    write_colored(msg, CallbackEvent::TOOL_CALL);
}

void CLI::show_tool_result(const std::string& summary, bool success) {
    // Indentation handled by write_colored via get_indent_for_event()
    std::string msg = summary + "\n";
    write_colored(msg, success ? CallbackEvent::TOOL_RESULT : CallbackEvent::ERROR);
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

void CLI::enter_generation_mode() {
    if (!interactive_mode || term_raw_mode) return;

    // Save current terminal settings and switch to raw mode
    if (tcgetattr(STDIN_FILENO, &original_term) == 0) {
        struct termios raw = original_term;
        raw.c_lflag &= ~(ICANON | ECHO);  // Disable canonical mode and echo
        raw.c_cc[VMIN] = 0;   // Non-blocking
        raw.c_cc[VTIME] = 0;
        tcsetattr(STDIN_FILENO, TCSANOW, &raw);
        term_raw_mode = true;
    }
}

void CLI::exit_generation_mode() {
    if (!term_raw_mode) return;

    // Restore original terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
    term_raw_mode = false;
}

bool CLI::check_escape_key() {
    if (!interactive_mode || !term_raw_mode) return false;

    // Simple non-blocking read (terminal already in raw mode)
    char c;
    if (read(STDIN_FILENO, &c, 1) == 1) {
        if (c == 27 || c == 3) {  // ESC or Ctrl+C
            return true;
        }
    }
    return false;
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
