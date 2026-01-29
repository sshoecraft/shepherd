
#include "cli.h"
#include "shepherd.h"
#include <cerrno>
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
#include <csignal>
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

void CLI::init(bool no_mcp_flag, bool no_tools_flag) {
    // Store flags for later use (e.g., fallback to local tools)
    no_mcp = no_mcp_flag;
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

                // Add tool result to session and generate next response
                {
                    auto lock = backend->acquire_lock();
                    add_message_to_session(Message::TOOL_RESPONSE, result.content, tool_name, tool_call_id);
                    generate_response();
                }
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
            case CallbackEvent::USER_PROMPT: {
                // Format with "> " prefix and newline (like TUI does)
                std::string formatted;
                std::istringstream stream(content);
                std::string line;
                while (std::getline(stream, line)) {
                    if (!formatted.empty()) {
                        formatted += "\n";
                    }
                    formatted += "> " + line;
                }
                formatted += "\n";
                write_colored(formatted, type);
                break;
            }
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
    init_tools(no_mcp, no_tools);

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

    // Connect to provider
    if (!connect_provider(provider_to_use->name)) {
        callback(CallbackEvent::SYSTEM, "Failed to connect to provider '" + provider_to_use->name + "'\n", "", "");
        return 1;
    }

    // If server_tools mode, fetch tools from server or fall back to local
    if (config->server_tools && !no_tools) {
        Provider* p = get_provider(current_provider);
        if (p && !p->base_url.empty()) {
            init_remote_tools(p->base_url, p->api_key);
        } else {
            callback(CallbackEvent::SYSTEM, "Warning: --server-tools requires an API provider with base_url, falling back to local tools\n", "", "");
            init_tools(no_mcp, no_tools, true);  // force_local = true
        }
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
        scheduler.set_fire_callback([this](const std::string& prompt) {
            add_scheduled_prompt(prompt);
        });
        scheduler.start();
        cli_debug(1, "Scheduler initialized with " + std::to_string(scheduler.list().size()) + " schedules");
    }

    // Start input reader thread (handles replxx/stdin in background)
    start_input_thread();
    cli_debug(1, "Input thread started");

    // Handle warmup if configured
    if (config->warmup && !config->warmup_message.empty()) {
        cli_debug(1, "Running warmup message...");
        auto lock = backend->acquire_lock();
        add_message_to_session(Message::USER, config->warmup_message);
        generate_response();
    }

    // Handle initial prompt from --prompt / -e (single query mode)
    if (!config->initial_prompt.empty()) {
        cli_debug(1, "Processing initial prompt from --prompt");
        callback(CallbackEvent::USER_PROMPT, config->initial_prompt, "", "");
        auto lock = backend->acquire_lock();
        add_message_to_session(Message::USER, config->initial_prompt);
        generate_response();
        // Exit after single query when using --prompt
        stop_input_thread();
        return 0;
    }

    // Main loop - wait for input from queue with timeout for scheduler polling
    while (true) {
        // Poll scheduler (prompts go to queue via add_input callback)
        if (!g_disable_scheduler) {
            scheduler.poll();
        }

        // Wait for input with 100ms timeout (allows scheduler polling)
        auto queued = wait_for_input(100);

        // Check for EOF
        if (eof_received) {
            cli_debug(1, "EOF received, exiting");
            break;
        }

        // Skip empty input but resume input thread
        if (queued.text.empty()) {
            resume_input();
            continue;
        }

        std::string user_input = queued.text;

        // Display user prompt via callback (unified with TUI and CLI Server)
        // All input sources now go through this path - replxx display is cleared first
        callback(CallbackEvent::USER_PROMPT, user_input, "", "");

        // Handle exit commands
        if (user_input == "exit" || user_input == "quit") {
            cli_debug(1, "User requested exit");
            break;
        }

        // Handle slash commands
        if (!user_input.empty() && user_input[0] == '/') {
            if (Frontend::handle_slash_commands(user_input, tools)) {
                resume_input();
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

        // Pause input thread during generation
        pause_input();

        // Enter raw mode for escape key detection during generation
        generation_cancelled = false;
        g_generation_cancelled = false;
        enter_generation_mode();
        {
            auto lock = backend->acquire_lock();
            add_message_to_session(Message::USER, user_input);
            generate_response();
        }
        exit_generation_mode();

        // Resume input thread
        resume_input();

        cli_debug(1, "tokens: " + std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

        // Show token count to stderr (only for GPU backends - API backends have their own display)
        if (backend->is_gpu) {
            fprintf(stderr, "tokens: %d/%zu\n", session.total_tokens, backend->context_size);
        }
    }

    // Cleanup
    cli_debug(1, "Stopping input thread...");
    stop_input_thread();

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

    // Format assistant content (LaTeX → Unicode, table alignment)
    std::string formatted = (type == CallbackEvent::CONTENT) ? format_output(text) : text;

    // Get indent from centralized config (frontend.h)
    int indent_spaces = get_indent_for_event(type);
    std::string indent(indent_spaces, ' ');

    const char* color = colors_enabled ? ansi_color(type) : "";
    const char* reset = colors_enabled ? ANSI_RESET : "";

    // Output with indentation at line starts
    // Process line-by-line to handle indentation while preserving UTF-8
    size_t pos = 0;
    while (pos < formatted.length()) {
        // Add indentation at line starts
        if (at_line_start && formatted[pos] != '\n' && !indent.empty()) {
            printf("%s", indent.c_str());
            at_line_start = false;
        }

        // Find end of current line (or end of text)
        size_t line_end = formatted.find('\n', pos);
        if (line_end == std::string::npos) {
            // No newline - print rest of text
            printf("%s%s%s", color, formatted.substr(pos).c_str(), reset);
            at_line_start = false;
            break;
        } else {
            // Print up to and including the newline
            printf("%s%s%s", color, formatted.substr(pos, line_end - pos + 1).c_str(), reset);
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
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_queue.push_back({input, needs_echo});
    }
    input_cv.notify_one();
}

void CLI::add_scheduled_prompt(const std::string& prompt) {
    std::lock_guard<std::mutex> lock(scheduled_mutex);
    scheduled_queue.push_back(prompt);
    cli_debug(1, "Scheduled prompt queued: " + prompt.substr(0, 50));
}

CLI::QueuedInput CLI::wait_for_input(int timeout_ms) {
    std::unique_lock<std::mutex> lock(input_mutex);
    if (timeout_ms < 0) {
        input_cv.wait(lock, [this] { return !input_queue.empty() || eof_received; });
    } else {
        input_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return !input_queue.empty() || eof_received; });
    }
    if (input_queue.empty()) return {"", false};
    QueuedInput item = input_queue.front();
    input_queue.pop_front();
    return item;
}

// Signal handler for interrupting the input thread
void CLI::input_signal_handler(int sig) {
    (void)sig;  // Just interrupt the blocking read, nothing else needed
}

void CLI::start_input_thread() {
    // Set up signal handler for thread interruption (no restart so reads get interrupted)
    struct sigaction sa;
    sa.sa_handler = input_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // Don't restart syscalls
    sigaction(SIGUSR1, &sa, nullptr);

    input_running = true;
    input_thread = std::thread(&CLI::input_reader_loop, this);
    input_thread_id = input_thread.native_handle();
}

void CLI::stop_input_thread() {
    input_running = false;
    ready_for_input = true;  // Unblock if waiting
    ready_cv.notify_all();
    if (input_thread_id != 0) {
        pthread_kill(input_thread_id, SIGUSR1);  // Interrupt blocking read
    }
    if (input_thread.joinable()) {
        input_thread.join();
    }
}

void CLI::pause_input() {
    ready_for_input = false;
}

void CLI::resume_input() {
    ready_for_input = true;
    ready_cv.notify_one();
}

void CLI::input_reader_loop() {
    while (input_running) {
        // Wait until ready for input (not during generation)
        {
            std::unique_lock<std::mutex> lock(ready_mutex);
            ready_cv.wait(lock, [this] { return ready_for_input.load() || !input_running; });
        }
        if (!input_running) break;

        std::string line;

        if (interactive_mode && replxx) {
            // Check for pending scheduled prompts before waiting for user input
            std::string scheduled_prompt;
            {
                std::lock_guard<std::mutex> lock(scheduled_mutex);
                if (!scheduled_queue.empty()) {
                    scheduled_prompt = scheduled_queue.front();
                    scheduled_queue.pop_front();
                }
            }

            if (!scheduled_prompt.empty()) {
                // Inject scheduled prompt via replxx emulated key presses
                cli_debug(1, "Injecting scheduled prompt: " + scheduled_prompt.substr(0, 50));
                for (char c : scheduled_prompt) {
                    replxx_emulate_key_press(replxx, static_cast<unsigned int>(c));
                }
                replxx_emulate_key_press(replxx, REPLXX_KEY_ENTER);
            }

            const char* result = replxx_input(replxx, "> ");
            if (result == nullptr) {
                if (!input_running) break;  // Signaled to stop
                if (errno == EINTR) continue;  // Signal interrupted, try again
                // Real EOF
                eof_received = true;
                input_cv.notify_one();
                break;
            }
            line = result;
            // Clear the line replxx displayed so callback can redisplay uniformly
            // \033[A = move up one line, \033[2K = clear line, \r = carriage return
            std::cout << "\033[A\033[2K\r" << std::flush;
            // Add to history in the input thread
            replxx_history_add(replxx, result);
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                history.push_back(line);
            }
        } else {
            // Non-interactive: check for scheduled prompts first
            {
                std::lock_guard<std::mutex> lock(scheduled_mutex);
                if (!scheduled_queue.empty()) {
                    line = scheduled_queue.front();
                    scheduled_queue.pop_front();
                    add_input(line, true);
                    ready_for_input = false;
                    continue;
                }
            }
            // Read from stdin
            if (!std::getline(std::cin, line)) {
                if (!input_running) break;
                if (errno == EINTR) continue;  // Signal interrupted, try again
                eof_received = true;
                input_cv.notify_one();
                break;
            }
        }

        add_input(line, true);  // Echo needed - we cleared replxx's display for uniform callback handling

        // Pause ourselves - main loop will resume when ready for more input
        ready_for_input = false;
    }
}

std::string CLI::read_input(const std::string& prompt) {
    // Legacy function - now just wraps wait_for_input for backwards compatibility
    (void)prompt;  // Prompt is now shown by the input reader thread
    auto queued = wait_for_input(-1);  // Block indefinitely
    return queued.text;
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
