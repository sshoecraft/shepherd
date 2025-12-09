#include "terminal_io.h"
#include "tui_screen.h"
#include "logger.h"
#include "vendor/replxx/include/replxx.h"
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <unistd.h>
#include <sys/select.h>

// External globals
extern int g_debug_level;
extern bool g_server_mode;

// Global instance
TerminalIO tio;

TerminalIO::TerminalIO()
    : interactive_mode(false)
    , colors_enabled(false)
    , tui_mode(false)
    , show_thinking(false)
    , last_char_was_newline(true)  // Start as if on new line
    , json_brace_depth(0)
    , tui_screen(nullptr)
    , replxx(nullptr)
    , term_raw_mode(false)
    , filter_state(NORMAL)
    , in_tool_call(false)
    , in_thinking(false)
    , in_code_block(false)
    , suppress_output(false) {
}

TerminalIO::~TerminalIO() {
    if (tui_screen) {
        delete tui_screen;
        tui_screen = nullptr;
    }
    if (replxx) {
        replxx_end(replxx);
        replxx = nullptr;
    }
    if (term_raw_mode) {
        restore_terminal();
    }
}

bool TerminalIO::init(int color_override, int tui_override) {
    // Detect if stdin is a TTY - check env var first (preserved across MPI re-exec)
    const char* interactive_env = getenv("SHEPHERD_INTERACTIVE");

    if (g_server_mode) {
        // Server modes are never interactive - no user input from stdin
        interactive_mode = false;
        LOG_DEBUG("TerminalIO: server mode, forcing interactive_mode=false");
    } else if (interactive_env) {
        interactive_mode = (std::string(interactive_env) == "1");
        LOG_DEBUG("TerminalIO using SHEPHERD_INTERACTIVE=" + std::string(interactive_env) +
                  " -> interactive_mode=" + std::to_string(interactive_mode));
    } else {
        interactive_mode = isatty(STDIN_FILENO);
        LOG_DEBUG("TerminalIO no env var, using isatty=" + std::to_string(interactive_mode));
    }

    // Determine color support
    if (color_override == 1) {
        // Force colors on
        colors_enabled = true;
    } else if (color_override == 0) {
        // Force colors off
        colors_enabled = false;
    } else {
        // Auto-detect: check environment variables
        colors_enabled = interactive_mode; // Default to interactive mode

        // Check NO_COLOR
        if (getenv("NO_COLOR") != nullptr) {
            colors_enabled = false;
        }

        // Check CLICOLOR
        const char* clicolor = getenv("CLICOLOR");
        if (clicolor && strcmp(clicolor, "0") == 0) {
            colors_enabled = false;
        }

        // Check TERM
        const char* term = getenv("TERM");
        if (term) {
            if (strcmp(term, "dumb") == 0 ||
                strcmp(term, "cons25") == 0 ||
                strcmp(term, "emacs") == 0) {
                colors_enabled = false;
            }
        }
    }

    // Determine TUI mode
    if (tui_override == 1) {
        tui_mode = true;
    } else if (tui_override == 0) {
        tui_mode = false;
    } else {
        // Auto: TUI mode only in interactive mode with color support
        tui_mode = interactive_mode && colors_enabled;
    }

    // Initialize TUI screen if in TUI mode
    if (tui_mode) {
        tui_screen = new TUIScreen();
        if (!tui_screen->init()) {
            LOG_ERROR("Failed to initialize TUI screen, falling back to standard mode");
            delete tui_screen;
            tui_screen = nullptr;
            tui_mode = false;
        } else {
            LOG_DEBUG("TUI mode enabled");
            g_tui_screen = tui_screen;

            // Set up input callback to queue input
            tui_screen->set_input_callback([this](const std::string& input) {
                add_input(input);
            });
        }
    }

    // Initialize replxx for interactive non-TUI mode
    // This single instance is shared - InputReader uses tio.replxx for input
    if (interactive_mode && !tui_mode) {
        replxx = replxx_init();
        if (!replxx) {
            std::cerr << "ERROR: Failed to initialize terminal (replxx)\n";
            return false;
        }

        // Configure replxx
        replxx_set_max_history_size(replxx, 1000);
        replxx_set_max_hint_rows(replxx, 3);
        replxx_set_indent_multiline(replxx, 1);
        replxx_enable_bracketed_paste(replxx);

        if (!colors_enabled) {
            replxx_set_no_color(replxx, 1);
        }
    }

    return true;
}

std::string TerminalIO::read(const char* prompt) {
    auto [text, needs_echo] = read_with_echo_flag(prompt);
    return text;
}

std::pair<std::string, bool> TerminalIO::read_with_echo_flag(const char* prompt) {
    // Wait for input to be available in the queue
    // InputReader thread populates the queue, we just consume
    std::unique_lock<std::mutex> lock(queue_mutex);

    // Wait for input
    queue_cv.wait(lock, [this] { return !input_queue.empty(); });

    QueuedInput item = input_queue.front();
    input_queue.pop_front();

    return {item.text, item.needs_echo};
}

void TerminalIO::write_raw(const char* text, size_t len, Color color) {
    // In TUI mode, route through TUIScreen
    if (tui_screen) {
        // Map Color to LineType
        TUIScreen::LineType type;
        switch (color) {
            case Color::GREEN:  type = TUIScreen::LineType::USER; break;
            case Color::YELLOW: type = TUIScreen::LineType::TOOL_CALL; break;
            case Color::CYAN:   type = TUIScreen::LineType::TOOL_RESULT; break;
            case Color::RED:
            case Color::GRAY:   type = TUIScreen::LineType::SYSTEM; break;
            default:            type = TUIScreen::LineType::ASSISTANT; break;
        }
        tui_screen->write_output(text, len, type);
        tui_screen->flush();
        return;
    }

    // Use stdout for all non-TUI output
    // Note: replxx_write is not thread-safe with replxx_input running in another thread
    if (colors_enabled && color != Color::DEFAULT) {
        const char* ansi_color = get_ansi_color(color);
        std::cout << ansi_color;
        std::cout.write(text, len);
        std::cout << "\033[0m";
    } else {
        std::cout.write(text, len);
    }
    std::cout.flush();
}

void TerminalIO::write(const char* text, size_t len, Color color) {
    // Process each character through the filtering state machine
    for (size_t i = 0; i < len; i++) {
        char c = text[i];

        // Track backticks for code block detection (``` toggles code block mode)
        if (c == '`') {
            backtick_buffer += c;
            if (backtick_buffer.length() >= 3) {
                // Found ```, toggle code block state
                in_code_block = !in_code_block;
                backtick_buffer.clear();
            }
        } else {
            backtick_buffer.clear();
        }

        switch (filter_state) {
            case NORMAL:
                if (c == '<' && !in_code_block) {
                    // Detect potential XML tag start (but not inside code blocks)
                    tag_buffer = "<";
                    filter_state = DETECTING_TAG;
                } else if (c == '{' && !in_code_block) {
                    // Detect potential JSON tool call start
                    tag_buffer = "{";
                    filter_state = DETECTING_TAG;
                } else {
                    if (!suppress_output) {
                        write_raw(&c, 1, color);
                    }
                }
                break;

            case DETECTING_TAG: {
                tag_buffer += c;

                // Check for closing tag start
                if (tag_buffer.length() == 2 && tag_buffer == "</") {
                    filter_state = CHECKING_CLOSE;
                    break;
                }

                // Check if we've matched a tool call start marker
                // Also check if we're one char past a marker with a valid boundary
                std::string matched_marker;
                bool is_tool_call = false;

                if (matches_any(tag_buffer, markers.tool_call_start, &matched_marker)) {
                    // Exact match - wait for boundary char to confirm
                    // Don't transition yet
                } else {
                    // Check if we're one char past a marker (boundary check)
                    for (const auto& marker : markers.tool_call_start) {
                        if (tag_buffer.length() == marker.length() + 1 &&
                            tag_buffer.substr(0, marker.length()) == marker) {
                            // We have marker + one more char
                            char boundary = tag_buffer.back();
                            if (boundary == '>' || boundary == ' ' || boundary == '/' ||
                                boundary == '\n' || boundary == '\t' || boundary == '\r' ||
                                boundary == ':') {  // : for JSON like {"name":
                                matched_marker = marker;
                                is_tool_call = true;
                                break;
                            }
                            // If boundary is a letter/digit, this is NOT a tool call (e.g., <reading vs <read)
                            // Fall through to could_match check
                        }
                    }
                }

                if (is_tool_call) {
                    in_tool_call = true;
                    current_tag = matched_marker;
                    filter_state = IN_TOOL_CALL;
                    suppress_output = true;  // Always suppress tool calls
                    buffered_tool_call = tag_buffer;  // Include opening tag
                    // For JSON tool calls, initialize brace depth (we've already seen the opening {)
                    if (matched_marker.length() > 0 && matched_marker[0] == '{') {
                        json_brace_depth = 1;
                    }
                    tag_buffer.clear();
                    break;
                }

                // Check if we've matched a thinking start marker
                if (matches_any(tag_buffer, markers.thinking_start, &matched_marker)) {
                    in_thinking = true;
                    current_tag = matched_marker;
                    filter_state = IN_THINKING;
                    // Set suppress_output based on show_thinking flag
                    if (!show_thinking) {
                        suppress_output = true;
                    } else {
                        // show_thinking=true: output the opening tag
                        write_raw(tag_buffer.c_str(), tag_buffer.length(), color);
                    }
                    tag_buffer.clear();
                    break;
                }

                // Check if tag could still match something (partial match)
                bool could_match = could_match_any(tag_buffer, markers.tool_call_start) ||
                                   could_match_any(tag_buffer, markers.thinking_start);

                if (!could_match) {
                    // Not a special tag, output buffered content
                    write_raw(tag_buffer.c_str(), tag_buffer.length(), color);
                    tag_buffer.clear();
                    filter_state = NORMAL;
                }
                break;
            }

            case IN_TOOL_CALL:
                if (c == '<') {
                    tag_buffer = "<";
                    filter_state = CHECKING_CLOSE;
                } else {
                    buffered_tool_call += c;
                    // For JSON tool calls, check if we've reached the end
                    // JSON tool calls end with } followed by newline or end of content
                    if (current_tag.length() > 0 && current_tag[0] == '{') {
                        // Track brace depth for JSON
                        if (c == '{') json_brace_depth++;
                        if (c == '}') json_brace_depth--;

                        // When we reach balanced braces, the JSON is complete
                        if (json_brace_depth == 0) {
                            in_tool_call = false;
                            suppress_output = false;
                            filter_state = NORMAL;
                        }
                    }
                }
                break;

            case IN_THINKING:
                if (c == '<') {
                    tag_buffer = "<";
                    filter_state = CHECKING_CLOSE;
                } else {
                    if (!suppress_output) {
                        write_raw(&c, 1, color);
                    } else {
                        buffered_thinking += c;
                    }
                }
                break;

            case CHECKING_CLOSE: {
                bool could_close;  // Declare at top of block before any gotos
                tag_buffer += c;

                // Check if closing tag matches expected end marker
                if (in_tool_call) {
                    for (const auto& end_marker : markers.tool_call_end) {
                        if (tag_buffer == end_marker) {
                            buffered_tool_call += tag_buffer;
                            in_tool_call = false;
                            suppress_output = false;  // Clear suppress when exiting tool call
                            filter_state = NORMAL;
                            tag_buffer.clear();
                            goto next_char;
                        }
                    }
                }

                if (in_thinking) {
                    for (const auto& end_marker : markers.thinking_end) {
                        if (tag_buffer == end_marker) {
                            if (!suppress_output) {
                                // show_thinking=true: output the closing tag
                                write_raw(tag_buffer.c_str(), tag_buffer.length(), color);
                            }
                            in_thinking = false;
                            suppress_output = false;  // Always clear suppress when exiting thinking
                            filter_state = NORMAL;
                            tag_buffer.clear();
                            goto next_char;
                        }
                    }
                }

                // Check for partial match
                could_close = false;
                if (in_tool_call) {
                    could_close = could_match_any(tag_buffer, markers.tool_call_end);
                } else if (in_thinking) {
                    could_close = could_match_any(tag_buffer, markers.thinking_end);
                }

                if (!could_close) {
                    // False alarm, not a closing tag
                    if (in_tool_call) {
                        buffered_tool_call += tag_buffer;
                        filter_state = IN_TOOL_CALL;
                    } else if (in_thinking) {
                        if (!suppress_output) {
                            write_raw(tag_buffer.c_str(), tag_buffer.length(), color);
                        } else {
                            buffered_thinking += tag_buffer;
                        }
                        filter_state = IN_THINKING;
                    }
                    tag_buffer.clear();
                }
                next_char:
                break;
            }
        }
    }
}

void TerminalIO::add_input(const std::string& input, bool needs_echo) {
    // Skip blank lines
    if (input.empty() || is_blank(input)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        input_queue.push_back({input, needs_echo});
    }

    // If generating and in TUI mode, show input as queued (gray)
    if (is_generating && tui_screen) {
        tui_screen->show_queued_input(input);
    }

    // Notify waiting consumers
    queue_cv.notify_one();
}

void TerminalIO::notify_input() {
    queue_cv.notify_one();
}

bool TerminalIO::wait_for_input(int timeout_ms) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    if (!input_queue.empty()) {
        return true;
    }
    if (timeout_ms < 0) {
        // Wait indefinitely
        queue_cv.wait(lock, [this] { return !input_queue.empty(); });
        return true;
    }
    return queue_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this] { return !input_queue.empty(); });
}

bool TerminalIO::has_pending_input() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return !input_queue.empty();
}

size_t TerminalIO::queue_size() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return input_queue.size();
}

void TerminalIO::clear_input_queue() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    input_queue.clear();
}

std::string TerminalIO::get_input_line(const char* prompt) {
    while (true) {
        std::string line;

        if (interactive_mode) {
            // Format prompt with color
            std::string colored_prompt;
            if (colors_enabled) {
                colored_prompt = std::string("\033[32m") + prompt + " \033[0m";
            } else {
                colored_prompt = std::string(prompt) + " ";
            }

            const char* input = replxx_input(replxx, colored_prompt.c_str());
            if (input == nullptr) {
                // EOF (Ctrl+D) or Ctrl+C
                // Exit gracefully
                printf("\n");
                exit(0);
            }

            line = input;

            // Add to history if non-empty
            if (!line.empty() && !is_blank(line)) {
                replxx_history_add(replxx, input);
            }
        } else {
            // Piped mode - no prompt
            if (!std::getline(std::cin, line)) {
                // EOF
                return "";
            }
        }

        // Skip blank lines
        if (is_blank(line)) {
            continue;
        }

        return line;
    }
}

bool TerminalIO::is_blank(const std::string& str) const {
    for (char c : str) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

const char* TerminalIO::get_ansi_color(Color color) const {
    switch (color) {
        case Color::RED:     return "\033[31m";
        case Color::YELLOW:  return "\033[33m";
        case Color::GREEN:   return "\033[32m";
        case Color::CYAN:    return "\033[36m";
        case Color::GRAY:    return "\033[90m";
        case Color::DEFAULT:
        default:             return "\033[0m";
    }
}

// Output filtering helpers
bool TerminalIO::matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched) const {
    for (const auto& marker : markers) {
        if (buffer == marker) {
            if (matched) *matched = marker;
            return true;
        }
    }
    return false;
}

bool TerminalIO::could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const {
    for (const auto& marker : markers) {
        if (marker.length() >= buffer.length() &&
            marker.substr(0, buffer.length()) == buffer) {
            return true;
        }
    }
    return false;
}

void TerminalIO::begin_response() {
    // Reset all filtering state for a new response
    buffered_tool_call.clear();
    buffered_thinking.clear();
    filter_state = NORMAL;
    in_tool_call = false;
    in_thinking = false;
    in_code_block = false;
    suppress_output = false;
    tag_buffer.clear();
    current_tag.clear();
    backtick_buffer.clear();
    last_char_was_newline = true;  // Start responses as if on new line
}

void TerminalIO::end_response() {
    // Flush any incomplete tags that weren't closed properly
    // If we're still in a tool call or thinking block, consume it silently
    if (in_tool_call) {
        // Tool call wasn't closed - it's spurious, consume it
        buffered_tool_call.clear();
    }

    if (in_thinking && !show_thinking) {
        // Thinking wasn't closed - consume it
        buffered_thinking.clear();
    }

    // If we have buffered tag data that never matched, output it
    if (!tag_buffer.empty() && !suppress_output) {
        write_raw(tag_buffer.c_str(), tag_buffer.length(), Color::DEFAULT);
    }

    // Reset state for next response
    filter_state = NORMAL;
    in_tool_call = false;
    in_thinking = false;
    suppress_output = false;
    tag_buffer.clear();
    current_tag.clear();
}

void TerminalIO::reset() {
    // Deprecated - use begin_response() instead
    begin_response();
}

// Terminal control methods (unused for now)
void TerminalIO::set_raw_mode() {
    if (term_raw_mode) return;

    struct termios raw;
    tcgetattr(STDIN_FILENO, &original_term);
    raw = original_term;

    // Disable canonical mode and echo
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;   // Non-blocking read
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    term_raw_mode = true;
}

void TerminalIO::restore_terminal() {
    if (!term_raw_mode) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
    term_raw_mode = false;
}

bool TerminalIO::check_escape_pressed() {
    // In TUI mode, check FTXUI's escape flag
    if (tui_screen) {
        if (tui_screen->check_escape_pressed()) {
            // Clear queued input on cancel
            clear_input_queue();
            return true;
        }
        return false;
    }

    // Non-TUI mode: check raw terminal input
    if (!term_raw_mode) return false;

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    struct timeval tv = {0, 0};  // No wait

    if (select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv) > 0) {
        char c;
        if (::read(STDIN_FILENO, &c, 1) == 1) {
            if (c == 27) {  // ESC key
                return true;
            }
        }
    }

    return false;
}

// TUI-specific methods
void TerminalIO::set_status(const std::string& left, const std::string& right) {
    if (tui_screen) {
        tui_screen->set_status(left, right);
    }
}

void TerminalIO::echo_user_input(const std::string& input) {
    // Display input in output window with > prefix on first line only
    // Continuation lines get indented with spaces
    std::string formatted;
    std::istringstream stream(input);
    std::string line;
    bool first = true;
    while (std::getline(stream, line)) {
        if (!first) formatted += "\n";
        if (first) {
            formatted += "> " + line;
            first = false;
        } else {
            formatted += "  " + line;  // 2-space indent for continuation
        }
    }
    formatted += "\n";
    write(formatted.c_str(), formatted.length(), Color::GREEN);
}
