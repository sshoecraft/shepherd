#include "tui_screen.h"
#include "terminal_io.h"

#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

// Global TUI screen instance
TUIScreen* g_tui_screen = nullptr;

// ANSI color codes
static const char* to_ansi_color(Color color) {
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

// Get terminal size
static void get_terminal_size(int& rows, int& cols) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        rows = w.ws_row;
        cols = w.ws_col;
    } else {
        rows = 24;
        cols = 80;
    }
}

TUIScreen::TUIScreen()
    : current_output_color(Color::DEFAULT)
    , current_output_type(LineType::ASSISTANT) {
}

TUIScreen::~TUIScreen() {
    shutdown();
}

bool TUIScreen::init() {
    int rows, cols;
    get_terminal_size(rows, cols);

    // Reserve bottom 4 lines for input area
    int scroll_bottom = rows - 4;

    // Set scroll region (top to scroll_bottom)
    printf("\033[1;%dr", scroll_bottom);

    // Enable bracketed paste
    printf("\033[?2004h");

    // Move cursor to scroll region
    printf("\033[%d;1H", scroll_bottom);

    // Clear input area
    for (int i = scroll_bottom + 1; i <= rows; i++) {
        printf("\033[%d;1H\033[2K", i);
    }

    // Draw initial input area
    redraw_input_area();

    fflush(stdout);
    return true;
}

void TUIScreen::shutdown() {
    int rows, cols;
    get_terminal_size(rows, cols);

    // Reset scroll region to full screen
    printf("\033[r");

    // Disable bracketed paste
    printf("\033[?2004l");

    // Show cursor, reset style
    printf("\033[?25h");
    printf("\033[2 q");

    // Move to bottom
    printf("\033[%d;1H\n", rows);

    fflush(stdout);
}

void TUIScreen::redraw_input_area() {
    int rows, cols;
    get_terminal_size(rows, cols);

    int input_start = rows - 3;  // 4 lines: queued(1), separator(1), input(1), status(1)

    // Save cursor position
    printf("\033[s");

    // Draw queued inputs (line 1 of input area)
    printf("\033[%d;1H\033[2K", input_start);
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        int shown = 0;
        for (int i = std::max(0, (int)queued_inputs.size() - 2); i < (int)queued_inputs.size() && shown < 2; i++, shown++) {
            auto& q = queued_inputs[i];
            if (q.is_processing) {
                printf("\033[36m%s\033[0m  ", q.text.c_str());  // Cyan
            } else {
                printf("\033[90m%s\033[0m  ", q.text.c_str());  // Gray
            }
        }
    }

    // Draw separator (line 2)
    printf("\033[%d;1H\033[2K", input_start + 1);
    for (int i = 0; i < cols; i++) printf("─");

    // Draw input prompt (line 3)
    printf("\033[%d;1H\033[2K", input_start + 2);
    printf("\033[32m>\033[0m %s", input_content.c_str());

    // Draw status line (line 4)
    printf("\033[%d;1H\033[2K", input_start + 3);
    printf("\033[90m%s", status_left.c_str());
    if (!status_right.empty()) {
        int right_pos = cols - (int)status_right.length();
        if (right_pos > (int)status_left.length() + 1) {
            printf("\033[%d;%dH%s", input_start + 3, right_pos, status_right.c_str());
        }
    }
    printf("\033[0m");

    // Restore cursor to input line, after prompt
    printf("\033[%d;%dH", input_start + 2, 3 + (int)input_content.length());

    fflush(stdout);
}

void TUIScreen::write_output(const std::string& text_str, LineType type) {
    write_output(text_str.c_str(), text_str.length(), type);
}

void TUIScreen::write_output(const char* text_data, size_t len, LineType type) {
    int rows, cols;
    get_terminal_size(rows, cols);
    int scroll_bottom = rows - 4;

    // Determine color from type
    Color color;
    switch (type) {
        case LineType::USER:        color = Color::GREEN; break;
        case LineType::TOOL_CALL:   color = Color::YELLOW; break;
        case LineType::TOOL_RESULT: color = Color::CYAN; break;
        case LineType::SYSTEM:      color = Color::RED; break;
        case LineType::ASSISTANT:
        default:                    color = Color::DEFAULT; break;
    }

    // Should we indent?
    bool indent = (type == LineType::ASSISTANT);

    // Save cursor
    printf("\033[s");

    // Move to bottom of scroll region
    printf("\033[%d;1H", scroll_bottom);

    // Set color
    if (color != Color::DEFAULT) {
        printf("%s", to_ansi_color(color));
    }

    // Print with indentation, handling newlines
    for (size_t i = 0; i < len; i++) {
        char c = text_data[i];
        if (c == '\n') {
            printf("\n");
            // After newline, we're at start of line - add indent if needed
            if (indent && i + 1 < len) {
                printf("  ");
            }
        } else if (c == '\r') {
            // Ignore
        } else {
            // First char of a line after indent decision
            if (i == 0 && indent) {
                printf("  ");
            }
            putchar(c);
        }
    }

    // Reset color
    if (color != Color::DEFAULT) {
        printf("\033[0m");
    }

    // Restore cursor (back to input area)
    printf("\033[u");

    fflush(stdout);
}

void TUIScreen::set_input_callback(InputCallback callback) {
    on_input = callback;
}

std::string TUIScreen::get_input_content() const {
    return input_content;
}

void TUIScreen::clear_input() {
    input_content.clear();
    redraw_input_area();
}

void TUIScreen::set_status(const std::string& left, const std::string& right) {
    status_left = left;
    status_right = right;
    redraw_input_area();
}

bool TUIScreen::run_once() {
    // In this mode, we need to handle input ourselves
    // For now, return true to keep running
    // Input handling will need to be integrated
    return !quit_requested;
}

void TUIScreen::request_refresh() {
    redraw_input_area();
}

void TUIScreen::set_input_content(const std::string& content) {
    input_content = content;
    redraw_input_area();
}

void TUIScreen::position_cursor_in_input(int /* col */) {
    redraw_input_area();
}

void TUIScreen::flush() {
    fflush(stdout);
}

void TUIScreen::show_queued_input(const std::string& input) {
    std::lock_guard<std::mutex> lock(output_mutex);
    queued_inputs.push_back({"> " + input, false});
    redraw_input_area();
}

void TUIScreen::mark_input_processing() {
    std::lock_guard<std::mutex> lock(output_mutex);
    for (auto& q : queued_inputs) {
        if (!q.is_processing) {
            q.is_processing = true;
            break;
        }
    }
    redraw_input_area();
}

void TUIScreen::clear_queued_input_display() {
    std::lock_guard<std::mutex> lock(output_mutex);
    queued_inputs.clear();
    redraw_input_area();
}
