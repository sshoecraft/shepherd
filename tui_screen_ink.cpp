#include "tui_screen.h"
#include "terminal_io.h"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/loop.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/terminal.hpp>

#include <algorithm>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

// Namespace alias to avoid Color conflict
namespace fx = ftxui;

// Global TUI screen instance
TUIScreen* g_tui_screen = nullptr;

// Convert our Color enum to ANSI escape code
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
    : screen(nullptr)
    , loop(nullptr)
    , current_output_color(Color::DEFAULT) {
}

TUIScreen::~TUIScreen() {
    shutdown();
}

bool TUIScreen::init() {
    // Enable bracketed paste mode
    printf("\033[?2004h");

    // Save cursor position, we'll manage the bottom area
    // Move cursor to bottom and clear the input area
    int rows, cols;
    get_terminal_size(rows, cols);

    // Reserve bottom 4 lines for: queued (up to 2), input box (1), status (1)
    input_area_height = 4;

    // Initial draw of input area
    redraw_input_area();

    fflush(stdout);
    return true;
}

void TUIScreen::shutdown() {
    // Disable bracketed paste mode
    printf("\033[?2004l");
    // Show cursor
    printf("\033[?25h");
    // Reset cursor style to steady block
    printf("\033[2 q");
    // Move to bottom and clear input area
    int rows, cols;
    get_terminal_size(rows, cols);
    printf("\033[%d;1H", rows - input_area_height + 1);
    for (int i = 0; i < input_area_height; i++) {
        printf("\033[2K\n");  // Clear line
    }
    printf("\033[%d;1H", rows - input_area_height + 1);  // Position cursor
    fflush(stdout);
}

void TUIScreen::redraw_input_area() {
    int rows, cols;
    get_terminal_size(rows, cols);

    // Save cursor position
    printf("\033[s");

    // Move to input area start (bottom - input_area_height)
    int input_start_row = rows - input_area_height + 1;

    // Draw queued inputs (up to 2 lines)
    printf("\033[%d;1H\033[2K", input_start_row);  // Line 1: queued
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        int queue_start = std::max(0, static_cast<int>(queued_inputs.size()) - 2);
        for (int i = queue_start; i < static_cast<int>(queued_inputs.size()); i++) {
            auto& q = queued_inputs[i];
            if (q.is_processing) {
                printf("\033[36m%s\033[0m", q.text.c_str());  // Cyan
            } else {
                printf("\033[90m%s\033[0m", q.text.c_str());  // Gray
            }
            if (i < static_cast<int>(queued_inputs.size()) - 1) printf("  ");
        }
    }

    // Draw separator line
    printf("\033[%d;1H\033[2K", input_start_row + 1);
    for (int i = 0; i < cols; i++) printf("─");

    // Draw input prompt
    printf("\033[%d;1H\033[2K", input_start_row + 2);
    printf("\033[32m>\033[0m %s", input_content.c_str());

    // Draw status line
    printf("\033[%d;1H\033[2K", input_start_row + 3);
    printf("\033[90m%s", status_left.c_str());
    // Right-align status_right
    int right_pos = cols - static_cast<int>(status_right.length());
    if (right_pos > static_cast<int>(status_left.length())) {
        printf("\033[%d;%dH%s", input_start_row + 3, right_pos, status_right.c_str());
    }
    printf("\033[0m");

    // Position cursor after input prompt
    printf("\033[%d;%dH", input_start_row + 2, 3 + static_cast<int>(input_content.length()));

    fflush(stdout);
}

std::shared_ptr<fx::ComponentBase> TUIScreen::build_ui() {
    // Not used in this implementation
    return nullptr;
}

void TUIScreen::on_input_submit() {
    if (input_content.empty()) return;

    std::string submitted = input_content;
    input_content.clear();

    // Add to history
    history.push_back(submitted);
    history_index = -1;
    saved_input.clear();

    // Call the callback if set
    if (on_input) {
        on_input(submitted);
    }

    redraw_input_area();
}

void TUIScreen::write_output(const std::string& text_str, Color color) {
    write_output(text_str.c_str(), text_str.length(), color);
}

void TUIScreen::write_output(const char* text_data, size_t len, Color color) {
    int rows, cols;
    get_terminal_size(rows, cols);

    // Save cursor, move to output area (above input area), print, restore
    printf("\033[s");  // Save cursor
    printf("\033[%d;1H", rows - input_area_height);  // Move to line above input area

    // Scroll up if needed (insert line at bottom of output area)
    printf("\033[S");  // Scroll up one line
    printf("\033[%d;1H", rows - input_area_height);  // Move back

    // Print with color
    if (color != Color::DEFAULT) {
        printf("%s", to_ansi_color(color));
    }

    // Print character by character, handling newlines
    for (size_t i = 0; i < len; i++) {
        char c = text_data[i];
        if (c == '\n') {
            printf("\033[S");  // Scroll up
            printf("\033[%d;1H", rows - input_area_height);  // Move to output line
        } else if (c != '\r') {
            putchar(c);
        }
    }

    if (color != Color::DEFAULT) {
        printf("\033[0m");
    }

    // Redraw input area (it may have scrolled)
    redraw_input_area();
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
    // Read input character by character in non-blocking mode
    // This is a simplified version - real implementation would need proper terminal handling

    // For now, just return true to keep running
    // Input will be handled separately
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
