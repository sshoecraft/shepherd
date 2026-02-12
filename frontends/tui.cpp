#include "tui.h"
#include "shepherd.h"
#include "ansi.h"
#include <climits>
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
#include "config.h"
#include "provider.h"
#include "scheduler.h"
#include "generation_thread.h"
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <chrono>
#include <locale.h>
#include <wchar.h>

// UTF-8 display width helpers
static int utf8_display_width(const std::string& str) {
    // Convert UTF-8 to wide chars and sum their display widths
    std::wstring wstr(str.length(), L'\0');
    size_t wlen = mbstowcs(&wstr[0], str.c_str(), str.length());
    if (wlen == (size_t)-1) return str.length();  // Fallback to byte count
    wstr.resize(wlen);

    int width = 0;
    for (wchar_t wc : wstr) {
        int w = wcwidth(wc);
        if (w > 0) width += w;
        else if (w == 0) width += 0;  // Zero-width chars
        else width += 1;  // Unprintable, assume 1
    }
    return width;
}

// Substring by display width (returns valid UTF-8 up to max_width display columns)
static std::string utf8_substr_width(const std::string& str, int max_width, int* actual_width = nullptr) {
    if (max_width <= 0) {
        if (actual_width) *actual_width = 0;
        return "";
    }

    int width = 0;
    size_t byte_pos = 0;

    while (byte_pos < str.length() && width < max_width) {
        // Decode one UTF-8 character
        unsigned char c = str[byte_pos];
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        // Make sure we have enough bytes
        if (byte_pos + char_len > str.length()) break;

        // Convert this char to wchar and get width
        wchar_t wc;
        std::string one_char = str.substr(byte_pos, char_len);
        if (mbtowc(&wc, one_char.c_str(), char_len) > 0) {
            int char_width = wcwidth(wc);
            if (char_width < 0) char_width = 1;
            if (width + char_width > max_width) break;  // Would exceed
            width += char_width;
        } else {
            width += 1;  // Invalid, count as 1
        }
        byte_pos += char_len;
    }

    if (actual_width) *actual_width = width;
    return str.substr(0, byte_pos);
}

// External globals
extern std::unique_ptr<Config> config;
extern bool g_disable_scheduler;

#ifdef _DEBUG
extern int g_debug_level;
// Debug helper
static void tui_debug(int level, const std::string& text) {
    if (g_debug_level >= level) {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        char timestamp[32];
        std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", std::localtime(&time));
        FILE* log = fopen("/tmp/shepherd_tui.log", "a");
        if (log) {
            fprintf(log, "[%s.%03d] %s\n", timestamp, (int)ms.count(), text.c_str());
            fclose(log);
        }
    }
}
#else
static void tui_debug(int, const std::string&) {}
#endif

// Global TUI instance
TUI* g_tui = nullptr;

// Bracketed paste escape sequences
static const char* PASTE_START = "\033[200~";
static const char* PASTE_END = "\033[201~";

// Color pair indices
enum ColorPairs {
    PAIR_DEFAULT = 0,
    PAIR_RED = 1,
    PAIR_GREEN = 2,
    PAIR_YELLOW = 3,
    PAIR_CYAN = 4,
    PAIR_GRAY = 5,
};

TUI::TUI()
    : output_pad(nullptr)
    , pending_win(nullptr)
    , input_win(nullptr)
    , status_win(nullptr)
    , term_rows(24)
    , term_cols(80)
    , output_height(20)
    , pending_height(0)
    , input_height(3)
    , status_height(1)
    , input_cursor_pos(0)
    , history_index(-1)
    , in_paste_mode(false) {
}

TUI::~TUI() {
    shutdown();
}

// Frontend interface - initialize tools and ncurses
void TUI::init(bool no_mcp_flag, bool no_tools_flag, bool no_rag_flag, bool mem_tools_flag) {
    // Store flags for later use (e.g., fallback to local tools)
    no_mcp = no_mcp_flag;
    no_tools = no_tools_flag;
    no_rag = no_rag_flag;
    mem_tools = mem_tools_flag;

    // Set up the event callback for streaming output
    // This callback is called by the backend for all streaming events
    callback = [this](CallbackEvent type, const std::string& content,
                            const std::string& tool_name, const std::string& tool_call_id) -> bool {
        return output_callback(type, content, tool_name, tool_call_id);
    };

    init_tools(no_mcp, no_tools, false, no_rag, mem_tools);
    init_ncurses();
    tui_debug(1, "TUI initialized");
}

bool TUI::init_ncurses() {
    // Enable wide character (Unicode/emoji) support
    setlocale(LC_ALL, "");

    // Set ESCDELAY before initscr - controls how long ncurses waits
    // to distinguish ESC key from escape sequences (default is 1000ms!)
    set_escdelay(25);  // 25ms - fast ESC response
    // Set TERM environment to prevent ncurses from using alternate screen
    // We need to use a terminfo entry that doesn't have smcup/rmcup
    // Alternative: use filter() to disable full-screen mode

    // Initialize ncurses without alternate screen buffer
    // filter() tells ncurses we only want single-line mode, which disables smcup/rmcup
    // But that's too restrictive - instead we manually handle it after initscr
    initscr();

    // Immediately switch back to primary screen if ncurses switched to alternate
    // This sequence disables alternate screen buffer
    printf("\033[?1049l");
    fflush(stdout);

    // Check if colors are supported
    if (has_colors()) {
        start_color();
        use_default_colors();

        // Initialize color pairs
        init_pair(PAIR_RED, COLOR_RED, -1);
        init_pair(PAIR_GREEN, COLOR_GREEN, -1);
        init_pair(PAIR_YELLOW, COLOR_YELLOW, -1);
        init_pair(PAIR_CYAN, COLOR_CYAN, -1);
        init_pair(PAIR_GRAY, COLOR_WHITE, -1);  // No gray in basic ncurses
    }

    // Configure terminal
    cbreak();              // Disable line buffering
    noecho();              // Don't echo input
    keypad(stdscr, TRUE);  // Enable special keys
    curs_set(1);           // Show cursor

    // Disable terminal's "mouse wheel sends arrow keys" feature
    // Try multiple escape sequences for different terminals
    printf("\033[?1007l");   // DEC 1007 - VTE/GNOME terminals
    printf("\033[?1003l");   // Disable all mouse tracking
    printf("\033[?1006l");   // Disable SGR mouse mode
    printf("\033[?1015l");   // Disable urxvt mouse mode
    printf("\033[?1000l");   // Disable basic mouse tracking
    fflush(stdout);

    // Clear screen once at start
    clear();
    refresh();

    // Get terminal size
    getmaxyx(stdscr, term_rows, term_cols);

    // Calculate layout and create windows
    calculate_layout();
    create_windows();

    // Enable bracketed paste mode
    enable_bracketed_paste();

    // Initial draw
    draw_output();
    draw_status();
    draw_input_box();

    doupdate();
    wrefresh(input_win);  // Make sure input window has focus

    return true;
}

void TUI::shutdown() {
    disable_bracketed_paste();
    destroy_windows();
    endwin();

    // Move cursor to bottom and print newline for clean shell prompt
    printf("\033[%d;1H\n", term_rows);
    fflush(stdout);
}

void TUI::calculate_layout() {
    // Status line at bottom (1 line)
    status_height = 1;

    // Input area: 3 lines minimum (border + content + border), grows up to max_input_lines + 2
    int current_input_lines = count_input_lines();
    input_height = std::min(current_input_lines + 2, max_input_lines + 2);
    if (input_height < 3) input_height = 3;

    // Pending area: grows based on queued items (0 when empty, up to max_pending_lines)
    // No border - just raw lines
    int queued_count = static_cast<int>(queued_input_display.size());
    if (queued_count == 0) {
        pending_height = 0;
    } else {
        pending_height = std::min(queued_count, max_pending_lines);
    }

    // Output area gets the rest
    output_height = term_rows - pending_height - input_height - status_height;
    if (output_height < 3) output_height = 3;
}

void TUI::create_windows() {
    // Destroy non-pad windows (preserve the pad!)
    if (pending_win) { delwin(pending_win); pending_win = nullptr; }
    if (input_win) { delwin(input_win); input_win = nullptr; }
    if (status_win) { delwin(status_win); status_win = nullptr; }

    // Only create pad if it doesn't exist yet
    if (!output_pad) {
        output_pad = newpad(config->tui_history, term_cols);
        scrollok(output_pad, TRUE);
        leaveok(output_pad, TRUE);

        // Position cursor at bottom of visible area so output grows upward
        pad_row = output_height - 1;
        wmove(output_pad, pad_row, 0);
        pad_view_top = 0;

        // Initial display of pad
        prefresh(output_pad, pad_view_top, 0, 0, 0, output_height - 1, term_cols - 1);
    }

    // Pending window below output (only if there are pending items)
    if (pending_height > 0) {
        pending_win = newwin(pending_height, term_cols, output_height, 0);
        leaveok(pending_win, TRUE);
    }

    // Input window below pending
    int input_row = output_height + pending_height;
    input_win = newwin(input_height, term_cols, input_row, 0);
    keypad(input_win, TRUE);
    leaveok(input_win, FALSE);  // Keep cursor where we put it
    nodelay(input_win, TRUE);   // Non-blocking input

    // Status window at bottom
    status_win = newwin(status_height, term_cols, term_rows - status_height, 0);
    leaveok(status_win, TRUE);  // Don't care about cursor in status
}

void TUI::destroy_windows() {
    if (output_pad) { delwin(output_pad); output_pad = nullptr; }
    if (pending_win) { delwin(pending_win); pending_win = nullptr; }
    if (input_win) { delwin(input_win); input_win = nullptr; }
    if (status_win) { delwin(status_win); status_win = nullptr; }
}

// Map FrontendColor (from frontend.h) to ncurses color pair
static int ncurses_from_color(FrontendColor color) {
    switch (color) {
        case FrontendColor::GREEN:   return PAIR_GREEN;
        case FrontendColor::YELLOW:  return PAIR_YELLOW;
        case FrontendColor::RED:     return PAIR_RED;
        case FrontendColor::CYAN:    return PAIR_CYAN;
        case FrontendColor::GRAY:    return PAIR_GRAY;
        case FrontendColor::DEFAULT: return PAIR_DEFAULT;
    }
    return PAIR_DEFAULT;
}

// Helper to get ncurses color pair for callback event type
static int get_color_for_event_ncurses(CallbackEvent event) {
    return ncurses_from_color(get_color_for_event(event));
}

void TUI::draw_output() {
    // Calculate which portion of the pad to show
    // pad_row is where we're currently writing
    // pad_view_top is the top of the viewport (for scrollback)

    // By default, show the most recent content (follow the output)
    int view_top = pad_view_top;
    if (view_top < 0) view_top = 0;

    // Make sure we don't show past the current write position
    int max_top = std::max(0, pad_row - output_height + 1);
    if (view_top > max_top) view_top = max_top;

    // prefresh(pad, pad_min_row, pad_min_col, screen_min_row, screen_min_col, screen_max_row, screen_max_col)
    prefresh(output_pad, view_top, 0, 0, 0, output_height - 1, term_cols - 1);
}

void TUI::draw_pending() {
    if (!pending_win || pending_height == 0) return;

    std::lock_guard<std::mutex> lock(output_mutex);

    werase(pending_win);

    // Draw queued items (newest at bottom, scroll if > max_pending_lines)
    int queued_count = static_cast<int>(queued_input_display.size());
    int start_idx = std::max(0, queued_count - max_pending_lines);

    int row = 0;
    for (int i = start_idx; i < queued_count && row < pending_height; i++, row++) {
        const auto& item = queued_input_display[i];
        if (item.is_processing) {
            wattron(pending_win, COLOR_PAIR(PAIR_CYAN));
        } else {
            wattron(pending_win, A_DIM);  // Gray effect
        }
        mvwprintw(pending_win, row, 0, "%s", item.text.c_str());
        if (item.is_processing) {
            wattroff(pending_win, COLOR_PAIR(PAIR_CYAN));
        } else {
            wattroff(pending_win, A_DIM);
        }
    }

    wnoutrefresh(pending_win);
}

void TUI::draw_input_box() {
    // NOTE: Caller must hold input_mutex if thread safety needed
    // Internal calls from handle_key already hold the lock

    werase(input_win);

    int win_height, win_width;
    getmaxyx(input_win, win_height, win_width);

    // Draw border using box drawing characters
    // Top border
    mvwaddch(input_win, 0, 0, ACS_ULCORNER);
    for (int i = 1; i < win_width - 1; i++) {
        waddch(input_win, ACS_HLINE);
    }
    waddch(input_win, ACS_URCORNER);

    // Side borders and content
    for (int row = 1; row < win_height - 1; row++) {
        mvwaddch(input_win, row, 0, ACS_VLINE);
        mvwaddch(input_win, row, win_width - 1, ACS_VLINE);
    }

    // Bottom border
    mvwaddch(input_win, win_height - 1, 0, ACS_LLCORNER);
    for (int i = 1; i < win_width - 1; i++) {
        waddch(input_win, ACS_HLINE);
    }
    waddch(input_win, ACS_LRCORNER);

    // Draw prompt and content
    if (has_colors()) {
        wattron(input_win, COLOR_PAIR(PAIR_GREEN));
    }
    mvwprintw(input_win, 1, 2, "> ");
    if (has_colors()) {
        wattroff(input_win, COLOR_PAIR(PAIR_GREEN));
    }

    // Draw input content (handle multiline)
    int content_width = win_width - 6;  // Account for borders and "> "
    int content_start_col = 4;

    // Split input into lines for display
    std::vector<std::string> display_lines;
    std::string remaining = input_content;

    while (!remaining.empty()) {
        size_t newline_pos = remaining.find('\n');
        if (newline_pos != std::string::npos) {
            display_lines.push_back(remaining.substr(0, newline_pos));
            remaining = remaining.substr(newline_pos + 1);
        } else {
            // Wrap long lines (UTF-8 aware)
            while (utf8_display_width(remaining) > content_width) {
                std::string line = utf8_substr_width(remaining, content_width);
                display_lines.push_back(line);
                remaining = remaining.substr(line.length());
            }
            display_lines.push_back(remaining);
            remaining.clear();
        }
    }

    if (display_lines.empty()) {
        display_lines.push_back("");
    }

    // Show last N lines that fit
    int max_display_lines = win_height - 2;
    int start_line = std::max(0, static_cast<int>(display_lines.size()) - max_display_lines);

    for (int i = 0; i < max_display_lines && (start_line + i) < static_cast<int>(display_lines.size()); i++) {
        mvwprintw(input_win, 1 + i, content_start_col, "%s", display_lines[start_line + i].c_str());
    }

    // Position cursor based on input_cursor_pos
    // Calculate which display line and column the cursor is on
    int cursor_row = 1;  // First content row
    int cursor_col = content_start_col;

    int chars_counted = 0;
    int line_idx = 0;
    for (size_t i = 0; i < display_lines.size(); i++) {
        int line_len = static_cast<int>(display_lines[i].length());
        if (chars_counted + line_len >= input_cursor_pos) {
            // Cursor is on this line
            line_idx = static_cast<int>(i);
            cursor_col = content_start_col + (input_cursor_pos - chars_counted);
            break;
        }
        chars_counted += line_len + 1;  // +1 for newline
        line_idx = static_cast<int>(i) + 1;
        cursor_col = content_start_col;
    }

    // Adjust for scrolling
    int visible_line_idx = line_idx - start_line;
    if (visible_line_idx >= 0 && visible_line_idx < max_display_lines) {
        cursor_row = 1 + visible_line_idx;
    } else if (visible_line_idx < 0) {
        cursor_row = 1;
    } else {
        cursor_row = 1 + max_display_lines - 1;
    }

    // Make sure cursor is within bounds
    if (cursor_col >= win_width - 1) {
        cursor_col = win_width - 2;
    }

    wmove(input_win, cursor_row, cursor_col);
    wnoutrefresh(input_win);

    // Move physical cursor to input window
    // Need to use leaveok(FALSE) and position cursor after refresh
}

void TUI::draw_status() {
    werase(status_win);

    if (has_colors()) {
        wattron(status_win, A_DIM);
    }

    // Left side
    mvwprintw(status_win, 0, 0, "%s", status_left.c_str());

    // Right side
    int right_col = term_cols - static_cast<int>(status_right.length());
    if (right_col > static_cast<int>(status_left.length()) + 2) {
        mvwprintw(status_win, 0, right_col, "%s", status_right.c_str());
    }

    if (has_colors()) {
        wattroff(status_win, A_DIM);
    }

    wnoutrefresh(status_win);
}

bool TUI::run_once() {
    if (quit_requested) return false;

    // Check for terminal resize
    int new_rows, new_cols;
    getmaxyx(stdscr, new_rows, new_cols);
    if (new_rows != term_rows || new_cols != term_cols) {
        term_rows = new_rows;
        term_cols = new_cols;
        calculate_layout();
        create_windows();
        draw_output();
        draw_status();
        draw_input_box();
        doupdate();
        wrefresh(input_win);
    }

    // Refresh if needed (from output writes)
    if (refresh_needed) {
        refresh_needed = false;
        draw_output();
        draw_pending();
        draw_status();
        draw_input_box();
        doupdate();
    }

    // Handle input - use short timeout so we return quickly
    wtimeout(input_win, 5);  // 5ms timeout for responsive streaming
    int ch = wgetch(input_win);

    if (ch == ERR) {
        // No input, just return
        return !quit_requested;
    }

    // Check for bracketed paste sequences and handle ESC
    if (ch == 27) {  // ESC
        // Read ahead to check for escape sequence
        wtimeout(input_win, 10);  // Short timeout for escape sequences
        int next = wgetch(input_win);
        if (next == '[') {
            // Could be paste sequence, read more
            char seq[10] = {0};
            int seq_len = 0;
            while (seq_len < 9) {
                int c = wgetch(input_win);
                if (c == ERR) break;
                seq[seq_len++] = c;
                if (c == '~') break;
            }
            seq[seq_len] = '\0';

            if (strcmp(seq, "200~") == 0) {
                // Paste start
                in_paste_mode = true;
                paste_buffer.clear();
            } else if (strcmp(seq, "201~") == 0) {
                // Paste end
                in_paste_mode = false;
                input_content += paste_buffer;
                input_cursor_pos = input_content.length();
                paste_buffer.clear();

                // Recalculate layout if input grew
                int old_height = input_height;
                calculate_layout();
                if (input_height != old_height) {
                    create_windows();
                    draw_output();
                    draw_status();
                }
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            // else: unknown escape sequence, ignore
        } else {
            // Just ESC key (next == ERR or some other char)
            // Handle double-ESC for clear, single-ESC for cancel
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_escape_time);
            last_escape_time = now;

            if (elapsed.count() < 500) {
                // Double ESC - clear input and queued inputs
                tui_debug(1, "Double ESC - clearing input");
                std::lock_guard<std::mutex> lock(input_mutex);
                input_content.clear();
                input_cursor_pos = 0;
                history_index = -1;
                saved_input.clear();
                input_queue.clear();
                refresh_needed = true;
            } else {
                // Single ESC - cancel generation
                tui_debug(1, "Single ESC - cancelling");
                set_escape_pressed();
            }
        }
    } else if (in_paste_mode) {
        // Inline paste char handling
        if (ch >= 32 && ch < 127) {
            paste_buffer += static_cast<char>(ch);
        } else if (ch == '\n' || ch == '\r') {
            paste_buffer += '\n';
        }
    } else {
        handle_key(ch);
    }

    return !quit_requested;
}

void TUI::handle_key(int ch) {
    // Lock ncurses for any drawing this might do

    switch (ch) {
        case 4:  // Ctrl+D
            quit_requested = true;
            break;

        case '\n':
        case '\r':
        case KEY_ENTER:
            submit_input();
            break;

        case KEY_BACKSPACE:
        case 127:
        case 8:
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (input_cursor_pos > 0) {
                    input_content.erase(input_cursor_pos - 1, 1);
                    input_cursor_pos--;
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case KEY_DC:  // Delete key
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (input_cursor_pos < static_cast<int>(input_content.length())) {
                    input_content.erase(input_cursor_pos, 1);
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case KEY_LEFT:
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (input_cursor_pos > 0) {
                    input_cursor_pos--;
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case KEY_RIGHT:
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (input_cursor_pos < static_cast<int>(input_content.length())) {
                    input_cursor_pos++;
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case KEY_HOME:
        case 1:  // Ctrl+A
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                input_cursor_pos = 0;
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            break;

        case KEY_END:
        case 5:  // Ctrl+E
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                input_cursor_pos = static_cast<int>(input_content.length());
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            break;

        case KEY_UP:
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (!history.empty()) {
                    if (history_index == -1) {
                        // Save current input before browsing history
                        saved_input = input_content;
                        history_index = static_cast<int>(history.size()) - 1;
                    } else if (history_index > 0) {
                        history_index--;
                    }
                    input_content = history[history_index];
                    input_cursor_pos = static_cast<int>(input_content.length());
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case KEY_DOWN:
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                if (history_index >= 0) {
                    if (history_index < static_cast<int>(history.size()) - 1) {
                        history_index++;
                        input_content = history[history_index];
                    } else {
                        // Back to current input
                        history_index = -1;
                        input_content = saved_input;
                    }
                    input_cursor_pos = static_cast<int>(input_content.length());
                    draw_input_box();
                    doupdate();
                    wrefresh(input_win);
                }
            }
            break;

        case 21:  // Ctrl+U - clear line
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                input_content.clear();
                input_cursor_pos = 0;
                calculate_layout();
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            break;

        case 11:  // Ctrl+K - kill to end of line
            {
                std::lock_guard<std::mutex> lock(input_mutex);
                input_content.erase(input_cursor_pos);
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            break;

        case KEY_PPAGE:  // Page Up - scroll output back
            {
                pad_view_top -= output_height / 2;
                if (pad_view_top < 0) pad_view_top = 0;
                draw_output();
                doupdate();
            }
            break;

        case KEY_NPAGE:  // Page Down - scroll output forward
            {
                int max_top = std::max(0, pad_row - output_height + 1);
                pad_view_top += output_height / 2;
                if (pad_view_top > max_top) pad_view_top = max_top;
                draw_output();
                doupdate();
            }
            break;

        default:
            if (ch >= 32 && ch < 127) {  // Printable ASCII
                std::lock_guard<std::mutex> lock(input_mutex);
                input_content.insert(input_cursor_pos, 1, static_cast<char>(ch));
                input_cursor_pos++;

                // Recalculate layout if input grew to new line
                int old_height = input_height;
                calculate_layout();
                if (input_height != old_height) {
                    create_windows();
                    draw_output();
                    draw_status();
                }
                draw_input_box();
                doupdate();
                wrefresh(input_win);
            }
            break;
    }
}

void TUI::submit_input() {
    if (input_content.empty()) return;

    std::string submitted = input_content;

    // Add to history
    history.push_back(submitted);
    history_index = -1;
    saved_input.clear();

    input_content.clear();
    input_cursor_pos = 0;

    // Reset input area size
    calculate_layout();
    create_windows();
    draw_output();
    draw_input_box();
    draw_status();

    // Queue input for processing by run() loop
    input_queue.push_back({submitted, false});
    input_cv.notify_one();
}

int TUI::count_input_lines() const {
    if (input_content.empty()) return 1;

    int lines = 1;
    int col = 0;
    int content_width = term_cols - 6;

    for (char c : input_content) {
        if (c == '\n') {
            lines++;
            col = 0;
        } else {
            col++;
            if (col >= content_width) {
                lines++;
                col = 0;
            }
        }
    }

    return lines;
}

void TUI::write_output(const std::string& text, CallbackEvent type) {
    // Format assistant content (LaTeX → Unicode, table alignment)
    if (type == CallbackEvent::CONTENT) {
        std::string formatted = format_output(text);
        write_output(formatted.c_str(), formatted.length(), type);
    } else {
        write_output(text.c_str(), text.length(), type);
    }
}

void TUI::write_output(const char* text_data, size_t len, CallbackEvent type) {
    // Set color for this output type (uses centralized mapping from frontend.h)
    int color = get_color_for_event_ncurses(type);
    if (color != PAIR_DEFAULT) {
        wattron(output_pad, COLOR_PAIR(color));
    }
    if (type == CallbackEvent::THINKING) {
        wattron(output_pad, A_DIM);
    }

    // Get indent from centralized config (frontend.h)
    int indent_spaces = get_indent_for_event(type);

    // Write to pad, handling UTF-8 properly
    size_t i = 0;
    while (i < len) {
        unsigned char c = static_cast<unsigned char>(text_data[i]);

        // Strip incoming ANSI escape sequences
        if (c == '\033') {
            in_escape_sequence = true;
            i++;
            continue;
        }
        if (in_escape_sequence) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                in_escape_sequence = false;
            }
            i++;
            continue;
        }

        if (c == '\r') { i++; continue; }  // Skip carriage returns

        // Add indentation at line starts
        if (at_line_start && c != '\n' && indent_spaces > 0) {
            for (int j = 0; j < indent_spaces; j++) {
                waddch(output_pad, ' ');
            }
            at_line_start = false;
        }

        // Determine UTF-8 character length
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;        // ASCII
        else if ((c & 0xE0) == 0xC0) char_len = 2; // 2-byte
        else if ((c & 0xF0) == 0xE0) char_len = 3; // 3-byte
        else if ((c & 0xF8) == 0xF0) char_len = 4; // 4-byte

        // Ensure we have the complete character
        if (i + char_len > len) char_len = len - i;

        if (char_len == 1) {
            // Single byte - use waddch
            waddch(output_pad, c);
        } else {
            // Multi-byte UTF-8 - use waddnstr to output the complete sequence
            waddnstr(output_pad, text_data + i, char_len);
        }

        if (c == '\n') {
            at_line_start = true;
        } else {
            at_line_start = false;
        }

        i += char_len;
    }

    // Turn off attributes
    if (type == CallbackEvent::THINKING) {
        wattroff(output_pad, A_DIM);
    }
    if (color != PAIR_DEFAULT) {
        wattroff(output_pad, COLOR_PAIR(color));
    }

    // Get actual cursor position from ncurses (more reliable than tracking manually)
    pad_row = getcury(output_pad);

    // Auto-follow: keep viewport showing current output
    pad_view_top = std::max(0, pad_row - output_height + 1);

    // Refresh the pad display
    prefresh(output_pad, pad_view_top, 0, 0, 0, output_height - 1, term_cols - 1);
}

std::string TUI::get_input_content() const {
    std::lock_guard<std::mutex> lock(input_mutex);
    return input_content;
}

void TUI::clear_input() {
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_content.clear();
    }
    calculate_layout();
    draw_input_box();
    doupdate();
    wrefresh(input_win);
}

void TUI::set_status(const std::string& left, const std::string& right) {
    status_left = left;
    status_right = right;
    draw_status();
    doupdate();
}

void TUI::update_status_bar() {
    // Left side: provider name and model
    std::string left = current_provider;
    if (backend && !backend->model_name.empty()) {
        left += " | " + backend->model_name;
    }

    // Right side: token count / context size
    std::string right;
    if (backend && backend->context_size > 0) {
        right = std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size);
    }

    set_status(left, right);
}

void TUI::set_input_content(const std::string& content) {
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_content = content;
    }
    calculate_layout();
    draw_input_box();
    doupdate();
    wrefresh(input_win);
}

void TUI::position_cursor_in_input(int /* col */) {
    // ncurses handles cursor positioning in draw_input_box
}

void TUI::show_queued_input(const std::string& input) {
    // Add input to queued display list (shown in gray until processing)
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        queued_input_display.push_back({"> " + input, false});
    }
    // Recalculate layout since pending area may need to grow
    calculate_layout();
    create_windows();
    refresh_needed = true;
}

void TUI::mark_input_processing() {
    // Mark first non-processing queued input as now processing (gray -> cyan)
    std::lock_guard<std::mutex> lock(output_mutex);

    for (auto& item : queued_input_display) {
        if (!item.is_processing) {
            item.is_processing = true;
            break;
        }
    }

    refresh_needed = true;
}

void TUI::clear_queued_input_display() {
    // Clear all queued input display items (called when generation completes)
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        queued_input_display.clear();
    }
    // Recalculate layout since pending area should shrink/disappear
    calculate_layout();
    create_windows();
    refresh_needed = true;
}

void TUI::enable_bracketed_paste() {
    // Send escape sequence to enable bracketed paste mode
    printf("\033[?2004h");
    fflush(stdout);
}

void TUI::disable_bracketed_paste() {
    // Send escape sequence to disable bracketed paste mode
    printf("\033[?2004l");
    fflush(stdout);
}

// ============================================================================
// Input Queue Methods
// ============================================================================

void TUI::add_input(const std::string& input, bool needs_echo) {
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_queue.push_back({input, needs_echo});
    }
    input_cv.notify_one();
}

bool TUI::has_pending_input() {
    std::lock_guard<std::mutex> lock(input_mutex);
    return !input_queue.empty();
}

bool TUI::wait_for_input(int timeout_ms) {
    std::unique_lock<std::mutex> lock(input_mutex);
    if (!input_queue.empty()) return true;
    return input_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                             [this]{ return !input_queue.empty() || piped_eof; });
}

// ============================================================================
// Tool Display Methods
// ============================================================================

void TUI::show_tool_call(const std::string& name, const std::string& params) {
    // Indentation handled by write_output via get_indent_for_event()
    std::string msg = name + "(" + params + ")\n";
    write_output(msg, CallbackEvent::TOOL_CALL);
}

void TUI::show_tool_result(const std::string& summary, bool success) {
    // Indentation handled by write_output via get_indent_for_event()
    std::string msg = summary + "\n";
    write_output(msg, success ? CallbackEvent::TOOL_RESULT : CallbackEvent::SYSTEM);
}

void TUI::show_error(const std::string& error) {
    write_output("Error: " + error + "\n", CallbackEvent::SYSTEM);
}

void TUI::show_cancelled() {
    write_output("[Cancelled]\n", CallbackEvent::SYSTEM);
}

// ============================================================================
// Output Callback
// ============================================================================

bool TUI::output_callback(CallbackEvent type, const std::string& content,
                          const std::string& tool_name, const std::string& tool_call_id) {
    if (type == CallbackEvent::TOOL_CALL) {
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
        ToolResult result = execute_tool(tools, tool_name, content, tool_call_id, session.user_id);

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
        return true;
    }

    // Display-only tool call (from remote server - no local execution)
    if (type == CallbackEvent::TOOL_DISP) {
        show_tool_call(tool_name, content);
        return true;
    }

    // Display-only tool result (from remote server)
    if (type == CallbackEvent::RESULT_DISP) {
        show_tool_result(content, tool_name != "error");
        return true;
    }

    if (type == CallbackEvent::USER_PROMPT) {
        std::string formatted;
        std::istringstream stream(content);
        std::string line;
        bool first = true;
        while (std::getline(stream, line)) {
            if (!first) formatted += "\n";
            if (first) {
                formatted += "> " + line;
                first = false;
            } else {
                formatted += "  " + line;
            }
        }
        formatted += "\n";
        write_output(formatted, CallbackEvent::USER_PROMPT);
        return true;
    }

    // Handle CONTENT (assistant) messages
    if (type == CallbackEvent::CONTENT) {
        // Debug: log what we receive
        dout(2) << "CONTENT callback: len=" << content.length() << " content=[" << content << "]" << std::endl;
        // Indentation handled by write_output via get_indent_for_event()
        write_output(content, CallbackEvent::CONTENT);
        return true;
    }

    // Handle CODEBLOCK - indentation handled by write_output
    if (type == CallbackEvent::CODEBLOCK) {
        write_output(content, CallbackEvent::CODEBLOCK);
        return true;
    }

    // Handle remaining event types
    switch (type) {
        case CallbackEvent::SYSTEM:
            write_output(content, CallbackEvent::SYSTEM);
            break;
        case CallbackEvent::THINKING:
            write_output(content, CallbackEvent::THINKING);
            break;
        case CallbackEvent::STATS:
            // Only show stats if enabled via --stats flag
            if (config->stats) {
                write_output(content, CallbackEvent::THINKING);  // Gray/dim like thinking
            }
            break;
        case CallbackEvent::ERROR:
            write_output("Error: " + content + "\n", CallbackEvent::SYSTEM);
            break;
        case CallbackEvent::STOP:
            // STOP signals completion - content is finish_reason, not for display
            // Ensure we're at a new line for next prompt
            if (!at_line_start) {
                write_output("\n", CallbackEvent::CONTENT);
                at_line_start = true;
            }
            break;
        default:
            break;
    }
    return true;
}

// ============================================================================
// Main Run Loop
// ============================================================================

int TUI::run(Provider* cmdline_provider) {
    tui_debug(1, "Starting TUI mode");

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
            init_tools(no_mcp, no_tools, true, no_rag, mem_tools);  // force_local = true
        }
    }

    // Register other providers as tools (unless tools disabled)
    if (!no_tools) {
        register_provider_tools(tools, current_provider);
    }

    // Update status bar with provider info
    update_status_bar();

    // Populate session.tools from our tools instance
    tools.populate_session_tools(session);

    // Copy tool names to backend for output filtering
    for (const auto& tool : session.tools) {
        backend->valid_tool_names.insert(tool.name);
    }

    // Configure session based on backend capabilities
    if (config->max_tokens == -1) {
        // -1 = max possible: no cap on completion tokens (use all available)
        session.desired_completion_tokens = INT_MAX;
    } else if (config->max_tokens > 0) {
        // Explicit value
        session.desired_completion_tokens = config->max_tokens;
    } else {
        // 0 = auto: calculate based on context size
        session.desired_completion_tokens = calculate_desired_completion_tokens(
            backend->context_size, backend->max_output_tokens);
    }
    session.auto_evict = (backend->context_size > 0 && !backend->is_gpu);

    Scheduler scheduler(config->scheduler_name);
    if (!g_disable_scheduler) {
        scheduler.load();
        scheduler.set_fire_callback([this](const std::string& prompt) {
            add_input(prompt, true);
        });
        scheduler.start();
        tui_debug(1, "Scheduler initialized");
    }

    // Start generation thread AFTER backend is connected
    GenerationThread gen_thread;
    gen_thread.init(this);  // Pass Frontend pointer for unified generation path
    gen_thread.start();
    g_generation_thread = &gen_thread;

    wtimeout(input_win, 10);

    std::string user_input;
    bool awaiting_generation = false;
    Response resp;

    while (!quit_requested) {
        if (!g_disable_scheduler) {
            scheduler.poll();
        }

        // run_once() handles all input including escape sequences
        run_once();

        if (awaiting_generation && check_escape_pressed()) {
            g_generation_cancelled = true;
            for (int i = 0; i < 50 && !gen_thread.is_complete(); i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            show_cancelled();
            clear_queued_input_display();
            awaiting_generation = false;
            gen_thread.reset();
            g_generation_cancelled = false;
            continue;
        }

        if (awaiting_generation && gen_thread.is_complete()) {
            // Generation complete - tool calls are handled recursively in callback
            awaiting_generation = false;
            resp = gen_thread.last_response;
            gen_thread.reset();

            if (!resp.success) {
                show_error(resp.error);
                clear_queued_input_display();
                continue;
            }

            if (resp.finish_reason == "cancelled") {
                show_cancelled();
                clear_queued_input_display();
                continue;
            }

            tui_debug(1, "Generation complete, length: " + std::to_string(resp.content.length()));
            write_output("\n", CallbackEvent::CONTENT);
            clear_queued_input_display();

            // Update status bar with current token count
            update_status_bar();
            continue;
        }

        if (!awaiting_generation && has_pending_input()) {
            std::lock_guard<std::mutex> lock(input_mutex);
            if (input_queue.empty()) continue;

            auto item = input_queue.front();
            input_queue.pop_front();
            user_input = item.text;

            if (user_input == "exit" || user_input == "quit") {
                request_quit();
                break;
            }

            // Handle slash commands
            if (!user_input.empty() && user_input[0] == '/') {
                if (Frontend::handle_slash_commands(user_input, tools)) {
                    continue;
                }
            }

            mark_input_processing();

            tui_debug(1, "User input: " + user_input);

            // Display user prompt via callback (unified with CLI and CLI Server)
            callback(CallbackEvent::USER_PROMPT, user_input, "", "");

            GenerationRequest req;
            req.role = Message::USER;
            req.content = user_input;

            req.callback = [this](CallbackEvent type, const std::string& content,
                                  const std::string& tool_name, const std::string& tool_call_id) -> bool {
                return output_callback(type, content, tool_name, tool_call_id);
            };

            gen_thread.submit(req);
            awaiting_generation = true;
        }

        if (piped_eof && !has_pending_input() && !awaiting_generation) {
            tui_debug(1, "EOF, exiting");
            break;
        }
    }

    g_generation_thread = nullptr;
    tui_debug(1, "TUI loop ended");

    return 0;
}
