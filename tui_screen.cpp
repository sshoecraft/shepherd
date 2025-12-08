#include "tui_screen.h"
#include "terminal_io.h"
#include <algorithm>
#include <cstring>
#include <unistd.h>

// Global TUI screen instance
TUIScreen* g_tui_screen = nullptr;

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

TUIScreen::TUIScreen()
    : output_win(nullptr)
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

TUIScreen::~TUIScreen() {
    shutdown();
}

bool TUIScreen::init() {
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

void TUIScreen::shutdown() {
    disable_bracketed_paste();
    destroy_windows();
    endwin();

    // Move cursor to bottom and print newline for clean shell prompt
    printf("\033[%d;1H\n", term_rows);
    fflush(stdout);
}

void TUIScreen::calculate_layout() {
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

void TUIScreen::create_windows() {
    destroy_windows();

    // Output window at top
    output_win = newwin(output_height, term_cols, 0, 0);
    scrollok(output_win, TRUE);
    leaveok(output_win, TRUE);  // Don't care about cursor in output

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

void TUIScreen::destroy_windows() {
    if (output_win) { delwin(output_win); output_win = nullptr; }
    if (pending_win) { delwin(pending_win); pending_win = nullptr; }
    if (input_win) { delwin(input_win); input_win = nullptr; }
    if (status_win) { delwin(status_win); status_win = nullptr; }
}

void TUIScreen::draw_output() {
    std::lock_guard<std::mutex> lock(output_mutex);

    werase(output_win);

    // Calculate visible lines (output grows from bottom)
    // Queued items are now in separate pending window
    int visible_lines = output_height;
    int total_lines = static_cast<int>(output_lines.size());
    bool has_partial = !current_output_line.empty() && scroll_offset == 0;
    if (has_partial) total_lines++;

    // Apply scroll offset (scroll_offset > 0 means scrolled back in history)
    int end_idx = total_lines - scroll_offset;
    if (end_idx < 0) end_idx = 0;
    int start_idx = std::max(0, end_idx - visible_lines);

    // Draw from bottom up (newest at bottom of visible area)
    int row = visible_lines - (end_idx - start_idx);
    if (row < 0) row = 0;

    // Draw completed lines
    int completed_lines = static_cast<int>(output_lines.size());
    int draw_end = std::min(end_idx, completed_lines);
    for (int i = start_idx; i < draw_end && row < visible_lines; i++, row++) {
        mvwprintw(output_win, row, 0, "%s", output_lines[i].c_str());
    }

    // Draw partial line (streaming output) - only if not scrolled back
    if (has_partial && row < visible_lines && scroll_offset == 0) {
        mvwprintw(output_win, row, 0, "%s", current_output_line.c_str());
    }

    wnoutrefresh(output_win);
}

void TUIScreen::draw_pending() {
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

void TUIScreen::draw_input_box() {
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
            // Wrap long lines
            while (static_cast<int>(remaining.length()) > content_width) {
                display_lines.push_back(remaining.substr(0, content_width));
                remaining = remaining.substr(content_width);
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

void TUIScreen::draw_status() {
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

bool TUIScreen::run_once() {
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

    // Handle input - use short timeout so we return quickly to drain output queue
    wtimeout(input_win, 5);  // 5ms timeout for responsive streaming
    int ch = wgetch(input_win);

    if (ch == ERR) {
        // No input, just return
        return !quit_requested;
    }

    // Check for bracketed paste sequences
    if (ch == 27) {  // ESC
        // Read ahead to check for paste sequence
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
        } else if (next != ERR) {
            // Not a bracket, could be Alt+key or just ESC
            // For now, ignore
        }
        // else: just ESC key, ignore
    } else if (in_paste_mode) {
        handle_paste_char(ch);
    } else {
        handle_key(ch);
    }

    return !quit_requested;
}

void TUIScreen::handle_key(int ch) {
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
                std::lock_guard<std::mutex> lock(output_mutex);
                int max_scroll = static_cast<int>(output_lines.size()) - output_height;
                if (max_scroll < 0) max_scroll = 0;
                scroll_offset += output_height / 2;
                if (scroll_offset > max_scroll) scroll_offset = max_scroll;
                refresh_needed = true;
            }
            break;

        case KEY_NPAGE:  // Page Down - scroll output forward
            {
                std::lock_guard<std::mutex> lock(output_mutex);
                scroll_offset -= output_height / 2;
                if (scroll_offset < 0) scroll_offset = 0;
                refresh_needed = true;
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

void TUIScreen::handle_paste_char(int ch) {
    if (ch >= 32 && ch < 127) {
        paste_buffer += static_cast<char>(ch);
    } else if (ch == '\n' || ch == '\r') {
        paste_buffer += '\n';
    }
}

void TUIScreen::submit_input() {
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

    // Call callback (outside lock)
    if (on_input) {
        on_input(submitted);
    }
}

int TUIScreen::count_input_lines() const {
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

void TUIScreen::write_output(const std::string& text, Color color) {
    write_output(text.c_str(), text.length(), color);
}

void TUIScreen::write_output(const char* text_data, size_t len, Color /* color */) {
    std::lock_guard<std::mutex> lock(output_mutex);

    for (size_t i = 0; i < len; i++) {
        char c = text_data[i];

        // Strip ANSI escape sequences (state persists across chunks)
        if (c == '\033') {
            in_escape_sequence = true;
            continue;
        }
        if (in_escape_sequence) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                in_escape_sequence = false;
            }
            continue;
        }

        if (c == '\n') {
            output_lines.push_back(current_output_line);
            current_output_line.clear();
        } else if (c != '\r') {
            current_output_line += c;
        }
    }

    // Trim scrollback
    while (static_cast<int>(output_lines.size()) > scrollback_limit) {
        output_lines.erase(output_lines.begin());
    }

    // Snap to bottom when new output arrives
    scroll_offset = 0;

    refresh_needed = true;
}

void TUIScreen::flush() {
    if (refresh_needed) {
        refresh_needed = false;
        draw_output();
        draw_pending();
        draw_status();
        draw_input_box();
        doupdate();
        wrefresh(input_win);
    }
}

void TUIScreen::set_input_callback(InputCallback callback) {
    on_input = callback;
}

std::string TUIScreen::get_input_content() const {
    std::lock_guard<std::mutex> lock(input_mutex);
    return input_content;
}

void TUIScreen::clear_input() {
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_content.clear();
    }
    calculate_layout();
    draw_input_box();
    doupdate();
    wrefresh(input_win);
}

void TUIScreen::set_status(const std::string& left, const std::string& right) {
    status_left = left;
    status_right = right;
    draw_status();
    doupdate();
}

void TUIScreen::request_refresh() {
    refresh_needed = true;
}

void TUIScreen::set_input_content(const std::string& content) {
    {
        std::lock_guard<std::mutex> lock(input_mutex);
        input_content = content;
    }
    calculate_layout();
    draw_input_box();
    doupdate();
    wrefresh(input_win);
}

void TUIScreen::position_cursor_in_input(int /* col */) {
    // ncurses handles cursor positioning in draw_input_box
}

int TUIScreen::get_color_pair(Color color) const {
    switch (color) {
        case Color::RED:    return PAIR_RED;
        case Color::GREEN:  return PAIR_GREEN;
        case Color::YELLOW: return PAIR_YELLOW;
        case Color::CYAN:   return PAIR_CYAN;
        case Color::GRAY:   return PAIR_GRAY;
        default:            return PAIR_DEFAULT;
    }
}

void TUIScreen::show_queued_input(const std::string& input) {
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

void TUIScreen::mark_input_processing() {
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

void TUIScreen::clear_queued_input_display() {
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

void TUIScreen::enable_bracketed_paste() {
    // Send escape sequence to enable bracketed paste mode
    printf("\033[?2004h");
    fflush(stdout);
}

void TUIScreen::disable_bracketed_paste() {
    // Send escape sequence to disable bracketed paste mode
    printf("\033[?2004l");
    fflush(stdout);
}
