#ifndef ANSI_H
#define ANSI_H

#include <string>

// https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
 
// ANSI Escape Sequence Definitions
// Base escape character
#define ANSI_ESC "\033"
#define ANSI_CSI "\033["

// ============================================================================
// Cursor Movement
// ============================================================================

// Move cursor to home position (0,0)
#define ANSI_CURSOR_HOME "\033[H"

// Move cursor to specific position (1-indexed)
#define ANSI_CURSOR_POS(row, col) ("\033[" + std::to_string(row) + ";" + std::to_string(col) + "H")

// Move cursor up N lines
#define ANSI_CURSOR_UP(n) ("\033[" + std::to_string(n) + "A")

// Move cursor down N lines
#define ANSI_CURSOR_DOWN(n) ("\033[" + std::to_string(n) + "B")

// Move cursor right N columns
#define ANSI_CURSOR_RIGHT(n) ("\033[" + std::to_string(n) + "C")

// Move cursor left N columns
#define ANSI_CURSOR_LEFT(n) ("\033[" + std::to_string(n) + "D")

// Move cursor to beginning of line N lines down
#define ANSI_CURSOR_NEXT_LINE(n) ("\033[" + std::to_string(n) + "E")

// Move cursor to beginning of line N lines up
#define ANSI_CURSOR_PREV_LINE(n) ("\033[" + std::to_string(n) + "F")

// Move cursor to column N
#define ANSI_CURSOR_COL(n) ("\033[" + std::to_string(n) + "G")

// Request cursor position (response: ESC[{row};{col}R)
#define ANSI_CURSOR_REQUEST_POS "\033[6n"

// Save cursor position (DEC)
#define ANSI_CURSOR_SAVE "\0337"

// Restore cursor position (DEC)
#define ANSI_CURSOR_RESTORE "\0338"

// Save cursor position (SCO)
#define ANSI_CURSOR_SAVE_SCO "\033[s"

// Restore cursor position (SCO)
#define ANSI_CURSOR_RESTORE_SCO "\033[u"

// ============================================================================
// Cursor Visibility
// ============================================================================

#define ANSI_CURSOR_SHOW "\033[?25h"
#define ANSI_CURSOR_HIDE "\033[?25l"

// Cursor shape (DECSCUSR)
#define ANSI_CURSOR_BLOCK_BLINK "\033[1 q"
#define ANSI_CURSOR_BLOCK "\033[2 q"
#define ANSI_CURSOR_UNDERLINE_BLINK "\033[3 q"
#define ANSI_CURSOR_UNDERLINE "\033[4 q"
#define ANSI_CURSOR_BAR_BLINK "\033[5 q"
#define ANSI_CURSOR_BAR "\033[6 q"

// ============================================================================
// Screen Erasing
// ============================================================================

// Erase from cursor to end of display
#define ANSI_ERASE_TO_END "\033[J"
#define ANSI_ERASE_TO_END_0 "\033[0J"

// Erase from start of display to cursor
#define ANSI_ERASE_TO_START "\033[1J"

// Clear entire screen
#define ANSI_ERASE_SCREEN "\033[2J"

// Erase saved lines (scrollback)
#define ANSI_ERASE_SAVED "\033[3J"

// Erase from cursor to end of line
#define ANSI_ERASE_LINE_TO_END "\033[K"
#define ANSI_ERASE_LINE_TO_END_0 "\033[0K"

// Erase from start of line to cursor
#define ANSI_ERASE_LINE_TO_START "\033[1K"

// Erase entire line
#define ANSI_ERASE_LINE "\033[2K"

// ============================================================================
// Scroll Region
// ============================================================================

// Set scroll region (top and bottom rows, 1-indexed)
#define ANSI_SCROLL_REGION(top, bottom) ("\033[" + std::to_string(top) + ";" + std::to_string(bottom) + "r")

// Reset scroll region to full screen
#define ANSI_SCROLL_REGION_RESET "\033[r"

// Scroll up N lines
#define ANSI_SCROLL_UP(n) ("\033[" + std::to_string(n) + "S")

// Scroll down N lines
#define ANSI_SCROLL_DOWN(n) ("\033[" + std::to_string(n) + "T")

// ============================================================================
// Text Formatting (SGR - Select Graphic Rendition)
// ============================================================================

#define ANSI_RESET "\033[0m"
#define ANSI_BOLD "\033[1m"
#define ANSI_DIM "\033[2m"
#define ANSI_ITALIC "\033[3m"
#define ANSI_UNDERLINE "\033[4m"
#define ANSI_BLINK "\033[5m"
#define ANSI_BLINK_RAPID "\033[6m"
#define ANSI_REVERSE "\033[7m"
#define ANSI_HIDDEN "\033[8m"
#define ANSI_STRIKETHROUGH "\033[9m"

// Reset specific attributes
#define ANSI_BOLD_OFF "\033[22m"
#define ANSI_DIM_OFF "\033[22m"
#define ANSI_ITALIC_OFF "\033[23m"
#define ANSI_UNDERLINE_OFF "\033[24m"
#define ANSI_BLINK_OFF "\033[25m"
#define ANSI_REVERSE_OFF "\033[27m"
#define ANSI_HIDDEN_OFF "\033[28m"
#define ANSI_STRIKETHROUGH_OFF "\033[29m"

// ============================================================================
// Standard Colors (8/16 colors)
// ============================================================================

// Foreground colors
#define ANSI_FG_BLACK "\033[30m"
#define ANSI_FG_RED "\033[31m"
#define ANSI_FG_GREEN "\033[32m"
#define ANSI_FG_YELLOW "\033[33m"
#define ANSI_FG_BLUE "\033[34m"
#define ANSI_FG_MAGENTA "\033[35m"
#define ANSI_FG_CYAN "\033[36m"
#define ANSI_FG_WHITE "\033[37m"
#define ANSI_FG_DEFAULT "\033[39m"

// Bright foreground colors
#define ANSI_FG_BRIGHT_BLACK "\033[90m"
#define ANSI_FG_BRIGHT_RED "\033[91m"
#define ANSI_FG_BRIGHT_GREEN "\033[92m"
#define ANSI_FG_BRIGHT_YELLOW "\033[93m"
#define ANSI_FG_BRIGHT_BLUE "\033[94m"
#define ANSI_FG_BRIGHT_MAGENTA "\033[95m"
#define ANSI_FG_BRIGHT_CYAN "\033[96m"
#define ANSI_FG_BRIGHT_WHITE "\033[97m"

// Background colors
#define ANSI_BG_BLACK "\033[40m"
#define ANSI_BG_RED "\033[41m"
#define ANSI_BG_GREEN "\033[42m"
#define ANSI_BG_YELLOW "\033[43m"
#define ANSI_BG_BLUE "\033[44m"
#define ANSI_BG_MAGENTA "\033[45m"
#define ANSI_BG_CYAN "\033[46m"
#define ANSI_BG_WHITE "\033[47m"
#define ANSI_BG_DEFAULT "\033[49m"

// Bright background colors
#define ANSI_BG_BRIGHT_BLACK "\033[100m"
#define ANSI_BG_BRIGHT_RED "\033[101m"
#define ANSI_BG_BRIGHT_GREEN "\033[102m"
#define ANSI_BG_BRIGHT_YELLOW "\033[103m"
#define ANSI_BG_BRIGHT_BLUE "\033[104m"
#define ANSI_BG_BRIGHT_MAGENTA "\033[105m"
#define ANSI_BG_BRIGHT_CYAN "\033[106m"
#define ANSI_BG_BRIGHT_WHITE "\033[107m"

// ============================================================================
// 256 Colors
// ============================================================================

// Foreground 256 color (0-255)
#define ANSI_FG_256(id) ("\033[38;5;" + std::to_string(id) + "m")

// Background 256 color (0-255)
#define ANSI_BG_256(id) ("\033[48;5;" + std::to_string(id) + "m")

// ============================================================================
// True Color (24-bit RGB)
// ============================================================================

// Foreground RGB color
#define ANSI_FG_RGB(r, g, b) ("\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m")

// Background RGB color
#define ANSI_BG_RGB(r, g, b) ("\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m")

// ============================================================================
// Screen Modes
// ============================================================================

// Alternative screen buffer
#define ANSI_ALT_BUFFER_ON "\033[?1049h"
#define ANSI_ALT_BUFFER_OFF "\033[?1049l"

// Save/restore screen
#define ANSI_SCREEN_SAVE "\033[?47h"
#define ANSI_SCREEN_RESTORE "\033[?47l"

// Bracketed paste mode
#define ANSI_BRACKETED_PASTE_ON "\033[?2004h"
#define ANSI_BRACKETED_PASTE_OFF "\033[?2004l"

// Mouse tracking
#define ANSI_MOUSE_ON "\033[?1000h"
#define ANSI_MOUSE_OFF "\033[?1000l"
#define ANSI_MOUSE_SGR_ON "\033[?1006h"
#define ANSI_MOUSE_SGR_OFF "\033[?1006l"

// Line wrapping
#define ANSI_LINE_WRAP_ON "\033[?7h"
#define ANSI_LINE_WRAP_OFF "\033[?7l"

// ============================================================================
// Tabs
// ============================================================================

#define ANSI_TAB_SET "\033H"
#define ANSI_TAB_CLEAR "\033[g"
#define ANSI_TAB_CLEAR_ALL "\033[3g"

// ============================================================================
// Terminal Control
// ============================================================================

// Soft reset
#define ANSI_SOFT_RESET "\033[!p"

// Full reset (RIS)
#define ANSI_FULL_RESET "\033c"

// Set window title (OSC)
#define ANSI_SET_TITLE(title) ("\033]0;" + std::string(title) + "\007")

// ============================================================================
// Helper Functions (inline for header-only use)
// ============================================================================

namespace ansi {

inline std::string cursor_pos(int row, int col) {
    return "\033[" + std::to_string(row) + ";" + std::to_string(col) + "H";
}

inline std::string cursor_up(int n = 1) {
    return "\033[" + std::to_string(n) + "A";
}

inline std::string cursor_down(int n = 1) {
    return "\033[" + std::to_string(n) + "B";
}

inline std::string cursor_right(int n = 1) {
    return "\033[" + std::to_string(n) + "C";
}

inline std::string cursor_left(int n = 1) {
    return "\033[" + std::to_string(n) + "D";
}

inline std::string cursor_col(int col) {
    return "\033[" + std::to_string(col) + "G";
}

inline std::string scroll_region(int top, int bottom) {
    return "\033[" + std::to_string(top) + ";" + std::to_string(bottom) + "r";
}

inline std::string scroll_up(int n = 1) {
    return "\033[" + std::to_string(n) + "S";
}

inline std::string scroll_down(int n = 1) {
    return "\033[" + std::to_string(n) + "T";
}

inline std::string fg_256(int id) {
    return "\033[38;5;" + std::to_string(id) + "m";
}

inline std::string bg_256(int id) {
    return "\033[48;5;" + std::to_string(id) + "m";
}

inline std::string fg_rgb(int r, int g, int b) {
    return "\033[38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

inline std::string bg_rgb(int r, int g, int b) {
    return "\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
}

inline std::string set_title(const std::string& title) {
    return "\033]0;" + title + "\007";
}

} // namespace ansi

#endif // ANSI_H
