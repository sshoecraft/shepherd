#include "tui_screen.h"
#include "terminal_io.h"
#include "logger.h"

#include <termios.h>
#include <unistd.h>

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/component/loop.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/terminal.hpp>

#include <algorithm>

// Namespace alias to avoid Color conflict
namespace fx = ftxui;

// Global TUI screen instance
TUIScreen* g_tui_screen = nullptr;

// Convert our Color enum to FTXUI color
static fx::Color to_ftxui_color(Color color) {
    switch (color) {
        case Color::RED:     return fx::Color::Red;
        case Color::YELLOW:  return fx::Color::Yellow;
        case Color::GREEN:   return fx::Color::Green;
        case Color::CYAN:    return fx::Color::Cyan;
        case Color::GRAY:    return fx::Color::GrayDark;
        case Color::DEFAULT:
        default:             return fx::Color::Default;
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
    // Use Fullscreen mode on primary screen (not alternate buffer)
    screen = new fx::ScreenInteractive(fx::ScreenInteractive::FullscreenPrimaryScreen());

    // Disable mouse so terminal handles it natively (copy/paste works)
    screen->TrackMouse(false);

    // Enable bracketed paste mode
    printf("\033[?2004h");
    fflush(stdout);

    // Build the UI component tree
    main_component = build_ui();

    // Create the event loop
    loop = new fx::Loop(screen, main_component);

    return true;
}

void TUIScreen::shutdown() {
    // Save output lines before destroying screen
    std::vector<ColoredLine> saved_lines;
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        saved_lines = output_lines;
        if (!current_output_line.empty()) {
            saved_lines.push_back({current_output_line, current_output_color, current_output_type});
        }
    }

    // Just signal exit - don't delete, let program cleanup handle it
    // Deleting causes crashes due to FTXUI's DEC mode cleanup lambdas
    if (screen) {
        screen->Exit();
    }
    // Note: intentionally NOT deleting loop and screen
    // FTXUI has issues with cleanup order of captured lambdas

    // Disable bracketed paste mode
    printf("\033[?2004l");
    // Restore cursor: show it and set to steady block
    printf("\033[?25h");   // Show cursor
    printf("\033[2 q");    // Steady block

    // Restore terminal to sane defaults
    struct termios term;
    tcgetattr(STDIN_FILENO, &term);
    term.c_iflag |= (BRKINT | ICRNL | IMAXBEL | IUTF8);
    term.c_oflag |= (OPOST | ONLCR);
    term.c_lflag |= (ICANON | ISIG | IEXTEN | ECHO | ECHOE | ECHOK | ECHOCTL | ECHOKE);
    tcsetattr(STDIN_FILENO, TCSANOW, &term);

    // Dump session history to terminal so it's in scrollback
    printf("\n--- Session History ---\n");
    for (const auto& line : saved_lines) {
        printf("%s\n", line.text.c_str());
    }
    printf("--- End Session ---\n\n");

    fflush(stdout);
}

std::shared_ptr<fx::ComponentBase> TUIScreen::build_ui() {
    // Input component - multiline enabled for paste support
    fx::InputOption input_option;
    input_option.multiline = true;
    input_option.cursor_position = &cursor_position;  // Bind cursor position
    input_option.insert = &insert_mode;  // Bind insert mode

    // Custom transform: don't invert the whole input when focused
    // FTXUI internally applies "focused" decorator to just the cursor character
    input_option.transform = [](fx::InputState state) {
        if (state.is_placeholder) {
            // Show just cursor block when empty (no placeholder text)
            if (state.focused) {
                return fx::text(" ") | fx::inverted;
            }
            return fx::text("");  // Nothing when not focused and empty
        }
        // Don't apply inverted - let the internal cursor handling work
        return state.element;
    };

    input_component = fx::Input(&input_content, "", input_option);

    // Handle Enter key to submit and manage insert mode based on cursor position
    input_component |= fx::CatchEvent([this](fx::Event event) {
        // Left arrow: enable insert mode when moving into text
        if (event == fx::Event::ArrowLeft) {
            if (cursor_position > 0) {
                insert_mode = true;  // Switch to insert mode when in middle of text
            }
            return false;  // Let FTXUI handle the actual cursor move
        }

        // Right arrow: disable insert mode when reaching end of text
        if (event == fx::Event::ArrowRight) {
            // After FTXUI processes this, cursor will be at cursor_position + 1
            // If that's at the end, switch back to overtype (block cursor)
            if (cursor_position + 1 >= static_cast<int>(input_content.size())) {
                insert_mode = false;
            }
            return false;  // Let FTXUI handle the actual cursor move
        }

        // Enter submits
        if (event == fx::Event::Return) {
            on_input_submit();
            return true;
        }
        return false;
    });

    // Catch Ctrl+D to quit
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::CtrlD) {
            quit_requested = true;
            screen->Exit();
            return true;
        }
        return false;
    });

    // Catch Escape: single = cancel generation, double = clear input
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::Escape) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_escape_time);
            last_escape_time = now;

            if (elapsed.count() < 500) {
                // Double escape - clear input
                input_content.clear();
                cursor_position = 0;
                insert_mode = false;  // Block cursor when empty
                history_index = -1;
                saved_input.clear();
            } else {
                // Single escape - cancel generation
                escape_pressed = true;
                // Clear queued input display
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    queued_inputs.clear();
                }
            }
            return true;
        }
        return false;
    });

    // Up arrow - previous history
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::ArrowUp) {
            if (!history.empty()) {
                if (history_index == -1) {
                    saved_input = input_content;
                    history_index = static_cast<int>(history.size()) - 1;
                } else if (history_index > 0) {
                    history_index--;
                }
                input_content = history[history_index];
                cursor_position = static_cast<int>(input_content.size());  // Move cursor to end
            }
            return true;
        }
        return false;
    });

    // Down arrow - next history
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::ArrowDown) {
            if (history_index >= 0) {
                if (history_index < static_cast<int>(history.size()) - 1) {
                    history_index++;
                    input_content = history[history_index];
                } else {
                    history_index = -1;
                    input_content = saved_input;
                }
                cursor_position = static_cast<int>(input_content.size());  // Move cursor to end
            }
            return true;
        }
        return false;
    });

    // Page Up - scroll output up
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::PageUp) {
            int available_height = fx::Terminal::Size().dimy - 4;
            int max_scroll = std::max(0, static_cast<int>(output_lines.size()) - available_height);
            scroll_offset = std::min(scroll_offset + available_height / 2, max_scroll);
            return true;
        }
        return false;
    });

    // Page Down - scroll output down
    input_component |= fx::CatchEvent([this](fx::Event event) {
        if (event == fx::Event::PageDown) {
            int available_height = fx::Terminal::Size().dimy - 4;
            scroll_offset = std::max(0, scroll_offset - available_height / 2);
            return true;
        }
        return false;
    });

    // Create the renderer that composes the full UI
    auto renderer = fx::Renderer(input_component, [this] {
        std::lock_guard<std::mutex> lock(output_mutex);

        // Build output elements - show lines based on scroll position
        fx::Elements output_elements;

        // Get terminal height to calculate visible lines
        int available_height = fx::Terminal::Size().dimy - 4;  // Reserve for input box + status
        if (available_height < 1) available_height = 1;

        // Calculate visible range with scroll offset
        int total_lines = static_cast<int>(output_lines.size());
        // When scrolled, don't show partial line
        bool show_partial = (scroll_offset == 0) && !current_output_line.empty();

        // End index is total_lines minus scroll_offset
        int end_idx = total_lines - scroll_offset;
        int start_idx = std::max(0, end_idx - available_height);
        if (end_idx < 0) end_idx = 0;

        for (int i = start_idx; i < end_idx && i < static_cast<int>(output_lines.size()); i++) {
            const auto& line = output_lines[i];
            // Only indent ASSISTANT output
            bool needs_indent = (line.type == LineType::ASSISTANT);
            std::string display = needs_indent ? ("  " + line.text) : line.text;
            auto elem = fx::text(display);
            if (line.color != Color::DEFAULT) {
                elem |= fx::color(to_ftxui_color(line.color));
            }
            output_elements.push_back(elem);
        }
        // Add partial line (streaming output) only when at bottom
        if (show_partial) {
            bool needs_indent = (current_output_type == LineType::ASSISTANT);
            std::string display = needs_indent ? ("  " + current_output_line) : current_output_line;
            auto elem = fx::text(display);
            if (current_output_color != Color::DEFAULT) {
                elem |= fx::color(to_ftxui_color(current_output_color));
            }
            output_elements.push_back(elem);
        }

        // If we have fewer lines than available space, pad with empty lines at top
        int padding = available_height - static_cast<int>(output_elements.size());
        fx::Elements padded_output;
        for (int i = 0; i < padding; i++) {
            padded_output.push_back(fx::text(""));
        }
        for (auto& elem : output_elements) {
            padded_output.push_back(std::move(elem));
        }

        // Build queued input display (max 3 lines, right above input box)
        fx::Elements queued_elements;
        int queue_start = std::max(0, static_cast<int>(queued_inputs.size()) - 3);
        for (int i = queue_start; i < static_cast<int>(queued_inputs.size()); i++) {
            auto& q = queued_inputs[i];
            auto color = q.is_processing ? fx::Color::Cyan : fx::Color::GrayDark;
            queued_elements.push_back(fx::text(q.text) | fx::color(color));
        }

        // Build the input box with border
        auto input_elem = input_component->Render();
        auto prompt_and_input = fx::hbox({
            fx::text(" > ") | fx::color(fx::Color::Green),
            input_elem,
        });
        auto input_box = fx::hbox({
            prompt_and_input,
            fx::text("") | fx::flex,  // Empty flex element to push input left
        }) | fx::border | fx::size(fx::HEIGHT, fx::LESS_THAN, 12);

        // Build status line
        auto status_line = fx::hbox({
            fx::text(status_left) | fx::color(fx::Color::GrayDark),
            fx::filler(),
            fx::text(status_right) | fx::color(fx::Color::GrayDark),
        });

        // Compose the full layout
        fx::Elements layout;
        layout.push_back(fx::vbox(std::move(padded_output)) | fx::flex);  // Output area
        if (!queued_elements.empty()) {
            layout.push_back(fx::vbox(std::move(queued_elements)));  // Queued inputs
        }
        layout.push_back(input_box);   // Input box
        layout.push_back(status_line); // Status line

        return fx::vbox(std::move(layout));
    });

    return renderer;
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
}

void TUIScreen::write_output(const std::string& text_str, LineType type) {
    write_output(text_str.c_str(), text_str.length(), type);
}

void TUIScreen::write_output(const char* text_data, size_t len, LineType type) {
    {
        std::lock_guard<std::mutex> lock(output_mutex);

        // Set type and derive color from it
        current_output_type = type;
        switch (type) {
            case LineType::USER:        current_output_color = Color::GREEN; break;
            case LineType::TOOL_CALL:   current_output_color = Color::YELLOW; break;
            case LineType::TOOL_RESULT: current_output_color = Color::CYAN; break;
            case LineType::SYSTEM:      current_output_color = Color::RED; break;
            case LineType::ASSISTANT:
            default:                    current_output_color = Color::DEFAULT; break;
        }

        // Get terminal width for wrapping (account for indent)
        int term_width = fx::Terminal::Size().dimx - 4;  // 2 for border, 2 for indent
        if (term_width < 20) term_width = 80;

        // Process text character by character, building lines
        // Strip ANSI escape sequences since FTXUI has its own color system
        static bool in_escape = false;

        for (size_t i = 0; i < len; i++) {
            char c = text_data[i];

            // Detect start of ANSI escape sequence
            if (c == '\033') {
                in_escape = true;
                continue;
            }

            // Skip characters until end of escape sequence
            if (in_escape) {
                // ANSI sequences end with a letter (a-zA-Z)
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                    in_escape = false;
                }
                continue;
            }

            if (c == '\n') {
                output_lines.push_back({current_output_line, current_output_color, current_output_type});
                current_output_line.clear();
            } else if (c == '\r') {
                // Ignore carriage returns
            } else {
                current_output_line += c;
                // Wrap long lines
                if (static_cast<int>(current_output_line.size()) >= term_width) {
                    output_lines.push_back({current_output_line, current_output_color, current_output_type});
                    current_output_line.clear();
                }
            }
        }

        // Trim scrollback if needed
        while (static_cast<int>(output_lines.size()) > scrollback_limit) {
            output_lines.erase(output_lines.begin());
        }

        // Auto-scroll to bottom on new output
        scroll_offset = 0;

        refresh_needed = true;
    }
    // Post outside lock to avoid potential deadlock
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

void TUIScreen::set_input_callback(InputCallback callback) {
    on_input = callback;
}

std::string TUIScreen::get_input_content() const {
    return input_content;
}

void TUIScreen::clear_input() {
    input_content.clear();
}

void TUIScreen::set_status(const std::string& left, const std::string& right) {
    status_left = left;
    status_right = right;
    refresh_needed = true;
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

bool TUIScreen::run_once() {
    if (!loop || quit_requested) return false;

    loop->RunOnce();
    return !quit_requested;
}

void TUIScreen::request_refresh() {
    refresh_needed = true;
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

// Legacy API compatibility
void TUIScreen::set_input_content(const std::string& content) {
    input_content = content;
    request_refresh();
}

void TUIScreen::position_cursor_in_input(int /* col */) {
    // FTXUI handles cursor positioning automatically
    // This is a no-op for compatibility
}

void TUIScreen::flush() {
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

void TUIScreen::show_queued_input(const std::string& input) {
    std::lock_guard<std::mutex> lock(output_mutex);
    queued_inputs.push_back({"> " + input, false});
    refresh_needed = true;
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

void TUIScreen::mark_input_processing() {
    std::lock_guard<std::mutex> lock(output_mutex);
    // Mark the first non-processing queued item as processing
    for (auto& q : queued_inputs) {
        if (!q.is_processing) {
            q.is_processing = true;
            break;
        }
    }
    refresh_needed = true;
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}

void TUIScreen::clear_queued_input_display() {
    std::lock_guard<std::mutex> lock(output_mutex);
    queued_inputs.clear();
    refresh_needed = true;
    if (screen) {
        screen->Post(fx::Event::Custom);
    }
}
