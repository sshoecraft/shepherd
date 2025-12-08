#include "input_reader.h"
#include "tui_screen.h"
#include "terminal_io.h"
#include "logger.h"
#include "vendor/replxx/include/replxx.h"
#include <iostream>
#include <cstring>

// External globals
extern TUIScreen* g_tui_screen;
extern TerminalIO tio;

InputReader::InputReader()
    : replxx(nullptr)
    , running(false)
    , should_stop(false)
    , interactive_mode(false)
    , colors_enabled(false)
    , on_input(nullptr) {
}

InputReader::~InputReader() {
    stop();
    // Note: replxx is owned by TerminalIO, not us - don't call replxx_end here
}

bool InputReader::init(bool interactive, bool colors, InputCallback callback) {
    interactive_mode = interactive;
    colors_enabled = colors;
    on_input = callback;

    // Use replxx from TerminalIO (shared instance for coordinated I/O)
    if (interactive_mode && !tio.tui_mode) {
        replxx = tio.get_replxx();
        if (!replxx) {
            LOG_ERROR("TerminalIO replxx not initialized");
            return false;
        }
    }

    return true;
}

void InputReader::start() {
    if (running) return;

    should_stop = false;
    running = true;
    reader_thread = std::thread(&InputReader::reader_loop, this);
    LOG_DEBUG("InputReader thread started");
}

void InputReader::stop() {
    should_stop = true;

    // Wake up thread if it's paused waiting on pause_cv
    {
        std::lock_guard<std::mutex> lock(pause_mutex);
        prompting_paused = false;
    }
    pause_cv.notify_one();

    // For interactive mode, we need to interrupt replxx_input
    // Send a newline to unblock it
    if (interactive_mode && replxx) {
        replxx_emulate_key_press(replxx, '\n');
    }

    // Always join the thread if joinable (it may have exited naturally on EOF)
    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    running = false;
    LOG_DEBUG("InputReader thread stopped");
}

void InputReader::pause_prompting() {
    prompting_paused = true;
}

void InputReader::resume_prompting() {
    {
        std::lock_guard<std::mutex> lock(pause_mutex);
        prompting_paused = false;
    }
    pause_cv.notify_one();
}

void InputReader::reader_loop() {
    while (!should_stop) {
        // Wait if prompting is paused (during generation)
        if (interactive_mode && prompting_paused) {
            std::unique_lock<std::mutex> lock(pause_mutex);
            pause_cv.wait(lock, [this] { return !prompting_paused || should_stop; });
            if (should_stop) break;
        }

        std::string line;

        if (interactive_mode) {
            line = read_line_interactive();
        } else {
            line = read_line_piped();
        }

        if (should_stop) break;

        // Empty line from interactive means continue prompting
        // Empty line from piped means EOF
        if (line.empty()) {
            if (!interactive_mode) {
                // EOF on stdin - signal with empty string
                if (on_input) {
                    on_input("");
                }
                break;
            }
            continue;
        }

        // Skip blank lines
        if (is_blank(line)) {
            continue;
        }

        // Call the callback to queue the input
        if (on_input) {
            on_input(line);
        }

        // In interactive non-TUI mode, pause after submitting input
        // Main loop will resume when ready for next input
        if (interactive_mode && !g_tui_screen) {
            prompting_paused = true;
        }
    }

    running = false;
}

std::string InputReader::read_line_interactive() {
    // In TUI mode, we don't use a prompt - the input box handles it
    std::string prompt;
    if (g_tui_screen) {
        // Just position cursor - box is already drawn by TUIScreen::refresh()
        g_tui_screen->position_cursor_in_input(0);
        prompt = "";  // No prompt - box already has "> "
    } else {
        if (colors_enabled) {
            prompt = "\033[32m> \033[0m";
        } else {
            prompt = "> ";
        }
    }

    const char* input = replxx_input(replxx, prompt.c_str());
    if (input == nullptr) {
        // EOF (Ctrl+D) or error
        return "";
    }

    std::string line = input;

    // Add to history if non-empty
    if (!line.empty() && !is_blank(line)) {
        replxx_history_add(replxx, input);

        // In TUI mode, clear the input box for next input
        // (echo happens in CLI when input is consumed)
        if (g_tui_screen) {
            g_tui_screen->set_input_content("");
        }
    }

    return line;
}

std::string InputReader::read_line_piped() {
    std::string line;
    if (!std::getline(std::cin, line)) {
        // EOF
        return "";
    }
    return line;
}

bool InputReader::is_blank(const std::string& str) const {
    for (char c : str) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

void InputReader::history_add(const std::string& line) {
    if (replxx && !line.empty()) {
        replxx_history_add(replxx, line.c_str());
    }
}

void InputReader::history_load(const std::string& path) {
    if (replxx) {
        replxx_history_load(replxx, path.c_str());
    }
}

void InputReader::history_save(const std::string& path) {
    if (replxx) {
        replxx_history_save(replxx, path.c_str());
    }
}
