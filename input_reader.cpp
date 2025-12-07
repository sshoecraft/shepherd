#include "input_reader.h"
#include "logger.h"
#include "vendor/replxx/include/replxx.h"
#include <iostream>
#include <cstring>

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
    if (replxx) {
        replxx_end(replxx);
        replxx = nullptr;
    }
}

bool InputReader::init(bool interactive, bool colors, InputCallback callback) {
    interactive_mode = interactive;
    colors_enabled = colors;
    on_input = callback;

    if (interactive_mode) {
        replxx = replxx_init();
        if (!replxx) {
            LOG_ERROR("Failed to initialize replxx");
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

void InputReader::start() {
    if (running) return;

    should_stop = false;
    running = true;
    reader_thread = std::thread(&InputReader::reader_loop, this);
    LOG_DEBUG("InputReader thread started");
}

void InputReader::stop() {
    if (!running) return;

    should_stop = true;

    // For interactive mode, we need to interrupt replxx_input
    // Send a newline to unblock it
    if (interactive_mode && replxx) {
        replxx_emulate_key_press(replxx, '\n');
    }

    if (reader_thread.joinable()) {
        reader_thread.join();
    }

    running = false;
    LOG_DEBUG("InputReader thread stopped");
}

void InputReader::reader_loop() {
    while (!should_stop) {
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
    }

    running = false;
}

std::string InputReader::read_line_interactive() {
    std::string colored_prompt;
    if (colors_enabled) {
        colored_prompt = "\033[32m> \033[0m";
    } else {
        colored_prompt = "> ";
    }

    const char* input = replxx_input(replxx, colored_prompt.c_str());
    if (input == nullptr) {
        // EOF (Ctrl+D) or error
        return "";
    }

    std::string line = input;

    // Add to history if non-empty
    if (!line.empty() && !is_blank(line)) {
        replxx_history_add(replxx, input);
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
