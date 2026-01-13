
#include "shepherd.h"
#include "backend.h"
#include <unordered_map>

// LaTeX symbol to Unicode mapping
static const std::unordered_map<std::string, std::string> latex_symbols = {
    // Greek letters
    {"\\alpha", "α"}, {"\\beta", "β"}, {"\\gamma", "γ"}, {"\\delta", "δ"},
    {"\\epsilon", "ε"}, {"\\zeta", "ζ"}, {"\\eta", "η"}, {"\\theta", "θ"},
    {"\\iota", "ι"}, {"\\kappa", "κ"}, {"\\lambda", "λ"}, {"\\mu", "μ"},
    {"\\nu", "ν"}, {"\\xi", "ξ"}, {"\\pi", "π"}, {"\\rho", "ρ"},
    {"\\sigma", "σ"}, {"\\tau", "τ"}, {"\\upsilon", "υ"}, {"\\phi", "φ"},
    {"\\chi", "χ"}, {"\\psi", "ψ"}, {"\\omega", "ω"},
    {"\\Gamma", "Γ"}, {"\\Delta", "Δ"}, {"\\Theta", "Θ"}, {"\\Lambda", "Λ"},
    {"\\Xi", "Ξ"}, {"\\Pi", "Π"}, {"\\Sigma", "Σ"}, {"\\Phi", "Φ"},
    {"\\Psi", "Ψ"}, {"\\Omega", "Ω"},

    // Arrows
    {"\\to", "→"}, {"\\rightarrow", "→"}, {"\\leftarrow", "←"},
    {"\\leftrightarrow", "↔"}, {"\\Rightarrow", "⇒"}, {"\\Leftarrow", "⇐"},
    {"\\Leftrightarrow", "⇔"}, {"\\mapsto", "↦"}, {"\\uparrow", "↑"},
    {"\\downarrow", "↓"}, {"\\nearrow", "↗"}, {"\\searrow", "↘"},
    {"\\xrightarrow", "→"}, {"\\xleftarrow", "←"},  // Extended arrows (ignore annotation)

    // Operators and relations
    {"\\times", "×"}, {"\\div", "÷"}, {"\\pm", "±"}, {"\\mp", "∓"},
    {"\\cdot", "·"}, {"\\ast", "∗"}, {"\\star", "⋆"},
    {"\\leq", "≤"}, {"\\le", "≤"}, {"\\geq", "≥"}, {"\\ge", "≥"},
    {"\\neq", "≠"}, {"\\ne", "≠"}, {"\\approx", "≈"}, {"\\equiv", "≡"},
    {"\\sim", "∼"}, {"\\simeq", "≃"}, {"\\cong", "≅"},
    {"\\subset", "⊂"}, {"\\supset", "⊃"}, {"\\subseteq", "⊆"}, {"\\supseteq", "⊇"},
    {"\\in", "∈"}, {"\\notin", "∉"}, {"\\ni", "∋"},
    {"\\cap", "∩"}, {"\\cup", "∪"}, {"\\setminus", "∖"},
    {"\\land", "∧"}, {"\\lor", "∨"}, {"\\neg", "¬"}, {"\\lnot", "¬"},
    {"\\forall", "∀"}, {"\\exists", "∃"}, {"\\nexists", "∄"},

    // Brackets and delimiters
    {"\\langle", "⟨"}, {"\\rangle", "⟩"},
    {"\\lfloor", "⌊"}, {"\\rfloor", "⌋"},
    {"\\lceil", "⌈"}, {"\\rceil", "⌉"},
    {"\\|", "‖"},

    // Miscellaneous
    {"\\infty", "∞"}, {"\\partial", "∂"}, {"\\nabla", "∇"},
    {"\\sum", "∑"}, {"\\prod", "∏"}, {"\\int", "∫"},
    {"\\sqrt", "√"}, {"\\angle", "∠"}, {"\\degree", "°"},
    {"\\circ", "∘"}, {"\\bullet", "•"}, {"\\ldots", "…"}, {"\\cdots", "⋯"},
    {"\\vdots", "⋮"}, {"\\ddots", "⋱"},
    {"\\emptyset", "∅"}, {"\\varnothing", "∅"},
    {"\\prime", "′"}, {"\\dagger", "†"}, {"\\ddagger", "‡"},
    {"\\ell", "ℓ"}, {"\\hbar", "ℏ"}, {"\\Re", "ℜ"}, {"\\Im", "ℑ"},

    // Spacing (convert to space or empty)
    {"\\,", " "}, {"\\;", " "}, {"\\:", " "}, {"\\!", ""},
    {"\\quad", "  "}, {"\\qquad", "    "},
    {"\\ ", " "},

    // Display/inline math delimiters (remove entirely)
    {"\\[", ""}, {"\\]", ""}, {"\\(", ""}, {"\\)", ""},

    // Common escapes
    {"\\{", "{"}, {"\\}", "}"}, {"\\%", "%"}, {"\\$", "$"},
    {"\\&", "&"}, {"\\#", "#"},

    // Structural commands (remove, let brace content flow through)
    {"\\frac", ""}, {"\\text", ""}, {"\\textbf", ""}, {"\\textit", ""},
    {"\\mathrm", ""}, {"\\mathbf", ""}, {"\\mathit", ""}, {"\\mathcal", ""},
    {"\\mathbb", ""}, {"\\left", ""}, {"\\right", ""}, {"\\big", ""},
    {"\\Big", ""}, {"\\bigg", ""}, {"\\Bigg", ""},
};

Backend::Backend(size_t ctx_size, Session& session, EventCallback cb)
    : context_size(ctx_size), callback(cb) {
    if (!callback) {
        throw std::invalid_argument("Internal error: callback not passed to backend constructor");
    }
    // Disable LaTeX conversion in server mode - UIs like OpenWebUI render LaTeX themselves
    convert_latex = !g_server_mode;
}

// Flush the output buffer to callback
// Returns true to continue, false if callback requested cancellation
bool Backend::flush_filter_buffer() {
    if (output_buffer.empty()) return true;

    CallbackEvent event = in_code_block ? CallbackEvent::CODEBLOCK : CallbackEvent::CONTENT;
    bool cont = callback(event, output_buffer, "", "");
    output_buffer.clear();
    return cont;
}

// Reset filter state (called before new generation)
void Backend::reset_output_state() {
    clear_tool_calls();  // Reset accumulated_tool_calls
    backtick_buffer.clear();
    output_buffer.clear();
    in_code_block = false;
    skip_to_newline = false;
    think_tag_buffer.clear();
    in_think_block = false;
    latex_buffer.clear();
    latex_brace_depth = 0;
    utf8_incomplete.clear();
}

// Returns the length of valid UTF-8 in text
// If result < text.size(), there's an incomplete multi-byte char at the end
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check last 1-3 bytes for incomplete sequences
    for (size_t i = 1; i <= 3 && i <= len; i++) {
        unsigned char c = text[len - i];
        if ((c & 0x80) == 0) {
            // ASCII byte - all previous bytes are complete
            return len;
        } else if ((c & 0xC0) == 0xC0) {
            // Start of multi-byte sequence
            int expected_len;
            if ((c & 0xE0) == 0xC0) expected_len = 2;
            else if ((c & 0xF0) == 0xE0) expected_len = 3;
            else if ((c & 0xF8) == 0xF0) expected_len = 4;
            else return len;  // Invalid, treat as complete

            // Check if we have all the bytes
            if (i >= (size_t)expected_len) {
                return len;  // Complete
            } else {
                return len - i;  // Incomplete, return valid portion
            }
        }
        // Continuation byte (10xxxxxx) - keep looking for start byte
    }
    return len;  // No incomplete sequence found
}

// Try to match latex_buffer against known symbols
// Returns (found, unicode) - found=true means in table (even if unicode is empty)
static std::pair<bool, std::string> lookup_latex(const std::string& seq) {
    auto it = latex_symbols.find(seq);
    if (it != latex_symbols.end()) {
        return {true, it->second};
    }
    return {false, ""};
}

// Find longest prefix of seq that matches a known symbol
// Returns (unicode, chars_consumed) or ("", 0) if no match
static std::pair<std::string, size_t> lookup_latex_prefix(const std::string& seq) {
    // Try longest to shortest
    for (size_t len = seq.length(); len >= 2; len--) {
        std::string prefix = seq.substr(0, len);
        auto it = latex_symbols.find(prefix);
        if (it != latex_symbols.end()) {
            return {it->second, len};
        }
    }
    return {"", 0};
}

// Flush any pending output (called at end of generation)
void Backend::flush_output() {
    // Flush any partial backticks as content
    if (!backtick_buffer.empty()) {
        output_buffer += backtick_buffer;
        backtick_buffer.clear();
    }
    // Flush any partial latex as content
    if (!latex_buffer.empty()) {
        output_buffer += latex_buffer;
        latex_buffer.clear();
    }
    flush_filter_buffer();
}

// Common output filtering - handles backticks, think blocks, and buffering
// Returns true to continue, false if callback requested cancellation
bool Backend::filter(const char* text, size_t len) {
    // Handle incomplete UTF-8 from previous call
    std::string combined;
    if (!utf8_incomplete.empty()) {
        dout(2) << "filter: prepending " << utf8_incomplete.length() << " incomplete UTF-8 bytes" << std::endl;
        combined = utf8_incomplete + std::string(text, len);
        utf8_incomplete.clear();
        text = combined.c_str();
        len = combined.length();
    }

    // Check for incomplete UTF-8 at the end
    size_t valid_len = validate_utf8(std::string(text, len));
    if (valid_len < len) {
        // Buffer incomplete bytes for next call
        dout(2) << "filter: buffering " << (len - valid_len) << " incomplete UTF-8 bytes at end" << std::endl;
        utf8_incomplete = std::string(text + valid_len, len - valid_len);
        len = valid_len;
        if (len == 0) return true;  // Nothing complete yet
    }

    // Debug: log high bytes (non-ASCII)
    for (size_t i = 0; i < len; i++) {
        unsigned char uc = static_cast<unsigned char>(text[i]);
        if (uc >= 0x80) {
            dout(2) << "filter: byte " << i << " = 0x" << std::hex << (int)uc << std::dec << std::endl;
        }
    }

    for (size_t i = 0; i < len; i++) {
        char c = text[i];

        // Think block detection: look for <think> and </think>
        if (c == '<' || !think_tag_buffer.empty()) {
            think_tag_buffer += c;

            // Check for complete tags
            if (think_tag_buffer == "<think>") {
                in_think_block = true;
                think_tag_buffer.clear();
                // If thinking enabled, output the tag; otherwise suppress
                if (config && config->thinking) {
                    output_buffer += "<think>";
                }
                continue;
            } else if (think_tag_buffer == "</think>") {
                in_think_block = false;
                think_tag_buffer.clear();
                // If thinking enabled, output the tag; otherwise suppress
                if (config && config->thinking) {
                    output_buffer += "</think>";
                }
                continue;
            } else if (think_tag_buffer.length() < 8) {
                // Could still be <think> or </think>, keep buffering
                // Check if it could still match
                const char* open_tag = "<think>";
                const char* close_tag = "</think>";
                bool could_match_open = (think_tag_buffer.length() <= 7 &&
                    think_tag_buffer == std::string(open_tag, think_tag_buffer.length()));
                bool could_match_close = (think_tag_buffer.length() <= 8 &&
                    think_tag_buffer == std::string(close_tag, think_tag_buffer.length()));
                if (could_match_open || could_match_close) {
                    continue;  // Keep buffering
                }
            }

            // Doesn't match - flush tag buffer as content (unless in suppressed think block)
            if (!in_think_block || (config && config->thinking)) {
                output_buffer += think_tag_buffer;
            }
            think_tag_buffer.clear();
            continue;
        }

        // If in think block and thinking disabled, suppress content
        if (in_think_block && config && !config->thinking) {
            continue;
        }

        // LaTeX to Unicode conversion (skip inside code blocks)
        // convert_latex is set false in server mode (UIs like OpenWebUI render LaTeX themselves)
        if (!in_code_block && convert_latex) {
            // If we have a pending latex command and see a new backslash, finalize it first
            if (c == '\\' && !latex_buffer.empty()) {
                auto it = latex_symbols.find(latex_buffer);
                if (it != latex_symbols.end()) {
                    output_buffer += it->second;
                } else {
                    auto [prefix_unicode, consumed] = lookup_latex_prefix(latex_buffer);
                    if (consumed > 0) {
                        output_buffer += prefix_unicode;
                        output_buffer += latex_buffer.substr(consumed);
                    } else {
                        output_buffer += latex_buffer;
                    }
                }
                latex_buffer.clear();
            }

            // Start of potential LaTeX sequence
            if (c == '\\' && latex_buffer.empty()) {
                latex_buffer = "\\";
                continue;
            }

            // Track braces after LaTeX commands (skip { and } but keep content)
            if (latex_brace_depth > 0) {
                if (c == '{') {
                    latex_brace_depth++;
                    continue;  // Skip opening brace
                } else if (c == '}') {
                    latex_brace_depth--;
                    continue;  // Skip closing brace
                }
                // Content inside braces flows through normally (but LaTeX inside is still processed above)
            }

            // Accumulating LaTeX sequence
            if (!latex_buffer.empty()) {
                // Check what kind of sequence this could be
                if (latex_buffer.length() == 1) {
                    // Just saw backslash - check for single-char escapes or start of command
                    if (c == '[' || c == ']' || c == '(' || c == ')' ||
                        c == '{' || c == '}' || c == '%' || c == '$' ||
                        c == '&' || c == '#' || c == '|' || c == ' ' ||
                        c == ',' || c == ';' || c == ':' || c == '!') {
                        // Single-char escape - look it up
                        latex_buffer += c;
                        auto [found, unicode] = lookup_latex(latex_buffer);
                        if (found) {
                            output_buffer += unicode;  // May be empty (delete sequence)
                        } else {
                            output_buffer += latex_buffer;  // Not in table, emit as-is
                        }
                        latex_buffer.clear();
                        continue;
                    } else if (std::isalpha(c)) {
                        // Start of command name
                        latex_buffer += c;
                        continue;
                    } else {
                        // Not a valid LaTeX sequence, flush backslash
                        output_buffer += latex_buffer;
                        latex_buffer.clear();
                        // Fall through to process current char normally
                    }
                } else {
                    // In a command name - keep accumulating letters
                    if (std::isalpha(c) && latex_buffer.length() < 20) {
                        latex_buffer += c;
                        continue;
                    }

                    // End of command name (non-alpha or too long) - find best match
                    auto it = latex_symbols.find(latex_buffer);
                    if (it != latex_symbols.end()) {
                        // Exact match
                        output_buffer += it->second;
                    } else {
                        // Try prefix match (e.g., \textIndex -> \text + Index)
                        auto [prefix_unicode, consumed] = lookup_latex_prefix(latex_buffer);
                        if (consumed > 0) {
                            output_buffer += prefix_unicode;
                            output_buffer += latex_buffer.substr(consumed);
                        } else {
                            output_buffer += latex_buffer;  // No match, emit as-is
                        }
                    }
                    latex_buffer.clear();

                    // Start tracking braces after commands
                    if (c == '{') {
                        latex_brace_depth = 1;
                        continue;
                    }
                    // Fall through to process current char normally
                }
            }
        } else {
            // Inside code block - flush any pending latex state
            if (!latex_buffer.empty()) {
                output_buffer += latex_buffer;
                latex_buffer.clear();
            }
            latex_brace_depth = 0;
        }

        // Track backticks for code block detection (``` toggles code block mode)
        // Suppress the ``` markers entirely from output
        if (c == '`') {
            backtick_buffer += c;
            if (backtick_buffer.length() >= 3) {
                in_code_block = !in_code_block;
                backtick_buffer.clear();
                // Skip language identifier after opening ```
                if (in_code_block) {
                    skip_to_newline = true;
                }
            }
            continue;  // Don't add backticks to output yet
        } else {
            // If we had partial backticks (1 or 2), flush them as content
            if (!backtick_buffer.empty()) {
                output_buffer += backtick_buffer;
                backtick_buffer.clear();
            }
        }

        // Skip language identifier on code block opening line
        if (skip_to_newline) {
            if (c == '\n') {
                skip_to_newline = false;
            }
            continue;
        }

        // Buffer regular output
        output_buffer += c;
    }

    // Flush after processing all input (stream per-token, not per-line)
    if (!output_buffer.empty()) {
        if (!flush_filter_buffer()) {
            return false;  // Cancelled
        }
    }
    return true;
}
