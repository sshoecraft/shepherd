
#pragma once

#include "backend.h"
#include "parser.h"
#include <memory>
#include <vector>
#include <string>

// Forward declarations
namespace ChatTemplates {
    struct ChatTemplateCaps;
}


// GpuBackend: Base class for GPU/local backends (llamacpp, tensorrt)
// Provides output filtering, harmony parsing, and chat template support
// API backends inherit directly from Backend (via ApiBackend) and don't need these features
class GpuBackend : public Backend {
public:
    GpuBackend(size_t context_size, Session& session, EventCallback callback);
    virtual ~GpuBackend();

    // Tool and thinking tag markers - GPU backends override with model-specific markers
    // (inherited from Backend, overridden here to emphasize GPU backends provide actual markers)
    std::vector<std::string> get_tool_call_markers() const override { return {}; }
    std::vector<std::string> get_tool_call_end_markers() const override { return {}; }
    std::vector<std::string> get_thinking_start_markers() const override { return {}; }
    std::vector<std::string> get_thinking_end_markers() const override { return {}; }

    // Get chat template capabilities - GPU backends override with actual capabilities
    const ChatTemplates::ChatTemplateCaps* get_chat_template_caps() const override { return nullptr; }

protected:
    // Streaming parser for output processing (GenericParser or HarmonyParser)
    // Created in reset_output_state() based on model capabilities
    std::unique_ptr<StreamParser::Parser> parser;

    // Output function - harmony parsing + GPU tag filtering, then calls filter()
    // Entry point for GPU backends - LlamaCpp/TensorRT call this for each generated token
    // Returns true to continue, false if callback requested cancellation
    bool output(const char* text, size_t len);
    bool output(const std::string& text) { return output(text.c_str(), text.length()); }

    // Reset filter state between requests - overrides Backend
    void reset_output_state() override;

    // Flush any pending output at end of response - overrides Backend
    void flush_output() override;

private:
    // Filtering state machine
    enum FilterState {
        FILTER_NORMAL,
        FILTER_DETECTING_TAG,
        FILTER_IN_THINKING,
        FILTER_IN_TOOL_CALL,
        FILTER_CHECKING_CLOSE
    };

    FilterState filter_state = FILTER_NORMAL;
    bool in_tool_call = false;
    bool in_thinking = false;
    // in_code_block, skip_to_newline, backtick_buffer, output_buffer inherited from Backend
    int json_brace_depth = 0;

    std::string tag_buffer;
    std::string current_tag;
    std::string buffered_tool_call;
    std::string buffered_thinking;
    std::string utf8_buffer;  // Buffer for incomplete UTF-8 sequences

    // Marker vectors (cached from virtual methods)
    std::vector<std::string> tool_call_start_markers;
    std::vector<std::string> tool_call_end_markers;
    std::vector<std::string> thinking_start_markers;
    std::vector<std::string> thinking_end_markers;
    bool markers_initialized = false;

    // Channel parsing flag (harmony parsing now done in LlamaCppBackend)
    bool channelParsingEnabled = false;

    // Filter helpers
    void ensure_markers_initialized();
    bool matches_any(const std::string& buffer, const std::vector<std::string>& markers, std::string* matched = nullptr) const;
    bool could_match_any(const std::string& buffer, const std::vector<std::string>& markers) const;
    void emit_tool_call();
};
