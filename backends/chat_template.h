#pragma once

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "models.h"
#include "message.h"
#include "session.h"

namespace ChatTemplates {

// Capability flags discovered by probing the template
struct ChatTemplateCaps {
    bool supports_tools = false;           // Can render tools array
    bool supports_tool_calls = false;      // Can format tool_calls in assistant messages
    bool supports_tool_responses = false;  // Can handle role="tool" messages
    bool supports_system_role = false;     // Has system message support
    bool requires_object_arguments = false; // Tool args as object vs string
    bool probed = false;                   // Whether probing has been done

    // Channel-based output (detected from template, not hardcoded)
    bool has_channels = false;             // Template uses channel-based output format
    std::string channel_extract_marker;    // e.g., "<|channel|>final<|message|>" - extracted from template
    std::string channel_end_marker;        // e.g., "<|end|>" - extracted from template

    // Thinking mode support
    bool supports_enable_thinking = false; // Template responds to enable_thinking parameter
    std::string generation_prompt_base;    // Base generation prompt (for injecting think block)
};

class ChatTemplate {
public:
    virtual ~ChatTemplate() = default;

    // Format a single message with role and content (legacy - for hardcoded templates)
    virtual std::string format_message(const Message& msg) const = 0;

    // Format system message with optional tools
    virtual std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const = 0;

    // Get the assistant generation prompt (e.g., "<|im_start|>assistant\n")
    virtual std::string get_generation_prompt() const = 0;

    // Get the assistant end tag (e.g., "<|im_end|>\n")
    virtual std::string get_assistant_end_tag() const = 0;

    // Get the model family this template is for
    virtual ModelFamily get_family() const = 0;

    // NEW: Format entire conversation at once (for full template rendering)
    virtual std::string format_conversation(
        const std::vector<Message>& messages,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const {
        // Default implementation for hardcoded templates: concatenate individual messages
        std::string result;
        for (const auto& msg : messages) {
            if (msg.role == Message::SYSTEM) {
                result += format_system_message(msg.content, tools);
            } else {
                result += format_message(msg);
            }
        }
        if (add_generation_prompt) {
            result += get_generation_prompt();
        }
        return result;
    }

    // NEW: Format a single message incrementally using full conversation context
    // Renders full conversation twice and returns the diff (like llama.cpp)
    virtual std::string format_message_incremental(
        const std::vector<Message>& all_messages,
        size_t target_index,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const {
        // Default: use format_conversation with diff
        if (target_index == 0) {
            // First message - just render it
            std::vector<Message> single(all_messages.begin(), all_messages.begin() + 1);
            return format_conversation(single, tools, add_generation_prompt && target_index == all_messages.size() - 1);
        }

        // Render prefix (messages before target)
        std::vector<Message> prefix(all_messages.begin(), all_messages.begin() + target_index);
        std::string prefix_str = format_conversation(prefix, tools, false);

        // Render full (including target)
        std::vector<Message> full(all_messages.begin(), all_messages.begin() + target_index + 1);
        std::string full_str = format_conversation(full, tools, add_generation_prompt);

        // Return the diff
        if (full_str.length() > prefix_str.length()) {
            return full_str.substr(prefix_str.length());
        }
        return "";
    }

    // NEW: Probe template capabilities (default: assume all features supported)
    virtual void probe_capabilities() { caps.probed = true; caps.supports_system_role = true; }

    // NEW: Get probed capabilities
    virtual const ChatTemplateCaps& get_capabilities() const { return caps; }

    // Extract content from raw model output (handles channel-based models)
    // For most models: returns raw_output unchanged
    // For channel-based models: extracts content from final channel
    virtual std::string extract_content(const std::string& raw_output) const {
        if (!caps.has_channels || caps.channel_extract_marker.empty()) {
            return raw_output;  // Pass-through for non-channel models
        }

        // Find the final channel marker
        size_t pos = raw_output.find(caps.channel_extract_marker);
        if (pos == std::string::npos) {
            // For channel-based models, no "final" channel means tool call only
            // Return empty string - the tool call is handled separately
            return "";
        }

        // Extract content after the marker
        std::string content = raw_output.substr(pos + caps.channel_extract_marker.length());

        // Check for end markers - content ends at <|end|>, <|return|>, or <|start|> (new turn)
        // Take the earliest one found as the end boundary
        size_t end_pos = std::string::npos;
        size_t end1 = content.find("<|end|>");
        size_t end2 = content.find("<|return|>");
        size_t end3 = content.find("<|start|>");  // New turn = end of current content

        // Find minimum of all valid positions
        if (end1 != std::string::npos) end_pos = end1;
        if (end2 != std::string::npos && (end_pos == std::string::npos || end2 < end_pos)) end_pos = end2;
        if (end3 != std::string::npos && (end_pos == std::string::npos || end3 < end_pos)) end_pos = end3;

        if (end_pos != std::string::npos) {
            content = content.substr(0, end_pos);
        }

        return content;
    }

protected:
    ChatTemplateCaps caps;
};

class ChatMLTemplate : public ChatTemplate {
public:
    ChatMLTemplate() = default;

    std::string format_message(const Message& msg) const override;
    std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const override;
    std::string get_generation_prompt() const override;
    std::string get_assistant_end_tag() const override;
    ModelFamily get_family() const override;
};

// Qwen3 thinking models - adds empty think block when thinking is disabled
class Qwen3ThinkingTemplate : public ChatMLTemplate {
public:
    Qwen3ThinkingTemplate() = default;

    std::string get_generation_prompt() const override;
    ModelFamily get_family() const override;
};

class Llama2Template : public ChatTemplate {
public:
    Llama2Template() = default;

    std::string format_message(const Message& msg) const override;
    std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const override;
    std::string get_generation_prompt() const override;
    std::string get_assistant_end_tag() const override;
    ModelFamily get_family() const override;
};

class Llama3Template : public ChatTemplate {
public:
    Llama3Template() = default;

    std::string format_message(const Message& msg) const override;
    std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const override;
    std::string get_generation_prompt() const override;
    std::string get_assistant_end_tag() const override;
    ModelFamily get_family() const override;
};

class GLM4Template : public ChatTemplate {
public:
    GLM4Template() = default;

    std::string format_message(const Message& msg) const override;
    std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const override;
    std::string get_generation_prompt() const override;
    std::string get_assistant_end_tag() const override;
    ModelFamily get_family() const override;
};

class MinjaTemplate : public ChatTemplate {
public:
    MinjaTemplate(const std::string& template_text, void* template_node_ptr,
                  const std::string& eos_token = "", const std::string& bos_token = "");
    ~MinjaTemplate() override;

    std::string format_message(const Message& msg) const override;
    std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const override;
    std::string get_generation_prompt() const override;
    std::string get_assistant_end_tag() const override;
    ModelFamily get_family() const override;

    // Override to use minja template for full conversation rendering
    std::string format_conversation(
        const std::vector<Message>& messages,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const override;

    // Override with caching for efficient incremental rendering
    std::string format_message_incremental(
        const std::vector<Message>& all_messages,
        size_t target_index,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const override;

    // Probe template to discover supported features
    void probe_capabilities() override;

    // Clear the rendering cache (call when conversation changes non-incrementally)
    void clear_cache();

    // Template text and node (public for inspection if needed)
    std::string template_text;
    void* template_node;
    std::string eos_token;
    std::string bos_token;

private:
    // Helper to render messages through minja
    std::string render_via_minja(
        const std::vector<Message>& messages,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const;

    // Helper to apply polyfills for missing features
    std::vector<Message> apply_polyfills(
        const std::vector<Message>& messages,
        const std::vector<Session::Tool>& tools) const;

    // Helper for probe rendering (catches exceptions)
    std::string try_render(
        const std::vector<Message>& messages,
        const std::vector<Session::Tool>& tools,
        bool add_generation_prompt) const;

    // Cache for incremental rendering
    mutable std::string cached_prefix;
    mutable size_t cached_message_count = 0;
};

class ChatTemplateFactory {
public:
    static std::unique_ptr<ChatTemplate> create(const std::string& template_text,
                                                  const ModelConfig& config,
                                                  void* template_node_ptr = nullptr,
                                                  const std::string& eos_token = "",
                                                  const std::string& bos_token = "");
};

} // namespace ChatTemplates
