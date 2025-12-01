#pragma once

#include <string>
#include <memory>
#include <vector>
#include "models.h"
#include "message.h"
#include "session.h"

namespace ChatTemplates {

class ChatTemplate {
public:
    virtual ~ChatTemplate() = default;

    // Format a single message with role and content
    virtual std::string format_message(const Message& msg) const = 0;

    // Format system message with optional tools
    virtual std::string format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const = 0;

    // Get the assistant generation prompt (e.g., "<|im_start|>assistant\n")
    virtual std::string get_generation_prompt() const = 0;

    // Get the assistant end tag (e.g., "<|im_end|>\n")
    virtual std::string get_assistant_end_tag() const = 0;

    // Get the model family this template is for
    virtual ModelFamily get_family() const = 0;
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

private:
    std::string template_text;
    void* template_node;
    std::string eos_token;
    std::string bos_token;
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
