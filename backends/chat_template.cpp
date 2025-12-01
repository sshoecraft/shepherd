#include "chat_template.h"
#include "logger.h"
#include "nlohmann/json.hpp"
#include <ctime>

// Minja includes
#include "llama.cpp/vendor/minja/minja.hpp"

// External config for thinking setting
#include "config.h"
extern std::unique_ptr<Config> config;

namespace ChatTemplates {

// ==================== ChatMLTemplate Implementation ====================

std::string ChatMLTemplate::format_message(const Message& msg) const {
    return "<|im_start|>" + msg.get_role() + "\n" + msg.content + "<|im_end|>\n";
}

std::string ChatMLTemplate::format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const {
    std::string formatted = "<|im_start|>system\n";

    if (content.empty()) {
        formatted += "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.";
    } else {
        formatted += content;
    }

    // Add tools if present
    if (!tools.empty()) {
        formatted += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n";
        formatted += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";

        for (const auto& tool : tools) {
            nlohmann::json tool_json;
            tool_json["type"] = "function";
            tool_json["function"]["name"] = tool.name;
            tool_json["function"]["description"] = tool.description;
            tool_json["function"]["parameters"] = tool.parameters;
            formatted += tool_json.dump() + "\n";
        }

        formatted += "</tools>\n\n";
        formatted += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n";
        formatted += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
    }

    formatted += "<|im_end|>\n";
    return formatted;
}

std::string ChatMLTemplate::get_generation_prompt() const {
    return "<|im_start|>assistant\n";
}

std::string ChatMLTemplate::get_assistant_end_tag() const {
    return "<|im_end|>\n";
}

ModelFamily ChatMLTemplate::get_family() const {
    return ModelFamily::QWEN_2_X;
}

// ==================== Qwen3ThinkingTemplate Implementation ====================

std::string Qwen3ThinkingTemplate::get_generation_prompt() const {
    // When thinking is disabled (config->thinking == false), add empty think block
    // to tell the model "thinking is done, just respond"
    // This matches llama-server behavior with --reasoning-format auto
    if (config && !config->thinking) {
        LOG_DEBUG("Qwen3ThinkingTemplate: thinking disabled, returning empty think block");
        return "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    // When thinking is enabled, let the model do its thinking
    LOG_DEBUG("Qwen3ThinkingTemplate: thinking enabled (config=" + std::string(config ? "valid" : "null") +
              ", thinking=" + std::string(config ? (config->thinking ? "true" : "false") : "n/a") + ")");
    return "<|im_start|>assistant\n";
}

ModelFamily Qwen3ThinkingTemplate::get_family() const {
    return ModelFamily::QWEN_3_X;
}

// ==================== Llama2Template Implementation ====================

std::string Llama2Template::format_message(const Message& msg) const {
    // Llama 2 uses [INST] tags for user messages
    if (msg.type == Message::USER) {
        return "[INST] " + msg.content + " [/INST]";
    } else if (msg.type == Message::ASSISTANT) {
        return msg.content + "</s>";
    } else if (msg.type == Message::TOOL) {
        // Tool results as user context
        return "[INST] Tool result: " + msg.content + " [/INST]";
    }
    return msg.content;
}

std::string Llama2Template::format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const {
    std::string system_content;

    if (content.empty()) {
        system_content = "You are a helpful assistant.";
    } else {
        system_content = content;
    }

    if (!tools.empty()) {
        system_content += "\n\nAvailable tools:\n";
        for (const auto& tool : tools) {
            system_content += "- " + tool.name + ": " + tool.description + "\n";
        }
        system_content += "\nTo use a tool, respond with JSON: {\"name\": \"tool_name\", \"parameters\": {...}}\n";
    }

    // Llama 2 system format uses <<SYS>> tags
    return "<s>[INST] <<SYS>>\n" + system_content + "\n<</SYS>>\n\n";
}

std::string Llama2Template::get_generation_prompt() const {
    return "";  // Llama 2 generation follows directly after [/INST]
}

std::string Llama2Template::get_assistant_end_tag() const {
    return "</s>";
}

ModelFamily Llama2Template::get_family() const {
    return ModelFamily::LLAMA_2_X;
}

// ==================== Llama3Template Implementation ====================

std::string Llama3Template::format_message(const Message& msg) const {
    std::string role = msg.get_role();

    // Llama 3 uses "ipython" role for tool results
    if (msg.type == Message::TOOL) {
        role = "ipython";
    }

    return "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + msg.content + "<|eot_id|>";
}

std::string Llama3Template::format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const {
    std::string system_content;

    if (content.empty()) {
        system_content = "You are a helpful assistant.";
    } else {
        system_content = content;
    }

    // Add JSON schemas for tools (BFCL-style)
    if (!tools.empty()) {
        system_content += "\n\nThe following functions are available IF needed to answer the user's request:\n\n";
        system_content += "IMPORTANT: Only call a function if you actually need external information or capabilities. ";
        system_content += "For greetings, casual conversation, or questions you can answer directly - respond normally without calling any function.\n\n";
        system_content += "When you DO need to call a function, respond with ONLY a JSON object in this format: ";
        system_content += "{\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\n";

        for (const auto& tool : tools) {
            system_content += "- " + tool.name + ": " + tool.description + "\n";
            if (!tool.parameters.empty()) {
                system_content += "  Parameters: " + tool.parameters.dump() + "\n";
            }
        }
    }

    return "<|start_header_id|>system<|end_header_id|>\n\n" + system_content + "<|eot_id|>";
}

std::string Llama3Template::get_generation_prompt() const {
    return "<|start_header_id|>assistant<|end_header_id|>\n\n";
}

std::string Llama3Template::get_assistant_end_tag() const {
    return "<|eot_id|>";
}

ModelFamily Llama3Template::get_family() const {
    return ModelFamily::LLAMA_3_X;
}

// ==================== GLM4Template Implementation ====================

std::string GLM4Template::format_message(const Message& msg) const {
    std::string role = msg.get_role();

    // GLM-4 uses "observation" role for tool results
    if (msg.type == Message::TOOL) {
        role = "observation";
    }

    return "<|" + role + "|>\n" + msg.content;
}

std::string GLM4Template::format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const {
    std::string system_content = "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n";
    system_content += "IMPORTANT: Always respond in English unless the user specifically requests another language.\n\n";

    if (!content.empty()) {
        system_content += content + "\n\n";
    }

    // Add tools in Chinese format
    if (!tools.empty()) {
        system_content += "# 可用工具\n\n";
        system_content += "**重要提示**: 读取大文件时，请使用 `head` 或 `tail` 参数只读取部分内容（例如前100行或后100行），以避免超出上下文窗口限制。\n\n";

        for (const auto& tool : tools) {
            system_content += "## " + tool.name + "\n";
            system_content += tool.description + "\n";
            system_content += "在调用上述函数时，请使用扁平 Json 格式：{\"name\": \"" + tool.name + "\", \"parameters\": {...}}\n\n";
        }
    }

    return "<|system|>\n" + system_content;
}

std::string GLM4Template::get_generation_prompt() const {
    return "<|assistant|>\n";
}

std::string GLM4Template::get_assistant_end_tag() const {
    return "";  // GLM-4 doesn't use an end tag
}

ModelFamily GLM4Template::get_family() const {
    return ModelFamily::GLM_4;
}

// ==================== MinjaTemplate Implementation ====================

MinjaTemplate::MinjaTemplate(const std::string& template_text, void* template_node_ptr,
                             const std::string& eos_tok, const std::string& bos_tok)
    : template_text(template_text), template_node(template_node_ptr),
      eos_token(eos_tok), bos_token(bos_tok) {
}

MinjaTemplate::~MinjaTemplate() {
    // template_node is owned by LlamaCppBackend, don't delete here
}

std::string MinjaTemplate::format_message(const Message& msg) const {
    // For assistant messages, use raw content (no template)
    // Templates are for formatting prompts, not parsing model outputs
    if (msg.type == Message::ASSISTANT) {
        return msg.content;
    }

    if (!template_node) {
        throw std::runtime_error("MinjaTemplate::format_message() called but template_node is null");
    }

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

    // Create minja context
    auto context = minja::Context::builtins();

    // Add date/time support
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        if (args.args.empty()) return minja::Value("");
        std::string format = args.args[0].get<std::string>();
        time_t now = time(nullptr);
        struct tm* tm_info = localtime(&now);
        char buffer[128];
        strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
        return minja::Value(std::string(buffer));
    });
    context->set("strftime_now", strftime_now);

    time_t now = time(nullptr);
    struct tm* tm_info = localtime(&now);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%d %b %Y", tm_info);
    context->set("date_string", minja::Value(std::string(date_buffer)));

    // Build messages array with single message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();
    msg_obj.set("role", minja::Value(msg.get_role()));
    msg_obj.set("content", minja::Value(msg.content));
    messages.push_back(msg_obj);

    context->set("messages", messages);
    context->set("bos_token", minja::Value(bos_token));
    context->set("eos_token", minja::Value(eos_token));
    context->set("add_generation_prompt", minja::Value(false));

    // Render through template
    std::string rendered = (*template_ptr)->render(context);

    LOG_DEBUG("MinjaTemplate rendered message: " + std::to_string(rendered.length()) + " chars");
    return rendered;
}

std::string MinjaTemplate::format_system_message(const std::string& content, const std::vector<Session::Tool>& tools) const {
    if (!template_node) {
        throw std::runtime_error("MinjaTemplate::format_system_message() called but template_node is null");
    }

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

    // Create minja context
    auto context = minja::Context::builtins();

    // Add date/time support
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        if (args.args.empty()) return minja::Value("");
        std::string format = args.args[0].get<std::string>();
        time_t now = time(nullptr);
        struct tm* tm_info = localtime(&now);
        char buffer[128];
        strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
        return minja::Value(std::string(buffer));
    });
    context->set("strftime_now", strftime_now);

    time_t now = time(nullptr);
    struct tm* tm_info = localtime(&now);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%d %b %Y", tm_info);
    context->set("date_string", minja::Value(std::string(date_buffer)));

    // Build system message
    auto messages = minja::Value::array();
    auto sys_msg = minja::Value::object();
    sys_msg.set("role", minja::Value("system"));
    sys_msg.set("content", minja::Value(content.empty() ? "You are a helpful assistant." : content));
    messages.push_back(sys_msg);

    context->set("messages", messages);
    context->set("bos_token", minja::Value(bos_token));
    context->set("eos_token", minja::Value(eos_token));
    context->set("add_generation_prompt", minja::Value(false));

    // Add tools if present
    if (!tools.empty()) {
        auto tools_array = minja::Value::array();
        for (const auto& tool : tools) {
            auto tool_obj = minja::Value::object();
            tool_obj.set("name", minja::Value(tool.name));
            tool_obj.set("description", minja::Value(tool.description));

            // Convert nlohmann::json to minja::Value by round-tripping through string
            // This avoids the ambiguous json type conflict
            tool_obj.set("parameters", minja::Value(nlohmann::ordered_json::parse(tool.parameters.dump())));

            tools_array.push_back(tool_obj);
        }
        context->set("tools", tools_array);
    }

    // Render through template
    std::string rendered = (*template_ptr)->render(context);

    LOG_DEBUG("MinjaTemplate rendered system message: " + std::to_string(rendered.length()) + " chars");
    return rendered;
}

std::string MinjaTemplate::get_generation_prompt() const {
    if (!template_node) {
        throw std::runtime_error("MinjaTemplate::get_generation_prompt() called but template_node is null");
    }

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

    // Render template with a dummy message twice:
    // once without add_generation_prompt, once with.
    // The difference is the generation prompt.
    auto context_without = minja::Context::make(minja::Value::object());
    auto context_with = minja::Context::make(minja::Value::object());

    // Build minimal messages array with single user message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();
    msg_obj.set("role", minja::Value("user"));
    msg_obj.set("content", minja::Value("x"));
    messages.push_back(msg_obj);

    context_without->set("messages", messages);
    context_without->set("bos_token", minja::Value(bos_token));
    context_without->set("eos_token", minja::Value(eos_token));
    context_without->set("add_generation_prompt", minja::Value(false));

    context_with->set("messages", messages);
    context_with->set("bos_token", minja::Value(bos_token));
    context_with->set("eos_token", minja::Value(eos_token));
    context_with->set("add_generation_prompt", minja::Value(true));

    std::string without_prompt = (*template_ptr)->render(context_without);
    std::string with_prompt = (*template_ptr)->render(context_with);

    // The generation prompt is the additional text
    if (with_prompt.length() > without_prompt.length() &&
        with_prompt.substr(0, without_prompt.length()) == without_prompt) {
        return with_prompt.substr(without_prompt.length());
    }

    throw std::runtime_error("MinjaTemplate::get_generation_prompt() failed to extract generation prompt from template");
}

std::string MinjaTemplate::get_assistant_end_tag() const {
    // For minja templates, the end tag is typically the eos_token
    // If eos_token wasn't provided, use a reasonable ChatML-style default
    if (eos_token.empty()) {
        return "<|im_end|>";
    }
    return eos_token;
}

ModelFamily MinjaTemplate::get_family() const {
    return ModelFamily::GENERIC;
}

// ==================== ChatTemplateFactory Implementation ====================

std::unique_ptr<ChatTemplate> ChatTemplateFactory::create(const std::string& template_text,
                                                            const ModelConfig& config,
                                                            void* template_node_ptr,
                                                            const std::string& eos_token,
                                                            const std::string& bos_token) {
    LOG_DEBUG("ChatTemplateFactory creating template for family: " + std::to_string(static_cast<int>(config.family)));

    switch (config.family) {
        case ModelFamily::QWEN_2_X:
            LOG_INFO("Creating ChatMLTemplate for Qwen 2.x");
            return std::make_unique<ChatMLTemplate>();

        case ModelFamily::QWEN_3_X:
            // Use thinking template for Qwen 3.x models that support thinking
            if (config.supports_thinking_mode) {
                LOG_INFO("Creating Qwen3ThinkingTemplate for thinking model");
                return std::make_unique<Qwen3ThinkingTemplate>();
            }
            LOG_INFO("Creating ChatMLTemplate for Qwen 3.x (non-thinking)");
            return std::make_unique<ChatMLTemplate>();

        case ModelFamily::LLAMA_2_X:
            // If model has a jinja template, use MinjaTemplate with tokens
            // Otherwise use hardcoded Llama2Template
            if (!template_text.empty() && template_node_ptr) {
                LOG_INFO("Creating MinjaTemplate for Llama 2.x with custom template");
                return std::make_unique<MinjaTemplate>(template_text, template_node_ptr, eos_token, bos_token);
            }
            LOG_INFO("Creating Llama2Template for Llama 2.x");
            return std::make_unique<Llama2Template>();

        case ModelFamily::LLAMA_3_X:
            LOG_INFO("Creating Llama3Template");
            return std::make_unique<Llama3Template>();

        case ModelFamily::GLM_4:
            LOG_INFO("Creating GLM4Template");
            return std::make_unique<GLM4Template>();

        case ModelFamily::MISTRAL:
        case ModelFamily::GEMMA:
        case ModelFamily::DEEPSEEK:
        case ModelFamily::COMMAND_R:
        case ModelFamily::PHI_3:
        case ModelFamily::FUNCTIONARY:
        case ModelFamily::GENERIC:
        default:
            LOG_INFO("Creating MinjaTemplate for unknown/generic family");
            return std::make_unique<MinjaTemplate>(template_text, template_node_ptr, eos_token, bos_token);
    }
}

} // namespace ChatTemplates
