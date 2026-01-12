#include "chat_template.h"
#include "shepherd.h"
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
    return "<|im_start|>assistant\n";
}

ModelFamily Qwen3ThinkingTemplate::get_family() const {
    return ModelFamily::QWEN_3_X;
}

// ==================== Llama2Template Implementation ====================

std::string Llama2Template::format_message(const Message& msg) const {
    // Llama 2 uses [INST] tags for user messages
    if (msg.role == Message::USER) {
        return "[INST] " + msg.content + " [/INST]";
    } else if (msg.role == Message::ASSISTANT) {
        return msg.content + "</s>";
    } else if (msg.role == Message::TOOL_RESPONSE) {
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
    if (msg.role == Message::TOOL_RESPONSE) {
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
    if (msg.role == Message::TOOL_RESPONSE) {
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
    if (msg.role == Message::ASSISTANT) {
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

    dout(2) << "MinjaTemplate rendered message: " + std::to_string(rendered.length()) + " chars" << std::endl;
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

    // Add tools if present - use OpenAI format with type/function wrapper
    // Most model templates expect this format (e.g., Mistral, GPT-OSS)
    if (!tools.empty()) {
        auto tools_array = minja::Value::array();
        for (const auto& tool : tools) {
            // Create the inner function object
            auto func_obj = minja::Value::object();
            func_obj.set("name", minja::Value(tool.name));
            func_obj.set("description", minja::Value(tool.description));

            // Convert nlohmann::json to minja::Value by round-tripping through string
            // This avoids the ambiguous json type conflict
            func_obj.set("parameters", minja::Value(nlohmann::ordered_json::parse(tool.parameters.dump())));

            // Wrap in OpenAI format: {"type": "function", "function": {...}}
            auto tool_obj = minja::Value::object();
            tool_obj.set("type", minja::Value("function"));
            tool_obj.set("function", func_obj);

            tools_array.push_back(tool_obj);
        }
        context->set("tools", tools_array);
    }

    // Render through template
    std::string rendered = (*template_ptr)->render(context);

    dout(2) << "MinjaTemplate rendered system message: " + std::to_string(rendered.length()) + " chars" << std::endl;
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

    // Add date/time support for templates that use strftime_now
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        std::string format = args.args.empty() ? "%Y-%m-%d" : args.args[0].get<std::string>();
        std::time_t now = std::time(nullptr);
        std::tm* tm_info = std::localtime(&now);
        char buffer[128];
        strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
        return minja::Value(std::string(buffer));
    });

    context_without->set("messages", messages);
    context_without->set("bos_token", minja::Value(bos_token));
    context_without->set("eos_token", minja::Value(eos_token));
    context_without->set("add_generation_prompt", minja::Value(false));
    context_without->set("strftime_now", strftime_now);

    context_with->set("messages", messages);
    context_with->set("bos_token", minja::Value(bos_token));
    context_with->set("eos_token", minja::Value(eos_token));
    context_with->set("add_generation_prompt", minja::Value(true));
    context_with->set("strftime_now", strftime_now);

    std::string without_prompt = (*template_ptr)->render(context_without);
    std::string with_prompt = (*template_ptr)->render(context_with);

    // The generation prompt is the additional text
    std::string gen_prompt;
    if (with_prompt.length() > without_prompt.length() &&
        with_prompt.substr(0, without_prompt.length()) == without_prompt) {
        gen_prompt = with_prompt.substr(without_prompt.length());
    } else {
        throw std::runtime_error("MinjaTemplate::get_generation_prompt() failed to extract generation prompt from template");
    }

    // Thinking suppression: inject empty think block to skip thinking phase
    // Model sees <think>\n\n</think> and believes it already "thought", continues from there
    if (config && !config->thinking && template_text.find("</think>") != std::string::npos) {
        gen_prompt += "<think>\n\n</think>\n\n";
        dout(1) << "Injected empty <think> block into generation prompt to suppress thinking" << std::endl;
    }

    // Harmony/GPT-OSS: reasoning controlled via reasoning_effort parameter, not injection
    // See: https://huggingface.co/blog/welcome-openai-gpt-oss#system-and-developer-messages

    return gen_prompt;
}

std::string MinjaTemplate::get_assistant_end_tag() const {
    // For minja templates, the end tag is the eos_token from the model
    // If eos_token wasn't provided, return empty (don't hardcode a fallback)
    return eos_token;
}

ModelFamily MinjaTemplate::get_family() const {
    return ModelFamily::GENERIC;
}

void MinjaTemplate::clear_cache() {
    cached_prefix.clear();
    cached_message_count = 0;
}

std::string MinjaTemplate::try_render(
    const std::vector<Message>& messages,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) const {
    try {
        return render_via_minja(messages, tools, add_generation_prompt);
    } catch (const std::exception& e) {
        dout(1) << "MinjaTemplate::try_render exception: " + std::string(e.what()) << std::endl;
        return "";
    }
}

std::string MinjaTemplate::render_via_minja(
    const std::vector<Message>& messages,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) const {

    if (!template_node) {
        throw std::runtime_error("MinjaTemplate::render_via_minja() called but template_node is null");
    }

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

    // Create minja context with builtins
    auto context = minja::Context::builtins();

    // Add date/time support
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        std::string format = args.args.empty() ? "%Y-%m-%d" : args.args[0].get<std::string>();
        std::time_t now = std::time(nullptr);
        std::tm* tm_info = std::localtime(&now);
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

    // Build messages array
    int msgs_with_tool_calls = 0;
    for (const auto& msg : messages) {
        if (!msg.tool_calls_json.empty()) msgs_with_tool_calls++;
    }
    dout(2) << "render_via_minja: " << messages.size() << " messages, " << msgs_with_tool_calls << " with tool_calls" << std::endl;

    auto msgs_array = minja::Value::array();
    for (const auto& msg : messages) {
        auto msg_obj = minja::Value::object();

        // Map role
        std::string role = msg.get_role();
        if (msg.role == Message::TOOL_RESPONSE) {
            role = "tool";  // Standard role for tool responses
        }
        msg_obj.set("role", minja::Value(role));
        msg_obj.set("content", minja::Value(msg.content));

        // Add tool metadata if present
        if (!msg.tool_name.empty()) {
            msg_obj.set("name", minja::Value(msg.tool_name));
        }
        if (!msg.tool_call_id.empty()) {
            msg_obj.set("tool_call_id", minja::Value(msg.tool_call_id));
        }

        // Add tool_calls if present (for assistant messages with tool calls)
        // tool_calls_json is a JSON string containing tool calls array
        if (!msg.tool_calls_json.empty()) {
            try {
                auto tool_calls_parsed = nlohmann::json::parse(msg.tool_calls_json);
                if (tool_calls_parsed.is_array()) {
                    auto tool_calls_array = minja::Value::array();
                    for (const auto& tc : tool_calls_parsed) {
                        auto tc_obj = minja::Value::object();
                        if (tc.contains("id")) {
                            tc_obj.set("id", minja::Value(tc["id"].get<std::string>()));
                        }
                        tc_obj.set("type", minja::Value("function"));

                        if (tc.contains("function")) {
                            auto func_obj = minja::Value::object();
                            std::string func_name;
                            if (tc["function"].contains("name")) {
                                func_name = tc["function"]["name"].get<std::string>();
                                func_obj.set("name", minja::Value(func_name));
                            }
                            // Always set arguments at tool_call level for templates that expect it
                            // (like Qwen3-Coder which uses tool_call.arguments|items)
                            // Parse the arguments string into an actual JSON object for minja
                            minja::Value args_obj = minja::Value::object();  // Default empty

                            if (tc["function"].contains("arguments")) {
                                std::string args_str = tc["function"]["arguments"].get<std::string>();

                                // Parse arguments string as JSON and convert to minja Value
                                // IMPORTANT: Set parsed object on BOTH func_obj AND tc_obj
                                // Some templates (like Qwen) do: {%- set tool_call = tool_call.function %}
                                // which means tool_call.arguments accesses function.arguments
                                try {
                                    auto parsed = nlohmann::ordered_json::parse(args_str);
                                    if (parsed.is_object()) {
                                        args_obj = minja::Value(parsed);
                                        // Set on BOTH locations for template compatibility
                                        func_obj.set("arguments", args_obj);
                                        dout(2) << "Parsed tool arguments as object: " << args_str << std::endl;
                                    } else {
                                        // Non-object JSON (array, primitive) - keep as string
                                        func_obj.set("arguments", minja::Value(args_str));
                                    }
                                } catch (const std::exception& e) {
                                    // Parse failed - keep as string
                                    func_obj.set("arguments", minja::Value(args_str));
                                    dout(2) << "Tool arguments kept as string: " << args_str << std::endl;
                                }
                            }
                            tc_obj.set("arguments", args_obj);
                            tc_obj.set("function", func_obj);

                            // Also set name at top-level for templates that use tool_call.name
                            if (!func_name.empty()) {
                                tc_obj.set("name", minja::Value(func_name));
                            }
                        }

                        tool_calls_array.push_back(tc_obj);
                    }
                    msg_obj.set("tool_calls", tool_calls_array);
                }
            } catch (const std::exception& e) {
                dout(1) << "Failed to parse tool_calls_json: " + std::string(e.what()) << std::endl;
            }
        }

        msgs_array.push_back(msg_obj);
    }

    context->set("messages", msgs_array);
    context->set("bos_token", minja::Value(bos_token));
    context->set("eos_token", minja::Value(eos_token));
    context->set("add_generation_prompt", minja::Value(add_generation_prompt));

    // Pass thinking parameters to template - different models use different names
    // Qwen3 uses "enable_thinking", DeepSeek uses "thinking"
    bool thinking_enabled = config ? config->thinking : true;
    context->set("enable_thinking", minja::Value(thinking_enabled));
    context->set("thinking", minja::Value(thinking_enabled));

    // GPT-OSS reasoning_effort: high when thinking enabled, low when disabled
    // From: https://huggingface.co/blog/welcome-openai-gpt-oss#system-and-developer-messages
    std::string reasoning_effort = thinking_enabled ? "high" : "low";
    context->set("reasoning_effort", minja::Value(reasoning_effort));
    dout(1) << "GPT-OSS reasoning_effort set to: " << reasoning_effort << std::endl;

    // Add tools if present
    if (!tools.empty()) {
        auto tools_array = minja::Value::array();
        for (const auto& tool : tools) {
            auto func_obj = minja::Value::object();
            func_obj.set("name", minja::Value(tool.name));
            func_obj.set("description", minja::Value(tool.description));
            func_obj.set("parameters", minja::Value(nlohmann::ordered_json::parse(tool.parameters.dump())));

            auto tool_obj = minja::Value::object();
            tool_obj.set("type", minja::Value("function"));
            tool_obj.set("function", func_obj);

            tools_array.push_back(tool_obj);
        }
        context->set("tools", tools_array);
    }

    // Render through template
    std::string rendered = (*template_ptr)->render(context);

    // Thinking suppression: inject end marker to skip thinking phase
    // If template references </think>, this model supports thinking mode
    dout(1) << "Thinking check: add_gen=" << add_generation_prompt
            << ", config=" << (config ? "yes" : "null")
            << ", thinking=" << (config ? (config->thinking ? "true" : "false") : "n/a") << std::endl;
    if (add_generation_prompt && config && !config->thinking) {
        if (template_text.find("</think>") != std::string::npos) {
            // Append </think> to signal "thinking done, respond directly"
            rendered += "</think>";
            dout(1) << "Injected </think> to suppress thinking mode" << std::endl;
        }
    }

    return rendered;
}

std::vector<Message> MinjaTemplate::apply_polyfills(
    const std::vector<Message>& messages,
    const std::vector<Session::Tool>& tools) const {

    std::vector<Message> adjusted = messages;

    // Polyfill: Inject tools into system message if template doesn't support tools natively
    if (!caps.supports_tools && !tools.empty()) {
        std::string tool_description = "\n\nAvailable tools:\n";
        for (const auto& tool : tools) {
            tool_description += "- " + tool.name + ": " + tool.description + "\n";
            if (!tool.parameters.empty()) {
                tool_description += "  Parameters: " + tool.parameters.dump() + "\n";
            }
        }
        tool_description += "\nTo use a tool, respond with JSON: {\"name\": \"tool_name\", \"parameters\": {...}}\n";

        if (!adjusted.empty() && adjusted[0].role == Message::SYSTEM) {
            adjusted[0].content += tool_description;
        } else {
            Message sys(Message::SYSTEM, "You are a helpful assistant." + tool_description, 0);
            adjusted.insert(adjusted.begin(), sys);
        }
    }

    // Polyfill: Merge system into first user message if no system role support
    if (!caps.supports_system_role && !adjusted.empty() &&
        adjusted[0].role == Message::SYSTEM) {
        std::string sys_content = adjusted[0].content;
        adjusted.erase(adjusted.begin());
        if (!adjusted.empty() && adjusted[0].role == Message::USER) {
            adjusted[0].content = sys_content + "\n\n" + adjusted[0].content;
        }
    }

    // Polyfill: Convert tool responses to user messages if not supported
    if (!caps.supports_tool_responses) {
        for (auto& msg : adjusted) {
            if (msg.role == Message::TOOL_RESPONSE) {
                msg.role = Message::USER;
                msg.content = "Tool result for " + msg.tool_name + ": " + msg.content;
            }
        }
    }

    return adjusted;
}

std::string MinjaTemplate::format_conversation(
    const std::vector<Message>& messages,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) const {

    // Apply polyfills if capabilities have been probed
    std::vector<Message> adjusted_messages;
    if (caps.probed) {
        adjusted_messages = apply_polyfills(messages, tools);
    } else {
        adjusted_messages = messages;
    }

    // Pass empty tools if we applied polyfills (tools already injected into system message)
    std::vector<Session::Tool> render_tools;
    if (caps.probed && !caps.supports_tools) {
        // Tools already polyfilled into system message
        render_tools = {};
    } else {
        render_tools = tools;
    }

    try {
        return render_via_minja(adjusted_messages, render_tools, add_generation_prompt);
    } catch (const std::exception& e) {
        dout(1) << "MinjaTemplate::format_conversation exception: " << e.what() << std::endl;
        throw std::runtime_error("Template rendering failed: " + std::string(e.what()));
    }
}

std::string MinjaTemplate::format_message_incremental(
    const std::vector<Message>& all_messages,
    size_t target_index,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) const {

    // Check if we can use cached prefix
    bool cache_valid = (target_index == cached_message_count);

    std::string prefix;
    if (cache_valid && target_index > 0) {
        prefix = cached_prefix;
        dout(2) << "MinjaTemplate: using cached prefix (" + std::to_string(prefix.length()) + " chars)" << std::endl;
    } else {
        if (target_index > 0) {
            // Render messages[0..target_index-1]
            std::vector<Message> prefix_msgs(all_messages.begin(), all_messages.begin() + target_index);
            prefix = format_conversation(prefix_msgs, tools, false);
            dout(2) << "MinjaTemplate: rendered prefix (" + std::to_string(prefix.length()) + " chars)" << std::endl;
        } else {
            prefix = "";
        }
    }

    // Render messages[0..target_index] with generation prompt if needed
    std::vector<Message> full_msgs(all_messages.begin(), all_messages.begin() + target_index + 1);
    std::string full = format_conversation(full_msgs, tools, add_generation_prompt);

    // Update cache for next call
    cached_prefix = full;
    cached_message_count = target_index + 1;

    // Return just the new part
    if (full.length() > prefix.length()) {
        std::string diff = full.substr(prefix.length());
        dout(2) << "MinjaTemplate: incremental diff (" + std::to_string(diff.length()) + " chars)" << std::endl;
        return diff;
    }

    dout(1) << "MinjaTemplate: WARNING - full not longer than prefix, returning empty" << std::endl;
    return "";
}

void MinjaTemplate::probe_capabilities() {
    if (caps.probed) {
        return;  // Already probed
    }

    dout(1) << "MinjaTemplate: probing template capabilities..." << std::endl;

    const std::string user_needle = "<<USER_NEEDLE_12345>>";
    const std::string sys_needle = "<<SYSTEM_NEEDLE_67890>>";
    const std::string tool_needle = "test_probe_tool";
    const std::string tool_result_needle = "<<TOOL_RESULT_NEEDLE>>";

    // Test system role support
    {
        Message sys(Message::SYSTEM, sys_needle, 0);
        Message user(Message::USER, user_needle, 0);
        std::vector<Message> msgs = {sys, user};
        std::string rendered = try_render(msgs, {}, false);
        caps.supports_system_role = (rendered.find(sys_needle) != std::string::npos);
        dout(1) << "  supports_system_role: " + std::string(caps.supports_system_role ? "true" : "false") << std::endl;
    }

    // Test tools support
    {
        Message user(Message::USER, user_needle, 0);
        std::vector<Message> msgs = {user};

        Session::Tool test_tool;
        test_tool.name = tool_needle;
        test_tool.description = "A test tool for probing";
        test_tool.parameters = nlohmann::json::object();
        test_tool.parameters["type"] = "object";

        std::string rendered = try_render(msgs, {test_tool}, false);
        caps.supports_tools = (rendered.find(tool_needle) != std::string::npos);
        dout(1) << "  supports_tools: " + std::string(caps.supports_tools ? "true" : "false") << std::endl;
    }

    // Test tool_calls in assistant messages
    {
        Message user(Message::USER, user_needle, 0);
        Message asst(Message::ASSISTANT, "", 0);
        // Build tool_calls_json as a proper JSON array
        nlohmann::json tc_json = nlohmann::json::array();
        nlohmann::json tc_obj;
        tc_obj["id"] = "call_probe_123";
        tc_obj["type"] = "function";
        tc_obj["function"]["name"] = tool_needle;
        tc_obj["function"]["arguments"] = "{\"arg\": 1}";
        tc_json.push_back(tc_obj);
        asst.tool_calls_json = tc_json.dump();

        std::vector<Message> msgs = {user, asst};
        std::string rendered = try_render(msgs, {}, false);
        caps.supports_tool_calls = (rendered.find(tool_needle) != std::string::npos);
        dout(1) << "  supports_tool_calls: " + std::string(caps.supports_tool_calls ? "true" : "false") << std::endl;
    }

    // Test tool response messages
    {
        Message user(Message::USER, user_needle, 0);
        Message asst(Message::ASSISTANT, "", 0);
        // Build tool_calls_json
        nlohmann::json tc_json = nlohmann::json::array();
        nlohmann::json tc_obj;
        tc_obj["id"] = "call_probe_123";
        tc_obj["type"] = "function";
        tc_obj["function"]["name"] = tool_needle;
        tc_obj["function"]["arguments"] = "{}";
        tc_json.push_back(tc_obj);
        asst.tool_calls_json = tc_json.dump();

        Message tool_resp(Message::TOOL_RESPONSE, tool_result_needle, 0);
        tool_resp.tool_name = tool_needle;
        tool_resp.tool_call_id = "call_probe_123";

        std::vector<Message> msgs = {user, asst, tool_resp};
        std::string rendered = try_render(msgs, {}, false);
        caps.supports_tool_responses = (rendered.find(tool_result_needle) != std::string::npos);
        dout(1) << "  supports_tool_responses: " + std::string(caps.supports_tool_responses ? "true" : "false") << std::endl;
    }

    // Detect channel-based output format by examining the template text
    // Look for patterns like "<|channel|>final" and "<|channel|>analysis"
    {
        caps.has_channels = false;
        caps.channel_extract_marker.clear();
        caps.channel_end_marker.clear();

        // Check if template uses channel markers
        if (template_text.find("<|channel|>") != std::string::npos) {
            // Template has channel markers - check for common patterns
            if (template_text.find("<|channel|>final") != std::string::npos ||
                template_text.find("channel|>final") != std::string::npos) {
                caps.has_channels = true;
                caps.channel_extract_marker = "<|channel|>final<|message|>";

                // Detect end marker from template
                if (template_text.find("<|end|>") != std::string::npos) {
                    caps.channel_end_marker = "<|end|>";
                } else if (template_text.find("<|return|>") != std::string::npos) {
                    caps.channel_end_marker = "<|return|>";
                }

                dout(1) << "  has_channels: true (detected channel-based output format)" << std::endl;
                dout(1) << "  channel_extract_marker: " + caps.channel_extract_marker << std::endl;
                dout(1) << "  channel_end_marker: " + caps.channel_end_marker << std::endl;
            }
        }

        if (!caps.has_channels) {
            dout(1) << "  has_channels: false" << std::endl;
        }
    }

    // Test enable_thinking support by rendering with both values
    // Also extract the base generation prompt for potential injection
    {
        caps.supports_enable_thinking = false;
        caps.generation_prompt_base.clear();

        if (!template_node) {
            dout(1) << "  supports_enable_thinking: false (no template node)" << std::endl;
        } else {
            auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

            // Build minimal context for probing
            auto messages = minja::Value::array();
            auto msg_obj = minja::Value::object();
            msg_obj.set("role", minja::Value("user"));
            msg_obj.set("content", minja::Value("test"));
            messages.push_back(msg_obj);

            auto context_without = minja::Context::make(minja::Value::object());
            auto context_with = minja::Context::make(minja::Value::object());

            // Add strftime_now for templates that use it (like GPT-OSS)
            auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
                std::string format = args.args.empty() ? "%Y-%m-%d" : args.args[0].get<std::string>();
                std::time_t now = std::time(nullptr);
                std::tm* tm_info = std::localtime(&now);
                char buffer[128];
                strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
                return minja::Value(std::string(buffer));
            });

            context_without->set("messages", messages);
            context_without->set("bos_token", minja::Value(bos_token));
            context_without->set("eos_token", minja::Value(eos_token));
            context_without->set("add_generation_prompt", minja::Value(true));
            context_without->set("enable_thinking", minja::Value(false));
            context_without->set("thinking", minja::Value(false));  // DeepSeek uses "thinking"
            context_without->set("strftime_now", strftime_now);

            context_with->set("messages", messages);
            context_with->set("bos_token", minja::Value(bos_token));
            context_with->set("eos_token", minja::Value(eos_token));
            context_with->set("add_generation_prompt", minja::Value(true));
            context_with->set("enable_thinking", minja::Value(true));
            context_with->set("thinking", minja::Value(true));  // DeepSeek uses "thinking"
            context_with->set("strftime_now", strftime_now);

            try {
                std::string rendered_without = (*template_ptr)->render(context_without);
                std::string rendered_with = (*template_ptr)->render(context_with);

                // Template supports enable_thinking if outputs differ
                caps.supports_enable_thinking = (rendered_without != rendered_with);

                // Extract generation prompt for potential injection
                // Render without generation prompt to get base, then with to get full
                auto context_no_gen = minja::Context::make(minja::Value::object());
                context_no_gen->set("messages", messages);
                context_no_gen->set("bos_token", minja::Value(bos_token));
                context_no_gen->set("eos_token", minja::Value(eos_token));
                context_no_gen->set("add_generation_prompt", minja::Value(false));
                context_no_gen->set("enable_thinking", minja::Value(true));
                context_no_gen->set("thinking", minja::Value(true));  // DeepSeek uses "thinking"
                context_no_gen->set("strftime_now", strftime_now);

                std::string without_gen = (*template_ptr)->render(context_no_gen);
                if (rendered_with.length() > without_gen.length() &&
                    rendered_with.substr(0, without_gen.length()) == without_gen) {
                    caps.generation_prompt_base = rendered_with.substr(without_gen.length());
                }

                dout(1) << "  supports_enable_thinking: " + std::string(caps.supports_enable_thinking ? "true" : "false") << std::endl;
                if (!caps.generation_prompt_base.empty()) {
                    dout(1) << "  generation_prompt_base: " + caps.generation_prompt_base.substr(0, 50) + "..." << std::endl;
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Chat template error during capability probing: " + std::string(e.what()));
            }
        }
    }

    caps.probed = true;
    dout(1) << "MinjaTemplate: capability probing complete" << std::endl;
}

// ==================== ChatTemplateFactory Implementation ====================

std::unique_ptr<ChatTemplate> ChatTemplateFactory::create(const std::string& template_text, const ModelConfig& config, void* template_node_ptr, const std::string& eos_token, const std::string& bos_token) {
    dout(2) << "ChatTemplateFactory creating template for family: " + std::to_string(static_cast<int>(config.family)) << std::endl;

    // NEW: Prioritize MinjaTemplate when model has embedded Jinja template
    // This ensures proper full-conversation rendering for any model
    if (!template_text.empty() && template_node_ptr) {
        // Special case: Qwen 3.x thinking models need special generation prompt handling
        // that MinjaTemplate doesn't provide, so use hardcoded template
        if (config.family == ModelFamily::QWEN_3_X && config.supports_thinking_mode) {
            dout(1) << "Creating Qwen3ThinkingTemplate for thinking model (special handling required)" << std::endl;
            return std::make_unique<Qwen3ThinkingTemplate>();
        }

        dout(1) << "Creating MinjaTemplate with model's embedded Jinja template" << std::endl;
        return std::make_unique<MinjaTemplate>(template_text, template_node_ptr, eos_token, bos_token);
    }

    // FALLBACK: Use hardcoded templates only when no embedded template available
    switch (config.family) {
        case ModelFamily::QWEN_2_X:
            dout(1) << "Creating ChatMLTemplate for Qwen 2.x (no embedded template)" << std::endl;
            return std::make_unique<ChatMLTemplate>();

        case ModelFamily::QWEN_3_X:
            if (config.supports_thinking_mode) {
                dout(1) << "Creating Qwen3ThinkingTemplate for thinking model (no embedded template)" << std::endl;
                return std::make_unique<Qwen3ThinkingTemplate>();
            }
            dout(1) << "Creating ChatMLTemplate for Qwen 3.x (no embedded template)" << std::endl;
            return std::make_unique<ChatMLTemplate>();

        case ModelFamily::LLAMA_2_X:
            dout(1) << "Creating Llama2Template (no embedded template)" << std::endl;
            return std::make_unique<Llama2Template>();

        case ModelFamily::LLAMA_3_X:
            dout(1) << "Creating Llama3Template (no embedded template)" << std::endl;
            return std::make_unique<Llama3Template>();

        case ModelFamily::GLM_4:
            dout(1) << "Creating GLM4Template (no embedded template)" << std::endl;
            return std::make_unique<GLM4Template>();

        case ModelFamily::MISTRAL:
        case ModelFamily::GEMMA:
        case ModelFamily::DEEPSEEK:
        case ModelFamily::COMMAND_R:
        case ModelFamily::PHI_3:
        case ModelFamily::FUNCTIONARY:
        case ModelFamily::GENERIC:
        default:
            // No template and unknown family - create MinjaTemplate anyway
            // It will fail gracefully if template_node is null
            dout(1) << "Creating MinjaTemplate for unknown/generic family (no embedded template)" << std::endl;
            return std::make_unique<MinjaTemplate>(template_text, template_node_ptr, eos_token, bos_token);
    }
}

} // namespace ChatTemplates
