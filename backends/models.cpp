
#include "models.h"
#include "logger.h"
#include "tools/tool.h"
#include "nlohmann/json.hpp"
#include "minja.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <ctime>

ModelConfig Models::detect_from_chat_template(const std::string& template_text, const std::string& model_path) {
    // Primary detection: Analyze chat template content (most reliable)
    ModelConfig config = detect_from_template_content(template_text, model_path);

    // If template detection succeeded, return it
    if (config.family != ModelFamily::GENERIC) {
        return config;
    }

    // Fallback: Try path-based detection if model_path provided
    if (!model_path.empty()) {
        LOG_DEBUG("Chat template detection failed, trying path analysis");
        return detect_from_path_analysis(model_path);
    }

    // Final fallback: Generic model
    LOG_WARN("Could not detect specific model family, using generic configuration");
    return ModelConfig::create_generic();
}

ModelConfig Models::detect_from_model_path(const std::string& model_path) {
    return detect_from_path_analysis(model_path);
}

ModelConfig Models::detect_from_model_name(const std::string& model_name) {
    // For now, treat API model names same as paths
    // Could be extended with API-specific detection logic
    return detect_from_path_analysis(model_name);
}

ModelConfig Models::detect_from_template_content(const std::string& template_text, const std::string& model_path) {
    if (template_text.empty()) {
        LOG_DEBUG("Empty chat template, cannot detect from content");
        return ModelConfig::create_generic();
    }

    LOG_DEBUG("Detecting model family from chat template...");

    // Llama 3.x: Has "Environment: ipython" and <|eom_id|>
    if (template_text.find("Environment: ipython") != std::string::npos &&
        template_text.find("<|eom_id|>") != std::string::npos) {
        LOG_INFO("Detected Llama 3.x model family from chat template");

        // Try to extract version from model path
        std::string version = extract_version_from_path(model_path, "3.");
        if (version.empty()) {
            version = "3.1"; // Default
        }

        return ModelConfig::create_llama_3x(version);
    }

    // GLM-4.x: Has <|observation|> role
    if (template_text.find("<|observation|>") != std::string::npos) {
        LOG_INFO("Detected GLM-4.x model family from chat template");

        std::string version = extract_version_from_path(model_path, "4.");
        if (version.empty()) {
            version = "4";
        }

        return ModelConfig::create_glm_4(version);
    }

    // Qwen 2.x: Has <|im_start|> and <|im_end|>
    if (template_text.find("<|im_start|>") != std::string::npos &&
        template_text.find("<|im_end|>") != std::string::npos) {
        LOG_INFO("Detected Qwen 2.x model family from chat template");

        std::string version = extract_version_from_path(model_path, "2.");
        if (version.empty()) {
            version = "2.5";
        }

        return ModelConfig::create_qwen_2x(version);
    }

    // No match - return generic
    return ModelConfig::create_generic();
}

ModelConfig Models::detect_from_path_analysis(const std::string& model_path) {
    if (model_path.empty()) {
        LOG_DEBUG("Empty model path, cannot detect");
        return ModelConfig::create_generic();
    }

    LOG_DEBUG("Detecting model family from path: " + model_path);

    // Convert to lowercase for case-insensitive matching
    std::string model_lower = model_path;
    std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);

    // Llama 3.x detection
    if (model_lower.find("llama-3") != std::string::npos ||
        model_lower.find("llama3") != std::string::npos) {
        LOG_INFO("Detected Llama 3.x from model path");
        std::string version = extract_version_from_path(model_lower, "3.");
        if (version.empty()) {
            version = "3.1";
        }
        return ModelConfig::create_llama_3x(version);
    }

    // GLM-4.x detection
    if (model_lower.find("glm-4") != std::string::npos ||
        model_lower.find("glm4") != std::string::npos) {
        LOG_INFO("Detected GLM-4 from model path");
        std::string version = extract_version_from_path(model_lower, "4.");
        if (version.empty()) {
            version = "4";
        }
        return ModelConfig::create_glm_4(version);
    }

    // Qwen 2.x detection
    if (model_lower.find("qwen2") != std::string::npos) {
        LOG_INFO("Detected Qwen 2.x from model path");
        std::string version = extract_version_from_path(model_lower, "2.");
        if (version.empty()) {
            version = "2.5";
        }
        return ModelConfig::create_qwen_2x(version);
    }

    // No match - return generic
    LOG_WARN("Could not detect specific model family from path, using generic configuration");
    return ModelConfig::create_generic();
}

std::string Models::extract_version_from_path(const std::string& model_path,
                                                     const std::string& pattern) {
    // Find the pattern in the path
    size_t pos = model_path.find(pattern);
    if (pos == std::string::npos) {
        return "";
    }

    // Extract version string (pattern + one more digit, e.g., "3.1", "4.5")
    if (pos + pattern.length() < model_path.length()) {
        char next_char = model_path[pos + pattern.length()];
        if (std::isdigit(next_char)) {
            return pattern + next_char;
        }
    }

    // Return just the major version
    return pattern.substr(0, pattern.length() - 1); // Remove trailing dot
}

std::string Models::format_system_message(const ModelConfig& config, const std::string& custom_system_prompt, ToolRegistry& registry, void* template_node) {
    std::string system_message;
    auto tool_descriptions = registry.list_tools_with_descriptions();

    // Qwen 2.x/3.x: Use minja template to render system message with tools
    if (config.family == ModelFamily::QWEN_2_X) {
        if (!template_node) {
            LOG_ERROR("Qwen requires template_node to format system message with tools");
            return custom_system_prompt.empty() ?
                "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." :
                custom_system_prompt;
        }

        // Build tools array for minja
        auto tools = minja::Value::array();
        for (const auto& tool_name : registry.list_tools()) {
            Tool* tool = registry.get_tool(tool_name);
            if (tool) {
                auto tool_obj = minja::Value::object();
                tool_obj.set("name", minja::Value(tool->name()));
                tool_obj.set("description", minja::Value(tool->description()));

                // Try to get structured schema first (same approach as api_backend.cpp)
                auto param_defs = tool->get_parameters_schema();
                nlohmann::json schema;

                if (!param_defs.empty()) {
                    // Build JSON schema from ParameterDef structs
                    schema["type"] = "object";
                    schema["properties"] = nlohmann::json::object();
                    nlohmann::json required_fields = nlohmann::json::array();

                    for (const auto& param : param_defs) {
                        nlohmann::json prop;
                        prop["type"] = param.type;
                        if (!param.description.empty()) {
                            prop["description"] = param.description;
                        }
                        schema["properties"][param.name] = prop;

                        if (param.required) {
                            required_fields.push_back(param.name);
                        }
                    }

                    if (!required_fields.empty()) {
                        schema["required"] = required_fields;
                    }
                } else {
                    // Fallback: try parsing legacy parameters()
                    try {
                        schema = nlohmann::json::parse(tool->parameters());
                    } catch (const std::exception& e) {
                        LOG_DEBUG("Tool " + tool->name() + " has plain text parameters, using empty schema");
                        // Create basic empty schema
                        schema = {
                            {"type", "object"},
                            {"properties", nlohmann::json::object()},
                            {"required", nlohmann::json::array()}
                        };
                    }
                }

                // Convert nlohmann::json to minja::Value (use json alias to avoid ambiguity)
                json ordered_schema = schema;
                tool_obj.set("parameters", minja::Value(ordered_schema));

                tools.push_back(tool_obj);
            }
        }

        // Create minja context with tools
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
        sys_msg.set("content", minja::Value(custom_system_prompt.empty() ?
            "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." :
            custom_system_prompt));
        messages.push_back(sys_msg);

        context->set("messages", messages);
        context->set("tools", tools);
        context->set("add_generation_prompt", minja::Value(false));
        context->set("bos_token", minja::Value(""));  // No BOS for single message

        // Render with the template
        auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);
        std::string rendered = (*template_ptr)->render(context);

        LOG_DEBUG("Qwen system message with tools: " + std::to_string(rendered.length()) + " chars (rendered by template)");
        return rendered;
    }

    // Llama 3.x: JSON schemas format (BFCL-style)
    if (config.family == ModelFamily::LLAMA_3_X) {
        if (!custom_system_prompt.empty()) {
            system_message = custom_system_prompt + "\n\n";
        } else {
            system_message = "You are a helpful assistant.\n\n";
        }

        // Add JSON schemas for tools
        system_message += "The following functions are available IF needed to answer the user's request:\n\n";
        system_message += "IMPORTANT: Only call a function if you actually need external information or capabilities. ";
        system_message += "For greetings, casual conversation, or questions you can answer directly - respond normally without calling any function.\n\n";
        system_message += "When you DO need to call a function, respond with ONLY a JSON object in this format: ";
        system_message += "{\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\n";

        // TODO: Add actual JSON schema generation from tool registry
        // For now, just list tools
        for (const auto& [tool_name, tool_desc] : tool_descriptions) {
            Tool* tool = registry.get_tool(tool_name);
            if (tool) {
                system_message += "- " + tool_name + ": " + tool_desc + "\n";
            }
        }

        LOG_DEBUG("Llama 3.x system message: " + std::to_string(system_message.length()) + " chars");
        return system_message;
    }

    // GLM-4.x: Chinese format with JSON schemas
    if (config.family == ModelFamily::GLM_4) {
        system_message = "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n";
        system_message += "IMPORTANT: Always respond in English unless the user specifically requests another language.\n\n";

        // Add tools in Chinese format
        if (!tool_descriptions.empty()) {
            system_message += "# 可用工具\n\n";
            system_message += "**重要提示**: 读取大文件时，请使用 `head` 或 `tail` 参数只读取部分内容（例如前100行或后100行），以避免超出上下文窗口限制。\n\n";

            for (const auto& [tool_name, tool_desc] : tool_descriptions) {
                system_message += "## " + tool_name + "\n";
                system_message += tool_desc + "\n";
                system_message += "在调用上述函数时，请使用扁平 Json 格式：{\"name\": \"" + tool_name + "\", \"parameters\": {...}}\n\n";
            }
        }

        LOG_DEBUG("GLM-4 system message: " + std::to_string(system_message.length()) + " chars");
        return system_message;
    }

    // Generic/fallback: Plain text tool list with JSON format instructions
    system_message = "IMPORTANT: Tool calls MUST be the entire message with EXACTLY this JSON format:\n";
    system_message += "{\"name\": \"function_name_here\", \"parameters\": {\"key\": \"value\"}}\n";
    system_message += "Use field \"name\" (not \"tool_name\") and \"parameters\" (not \"params\").\n\n";

    if (!custom_system_prompt.empty()) {
        system_message += custom_system_prompt + "\n\n";
    }

    // Plain text tool list
    system_message += "Here are the available tools:\n\n";
    for (const auto& [tool_name, tool_desc] : tool_descriptions) {
        Tool* tool = registry.get_tool(tool_name);
        if (tool) {
            system_message += "- " + tool_name + ": " + tool_desc + " (parameters: " + tool->parameters() + ")\n";
        }
    }

    LOG_DEBUG("Generic system message: " + std::to_string(system_message.length()) + " chars");
    return system_message;
}
