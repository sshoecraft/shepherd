
#include "models.h"
#include "logger.h"
#include "tools/tool.h"
#include "nlohmann/json.hpp"
#include "llama.cpp/vendor/minja/minja.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <ctime>
#include <fstream>
#include <filesystem>

std::map<std::string, ModelConfig> Models::model_database;
std::map<std::string, ModelConfig> Models::pattern_database;
std::map<std::string, ModelConfig> Models::provider_defaults;
bool Models::initialized = false;
std::string Models::custom_models_path;

ModelConfig Models::detect_from_chat_template(const std::string& template_text, const std::string& model_path) {
    // Primary detection: Analyze chat template content (most reliable)
    ModelConfig config = detect_from_template_content(template_text, model_path);

    // If template detection succeeded, return it
    if (config.family != ModelFamily::GENERIC) {
        return config;
    }

    // Return GENERIC - caller should try config.json before path analysis
    LOG_DEBUG("Chat template detection returned generic, caller should try config.json");
    return ModelConfig::create_generic();
}

ModelConfig Models::detect_from_model_path(const std::string& model_path) {
    return detect_from_path_analysis(model_path);
}

ModelConfig Models::detect_from_config_file(const std::string& model_dir) {
    std::string config_file = model_dir + "/config.json";
    std::ifstream file(config_file);
    if (!file.is_open()) {
        LOG_DEBUG("No config.json found at: " + config_file);
        return ModelConfig::create_generic();
    }

    std::string config_json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    LOG_DEBUG("Detecting model family from config.json...");

    // Helper lambda to extract string value after a key
    auto extract_string = [&config_json](const std::string& key) -> std::string {
        size_t pos = config_json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        size_t colon = config_json.find(":", pos);
        if (colon == std::string::npos) return "";
        size_t quote_start = config_json.find("\"", colon);
        if (quote_start == std::string::npos) return "";
        size_t quote_end = config_json.find("\"", quote_start + 1);
        if (quote_end == std::string::npos) return "";
        return config_json.substr(quote_start + 1, quote_end - quote_start - 1);
    };

    // Check architecture field (TensorRT-LLM engine format, also works for HF)
    std::string architecture = extract_string("architecture");
    if (!architecture.empty()) {
        LOG_DEBUG("Found architecture: " + architecture);

        // Map architecture class names to model families
        if (architecture.find("Llama") != std::string::npos) {
            // Check for Llama 3.x specific tokens in the config
            if (config_json.find("<|begin_of_text|>") != std::string::npos ||
                config_json.find("<|eom_id|>") != std::string::npos) {
                LOG_INFO("Detected Llama 3.x from config.json architecture");
                return ModelConfig::create_llama_3x();
            } else {
                LOG_INFO("Detected Llama 2.x from config.json architecture");
                return ModelConfig::create_llama_2x();
            }
        }
        else if (architecture.find("Qwen2") != std::string::npos && architecture.find("Qwen3") == std::string::npos) {
            LOG_INFO("Detected Qwen 2.x from config.json architecture");
            return ModelConfig::create_qwen_2x();
        }
        else if (architecture.find("Qwen") != std::string::npos) {
            // Qwen3 or generic Qwen
            LOG_INFO("Detected Qwen 3.x from config.json architecture");
            return ModelConfig::create_qwen_3x();
        }
        else if (architecture.find("GLM") != std::string::npos || architecture.find("ChatGLM") != std::string::npos) {
            LOG_INFO("Detected GLM-4 from config.json architecture");
            return ModelConfig::create_glm_4();
        }
        else if (architecture.find("Mistral") != std::string::npos) {
            LOG_INFO("Detected Mistral from config.json architecture");
            // Mistral uses similar format to Llama 2
            return ModelConfig::create_llama_2x();
        }
    }

    // Check model_type field (standard HuggingFace format)
    std::string model_type = extract_string("model_type");
    if (!model_type.empty()) {
        LOG_DEBUG("Found model_type: " + model_type);

        // Convert to lowercase for comparison
        std::transform(model_type.begin(), model_type.end(), model_type.begin(), ::tolower);

        if (model_type == "llama") {
            if (config_json.find("<|begin_of_text|>") != std::string::npos ||
                config_json.find("<|eom_id|>") != std::string::npos) {
                LOG_INFO("Detected Llama 3.x from config.json model_type");
                return ModelConfig::create_llama_3x();
            } else {
                LOG_INFO("Detected Llama 2.x from config.json model_type");
                return ModelConfig::create_llama_2x();
            }
        }
        else if (model_type == "qwen2") {
            LOG_INFO("Detected Qwen 2.x from config.json model_type");
            return ModelConfig::create_qwen_2x();
        }
        else if (model_type == "qwen" || model_type == "qwen3") {
            LOG_INFO("Detected Qwen 3.x from config.json model_type");
            return ModelConfig::create_qwen_3x();
        }
        else if (model_type == "chatglm" || model_type == "glm") {
            LOG_INFO("Detected GLM-4 from config.json model_type");
            return ModelConfig::create_glm_4();
        }
        else if (model_type == "mistral") {
            LOG_INFO("Detected Mistral from config.json model_type");
            return ModelConfig::create_llama_2x();
        }
    }

    // Check qwen_type field (TensorRT-LLM Qwen format)
    std::string qwen_type = extract_string("qwen_type");
    if (!qwen_type.empty()) {
        LOG_DEBUG("Found qwen_type: " + qwen_type);
        if (qwen_type == "qwen3") {
            LOG_INFO("Detected Qwen 3.x from config.json qwen_type");
            return ModelConfig::create_qwen_3x();
        } else if (qwen_type == "qwen2") {
            LOG_INFO("Detected Qwen 2.x from config.json qwen_type");
            return ModelConfig::create_qwen_2x();
        }
    }

    LOG_DEBUG("Could not detect model family from config.json");
    return ModelConfig::create_generic();
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

    // Qwen 2.x/3.x: Has <|im_start|> and <|im_end|>
    if (template_text.find("<|im_start|>") != std::string::npos &&
        template_text.find("<|im_end|>") != std::string::npos) {

        // Check if it's Qwen3 or a derivative like MindLink
        std::string model_lower = model_path;
        std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);

        if (model_lower.find("qwen3") != std::string::npos ||
            model_lower.find("qwen-3") != std::string::npos ||
            model_lower.find("mindlink") != std::string::npos) {
            LOG_INFO("Detected Qwen 3.x model family (or MindLink derivative) from chat template");

            std::string version = extract_version_from_path(model_path, "3.");
            if (version.empty()) {
                version = "3";
            }

            // Check if this is a thinking model by looking for <think> in the template
            // Don't assume based on model name - check the actual template content
            bool is_thinking = (template_text.find("<think>") != std::string::npos) ||
                               (template_text.find("</think>") != std::string::npos) ||
                               (model_lower.find("thinking") != std::string::npos);
            if (is_thinking) {
                LOG_INFO("Detected Thinking model variant");
            }

            return ModelConfig::create_qwen_3x(version, is_thinking);
        } else {
            LOG_INFO("Detected Qwen 2.x model family from chat template");

            std::string version = extract_version_from_path(model_path, "2.");
            if (version.empty()) {
                version = "2.5";
            }

            return ModelConfig::create_qwen_2x(version);
        }
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

    // Qwen 3.x detection (includes MindLink)
    if (model_lower.find("qwen3") != std::string::npos ||
        model_lower.find("qwen-3") != std::string::npos ||
        model_lower.find("mindlink") != std::string::npos) {
        LOG_INFO("Detected Qwen 3.x (or MindLink) from model path");
        std::string version = extract_version_from_path(model_lower, "3.");
        if (version.empty()) {
            version = "3";
        }
        // Only detect thinking mode if "thinking" is in the name
        // Path-based detection can't check template content, so be conservative
        bool is_thinking = (model_lower.find("thinking") != std::string::npos);
        if (is_thinking) {
            LOG_INFO("Detected Thinking model variant from path");
        }
        return ModelConfig::create_qwen_3x(version, is_thinking);
    }

    // Qwen 2.x detection
    if (model_lower.find("qwen2") != std::string::npos ||
        model_lower.find("qwen-2") != std::string::npos) {
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
    if (config.family == ModelFamily::QWEN_2_X || config.family == ModelFamily::QWEN_3_X) {
        // If no tools, just return the system prompt
        if (tool_descriptions.empty()) {
            return custom_system_prompt.empty() ?
                "You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks." :
                custom_system_prompt;
        }

        // If tools exist but no template_node, fall back to generic formatting
        if (!template_node) {
            LOG_WARN("Qwen model has tools but no chat template - using generic formatting");
            // Fall through to generic formatting below
        } else {

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
        }  // end else (template_node available)
    }  // end if (Qwen family)

    // Llama 3.x: JSON schemas format (BFCL-style)
    if (config.family == ModelFamily::LLAMA_3_X) {
        if (!custom_system_prompt.empty()) {
            system_message = custom_system_prompt;
        } else {
            system_message = "You are a helpful assistant.";
        }

        // Add JSON schemas for tools (only if tools exist)
        if (!tool_descriptions.empty()) {
            system_message += "\n\nThe following functions are available IF needed to answer the user's request:\n\n";
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

void Models::ensure_initialized() {
    if (!initialized) {
        init(custom_models_path);
    }
}

void Models::init(const std::string& custom_path) {
    if (initialized && custom_path.empty()) {
        return;
    }

    LOG_DEBUG("Initializing models database...");

    custom_models_path = custom_path;
    initialized = true;

    std::string json_content;
    bool loaded = false;

    if (!custom_path.empty()) {
        loaded = load_models_database(custom_path);
        if (loaded) {
            LOG_INFO("Loaded models database from: " + custom_path);
            return;
        }
        LOG_WARN("Failed to load custom models file: " + custom_path);
    }

    // Check XDG config location first
    std::string config_home;
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        config_home = xdg_config;
    } else {
        std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
        if (!home.empty()) {
            config_home = home + "/.config";
        }
    }
    if (!config_home.empty()) {
        std::string user_path = config_home + "/shepherd/models.json";
        if (std::filesystem::exists(user_path)) {
            loaded = load_models_database(user_path);
            if (loaded) {
                LOG_INFO("Loaded models database from: " + user_path);
                return;
            }
        }
    }

    json_content = get_embedded_models_json();
    if (parse_models_json(json_content)) {
        LOG_INFO("Loaded embedded models database");
        return;
    }

    LOG_WARN("Failed to load models database, using provider defaults only");
}

bool Models::load_models_database(const std::string& path) {
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            LOG_DEBUG("Could not open models file: " + path);
            return false;
        }

        std::string json_content((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
        return parse_models_json(json_content);
    } catch (const std::exception& e) {
        LOG_ERROR("Error loading models database from " + path + ": " + e.what());
        return false;
    }
}

std::string Models::get_embedded_models_json() {
    return R"({
  "version": "1.0",
  "providers": {
    "openai": {
      "context_window": 8192,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supports_temperature": true,
      "supports_streaming": true
    },
    "anthropic": {
      "context_window": 200000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens",
      "supports_temperature": true,
      "supports_streaming": true
    },
    "gemini": {
      "context_window": 1000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens",
      "supports_temperature": true,
      "supports_streaming": true
    }
  },
  "models": [
    {
      "name": "gpt-5",
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"],
      "supports_temperature": true,
      "supports_streaming": true,
      "vision_support": true,
      "function_calling_support": true,
      "training_cutoff_date": "2025-08"
    },
    {
      "name": "gpt-5-pro",
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"]
    },
    {
      "name": "gpt-5-codex",
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "gpt-5-mini",
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"]
    },
    {
      "name": "gpt-5-nano",
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o3",
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_cot_tokens": 100000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o3-mini",
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_cot_tokens": 65536,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o4-mini",
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o1",
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_cot_tokens": 32768,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o1-mini",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 65536,
      "max_cot_tokens": 65536,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "o1-pro",
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_tokens_param_name": "max_completion_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "gpt-4o",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"],
      "vision_support": true,
      "fine_tunable": true
    },
    {
      "name": "gpt-4o-mini",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"],
      "fine_tunable": true
    },
    {
      "name": "gpt-4o-realtime-preview",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/realtime"],
      "realtime_capable": true
    },
    {
      "name": "gpt-4-turbo",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"],
      "vision_support": true
    },
    {
      "name": "gpt-4.1",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"]
    },
    {
      "name": "gpt-4.1-mini",
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"]
    },
    {
      "name": "gpt-4",
      "provider": "openai",
      "context_window": 8192,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"]
    },
    {
      "name": "gpt-4-0613",
      "provider": "openai",
      "context_window": 8192,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "gpt-3.5-turbo",
      "provider": "openai",
      "context_window": 16384,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions", "/v1/assistants"],
      "fine_tunable": true
    },
    {
      "name": "gpt-3.5-turbo-16k",
      "provider": "openai",
      "context_window": 16384,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/chat/completions"]
    },
    {
      "name": "claude-sonnet-4-5",
      "provider": "anthropic",
      "aliases": ["claude-sonnet-4-5-20250929"],
      "context_window": 200000,
      "max_output_tokens": 64000,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true,
      "training_cutoff_date": "2025-09"
    },
    {
      "name": "claude-sonnet-4",
      "provider": "anthropic",
      "aliases": ["claude-sonnet-4-20250514"],
      "context_window": 200000,
      "max_output_tokens": 64000,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-3-7-sonnet-20250219",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_tokens",
      "special_headers": {
        "anthropic-beta": "output-128k-2025-02-19"
      },
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-opus-4-1",
      "provider": "anthropic",
      "aliases": ["claude-opus-4-1-20250805"],
      "context_window": 200000,
      "max_output_tokens": 32000,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-opus-4",
      "provider": "anthropic",
      "aliases": ["claude-opus-4-20250514"],
      "context_window": 200000,
      "max_output_tokens": 32000,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-3-5-sonnet-20240620",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-3-5-haiku-20241022",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"]
    },
    {
      "name": "claude-3-haiku-20240307",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "claude-3-opus-20240229",
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens",
      "supported_endpoints": ["/v1/messages"],
      "vision_support": true
    },
    {
      "name": "gemini-2.0-flash-exp",
      "provider": "gemini",
      "context_window": 1000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens",
      "supported_endpoints": ["/v1beta/models/{model}:generateContent"]
    },
    {
      "name": "gemini-2.5-pro",
      "provider": "gemini",
      "context_window": 2000000,
      "max_output_tokens": 65536,
      "max_tokens_param_name": "maxOutputTokens",
      "supported_endpoints": ["/v1beta/models/{model}:generateContent"],
      "vision_support": true
    },
    {
      "name": "gemini-1.5-pro",
      "provider": "gemini",
      "context_window": 2000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens",
      "supported_endpoints": ["/v1beta/models/{model}:generateContent"],
      "vision_support": true
    },
    {
      "name": "gemini-1.5-flash",
      "provider": "gemini",
      "context_window": 1000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens",
      "supported_endpoints": ["/v1beta/models/{model}:generateContent"]
    },
    {
      "name": "gemini-pro",
      "provider": "gemini",
      "context_window": 32000,
      "max_output_tokens": 2048,
      "max_tokens_param_name": "maxOutputTokens",
      "supported_endpoints": ["/v1beta/models/{model}:generateContent"]
    }
  ],
  "patterns": {
    "gpt-5*": {
      "provider": "openai",
      "context_window": 400000,
      "max_output_tokens": 128000,
      "max_tokens_param_name": "max_completion_tokens"
    },
    "o3*": {
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_tokens_param_name": "max_completion_tokens"
    },
    "o1*": {
      "provider": "openai",
      "context_window": 200000,
      "max_output_tokens": 100000,
      "max_tokens_param_name": "max_completion_tokens"
    },
    "gpt-4o*": {
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens"
    },
    "gpt-4.1*": {
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 16384,
      "max_tokens_param_name": "max_tokens"
    },
    "gpt-4*": {
      "provider": "openai",
      "context_window": 128000,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens"
    },
    "gpt-3.5*": {
      "provider": "openai",
      "context_window": 16384,
      "max_output_tokens": 4096,
      "max_tokens_param_name": "max_tokens"
    },
    "claude-*": {
      "provider": "anthropic",
      "context_window": 200000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "max_tokens"
    },
    "gemini-2*": {
      "provider": "gemini",
      "context_window": 1000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens"
    },
    "gemini-1.5*": {
      "provider": "gemini",
      "context_window": 1000000,
      "max_output_tokens": 8192,
      "max_tokens_param_name": "maxOutputTokens"
    }
  }
})";
}

bool Models::parse_models_json(const std::string& json_content) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_content);

        if (j.contains("providers") && j["providers"].is_object()) {
            for (auto& [provider_name, provider_config] : j["providers"].items()) {
                ModelConfig default_config = ModelConfig::create_generic();
                default_config.provider = provider_name;
                
                if (provider_config.contains("context_window"))
                    default_config.context_window = provider_config["context_window"];
                if (provider_config.contains("max_output_tokens"))
                    default_config.max_output_tokens = provider_config["max_output_tokens"];
                if (provider_config.contains("max_tokens_param_name"))
                    default_config.max_tokens_param_name = provider_config["max_tokens_param_name"];
                if (provider_config.contains("supports_temperature"))
                    default_config.supports_temperature = provider_config["supports_temperature"];
                if (provider_config.contains("supports_streaming"))
                    default_config.supports_streaming = provider_config["supports_streaming"];

                provider_defaults[provider_name] = default_config;
            }
        }

        if (j.contains("models") && j["models"].is_array()) {
            for (const auto& model_entry : j["models"]) {
                ModelConfig config = ModelConfig::create_generic();
                
                if (model_entry.contains("name"))
                    config.model_name = model_entry["name"];
                if (model_entry.contains("provider"))
                    config.provider = model_entry["provider"];
                if (model_entry.contains("context_window"))
                    config.context_window = model_entry["context_window"];
                if (model_entry.contains("max_output_tokens"))
                    config.max_output_tokens = model_entry["max_output_tokens"];
                if (model_entry.contains("max_cot_tokens"))
                    config.max_cot_tokens = model_entry["max_cot_tokens"];
                if (model_entry.contains("max_tokens_param_name"))
                    config.max_tokens_param_name = model_entry["max_tokens_param_name"];
                if (model_entry.contains("supports_temperature"))
                    config.supports_temperature = model_entry["supports_temperature"];
                if (model_entry.contains("supports_streaming"))
                    config.supports_streaming = model_entry["supports_streaming"];
                if (model_entry.contains("vision_support"))
                    config.vision_support = model_entry["vision_support"];
                if (model_entry.contains("audio_support"))
                    config.audio_support = model_entry["audio_support"];
                if (model_entry.contains("function_calling_support"))
                    config.function_calling_support = model_entry["function_calling_support"];
                if (model_entry.contains("realtime_capable"))
                    config.realtime_capable = model_entry["realtime_capable"];
                if (model_entry.contains("fine_tunable"))
                    config.fine_tunable = model_entry["fine_tunable"];
                if (model_entry.contains("training_cutoff_date"))
                    config.training_cutoff_date = model_entry["training_cutoff_date"];
                if (model_entry.contains("deprecated"))
                    config.deprecated = model_entry["deprecated"];
                if (model_entry.contains("replacement_model"))
                    config.replacement_model = model_entry["replacement_model"];
                if (model_entry.contains("notes"))
                    config.notes = model_entry["notes"];

                if (model_entry.contains("supported_endpoints") && model_entry["supported_endpoints"].is_array()) {
                    for (const auto& endpoint : model_entry["supported_endpoints"]) {
                        config.supported_endpoints.push_back(endpoint);
                    }
                }

                if (model_entry.contains("special_headers") && model_entry["special_headers"].is_object()) {
                    for (auto& [key, value] : model_entry["special_headers"].items()) {
                        config.special_headers[key] = value;
                    }
                }

                if (model_entry.contains("aliases") && model_entry["aliases"].is_array()) {
                    for (const auto& alias : model_entry["aliases"]) {
                        config.aliases.push_back(alias);
                        model_database[alias] = config;
                    }
                }

                if (!config.model_name.empty()) {
                    model_database[config.model_name] = config;
                }
            }
        }

        if (j.contains("patterns") && j["patterns"].is_object()) {
            for (auto& [pattern, pattern_config] : j["patterns"].items()) {
                ModelConfig config = ModelConfig::create_generic();
                
                if (pattern_config.contains("provider"))
                    config.provider = pattern_config["provider"];
                if (pattern_config.contains("context_window"))
                    config.context_window = pattern_config["context_window"];
                if (pattern_config.contains("max_output_tokens"))
                    config.max_output_tokens = pattern_config["max_output_tokens"];
                if (pattern_config.contains("max_tokens_param_name"))
                    config.max_tokens_param_name = pattern_config["max_tokens_param_name"];

                pattern_database[pattern] = config;
            }
        }

        LOG_DEBUG("Parsed models database: " + std::to_string(model_database.size()) + 
                  " models, " + std::to_string(pattern_database.size()) + " patterns");
        return true;

    } catch (const nlohmann::json::exception& e) {
        LOG_ERROR("Failed to parse models JSON: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Error parsing models JSON: " + std::string(e.what()));
        return false;
    }
}

bool Models::matches_pattern(const std::string& model_name, const std::string& pattern) {
    if (pattern.find('*') == std::string::npos) {
        return model_name == pattern;
    }

    size_t star_pos = pattern.find('*');
    std::string prefix = pattern.substr(0, star_pos);
    std::string suffix = pattern.substr(star_pos + 1);

    if (!prefix.empty() && model_name.find(prefix) != 0) {
        return false;
    }

    if (!suffix.empty()) {
        if (model_name.length() < suffix.length()) {
            return false;
        }
        if (model_name.substr(model_name.length() - suffix.length()) != suffix) {
            return false;
        }
    }

    return true;
}

ModelConfig Models::get_provider_default(const std::string& provider) {
    auto it = provider_defaults.find(provider);
    if (it != provider_defaults.end()) {
        return it->second;
    }

    ModelConfig config = ModelConfig::create_generic();
    config.provider = provider;
    config.context_window = 8192;
    config.max_output_tokens = 4096;
    config.max_tokens_param_name = "max_tokens";
    return config;
}

ModelConfig Models::detect_from_api_model(const std::string& provider, const std::string& model_name) {
    ensure_initialized();

    auto exact_match = model_database.find(model_name);
    if (exact_match != model_database.end()) {
        LOG_DEBUG("Found exact model match: " + model_name);
        return exact_match->second;
    }

    for (const auto& [pattern, pattern_config] : pattern_database) {
        if (matches_pattern(model_name, pattern)) {
            LOG_DEBUG("Matched model pattern: " + pattern + " for " + model_name);
            ModelConfig config = pattern_config;
            config.model_name = model_name;
            if (config.provider.empty()) {
                config.provider = provider;
            }
            return config;
        }
    }

    LOG_DEBUG("Unknown model: " + model_name + ", using provider defaults for " + provider);
    ModelConfig config = get_provider_default(provider);
    config.model_name = model_name;
    return config;
}

bool Models::supports_endpoint(const std::string& model_name, const std::string& endpoint_path) {
    ensure_initialized();

    auto it = model_database.find(model_name);
    if (it == model_database.end()) {
        return true;
    }

    if (it->second.supported_endpoints.empty()) {
        return true;
    }

    for (const auto& supported : it->second.supported_endpoints) {
        if (supported == endpoint_path) {
            return true;
        }
    }

    return false;
}

std::vector<std::string> Models::get_compatible_models(const std::string& endpoint_path) {
    ensure_initialized();

    std::vector<std::string> compatible;
    for (const auto& [model_name, config] : model_database) {
        if (config.supported_endpoints.empty()) {
            compatible.push_back(model_name);
            continue;
        }

        for (const auto& endpoint : config.supported_endpoints) {
            if (endpoint == endpoint_path) {
                compatible.push_back(model_name);
                break;
            }
        }
    }

    return compatible;
}

bool Models::load_generation_config(const std::string& model_dir_path,
                                     float& temperature,
                                     float& top_p,
                                     int& top_k) {
    std::filesystem::path gen_config_path = std::filesystem::path(model_dir_path) / "generation_config.json";

    if (!std::filesystem::exists(gen_config_path)) {
        LOG_DEBUG("No generation_config.json found at: " + gen_config_path.string());
        return false;
    }

    try {
        std::ifstream file(gen_config_path);
        if (!file.is_open()) {
            LOG_WARN("Could not open generation_config.json: " + gen_config_path.string());
            return false;
        }

        nlohmann::json config;
        file >> config;

        bool found_any = false;

        if (config.contains("temperature")) {
            temperature = config["temperature"].get<float>();
            LOG_DEBUG("Loaded temperature from generation_config.json: " + std::to_string(temperature));
            found_any = true;
        }

        if (config.contains("top_p")) {
            top_p = config["top_p"].get<float>();
            LOG_DEBUG("Loaded top_p from generation_config.json: " + std::to_string(top_p));
            found_any = true;
        }

        if (config.contains("top_k")) {
            top_k = config["top_k"].get<int>();
            LOG_DEBUG("Loaded top_k from generation_config.json: " + std::to_string(top_k));
            found_any = true;
        }

        if (found_any) {
            LOG_INFO("Loaded sampling parameters from: " + gen_config_path.string());
        }

        return found_any;

    } catch (const std::exception& e) {
        LOG_WARN("Failed to parse generation_config.json: " + std::string(e.what()));
        return false;
    }
}

// Default chat templates embedded from https://github.com/chujiezheng/chat_templates
// These serve as fallbacks when tokenizer_config.json doesn't provide a chat_template
namespace {
    // ChatML format - used by Qwen2, Yi, and other models with <|im_start|>/<|im_end|> tokens
    constexpr const char* CHATML_TEMPLATE = R"({{ bos_token }}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %})";

    // Llama 3 format - uses <|start_header_id|> and <|end_header_id|>
    constexpr const char* LLAMA3_TEMPLATE = R"({% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}

{{ bos_token }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}
{% endif %})";

    // Llama 2 format - uses [INST] and <<SYS>>
    constexpr const char* LLAMA2_TEMPLATE = R"({% if messages[0]['role'] == 'system' %}
    {% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + content | trim + ' ' + eos_token }}
    {% endif %}
{% endfor %})";

    // Mistral format - similar to Llama 2 but slightly different
    constexpr const char* MISTRAL_TEMPLATE = R"({% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] | trim + '\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ bos_token + system_message}}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + message['content'] | trim + eos_token }}
    {% endif %}
{% endfor %})";
}

std::string Models::get_default_chat_template(ModelFamily family) {
    switch (family) {
        case ModelFamily::LLAMA_3_X:
            return LLAMA3_TEMPLATE;
        case ModelFamily::QWEN_2_X:
        case ModelFamily::QWEN_3_X:
            return CHATML_TEMPLATE;
        case ModelFamily::MISTRAL:
            return MISTRAL_TEMPLATE;
        case ModelFamily::GENERIC:
        default:
            // Return ChatML as a generic default for unknown models
            // ChatML is widely compatible with many models
            return CHATML_TEMPLATE;
    }
}

ModelFamily Models::detect_from_tokenizer_class(const std::string& tokenizer_class) {
    // Detect model family from tokenizer class name
    // This is used as a fallback when chat_template is missing

    if (tokenizer_class.find("Llama") != std::string::npos) {
        // LlamaTokenizer or LlamaTokenizerFast
        // Default to Llama 3 as it's more common now
        return ModelFamily::LLAMA_3_X;
    }

    if (tokenizer_class.find("Qwen2") != std::string::npos) {
        // Qwen2Tokenizer - uses ChatML format
        return ModelFamily::QWEN_2_X;
    }

    if (tokenizer_class.find("Qwen") != std::string::npos) {
        // QwenTokenizer or other Qwen variants
        // Qwen 3.x also uses ChatML, so default to 3.x
        return ModelFamily::QWEN_3_X;
    }

    if (tokenizer_class.find("Mistral") != std::string::npos) {
        return ModelFamily::MISTRAL;
    }

    // Default to GENERIC which uses ChatML format
    return ModelFamily::GENERIC;
}
