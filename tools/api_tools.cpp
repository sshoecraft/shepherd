#include "shepherd.h"
#include "api_tools.h"
#include "tools.h"
#include "../shepherd.h"
#include "../message.h"

#include <sstream>

// ============================================================================
// APIToolAdapter implementation
// ============================================================================

APIToolAdapter::APIToolAdapter(const Provider& p, Tools* tools)
    : provider(p), tools_ptr(tools), connected(false) {
    tool_session.system_message = "You are a helpful AI assistant.";
}

std::string APIToolAdapter::unsanitized_name() const {
    return "ask_" + provider.name;
}

std::string APIToolAdapter::description() const {
    std::ostringstream desc;

    if (provider.type == "anthropic") {
        desc << "Call Claude AI (" << provider.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need Claude's analysis, expertise, or viewpoint on a question or problem.";
    } else if (provider.type == "openai") {
        desc << "Call ChatGPT/GPT (" << provider.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need GPT's analysis, expertise, or viewpoint on a question or problem.";
    } else if (provider.type == "gemini") {
        desc << "Call Google Gemini (" << provider.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need Gemini's analysis, expertise, or viewpoint on a question or problem.";
    } else {
        desc << "Call another AI model (" << provider.type << "/" << provider.model << ") to get a second opinion. ";
        desc << "Use this to get a different AI's perspective on a question or problem.";
    }

    return desc.str();
}

std::string APIToolAdapter::parameters() const {
    return "prompt: string (the question or task)";
}

std::vector<ParameterDef> APIToolAdapter::get_parameters_schema() const {
    std::vector<ParameterDef> params;

    ParameterDef prompt_param;
    prompt_param.name = "prompt";
    prompt_param.type = "string";
    prompt_param.description = "The question or task to ask " + provider.type;
    prompt_param.required = true;
    params.push_back(prompt_param);

    ParameterDef model_param;
    model_param.name = "model";
    model_param.type = "string";
    model_param.description = "Override model for this call (optional)";
    model_param.required = false;
    model_param.default_value = provider.model;
    params.push_back(model_param);

    ParameterDef max_tokens_param;
    max_tokens_param.name = "max_tokens";
    max_tokens_param.type = "number";
    max_tokens_param.description = "Maximum tokens to generate (optional)";
    max_tokens_param.required = false;
    int default_max = provider.max_tokens;
    if (provider.type == "ollama" && provider.num_predict > 0) {
        default_max = provider.num_predict;
    }
    if (default_max > 0) {
        max_tokens_param.default_value = std::to_string(default_max);
    }
    params.push_back(max_tokens_param);

    return params;
}

void APIToolAdapter::populate_session_tools() {
    if (!tools_ptr) return;

    tool_session.tools.clear();

    for (Tool* tool : tools_ptr->all_tools) {
        // Skip this tool to prevent infinite self-recursion
        if (tool->name() == this->name()) {
            continue;
        }
        // Skip disabled tools
        if (!tools_ptr->is_enabled(tool->name())) {
            continue;
        }

        Session::Tool session_tool;
        session_tool.name = tool->name();
        session_tool.description = tool->description();

        // Build JSON schema from parameter definitions
        auto params = tool->get_parameters_schema();
        nlohmann::json schema;
        schema["type"] = "object";
        schema["properties"] = nlohmann::json::object();
        nlohmann::json required = nlohmann::json::array();

        for (const auto& param : params) {
            nlohmann::json prop;
            prop["type"] = param.type;
            prop["description"] = param.description;
            schema["properties"][param.name] = prop;
            if (param.required) {
                required.push_back(param.name);
            }
        }

        schema["required"] = required;
        session_tool.parameters = schema;
        tool_session.tools.push_back(session_tool);
    }

    dout(1) << "APIToolAdapter: sub-session has " + std::to_string(tool_session.tools.size()) + " tools" << std::endl;
}

void APIToolAdapter::ensure_connected() {
    if (connected) return;

    // Create callback that captures all events
    Backend::EventCallback cb = [this](CallbackEvent event, const std::string& content,
                                        const std::string& name, const std::string& id) {
        dout(2) << "APIToolAdapter callback: event=" + std::to_string(static_cast<int>(event)) +
                   " content.len=" + std::to_string(content.length()) << std::endl;
        switch (event) {
            case CallbackEvent::CONTENT:
                accumulated_content += content;
                break;

            case CallbackEvent::TOOL_CALL: {
                ToolParser::ToolCall tc;
                tc.name = name;
                tc.tool_call_id = id;
                tc.raw_json = content;

                // Parse the JSON content for parameters
                try {
                    nlohmann::json params = nlohmann::json::parse(content);
                    for (auto it = params.begin(); it != params.end(); ++it) {
                        if (it.value().is_string()) {
                            tc.parameters[it.key()] = it.value().get<std::string>();
                        } else if (it.value().is_number_integer()) {
                            tc.parameters[it.key()] = it.value().get<int>();
                        } else if (it.value().is_number_float()) {
                            tc.parameters[it.key()] = it.value().get<double>();
                        } else if (it.value().is_boolean()) {
                            tc.parameters[it.key()] = it.value().get<bool>();
                        } else {
                            tc.parameters[it.key()] = it.value().dump();
                        }
                    }
                } catch (...) {}

                // Validate tool exists
                if (tools_ptr) {
                    bool valid = false;
                    for (Tool* tool : tools_ptr->all_tools) {
                        if (tool->name() == tc.name && tools_ptr->is_enabled(tool->name())) {
                            valid = true;
                            break;
                        }
                    }
                    if (!valid) {
                        dout(1) << "APIToolAdapter: tool '" + tc.name + "' not found, skipping" << std::endl;
                        break;
                    }
                }

                pending_tool_calls.push_back(tc);
                break;
            }

            case CallbackEvent::ERROR:
                cb_success = false;
                cb_error = content;
                break;

            default:
                break;
        }
        return true;
    };

    // Disable streaming for sub-backends (they need to complete before we can return result)
    extern std::unique_ptr<Config> config;
    bool orig_streaming = config->streaming;
    config->streaming = false;

    // Use Provider.connect() - the standard flow (uses provider's sampling params)
    backend = provider.connect(tool_session, cb);

    // Restore streaming setting
    config->streaming = orig_streaming;

    if (backend) {
        connected = true;
        dout(1) << "APIToolAdapter: connected to " + provider.name + " (" + provider.type + ")" << std::endl;
    } else {
        dout(1) << "APIToolAdapter: failed to connect to " + provider.name << std::endl;
    }
}

std::map<std::string, std::any> APIToolAdapter::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    // Extract parameters
    std::string prompt = tool_utils::get_string(args, "prompt");
    if (prompt.empty()) {
        prompt = tool_utils::get_string(args, "query");
    }
    int default_max = provider.max_tokens;
    if (provider.type == "ollama" && provider.num_predict > 0) {
        default_max = provider.num_predict;
    }
    int max_tokens = tool_utils::get_int(args, "max_tokens", default_max);
    if (max_tokens <= 0) max_tokens = 1024;

    if (prompt.empty()) {
        result["error"] = "prompt parameter is required";
        result["success"] = false;
        return result;
    }

    // Ensure backend is connected (lazy init on first use)
    ensure_connected();

    if (!backend) {
        result["error"] = "Failed to connect to provider: " + provider.name;
        result["success"] = false;
        return result;
    }

    dout(1) << "APIToolAdapter::execute(" + provider.name + "): prompt=\"" +
               prompt.substr(0, 50) + "...\" max_tokens=" + std::to_string(max_tokens) << std::endl;

    // Reset state for this call
    accumulated_content.clear();
    pending_tool_calls.clear();
    cb_success = true;
    cb_error.clear();

    // Populate tools in session (may have changed since last call)
    populate_session_tools();

    // Add user message to session
    Message user_msg(Message::USER, prompt);
    tool_session.messages.push_back(user_msg);

    // Tool execution loop
    const int MAX_TOOL_ITERATIONS = 10;

    for (int iteration = 0; iteration < MAX_TOOL_ITERATIONS; iteration++) {
        // Clear tool calls for this iteration
        pending_tool_calls.clear();

        // Generate response
        backend->generate_from_session(tool_session, max_tokens);

        dout(1) << "APIToolAdapter iteration " + std::to_string(iteration) + ": success=" + (cb_success ? "true" : "false") +
                 ", content.length=" + std::to_string(accumulated_content.length()) +
                 ", tool_calls=" + std::to_string(pending_tool_calls.size()) +
                 ", error=" + cb_error << std::endl;

        if (!cb_success) {
            break;
        }

        // If no tool calls, we're done
        if (pending_tool_calls.empty()) {
            break;
        }

        // Execute each tool call
        for (const auto& tool_call : pending_tool_calls) {
            // Show nested tool call to user
            printf("    %s(%s)\n", tool_call.name.c_str(),
                   tool_call.raw_json.length() > 60
                   ? (tool_call.raw_json.substr(0, 57) + "...").c_str()
                   : tool_call.raw_json.c_str());
            fflush(stdout);

            dout(1) << "APIToolAdapter executing tool: " + tool_call.name << std::endl;

            // Find the tool
            Tool* tool = nullptr;
            if (tools_ptr) {
                for (Tool* t : tools_ptr->all_tools) {
                    if (t->name() == tool_call.name) {
                        tool = t;
                        break;
                    }
                }
            }

            std::string tool_result;
            if (tool) {
                auto tool_result_map = tool->execute(tool_call.parameters);

                nlohmann::json result_json;
                for (const auto& [key, value] : tool_result_map) {
                    try {
                        if (value.type() == typeid(std::string)) {
                            result_json[key] = std::any_cast<std::string>(value);
                        } else if (value.type() == typeid(int)) {
                            result_json[key] = std::any_cast<int>(value);
                        } else if (value.type() == typeid(double)) {
                            result_json[key] = std::any_cast<double>(value);
                        } else if (value.type() == typeid(bool)) {
                            result_json[key] = std::any_cast<bool>(value);
                        }
                    } catch (...) {
                        result_json[key] = "[conversion error]";
                    }
                }
                tool_result = result_json.dump();
            } else {
                tool_result = "{\"error\": \"Tool not found: " + tool_call.name + "\"}";
            }

            // Show tool result to user - extract content or summary from JSON
            std::string display_result;
            try {
                nlohmann::json result_j = nlohmann::json::parse(tool_result);
                if (result_j.contains("summary") && result_j["summary"].is_string()) {
                    display_result = result_j["summary"].get<std::string>();
                } else if (result_j.contains("content") && result_j["content"].is_string()) {
                    display_result = result_j["content"].get<std::string>();
                } else {
                    display_result = tool_result;
                }
            } catch (...) {
                display_result = tool_result;
            }
            if (display_result.length() > 60) {
                display_result = display_result.substr(0, 57) + "...";
            }
            printf("      %s\n", display_result.c_str());
            fflush(stdout);

            // Add tool result to session
            Message tool_msg(Message::TOOL_RESPONSE, tool_result);
            tool_msg.tool_call_id = tool_call.tool_call_id;
            tool_msg.tool_name = tool_call.name;
            tool_session.messages.push_back(tool_msg);
        }
    }

    // Build result
    if (cb_success) {
        result["content"] = accumulated_content;
        result["success"] = true;
        result["summary"] = accumulated_content.substr(0, std::min(size_t(100), accumulated_content.size()));
    } else {
        result["error"] = cb_error.empty() ? "Unknown error" : cb_error;
        result["success"] = false;
    }

    // Clear session for next call (stateless)
    tool_session.messages.clear();
    tool_session.total_tokens = 0;

    return result;
}

// ============================================================================
// Provider tool registration functions
// ============================================================================

void register_provider_tools(Tools& tools, const std::string& active_provider) {
    tools.api_tools.clear();

    auto providers = Provider::load_providers();

    for (const auto& p : providers) {
        // Skip the active provider - don't want to call ourselves
        if (p.name == active_provider) {
            continue;
        }

        // Skip ephemeral providers (priority 0 is reserved for command-line providers)
        if (p.priority == 0) {
            continue;
        }

        // Only register API providers (not local models)
        if (!p.is_api()) {
            continue;
        }

        auto adapter = std::make_unique<APIToolAdapter>(p, &tools);
        tools.register_tool(std::move(adapter), "api");
    }

    tools.build_all_tools();
    dout(1) << "Registered " + std::to_string(tools.api_tools.size()) + " provider tools" << std::endl;
}

void register_provider_as_tool(Tools& tools, const std::string& provider_name) {
    auto providers = Provider::load_providers();

    const Provider* found = nullptr;
    for (const auto& p : providers) {
        if (p.name == provider_name) {
            found = &p;
            break;
        }
    }

    if (!found) {
        dout(1) << "WARNING: Provider not found: " + provider_name << std::endl;
        return;
    }

    if (!found->is_api()) {
        dout(1) << "Skipping non-API provider: " + provider_name << std::endl;
        return;
    }

    auto adapter = std::make_unique<APIToolAdapter>(*found, &tools);
    std::string tool_name = adapter->name();
    tools.register_tool(std::move(adapter), "api");

    tools.build_all_tools();
    dout(1) << "Registered provider as tool: " + tool_name << std::endl;
}

void unregister_provider_tool(Tools& tools, const std::string& provider_name) {
    std::string tool_name = "ask_" + provider_name;
    tools.remove_tool(tool_name);
    dout(1) << "Unregistered provider tool: " + tool_name << std::endl;
}
