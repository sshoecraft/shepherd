#include "shepherd.h"
#include "api_tools.h"
#include "tools.h"
#include "../shepherd.h"
#include "backend.h"
#include "../backends/factory.h"
#include "../message.h"
#include "../session.h"

#include <sstream>

// ============================================================================
// APIToolAdapter implementation
// ============================================================================

APIToolAdapter::APIToolAdapter(const APIToolEntry& entry, Tools* tools)
    : config(entry), tools_ptr(tools) {
}

std::string APIToolAdapter::unsanitized_name() const {
    return config.name;
}

std::string APIToolAdapter::description() const {
    std::ostringstream desc;

    // Make the description more explicit about what this tool does
    if (config.backend == "anthropic") {
        desc << "Call Claude AI (" << config.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need Claude's analysis, expertise, or viewpoint on a question or problem.";
    } else if (config.backend == "openai") {
        desc << "Call ChatGPT/GPT (" << config.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need GPT's analysis, expertise, or viewpoint on a question or problem.";
    } else if (config.backend == "gemini") {
        desc << "Call Google Gemini (" << config.model << ") to get a second opinion or different perspective. ";
        desc << "Use this when you need Gemini's analysis, expertise, or viewpoint on a question or problem.";
    } else {
        desc << "Call another AI model (" << config.backend << "/" << config.model << ") to get a second opinion. ";
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
    prompt_param.description = "The question or task to ask " + config.backend;
    prompt_param.required = true;
    params.push_back(prompt_param);

    ParameterDef model_param;
    model_param.name = "model";
    model_param.type = "string";
    model_param.description = "Override model for this call (optional)";
    model_param.required = false;
    model_param.default_value = config.model;
    params.push_back(model_param);

    ParameterDef max_tokens_param;
    max_tokens_param.name = "max_tokens";
    max_tokens_param.type = "number";
    max_tokens_param.description = "Maximum tokens to generate (optional)";
    max_tokens_param.required = false;
    if (config.max_tokens > 0) {
        max_tokens_param.default_value = std::to_string(config.max_tokens);
    }
    params.push_back(max_tokens_param);

    return params;
}

std::map<std::string, std::any> APIToolAdapter::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    // Extract parameters - accept both "prompt" and "query" since models use both
    std::string prompt = tool_utils::get_string(args, "prompt");
    if (prompt.empty()) {
        prompt = tool_utils::get_string(args, "query");
    }
    std::string model = tool_utils::get_string(args, "model", config.model);
    int max_tokens = tool_utils::get_int(args, "max_tokens", config.max_tokens);

    if (prompt.empty()) {
        result["error"] = "prompt parameter is required";
        result["success"] = false;
        return result;
    }

    // Save the current global config (outside try block so catch can access it)
    std::unique_ptr<Config> original_config = std::move(::config);

    try {

        // Create temporary config for this tool
        auto temp_config = std::make_unique<Config>();
        temp_config->backend = this->config.backend;
        temp_config->model = model;
        temp_config->key = this->config.api_key;
        temp_config->api_base = this->config.api_base;
        temp_config->context_size = this->config.context_size;
        temp_config->streaming = false;  // Disable streaming for sub-backends

        // Swap in temporary config
        ::config = std::move(temp_config);

        // Use configured context size (0 = auto-detect from models database)
        size_t context_size = this->config.context_size;

        // Create a simple session with just the user prompt
        Session temp_session;
        temp_session.system_message = "You are a helpful AI assistant.";

        // Create backend instance with accumulating callback
        std::string backend_name = this->config.backend;
        std::string accumulated_content;
        Backend::EventCallback stub_callback = [&accumulated_content](CallbackEvent event, const std::string& content,
                                                   const std::string&, const std::string&) {
            if (event == CallbackEvent::CONTENT) {
                accumulated_content += content;
            }
            return true;
        };
        auto backend = BackendFactory::create_backend(backend_name, context_size, temp_session, stub_callback);

        if (!backend) {
            // Restore original config
            ::config = std::move(original_config);

            result["error"] = "Failed to create backend: " + this->config.backend;
            result["success"] = false;
            return result;
        }

        // Add user message to session
        Message user_msg(Message::USER, prompt);
        temp_session.messages.push_back(user_msg);

        // Populate tools (excluding this tool to prevent self-recursion)
        if (tools_ptr) {
            for (Tool* tool : tools_ptr->all_tools) {
                // Skip this tool to prevent infinite self-recursion
                // (e.g., ask_localhost calling ask_localhost)
                // But allow other ask_* tools (e.g., ask_localhost can call ask_sonnet)
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
                    nlohmann::json param_schema;
                    param_schema["type"] = param.type;
                    if (!param.description.empty()) {
                        param_schema["description"] = param.description;
                    }
                    schema["properties"][param.name] = param_schema;

                    if (param.required) {
                        required.push_back(param.name);
                    }
                }

                if (!required.empty()) {
                    schema["required"] = required;
                }

                session_tool.parameters = schema;
                temp_session.tools.push_back(session_tool);
            }
        }

        dout(1) << "APIToolAdapter: sub-session has " + std::to_string(temp_session.tools.size()) + " tools" << std::endl;

        // Backend initialization happens in constructor - no separate initialize() call needed

        // Use configured max_tokens (0 = auto-calculate from available space)
        int generation_max_tokens = max_tokens;

        // Tool execution loop - handle tool calls from sub-model
        const int MAX_TOOL_ITERATIONS = 10;
        int tool_iteration = 0;

        // Callback state for accumulating responses
        struct CallbackState {
            std::string content;
            std::vector<ToolParser::ToolCall> tool_calls;
            std::string finish_reason;
            std::string error;
            bool success = true;
        };
        CallbackState cb_state;

        // Set up callback to capture output
        Backend::EventCallback capture_callback = [&cb_state](CallbackEvent event, const std::string& content,
                                                               const std::string& name, const std::string& id) {
            switch (event) {
                case CallbackEvent::CONTENT:
                    cb_state.content += content;
                    break;
                case CallbackEvent::TOOL_CALL: {
                    ToolParser::ToolCall tc;
                    tc.name = name;
                    tc.tool_call_id = id;
                    tc.raw_json = content;
                    // Parse parameters from JSON
                    try {
                        auto params = nlohmann::json::parse(content);
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
                    cb_state.tool_calls.push_back(tc);
                    break;
                }
                case CallbackEvent::ERROR:
                    cb_state.error = content;
                    cb_state.success = false;
                    break;
                case CallbackEvent::STOP:
                    cb_state.finish_reason = content;
                    break;
                default:
                    break;
            }
            return true;
        };

        // Replace stub callback with capture callback to get TOOL_CALL events
        backend->callback = capture_callback;

        while (tool_iteration < MAX_TOOL_ITERATIONS) {
            // Reset callback state
            cb_state = CallbackState{};

            // Generate response (output flows through callback)
            backend->generate_from_session(temp_session, generation_max_tokens);

            dout(1) << "APIToolAdapter iteration " + std::to_string(tool_iteration) +
                      ": success=" + std::string(cb_state.success ? "true" : "false") +
                      ", content.length=" + std::to_string(cb_state.content.length()) +
                      ", tool_calls=" + std::to_string(cb_state.tool_calls.size()) +
                      ", error=" + cb_state.error << std::endl;

            // Check for errors
            if (!cb_state.success) {
                break;
            }

            // If no tool calls, we're done
            if (cb_state.tool_calls.empty()) {
                break;
            }

            // Add assistant message with tool calls to session
            Message asst_msg(Message::ASSISTANT, cb_state.content);
            temp_session.messages.push_back(asst_msg);

            // Execute each tool call
            for (const auto& tool_call : cb_state.tool_calls) {
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
                    // Execute the tool
                    auto result_map = tool->execute(tool_call.parameters);
                    // Convert result map to JSON
                    nlohmann::json result_json;
                    for (const auto& [key, value] : result_map) {
                        if (auto* str = std::any_cast<std::string>(&value)) {
                            result_json[key] = *str;
                        } else if (auto* i = std::any_cast<int>(&value)) {
                            result_json[key] = *i;
                        } else if (auto* d = std::any_cast<double>(&value)) {
                            result_json[key] = *d;
                        } else if (auto* b = std::any_cast<bool>(&value)) {
                            result_json[key] = *b;
                        } else {
                            result_json[key] = "<unknown type>";
                        }
                    }
                    tool_result = result_json.dump();
                } else {
                    tool_result = "{\"error\": \"Tool not found: " + tool_call.name + "\"}";
                }

                // Add tool result to session
                Message tool_msg(Message::TOOL_RESPONSE, tool_result);
                tool_msg.tool_call_id = tool_call.tool_call_id;
                tool_msg.tool_name = tool_call.name;
                temp_session.messages.push_back(tool_msg);
            }

            tool_iteration++;
        }

        // Build result
        if (cb_state.success) {
            result["content"] = cb_state.content;
            result["success"] = true;
        } else {
            result["error"] = cb_state.error.empty() ? "API call failed" : cb_state.error;
            result["success"] = false;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("Exception: ") + e.what();
        result["success"] = false;
    }

    // Restore original config (always execute this, whether success or exception)
    ::config = std::move(original_config);

    return result;
}

// ============================================================================
// Provider tool registration functions
// ============================================================================

APIToolEntry provider_to_tool_entry(const Provider& provider) {
    APIToolEntry entry;
    entry.name = "ask_" + provider.name;
    entry.backend = provider.type;
    entry.model = provider.model;
    entry.context_size = provider.context_size;
    entry.api_key = provider.api_key;
    entry.api_base = provider.base_url;
    entry.max_tokens = provider.max_tokens;

    // Ollama uses num_predict instead of max_tokens
    if (provider.type == "ollama" && provider.num_predict > 0) {
        entry.max_tokens = provider.num_predict;
    }

    return entry;
}

void register_provider_tools(Tools& tools, const std::string& active_provider) {
    // Clear existing API tools first
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

        // Create APIToolEntry from provider
        APIToolEntry entry = provider_to_tool_entry(p);

        // Create adapter and register directly (pass tools pointer for sub-session population)
        auto adapter = std::make_unique<APIToolAdapter>(entry, &tools);
        tools.register_tool(std::move(adapter), "api");
    }

    tools.build_all_tools();
    dout(1) << "Registered " + std::to_string(tools.api_tools.size()) + " provider tools" << std::endl;
}

void register_provider_as_tool(Tools& tools, const std::string& provider_name) {
    auto providers = Provider::load_providers();

    // Find the provider by name
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

    // Create APIToolEntry from provider
    APIToolEntry entry = provider_to_tool_entry(*found);

    // Create adapter and register directly (pass tools pointer for sub-session population)
    auto adapter = std::make_unique<APIToolAdapter>(entry, &tools);
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
