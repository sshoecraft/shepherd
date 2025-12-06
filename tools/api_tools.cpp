#include "api_tools.h"
#include "tools.h"
#include "../shepherd.h"
#include "../backends/backend.h"
#include "../backends/factory.h"
#include "../message.h"
#include "../session.h"
#include "../logger.h"
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

        // Create backend instance
        std::string backend_name = this->config.backend;
        auto backend = BackendFactory::create_backend(backend_name, context_size);

        if (!backend) {
            // Restore original config
            ::config = std::move(original_config);

            result["error"] = "Failed to create backend: " + this->config.backend;
            result["success"] = false;
            return result;
        }

        // Create a simple session with just the user prompt
        Session temp_session;
        temp_session.system_message = "You are a helpful AI assistant.";
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

        LOG_DEBUG("APIToolAdapter: sub-session has " + std::to_string(temp_session.tools.size()) + " tools");

        // Initialize backend (validates API key, etc.)
        backend->initialize(temp_session);

        // Use configured max_tokens (0 = auto-calculate from available space)
        int generation_max_tokens = max_tokens;

        // Tool execution loop - handle tool calls from sub-model
        const int MAX_TOOL_ITERATIONS = 10;
        int tool_iteration = 0;
        Response resp;

        while (tool_iteration < MAX_TOOL_ITERATIONS) {
            // Generate response
            resp = backend->generate_from_session(temp_session, generation_max_tokens);

            LOG_DEBUG("APIToolAdapter iteration " + std::to_string(tool_iteration) +
                      ": resp.code=" + std::to_string(resp.code) +
                      ", content.length=" + std::to_string(resp.content.length()) +
                      ", tool_calls=" + std::to_string(resp.tool_calls.size()) +
                      ", error=" + resp.error);

            // Check for errors
            if (resp.code != Response::SUCCESS) {
                break;
            }

            // If no tool calls, we're done
            if (resp.tool_calls.empty()) {
                break;
            }

            // Add assistant message with tool calls to session
            Message asst_msg(Message::ASSISTANT, resp.content);
            asst_msg.tool_calls_json = resp.tool_calls_json;
            temp_session.messages.push_back(asst_msg);

            // Execute each tool call
            for (const auto& tool_call : resp.tool_calls) {
                LOG_DEBUG("APIToolAdapter executing tool: " + tool_call.name);

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
                Message tool_msg(Message::TOOL, tool_result);
                tool_msg.tool_call_id = tool_call.tool_call_id;
                tool_msg.tool_name = tool_call.name;
                temp_session.messages.push_back(tool_msg);
            }

            tool_iteration++;
        }

        // Build result
        if (resp.code == Response::SUCCESS) {
            result["content"] = resp.content;
            result["success"] = true;

            // Include token usage if available
            if (resp.prompt_tokens > 0) {
                result["prompt_tokens"] = resp.prompt_tokens;
                result["completion_tokens"] = resp.completion_tokens;
                result["total_tokens"] = resp.prompt_tokens + resp.completion_tokens;
            }
        } else {
            result["error"] = resp.error.empty() ? "API call failed" : resp.error;
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
    LOG_INFO("Registered " + std::to_string(tools.api_tools.size()) + " provider tools");
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
        LOG_WARN("Provider not found: " + provider_name);
        return;
    }

    if (!found->is_api()) {
        LOG_DEBUG("Skipping non-API provider: " + provider_name);
        return;
    }

    // Create APIToolEntry from provider
    APIToolEntry entry = provider_to_tool_entry(*found);

    // Create adapter and register directly (pass tools pointer for sub-session population)
    auto adapter = std::make_unique<APIToolAdapter>(entry, &tools);
    std::string tool_name = adapter->name();
    tools.register_tool(std::move(adapter), "api");

    tools.build_all_tools();
    LOG_INFO("Registered provider as tool: " + tool_name);
}

void unregister_provider_tool(Tools& tools, const std::string& provider_name) {
    std::string tool_name = "ask_" + provider_name;
    tools.remove_tool(tool_name);
    LOG_INFO("Unregistered provider tool: " + tool_name);
}
