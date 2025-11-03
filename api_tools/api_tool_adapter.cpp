#include "api_tool_adapter.h"
#include "../shepherd.h"
#include "../backends/backend.h"
#include "../backends/factory.h"
#include "../message.h"
#include <sstream>

APIToolAdapter::APIToolAdapter(const APIToolEntry& entry)
    : config(entry) {
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

    // Extract parameters
    std::string prompt = tool_utils::get_string(args, "prompt");
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

        // Initialize backend (validates API key, etc.)
        backend->initialize(temp_session);

        // Use configured max_tokens (0 = auto-calculate from available space)
        int generation_max_tokens = max_tokens;

        // Generate response
        Response resp = backend->generate_from_session(temp_session, generation_max_tokens);

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
