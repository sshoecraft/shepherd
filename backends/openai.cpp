
#include "shepherd.h"
#include "openai.h"
#include "nlohmann/json.hpp"
#include "sse_parser.h"
#include "../tools/utf8_sanitizer.h"

using json = nlohmann::json;

OpenAIBackend::OpenAIBackend(size_t context_size, Session& session, EventCallback callback)
    : ApiBackend(context_size, session, callback) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Load Azure OpenAI specific config
    if (!config->json.is_null() && !config->json.empty()) {
        if (config->json.contains("deployment_name")) {
            deployment_name = config->json["deployment_name"].get<std::string>();
            dout(1) << "Azure deployment_name: " + deployment_name << std::endl;
        }
        if (config->json.contains("api_version")) {
            api_version = config->json["api_version"].get<std::string>();
            dout(1) << "Azure api_version: " + api_version << std::endl;
        }
        if (config->json.contains("openai_strict")) {
            openai_strict = config->json["openai_strict"].get<bool>();
            dout(1) << "OpenAI strict mode: " + std::string(openai_strict ? "enabled" : "disabled") << std::endl;
        }
    }

    // Initialize model_config with generic defaults to avoid uninitialized memory
    model_config = ModelConfig::create_generic();

    // Detect model configuration from Models database (if model is specified)
    // If model is empty, it will be auto-detected below
    if (!model_name.empty()) {
        model_config = Models::detect_from_api_model("openai", model_name);
        max_output_tokens = model_config.max_output_tokens;
        dout(1) << "Detected model config: context=" + std::to_string(model_config.context_window) +
                  ", max_output=" + std::to_string(model_config.max_output_tokens) +
                  ", param_name=" + model_config.max_tokens_param_name << std::endl;
    }

    // Override max_tokens_param_name from provider config if specified
    if (!config->json.is_null() && config->json.contains("max_tokens_param_name")) {
        std::string override_param = config->json["max_tokens_param_name"].get<std::string>();
        if (!override_param.empty()) {
            model_config.max_tokens_param_name = override_param;
            dout(1) << "Provider override max_tokens_param_name: " + override_param << std::endl;
        }
    }

    // Set API endpoint from config (api_base or default)
    if (!config->api_base.empty()) {
        // User specified custom API base
        api_endpoint = config->api_base;

        // Check if this is Azure OpenAI format (has deployment_name and api_version)
        if (!deployment_name.empty() && !api_version.empty()) {
            // Azure OpenAI format: {base}/openai/deployments/{deployment}/chat/completions?api-version={version}
            // Remove trailing slash if present
            if (api_endpoint.back() == '/') {
                api_endpoint.pop_back();
            }
            // Build Azure URL
            api_endpoint = api_endpoint + "/openai/deployments/" + deployment_name + "/chat/completions";
            dout(1) << "Using Azure OpenAI endpoint: " + api_endpoint + "?api-version=" + api_version << std::endl;
        } else {
            // Standard OpenAI format
            // Ensure it has /chat/completions endpoint
            if (api_endpoint.find("/chat/completions") == std::string::npos) {
                if (api_endpoint.back() == '/') {
                    api_endpoint += "chat/completions";
                } else {
                    api_endpoint += "/chat/completions";
                }
            }
            dout(1) << "Using custom API endpoint: " + api_endpoint << std::endl;
        }
    }
    // else: keep default api_endpoint = "https://api.openai.com/v1/chat/completions" from header

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config();

    // --- Initialization (formerly in initialize()) ---

    // Auto-detect Azure if deployment_name and api_version are set
    bool is_azure = (!deployment_name.empty() && !api_version.empty());
    if (is_azure && !openai_strict) {
        openai_strict = true;  // Azure endpoints don't support non-standard params
        dout(1) << "Auto-enabled openai_strict for Azure endpoint" << std::endl;
    }

    // Validate API key (only required for actual OpenAI API, not OAuth-based endpoints)
    bool requires_api_key = (api_endpoint.find("api.openai.com") != std::string::npos) ||
                            (api_endpoint.find("openai.azure.com") != std::string::npos);
    if (api_key.empty() && requires_api_key) {
        std::cerr << "OpenAI API key is required for api.openai.com" << std::endl;
        throw std::runtime_error("OpenAI API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        dout(1) << "No model specified, querying server for available models" << std::endl;
        auto models = get_models();
        if (!models.empty()) {
            model_name = models[0];
            dout(1) << "Using model from server: " + model_name << std::endl;
            model_config = Models::detect_from_api_model("openai", model_name);
            max_output_tokens = model_config.max_output_tokens;
        } else {
            dout(1) << std::string("WARNING: ") +"Failed to query server for model, will use first API response to determine" << std::endl;
        }
    }

    // Query API for context size first (server knows its own limits)
    if (this->context_size == 0) {
        size_t api_context_size = query_model_context_size(model_name);
        if (api_context_size > 0) {
            this->context_size = api_context_size;
            dout(1) << "Using API's context size: " + std::to_string(this->context_size) << std::endl;
        }
    }

    // Fall back to local model config if API didn't provide context size
    if (this->context_size == 0 && model_config.context_window > 0) {
        this->context_size = model_config.context_window;
        dout(1) << "Using model config's context size: " + std::to_string(this->context_size) << std::endl;
    }

    // If we still don't have a context size, default to 16k with warning
    if (this->context_size == 0) {
        this->context_size = 16384;
        std::cerr << "Warning: Cannot determine context size for model '" << model_name
                  << "'. Defaulting to 16k. Use --context-size to specify explicitly." << std::endl;
    }

    // NOTE: Context safety margin disabled - now sending proper sampling params (top_p, top_k)
    // which should prevent runaway generation at high context
    // // Apply safety margin to auto-detected context size
    // // This accounts for server-side prompt caching and other overhead
    // // Reduce by 10% or 4096 tokens, whichever is smaller
    // size_t original_context = this->context_size;
    // size_t margin = std::min(this->context_size / 10, (size_t)4096);
    // if (margin > 0 && this->context_size > margin) {
    //     this->context_size -= margin;
    //     dout(1) << "Applied context safety margin: " + std::to_string(original_context) +
    //               " -> " + std::to_string(this->context_size) +
    //               " (reduced by " + std::to_string(margin) + " tokens)" << std::endl;
    // }

    // Calibrate token counts (if enabled in config)
    if (config->calibration) {
        dout(1) << "Calibrating token counts..." << std::endl;
        calibrate_token_counts(session);
    } else {
        dout(1) << "Calibration disabled, using default estimates" << std::endl;
        session.system_message_tokens = estimate_message_tokens(session.system_message);
        session.last_prompt_tokens = session.system_message_tokens;
    }

    dout(1) << "OpenAI backend initialized successfully" << std::endl;
}

OpenAIBackend::~OpenAIBackend() {
}

Response OpenAIBackend::parse_http_response(const HttpResponse& http_response) {
    Response resp;

    // Check HTTP status
    if (!http_response.is_success()) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.finish_reason = "error";

        // Try to parse error JSON
        try {
            nlohmann::json error_json = nlohmann::json::parse(http_response.body);
            if (error_json.contains("error")) {
                // OpenAI format: {"error": {"message": "..."}} or {"error": "..."}
                if (error_json["error"].is_object() && error_json["error"].contains("message")) {
                    resp.error = error_json["error"]["message"].get<std::string>();
                } else if (error_json["error"].is_string()) {
                    resp.error = error_json["error"].get<std::string>();
                }
            } else if (error_json.contains("message")) {
                // TRT-LLM/vLLM format: {"object":"error","message":"...","type":"..."}
                resp.error = error_json["message"].get<std::string>();
            }
        } catch (...) {
            // If JSON parsing fails, use raw error
            resp.error = http_response.error_message.empty() ? "API request failed" : http_response.error_message;
        }

        if (resp.error.empty()) {
            resp.error = http_response.error_message.empty() ? "Unknown API error" : http_response.error_message;
        }

        // Check if this is a MAX_TOKENS_TOO_HIGH error (vLLM format)
        // Format: "'max_tokens' or 'max_completion_tokens' is too large: 16744. This model's maximum context length is 32768 tokens and your request has 20956 input tokens"
        if (resp.error.find("max_tokens") != std::string::npos &&
            resp.error.find("is too large") != std::string::npos &&
            resp.error.find("your request has") != std::string::npos) {

            dout(1) << "Detected MAX_TOKENS_TOO_HIGH error, parsing..." << std::endl;

            // Parse actual prompt tokens: "your request has 20956 input tokens"
            size_t request_pos = resp.error.find("your request has ");
            if (request_pos != std::string::npos) {
                try {
                    size_t start = request_pos + 17;  // Length of "your request has "
                    size_t end = resp.error.find(" ", start);
                    resp.actual_prompt_tokens = std::stoi(resp.error.substr(start, end - start));

                    // Parse max_tokens requested: "is too large: 16744"
                    size_t large_pos = resp.error.find("is too large: ");
                    int max_tokens_requested = 0;
                    if (large_pos != std::string::npos) {
                        start = large_pos + 14;  // Length of "is too large: "
                        end = resp.error.find(".", start);
                        max_tokens_requested = std::stoi(resp.error.substr(start, end - start));
                    }

                    // Parse context size: "maximum context length is 32768 tokens"
                    size_t max_pos = resp.error.find("maximum context length is ");
                    if (max_pos != std::string::npos) {
                        start = max_pos + 26;  // Length of "maximum context length is "
                        end = resp.error.find(" tokens", start);
                        int max_context = std::stoi(resp.error.substr(start, end - start));

                        // overflow_tokens = how much we need to reduce the prompt by
                        // Formula: overflow = actual_prompt + max_tokens_requested - max_context
                        resp.overflow_tokens = resp.actual_prompt_tokens + max_tokens_requested - max_context;
                        if (resp.overflow_tokens < 0) resp.overflow_tokens = 0;

                        resp.code = Response::MAX_TOKENS_TOO_HIGH;
                        dout(1) << "Parsed MAX_TOKENS_TOO_HIGH: actual_prompt=" + std::to_string(resp.actual_prompt_tokens) +
                                  ", max_tokens=" + std::to_string(max_tokens_requested) +
                                  ", max_context=" + std::to_string(max_context) +
                                  ", overflow=" + std::to_string(resp.overflow_tokens) << std::endl;
                    }
                } catch (const std::exception& e) {
                    dout(1) << "Failed to parse MAX_TOKENS_TOO_HIGH details: " + std::string(e.what()) << std::endl;
                }
            }
        }

        return resp;
    }

    // Parse successful response
    try {
        std::string sanitized_body = utf8_sanitizer::sanitize_utf8(http_response.body);
        nlohmann::json json_resp = nlohmann::json::parse(sanitized_body);

        // Extract content
        if (json_resp.contains("choices") && !json_resp["choices"].empty()) {
            const auto& choice = json_resp["choices"][0];

            // Get finish_reason
            if (choice.contains("finish_reason")) {
                resp.finish_reason = choice["finish_reason"].get<std::string>();
            }

            // Get content from message
            if (choice.contains("message")) {
                const auto& message = choice["message"];
                if (message.contains("content") && !message["content"].is_null()) {
                    resp.content = message["content"].get<std::string>();
                }

                // Parse tool calls if present
                if (message.contains("tool_calls") && message["tool_calls"].is_array() && !message["tool_calls"].empty()) {
                    // Store raw tool_calls JSON for message persistence (only if non-empty)
                    resp.tool_calls_json = message["tool_calls"].dump();

                    for (const auto& tc : message["tool_calls"]) {
                        ToolParser::ToolCall tool_call;

                        if (tc.contains("id")) {
                            tool_call.tool_call_id = tc["id"].get<std::string>();
                        }

                        if (tc.contains("function")) {
                            const auto& func = tc["function"];
                            if (func.contains("name")) {
                                tool_call.name = func["name"].get<std::string>();
                            }
                            if (func.contains("arguments")) {
                                std::string args_str = func["arguments"].get<std::string>();
                                tool_call.raw_json = args_str;

                                // Parse arguments JSON into parameters map
                                try {
                                    nlohmann::json args_json = nlohmann::json::parse(args_str);
                                    for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                                        // Convert JSON values to std::any
                                        if (it.value().is_string()) {
                                            tool_call.parameters[it.key()] = it.value().get<std::string>();
                                        } else if (it.value().is_number_integer()) {
                                            tool_call.parameters[it.key()] = it.value().get<int>();
                                        } else if (it.value().is_number_float()) {
                                            tool_call.parameters[it.key()] = it.value().get<double>();
                                        } else if (it.value().is_boolean()) {
                                            tool_call.parameters[it.key()] = it.value().get<bool>();
                                        } else {
                                            // For complex types, store as string
                                            tool_call.parameters[it.key()] = it.value().dump();
                                        }
                                    }
                                } catch (const std::exception& e) {
                                    dout(1) << std::string("WARNING: ") +"Failed to parse tool call arguments: " + std::string(e.what()) << std::endl;
                                }
                            }
                        }

                        resp.tool_calls.push_back(tool_call);
                    }
                }
            }
        }

        // Extract token usage
        if (json_resp.contains("usage")) {
            const auto& usage = json_resp["usage"];
            if (usage.contains("prompt_tokens")) {
                resp.prompt_tokens = usage["prompt_tokens"].get<int>();
            }
            if (usage.contains("completion_tokens")) {
                resp.completion_tokens = usage["completion_tokens"].get<int>();
            }

            // Log full usage info
            int total = usage.value("total_tokens", resp.prompt_tokens + resp.completion_tokens);
            dout(1) << "API Usage - prompt_tokens: " + std::to_string(resp.prompt_tokens) +
                     ", completion_tokens: " + std::to_string(resp.completion_tokens) +
                     ", total_tokens: " + std::to_string(total) << std::endl;
            dout(1) << "Full usage JSON: " + usage.dump() << std::endl;
        }

        resp.success = true;
        resp.code = Response::SUCCESS;

    } catch (const std::exception& e) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.finish_reason = "error";
        resp.error = "Failed to parse API response: " + std::string(e.what());
    }

    return resp;
}

std::vector<std::string> OpenAIBackend::fetch_models() {
    std::vector<std::string> result;

    if (!http_client) {
        return result;
    }

    std::string response = make_get_request("/models");
    if (response.empty()) {
        return result;
    }

    try {
        auto j = json::parse(response);
        if (j.contains("data") && j["data"].is_array()) {
            for (const auto& model : j["data"]) {
                if (model.contains("id") && model["id"].is_string()) {
                    result.push_back(model["id"].get<std::string>());
                }
            }
        }
    } catch (const json::exception& e) {
        dout(1) << "Failed to parse /models response: " + std::string(e.what()) << std::endl;
    }

    return result;
}

std::string OpenAIBackend::make_get_request(const std::string& endpoint) {
    if (!http_client) {
        dout(1) << "HTTP client not initialized" << std::endl;
        return "";
    }

    // Build full URL
    std::string base_url = api_endpoint;
    // Extract base (remove /chat/completions if present)
    size_t pos = base_url.find("/chat/completions");
    if (pos != std::string::npos) {
        base_url = base_url.substr(0, pos);
    }
    std::string full_url = base_url + endpoint;

    // Prepare headers
    std::map<std::string, std::string> headers;
    headers["Authorization"] = "Bearer " + api_key;

    // Make GET request
    HttpResponse response = http_client->get(full_url, headers);

    if (!response.is_success()) {
        // Check for connection errors (status 0 = couldn't connect)
        if (response.status_code == 0) {
            std::string error_msg = response.error_message.empty()
                ? "Could not connect to server"
                : response.error_message;
            dout(1) << "Connection failed: " + error_msg << std::endl;
            throw BackendError("Failed to connect to API server at " + full_url + ": " + error_msg +
                             "\nPlease check that the server is running and accessible.");
        }

        // Check for authentication errors - these should fail immediately
        if (response.status_code == 401 || response.status_code == 403) {
            std::string error_msg = "Authentication failed";
            try {
                json error_json = json::parse(response.body);
                if (error_json.contains("error") && error_json["error"].contains("message")) {
                    error_msg = error_json["error"]["message"].get<std::string>();
                }
            } catch (...) {
                error_msg = response.error_message.empty() ? error_msg : response.error_message;
            }
            dout(1) << "Authentication failed: " + error_msg << std::endl;
            throw BackendError("Authentication failed: " + error_msg);
        }

        dout(1) << "GET request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message << std::endl;
        return "";
    }

    return response.body;
}
int OpenAIBackend::extract_tokens_to_evict(const HttpResponse& response) {
    // Extract error message from HTTP response
    std::string error_message = response.error_message;
    if (error_message.empty() && !response.body.empty()) {
        // Try to parse JSON error message
        try {
            auto json_body = nlohmann::json::parse(response.body);
            if (json_body.contains("error") && json_body["error"].contains("message")) {
                error_message = json_body["error"]["message"].get<std::string>();
            }
        } catch (...) {
            error_message = response.body;
        }
    }

    // OpenAI format: "This model's maximum context length is 16385 tokens. However, your messages resulted in 44366 tokens."
    // Shepherd server format: "would need 54721 tokens but limit is 32768 tokens"

    dout(1) << "extract_tokens_to_evict: parsing error message: " + error_message << std::endl;

    int actual_tokens = -1;
    int max_tokens = -1;

    // Try shepherd server format first: "would need X tokens but limit is Y tokens"
    size_t need_pos = error_message.find("would need ");
    size_t limit_pos = error_message.find("but limit is ");
    if (need_pos != std::string::npos && limit_pos != std::string::npos) {
        dout(1) << "Found shepherd server format markers" << std::endl;
        try {
            size_t start = need_pos + 11;
            size_t end = error_message.find(" tokens", start);
            actual_tokens = std::stoi(error_message.substr(start, end - start));

            start = limit_pos + 13;
            end = error_message.find(" tokens", start);
            max_tokens = std::stoi(error_message.substr(start, end - start));
            dout(1) << "Parsed shepherd format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens) << std::endl;
        } catch (const std::exception& e) {
            dout(1) << "Exception parsing shepherd format: " + std::string(e.what()) << std::endl;
        }
    }

    // Try OpenAI classic format: "maximum context length is X tokens. However, your messages resulted in Y tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t resulted_pos = error_message.find("resulted in ");
        dout(1) << "OpenAI classic format search: max_pos=" + std::to_string(max_pos) + ", resulted_pos=" + std::to_string(resulted_pos) << std::endl;
        if (max_pos != std::string::npos && resulted_pos != std::string::npos) {
            dout(1) << "Found OpenAI classic format markers" << std::endl;
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                dout(1) << "Parsing max_tokens from position " + std::to_string(start) + " to " + std::to_string(end) << std::endl;
                std::string max_str = error_message.substr(start, end - start);
                dout(1) << "max_tokens string: '" + max_str + "'" << std::endl;
                max_tokens = std::stoi(max_str);

                start = resulted_pos + 12;
                end = error_message.find(" tokens", start);
                dout(1) << "Parsing actual_tokens from position " + std::to_string(start) + " to " + std::to_string(end) << std::endl;
                std::string actual_str = error_message.substr(start, end - start);
                dout(1) << "actual_tokens string: '" + actual_str + "'" << std::endl;
                actual_tokens = std::stoi(actual_str);
                dout(1) << "Parsed OpenAI classic format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens) << std::endl;
            } catch (const std::exception& e) {
                dout(1) << "Exception parsing OpenAI classic format: " + std::string(e.what()) << std::endl;
            }
        }
    }

    // Try vLLM MAX_TOKENS_TOO_HIGH format: "is too large: X. ... maximum context length is Y ... your request has Z input tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t too_large_pos = error_message.find("is too large: ");
        size_t max_pos = error_message.find("maximum context length is ");
        size_t request_pos = error_message.find("your request has ");
        if (too_large_pos != std::string::npos && max_pos != std::string::npos && request_pos != std::string::npos) {
            dout(1) << "Found vLLM MAX_TOKENS_TOO_HIGH format markers" << std::endl;
            try {
                // Parse max_tokens_requested from "is too large: 27790"
                size_t start = too_large_pos + 14;  // After "is too large: "
                size_t end = error_message.find(".", start);
                int max_tokens_requested = std::stoi(error_message.substr(start, end - start));

                // Parse max_context from "maximum context length is 32768"
                start = max_pos + 26;
                end = error_message.find(" tokens", start);
                int max_context = std::stoi(error_message.substr(start, end - start));

                // Parse actual_prompt from "your request has 30021 input tokens"
                start = request_pos + 17;
                end = error_message.find(" ", start);
                int actual_prompt = std::stoi(error_message.substr(start, end - start));

                // Calculate overflow: how many tokens we need to free
                int overflow = actual_prompt + max_tokens_requested - max_context;
                dout(1) << "Parsed vLLM MAX_TOKENS_TOO_HIGH: actual_prompt=" + std::to_string(actual_prompt) +
                         ", max_tokens_requested=" + std::to_string(max_tokens_requested) +
                         ", max_context=" + std::to_string(max_context) +
                         ", overflow=" + std::to_string(overflow) << std::endl;
                return overflow > 0 ? overflow : -1;
            } catch (const std::exception& e) {
                dout(1) << "Exception parsing vLLM MAX_TOKENS_TOO_HIGH format: " + std::string(e.what()) << std::endl;
            }
        }
    }

    // Try vLLM/modern format: "your request has X input tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t request_pos = error_message.find("your request has ");
        dout(1) << "vLLM format search: max_pos=" + std::to_string(max_pos) + ", request_pos=" + std::to_string(request_pos) << std::endl;
        if (max_pos != std::string::npos && request_pos != std::string::npos) {
            dout(1) << "Found vLLM format markers" << std::endl;
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                max_tokens = std::stoi(error_message.substr(start, end - start));

                start = request_pos + 17;
                end = error_message.find(" ", start);
                actual_tokens = std::stoi(error_message.substr(start, end - start));
                dout(1) << "Parsed vLLM format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens) << std::endl;
            } catch (const std::exception& e) {
                dout(1) << "Exception parsing vLLM format: " + std::string(e.what()) << std::endl;
            }
        }
    }

    // Try OpenAI/vLLM detailed format: "you requested X tokens (Y in the messages, Z in the completion)"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t requested_pos = error_message.find("you requested ");
        dout(1) << "OpenAI detailed format search: max_pos=" + std::to_string(max_pos) + ", requested_pos=" + std::to_string(requested_pos) << std::endl;
        if (max_pos != std::string::npos && requested_pos != std::string::npos) {
            dout(1) << "Found OpenAI detailed format markers" << std::endl;
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                max_tokens = std::stoi(error_message.substr(start, end - start));

                start = requested_pos + 14;
                end = error_message.find(" tokens", start);
                actual_tokens = std::stoi(error_message.substr(start, end - start));
                dout(1) << "Parsed OpenAI detailed format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens) << std::endl;
            } catch (const std::exception& e) {
                dout(1) << "Exception parsing OpenAI detailed format: " + std::string(e.what()) << std::endl;
            }
        }
    }

    if (actual_tokens > 0 && max_tokens > 0) {
        int to_evict = actual_tokens - max_tokens;
        dout(1) << "Tokens to evict: " + std::to_string(to_evict) << std::endl;
        return to_evict;
    }

    // Can't parse - return error
    dout(1) << "Failed to parse token count from error message: " + error_message << std::endl;
    return -1;
}

// Implement required ApiBackend pure virtual methods
nlohmann::json OpenAIBackend::build_request_from_session(const Session& session, int max_tokens) {
    nlohmann::json request;
    request["model"] = model_name;
    if (!session.user_id.empty()) {
        request["user"] = session.user_id;
    }

    // Build messages array from complete session
    nlohmann::json messages = nlohmann::json::array();

    // Add system message
    if (!session.system_message.empty()) {
        messages.push_back({{"role", "system"}, {"content", session.system_message}});
    }

    // Add all messages from session
    for (const auto& msg : session.messages) {
        nlohmann::json jmsg;
        jmsg["role"] = msg.get_role();
        jmsg["content"] = utf8_sanitizer::sanitize_utf8(msg.content);

        // Restore tool_calls for assistant messages that made tool calls
        if (msg.role == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = nlohmann::json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                dout(1) << std::string("WARNING: ") +"Failed to parse stored tool_calls: " + std::string(e.what()) << std::endl;
            }
        }

        if (msg.role == Message::TOOL_RESPONSE && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    request["messages"] = messages;

    // Add tools if present
    dout(1) << "Session has " + std::to_string(session.tools.size()) + " tools" << std::endl;
    if (!session.tools.empty()) {
        nlohmann::json tools = nlohmann::json::array();
        for (const auto& tool : session.tools) {
            // OpenAI requires array properties to have 'items' field
            // Fix up the schema if needed
            nlohmann::json params = tool.parameters;
            if (params.contains("properties") && params["properties"].is_object()) {
                for (auto& [key, prop] : params["properties"].items()) {
                    if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                        // Add default items schema
                        prop["items"] = {{"type", "object"}};
                    }
                }
            }

            nlohmann::json jtool;
            jtool["type"] = "function";
            jtool["function"] = {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", params}
            };
            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add max_tokens if specified (use model-specific parameter name)
    // Note: max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    if (max_tokens > 0) {
        request[model_config.max_tokens_param_name] = max_tokens;
    }

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Only send sampling parameters if non-default (reasoning models like o1/gpt-5.x reject them)
        // OpenAI defaults: temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0
        if (!openai_strict || (temperature >= 0.01f && temperature <= 1.99f && temperature != 1.0f)) {
            request["temperature"] = temperature;
        }
        if (!openai_strict || (top_p < 0.99f)) {
            request["top_p"] = top_p;
        }
        if (!openai_strict || frequency_penalty != 0.0f) {
            request["frequency_penalty"] = frequency_penalty;
        }
        if (!openai_strict || presence_penalty != 0.0f) {
            request["presence_penalty"] = presence_penalty;
        }

        // Only send non-standard parameters to non-strict endpoints (Shepherd servers, vLLM, etc.)
        // OpenAI/Azure rejects unknown parameters like top_k and repetition_penalty
        if (!openai_strict) {
            if (top_k > 0) {
                request["top_k"] = top_k;
            }
            if (repeat_penalty != 0.0f) {
                request["repetition_penalty"] = repeat_penalty;
            }
        }
    }

    // Add stop sequences if configured
    dout(1) << "stop_sequences.size()=" + std::to_string(stop_sequences.size()) << std::endl;
    if (!stop_sequences.empty()) {
        json stop_array = json::array();
        for (const auto& seq : stop_sequences) {
            stop_array.push_back(seq);
        }
        request["stop"] = stop_array;
        dout(1) << "Added stop sequences to request: " + stop_array.dump() << std::endl;
    }

    // Add special headers if any
    if (!model_config.special_headers.empty()) {
        dout(1) << "Model has " + std::to_string(model_config.special_headers.size()) + " special headers" << std::endl;
    }

    return request;
}

nlohmann::json OpenAIBackend::build_request(const Session& session,
                                             Message::Role role,
                                             const std::string& content,
                                             const std::string& tool_name,
                                             const std::string& tool_id,
                                             int max_tokens) {
    nlohmann::json request;
    request["model"] = model_name;
    if (!session.user_id.empty()) {
        request["user"] = session.user_id;
    }

    // Build messages array
    nlohmann::json messages = nlohmann::json::array();

    // Add system message
    if (!session.system_message.empty()) {
        messages.push_back({{"role", "system"}, {"content", session.system_message}});
    }

    // Add existing messages
    for (const auto& msg : session.messages) {
        nlohmann::json jmsg;
        jmsg["role"] = msg.get_role();
        jmsg["content"] = utf8_sanitizer::sanitize_utf8(msg.content);

        // Restore tool_calls for assistant messages that made tool calls
        if (msg.role == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = nlohmann::json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                dout(1) << std::string("WARNING: ") +"Failed to parse stored tool_calls: " + std::string(e.what()) << std::endl;
            }
        }

        if (msg.role == Message::TOOL_RESPONSE && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    // Add the new message being sent
    nlohmann::json new_msg;
    // Convert Message::Role to role string
    std::string role_str;
    switch (role) {
        case Message::SYSTEM: role_str = "system"; break;
        case Message::USER: role_str = "user"; break;
        case Message::ASSISTANT: role_str = "assistant"; break;
        case Message::TOOL_RESPONSE: role_str = "tool"; break;
        case Message::FUNCTION: role_str = "function"; break;
    }
    new_msg["role"] = role_str;
    new_msg["content"] = utf8_sanitizer::sanitize_utf8(content);
    if (!tool_name.empty()) new_msg["name"] = tool_name;
    if (!tool_id.empty()) new_msg["tool_call_id"] = tool_id;
    messages.push_back(new_msg);
    
    request["messages"] = messages;

    dout(1) << "Built request with " + std::to_string(messages.size()) + " messages (session has " +
             std::to_string(session.messages.size()) + " messages" << std::endl;
    dout(1) << "Session has " + std::to_string(session.tools.size()) + " tools" << std::endl;

    // Add tools if present
    if (!session.tools.empty()) {
        nlohmann::json tools = nlohmann::json::array();
        for (const auto& tool : session.tools) {
            // OpenAI requires array properties to have 'items' field
            // Fix up the schema if needed
            nlohmann::json params = tool.parameters;
            if (params.contains("properties") && params["properties"].is_object()) {
                for (auto& [key, prop] : params["properties"].items()) {
                    if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                        // Add default items schema
                        prop["items"] = {{"type", "object"}};
                    }
                }
            }

            nlohmann::json jtool;
            jtool["type"] = "function";
            jtool["function"] = {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", params}
            };
            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add max_tokens if specified (use model-specific parameter name)
    // Note: max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    if (max_tokens > 0) {
        request[model_config.max_tokens_param_name] = max_tokens;
    }

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Only send sampling parameters if non-default (reasoning models like o1/gpt-5.x reject them)
        // OpenAI defaults: temperature=1.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0
        if (!openai_strict || (temperature >= 0.01f && temperature <= 1.99f && temperature != 1.0f)) {
            request["temperature"] = temperature;
        }
        if (!openai_strict || (top_p < 0.99f)) {
            request["top_p"] = top_p;
        }
        if (!openai_strict || frequency_penalty != 0.0f) {
            request["frequency_penalty"] = frequency_penalty;
        }
        if (!openai_strict || presence_penalty != 0.0f) {
            request["presence_penalty"] = presence_penalty;
        }

        // Only send non-standard parameters to non-strict endpoints (Shepherd servers, vLLM, etc.)
        // OpenAI/Azure rejects unknown parameters like top_k and repetition_penalty
        if (!openai_strict) {
            if (top_k > 0) {
                request["top_k"] = top_k;
            }
            if (repeat_penalty != 0.0f) {
                request["repetition_penalty"] = repeat_penalty;
            }
        }
    }

    // Add stop sequences if configured
    dout(1) << "stop_sequences.size()=" + std::to_string(stop_sequences.size()) << std::endl;
    if (!stop_sequences.empty()) {
        json stop_array = json::array();
        for (const auto& seq : stop_sequences) {
            stop_array.push_back(seq);
        }
        request["stop"] = stop_array;
        dout(1) << "Added stop sequences to request: " + stop_array.dump() << std::endl;
    }

    return request;
}

std::string OpenAIBackend::parse_response(const nlohmann::json& response) {
    if (response.contains("choices") && !response["choices"].empty()) {
        const auto& choice = response["choices"][0];
        if (choice.contains("message") && choice["message"].contains("content")) {
            return choice["message"]["content"].get<std::string>();
        }
    }
    throw std::runtime_error("Invalid OpenAI response format");
}


std::map<std::string, std::string> OpenAIBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
#ifdef ENABLE_API_BACKENDS
    headers["Content-Type"] = "application/json";

    // Ensure valid OAuth token if configured, otherwise use API key
    if (ensure_valid_oauth_token() && !oauth_token_.access_token.empty()) {
        // Use OAuth bearer token
        headers["Authorization"] = oauth_token_.token_type + " " + oauth_token_.access_token;
        dout(1) << "Using OAuth token for authorization" << std::endl;
    } else if (!api_key.empty()) {
        // Use API key
        headers["Authorization"] = "Bearer " + api_key;
        dout(1) << "Using API key for authorization" << std::endl;
    } else {
        dout(1) << std::string("WARNING: ") +"No authentication configured (neither OAuth nor API key)" << std::endl;
    }

    // Add model-specific special headers if any
    for (const auto& [key, value] : model_config.special_headers) {
        headers[key] = value;
        dout(1) << "Adding special header: " + key + " = " + value << std::endl;
    }
#endif
    return headers;
}

std::string OpenAIBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    // Add api-version query parameter for Azure OpenAI
    if (!api_version.empty()) {
        return api_endpoint + "?api-version=" + api_version;
    }
    return api_endpoint;
#else
    return "";
#endif
}

// NOTE: add_message() removed - use Frontend::add_message_to_session() + generate_response() instead

void OpenAIBackend::generate_from_session(Session& session, int max_tokens) {
    // If streaming disabled, use base class non-streaming implementation
    if (!config->streaming) {
        ApiBackend::generate_from_session(session, max_tokens);
        return;
    }

    reset_output_state();

    dout(1) << "OpenAIBackend::generate_from_session (streaming): max_tokens=" + std::to_string(max_tokens) << std::endl;

    // Build request from session
    nlohmann::json request = build_request_from_session(session, max_tokens);

    // Add streaming flag
    request["stream"] = true;

    // Request usage information in streaming response (Azure OpenAI support)
    request["stream_options"] = {{"include_usage", true}};

    dout(1) << "Sending streaming request to OpenAI API (generate_from_session)" << std::endl;

    // Get headers and endpoint
    auto headers = get_api_headers();
    std::string endpoint = get_api_endpoint();

    // Enforce rate limits before making request
    enforce_rate_limits();

    // Streaming state
    std::string accumulated_content;
    SSEParser sse_parser;
    bool stream_complete = false;
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::vector<ToolParser::ToolCall> tool_calls;
    clear_tool_calls();  // Clear any previous tool calls

    // Streaming callback to process SSE chunks
    auto stream_handler = [&](const std::string& chunk, void* user_data) -> bool {
        // Return value from process_chunk indicates if callback requested stop
        return sse_parser.process_chunk(chunk,
            [&](const std::string& event, const std::string& data, const std::string& id) -> bool {
                // Handle [DONE] sentinel
                if (data == "[DONE]") {
                    stream_complete = true;
                    return false;
                }

                try {
                    dout(3) << "SSE raw data: " << data.substr(0, 300) << std::endl;

                    // Sanitize UTF-8 before parsing JSON
                    std::string sanitized_data = utf8_sanitizer::sanitize_utf8(data);
                    json delta_json = json::parse(sanitized_data);

                    dout(3) << "SSE JSON: " << delta_json.dump().substr(0, 200) << std::endl;

                    // Extract delta content
                    if (delta_json.contains("choices") && !delta_json["choices"].empty()) {
                        const auto& choice = delta_json["choices"][0];

                        // Get finish_reason if present
                        if (choice.contains("finish_reason") && !choice["finish_reason"].is_null()) {
                            finish_reason = choice["finish_reason"].get<std::string>();
                        }

                        // Get delta content
                        if (choice.contains("delta")) {
                            const auto& delta = choice["delta"];

                            // Handle content delta
                            if (delta.contains("content") && !delta["content"].is_null()) {
                                std::string delta_text = delta["content"].get<std::string>();
                                accumulated_content += delta_text;

                                // Only stream content if we haven't seen tool calls
                                if (tool_calls.empty()) {
                                    if (!output(delta_text)) {
                                        return false;  // User cancelled
                                    }
                                }
                            }

                            // Handle tool_calls delta (incremental)
                            if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
                                for (const auto& tc_delta : delta["tool_calls"]) {
                                    int index = tc_delta.value("index", 0);

                                    while (tool_calls.size() <= static_cast<size_t>(index)) {
                                        tool_calls.push_back(ToolParser::ToolCall());
                                    }

                                    auto& tool_call = tool_calls[index];

                                    if (tc_delta.contains("id")) {
                                        tool_call.tool_call_id = tc_delta["id"].get<std::string>();
                                    }

                                    if (tc_delta.contains("function")) {
                                        const auto& func = tc_delta["function"];
                                        if (func.contains("name")) {
                                            tool_call.name = func["name"].get<std::string>();
                                        }
                                        if (func.contains("arguments")) {
                                            tool_call.raw_json += func["arguments"].get<std::string>();
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Extract usage data (usually in final chunk)
                    if (delta_json.contains("usage") && delta_json["usage"].is_object()) {
                        const auto& usage = delta_json["usage"];
                        prompt_tokens = usage.value("prompt_tokens", 0);
                        completion_tokens = usage.value("completion_tokens", 0);
                        dout(1) << "SSE parsed usage: prompt_tokens=" << prompt_tokens
                                << ", completion_tokens=" << completion_tokens << std::endl;
                    }
                    // Fallback: parse llama.cpp timings format
                    else if (delta_json.contains("timings")) {
                        const auto& timings = delta_json["timings"];
                        prompt_tokens = timings.value("prompt_n", 0);
                        completion_tokens = timings.value("predicted_n", 0);
                    }

                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") + "Failed to parse SSE data: " + std::string(e.what()) << std::endl;
                }

                return true;
            });
    };

    // Make streaming HTTP call
    HttpResponse http_response = http_client->post_stream_cancellable(endpoint, request.dump(), headers,
                                                                       stream_handler, nullptr);

    // Check for HTTP errors
    if (!http_response.is_success()) {
        std::string error_msg = http_response.error_message;
        if (error_msg.empty() && !http_response.body.empty()) {
            try {
                auto json_body = nlohmann::json::parse(http_response.body);
                if (json_body.contains("error") && json_body["error"].contains("message")) {
                    error_msg = json_body["error"]["message"].get<std::string>();
                }
            } catch (...) {
                error_msg = http_response.body;
            }
        }
        if (error_msg.empty()) {
            error_msg = "HTTP error " + std::to_string(http_response.status_code);
        }

        callback(CallbackEvent::ERROR, error_msg, "api_error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return;
    }

    flush_output();

    // Update session token counts using delta tracking
    update_session_tokens(session, prompt_tokens, completion_tokens);

    // Record tool calls for emission after STOP
    for (const auto& tc : tool_calls) {
        if (!tc.name.empty()) {
            record_tool_call(tc.name, tc.raw_json.empty() ? "{}" : tc.raw_json, tc.tool_call_id);
        }
    }

    // Signal completion
    callback(CallbackEvent::STOP, finish_reason, "", "");

    // Send tool calls AFTER STOP
    for (const auto& tc : tool_calls) {
        if (!tc.name.empty()) {
            std::string args = tc.raw_json.empty() ? "{}" : tc.raw_json;
            callback(CallbackEvent::TOOL_CALL, args, tc.name, tc.tool_call_id);
        }
    }
}

size_t OpenAIBackend::query_model_context_size(const std::string& model_name) {
    // Check prerequisites for making API call
    if (!http_client) {
        dout(1) << "HTTP client not available for model query" << std::endl;
        return 0;
    }

    // Make GET request to /models (list endpoint)
    dout(1) << "Querying model list from /models" << std::endl;
    std::string response = make_get_request("/models");
    dout(1) << "Model list response (" + std::to_string(response.length()) + " bytes): " +
             (response.length() > 200 ? response.substr(0, 200) + "..." : response) << std::endl;

    // Parse JSON response to find our model and extract context size
    if (!response.empty()) {
        try {
            auto j = json::parse(response);

            // Check if this is a list response with data array (vLLM/OpenAI format)
            if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
                // First, try exact match by ID
                for (const auto& model_obj : j["data"]) {
                    if (model_obj.contains("id") && model_obj["id"].get<std::string>() == model_name) {
                        dout(1) << "Found exact model match in list: " + model_name << std::endl;

                        // Try max_model_len (vLLM/llama.cpp format)
                        if (model_obj.contains("max_model_len") && model_obj["max_model_len"].is_number()) {
                            size_t context_size = model_obj["max_model_len"].get<size_t>();
                            dout(1) << "Parsed max_model_len from API: " + std::to_string(context_size) << std::endl;
                            return context_size;
                        }

                        // Try context_window (some OpenAI-compatible APIs)
                        if (model_obj.contains("context_window") && model_obj["context_window"].is_number()) {
                            size_t context_size = model_obj["context_window"].get<size_t>();
                            dout(1) << "Parsed context_window from API: " + std::to_string(context_size) << std::endl;
                            return context_size;
                        }

                        // Try context_length (official OpenAI format)
                        if (model_obj.contains("context_length") && model_obj["context_length"].is_number()) {
                            size_t context_size = model_obj["context_length"].get<size_t>();
                            dout(1) << "Parsed context_length from API: " + std::to_string(context_size) << std::endl;
                            return context_size;
                        }

                        // Try meta.n_ctx_train (llama.cpp format)
                        if (model_obj.contains("meta") && model_obj["meta"].is_object()) {
                            if (model_obj["meta"].contains("n_ctx_train") && model_obj["meta"]["n_ctx_train"].is_number()) {
                                size_t context_size = model_obj["meta"]["n_ctx_train"].get<size_t>();
                                dout(1) << "Parsed n_ctx_train from API: " + std::to_string(context_size) << std::endl;
                                return context_size;
                            }
                        }
                    }
                }

                // No exact match found - use first available model (common for llama.cpp/single-model servers)
                dout(1) << "No exact match for '" + model_name + "', using first available model from list" << std::endl;
                const auto& model_obj = j["data"][0];

                std::string actual_model_id = model_obj.contains("id") ? model_obj["id"].get<std::string>() : "unknown";
                dout(1) << "Using model: " + actual_model_id << std::endl;

                // Try max_model_len (vLLM/llama.cpp format)
                if (model_obj.contains("max_model_len") && model_obj["max_model_len"].is_number()) {
                    size_t context_size = model_obj["max_model_len"].get<size_t>();
                    dout(1) << "Parsed max_model_len from API: " + std::to_string(context_size) << std::endl;
                    return context_size;
                }

                // Try context_window (some OpenAI-compatible APIs)
                if (model_obj.contains("context_window") && model_obj["context_window"].is_number()) {
                    size_t context_size = model_obj["context_window"].get<size_t>();
                    dout(1) << "Parsed context_window from API: " + std::to_string(context_size) << std::endl;
                    return context_size;
                }

                // Try context_length (official OpenAI format)
                if (model_obj.contains("context_length") && model_obj["context_length"].is_number()) {
                    size_t context_size = model_obj["context_length"].get<size_t>();
                    dout(1) << "Parsed context_length from API: " + std::to_string(context_size) << std::endl;
                    return context_size;
                }

                // Try meta.n_ctx_train (llama.cpp format)
                if (model_obj.contains("meta") && model_obj["meta"].is_object()) {
                    if (model_obj["meta"].contains("n_ctx_train") && model_obj["meta"]["n_ctx_train"].is_number()) {
                        size_t context_size = model_obj["meta"]["n_ctx_train"].get<size_t>();
                        dout(1) << "Parsed n_ctx_train from API: " + std::to_string(context_size) << std::endl;
                        return context_size;
                    }
                }

                dout(1) << "Model found but no context size field detected" << std::endl;
            }
        } catch (const json::exception& e) {
            dout(1) << std::string("WARNING: ") +"Failed to parse model list JSON: " + std::string(e.what()) << std::endl;
        }
    }

    // Unable to query context size from API - return 0 and let caller handle fallback
    dout(1) << std::string("WARNING: ") +"Could not query context size from API for model: " + model_name << std::endl;
    return 0;
}
