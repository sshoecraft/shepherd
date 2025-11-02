
#include "shepherd.h"
#include "openai.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

OpenAIBackend::OpenAIBackend(size_t context_size) : ApiBackend(context_size) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Detect model configuration from Models database (if model is specified)
    // If model is empty, it will be auto-detected in initialize() and config will be set there
    if (!model_name.empty()) {
        model_config = Models::detect_from_api_model("openai", model_name);
        LOG_DEBUG("Detected model config: context=" + std::to_string(model_config.context_window) +
                  ", max_output=" + std::to_string(model_config.max_output_tokens) +
                  ", param_name=" + model_config.max_tokens_param_name);
    }

    // Set API endpoint from config (api_base or default)
    if (!config->api_base.empty()) {
        // User specified custom API base (e.g., local server)
        api_endpoint = config->api_base;
        // Ensure it has /chat/completions endpoint
        if (api_endpoint.find("/chat/completions") == std::string::npos) {
            if (api_endpoint.back() == '/') {
                api_endpoint += "chat/completions";
            } else {
                api_endpoint += "/chat/completions";
            }
        }
        LOG_INFO("Using custom API endpoint: " + api_endpoint);
    }
    // else: keep default api_endpoint = "https://api.openai.com/v1/chat/completions" from header

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config(config->backend_config("openai"));
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
                if (error_json["error"].is_object() && error_json["error"].contains("message")) {
                    resp.error = error_json["error"]["message"].get<std::string>();
                } else if (error_json["error"].is_string()) {
                    resp.error = error_json["error"].get<std::string>();
                }
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

            LOG_DEBUG("Detected MAX_TOKENS_TOO_HIGH error, parsing...");

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
                        LOG_DEBUG("Parsed MAX_TOKENS_TOO_HIGH: actual_prompt=" + std::to_string(resp.actual_prompt_tokens) +
                                  ", max_tokens=" + std::to_string(max_tokens_requested) +
                                  ", max_context=" + std::to_string(max_context) +
                                  ", overflow=" + std::to_string(resp.overflow_tokens));
                    }
                } catch (const std::exception& e) {
                    LOG_DEBUG("Failed to parse MAX_TOKENS_TOO_HIGH details: " + std::string(e.what()));
                }
            }
        }

        return resp;
    }

    // Parse successful response
    try {
        nlohmann::json json_resp = nlohmann::json::parse(http_response.body);

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
                if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
                    // Store raw tool_calls JSON for message persistence
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
                                    LOG_WARN("Failed to parse tool call arguments: " + std::string(e.what()));
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
            LOG_DEBUG("API Usage - prompt_tokens: " + std::to_string(resp.prompt_tokens) +
                     ", completion_tokens: " + std::to_string(resp.completion_tokens) +
                     ", total_tokens: " + std::to_string(total));
            LOG_DEBUG("Full usage JSON: " + usage.dump());
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

void OpenAIBackend::initialize(Session& session) {
    LOG_INFO("Initializing OpenAI backend...");

    // Validate API key (only required for actual OpenAI API, not local servers)
    bool is_openai_api = (api_endpoint.find("api.openai.com") != std::string::npos);
    if (api_key.empty() && is_openai_api) {
        LOG_ERROR("OpenAI API key is required for api.openai.com");
        throw std::runtime_error("OpenAI API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        LOG_INFO("No model specified, querying server for available models");
        std::string queried_model = query_available_model();
        if (!queried_model.empty()) {
            model_name = queried_model;
            LOG_INFO("Using model from server: " + model_name);

            // Save context_window from server query before detect_from_api_model overwrites it
            size_t server_context = model_config.context_window;
            model_config = Models::detect_from_api_model("openai", model_name);

            // Restore server-reported context if it was found
            if (server_context > 0) {
                model_config.context_window = server_context;
                LOG_INFO("Using server-reported context size: " + std::to_string(server_context));
            }
        } else {
            LOG_WARN("Failed to query server for model, will use first API response to determine");
        }
    }

    // Set context size from model config if not already set
    if (context_size == 0 && model_config.context_window > 0) {
        context_size = model_config.context_window;
        LOG_INFO("Using model's context size: " + std::to_string(context_size));
    }

    // Call base class initialize() which handles calibration
    ApiBackend::initialize(session);

    LOG_INFO("OpenAI backend initialized successfully");
}

std::string OpenAIBackend::query_available_model() {
    // Check prerequisites for making API call
    if (!http_client || api_key.empty()) {
        LOG_ERROR("HTTP client or API key not available for model query");
        return "";
    }

    // Make GET request to /models endpoint
    LOG_INFO("Querying server for available models");
    std::string response = make_get_request("/models");

    if (response.empty()) {
        LOG_WARN("Failed to query /models endpoint");
        return "";
    }

    try {
        auto j = json::parse(response);

        if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
            // Get the first model from the list
            auto first_model = j["data"][0];
            if (first_model.contains("id") && first_model["id"].is_string()) {
                std::string model_id = first_model["id"].get<std::string>();
                LOG_INFO("Found model from server: " + model_id);

                // Extract max_model_len if available (vLLM/Shepherd server format)
                if (first_model.contains("max_model_len") && first_model["max_model_len"].is_number()) {
                    int max_context = first_model["max_model_len"].get<int>();
                    model_config.context_window = max_context;
                    LOG_INFO("Server reported context size: " + std::to_string(max_context));
                }

                return model_id;
            }
        }

        LOG_WARN("No models found in /v1/models response");
        return "";
    } catch (const json::exception& e) {
        LOG_WARN("Failed to parse /v1/models response: " + std::string(e.what()));
        return "";
    }
}



std::string OpenAIBackend::make_get_request(const std::string& endpoint) {
    if (!http_client) {
        LOG_ERROR("HTTP client not initialized");
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
            LOG_ERROR("Authentication failed: " + error_msg);
            throw BackendError("Authentication failed: " + error_msg);
        }

        LOG_ERROR("GET request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message);
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

    LOG_DEBUG("extract_tokens_to_evict: parsing error message: " + error_message);

    int actual_tokens = -1;
    int max_tokens = -1;

    // Try shepherd server format first: "would need X tokens but limit is Y tokens"
    size_t need_pos = error_message.find("would need ");
    size_t limit_pos = error_message.find("but limit is ");
    if (need_pos != std::string::npos && limit_pos != std::string::npos) {
        LOG_DEBUG("Found shepherd server format markers");
        try {
            size_t start = need_pos + 11;
            size_t end = error_message.find(" tokens", start);
            actual_tokens = std::stoi(error_message.substr(start, end - start));

            start = limit_pos + 13;
            end = error_message.find(" tokens", start);
            max_tokens = std::stoi(error_message.substr(start, end - start));
            LOG_DEBUG("Parsed shepherd format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens));
        } catch (const std::exception& e) {
            LOG_DEBUG("Exception parsing shepherd format: " + std::string(e.what()));
        }
    }

    // Try OpenAI classic format: "maximum context length is X tokens. However, your messages resulted in Y tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t resulted_pos = error_message.find("resulted in ");
        LOG_DEBUG("OpenAI classic format search: max_pos=" + std::to_string(max_pos) + ", resulted_pos=" + std::to_string(resulted_pos));
        if (max_pos != std::string::npos && resulted_pos != std::string::npos) {
            LOG_DEBUG("Found OpenAI classic format markers");
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                LOG_DEBUG("Parsing max_tokens from position " + std::to_string(start) + " to " + std::to_string(end));
                std::string max_str = error_message.substr(start, end - start);
                LOG_DEBUG("max_tokens string: '" + max_str + "'");
                max_tokens = std::stoi(max_str);

                start = resulted_pos + 12;
                end = error_message.find(" tokens", start);
                LOG_DEBUG("Parsing actual_tokens from position " + std::to_string(start) + " to " + std::to_string(end));
                std::string actual_str = error_message.substr(start, end - start);
                LOG_DEBUG("actual_tokens string: '" + actual_str + "'");
                actual_tokens = std::stoi(actual_str);
                LOG_DEBUG("Parsed OpenAI classic format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens));
            } catch (const std::exception& e) {
                LOG_DEBUG("Exception parsing OpenAI classic format: " + std::string(e.what()));
            }
        }
    }

    // Try vLLM MAX_TOKENS_TOO_HIGH format: "is too large: X. ... maximum context length is Y ... your request has Z input tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t too_large_pos = error_message.find("is too large: ");
        size_t max_pos = error_message.find("maximum context length is ");
        size_t request_pos = error_message.find("your request has ");
        if (too_large_pos != std::string::npos && max_pos != std::string::npos && request_pos != std::string::npos) {
            LOG_DEBUG("Found vLLM MAX_TOKENS_TOO_HIGH format markers");
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
                LOG_DEBUG("Parsed vLLM MAX_TOKENS_TOO_HIGH: actual_prompt=" + std::to_string(actual_prompt) +
                         ", max_tokens_requested=" + std::to_string(max_tokens_requested) +
                         ", max_context=" + std::to_string(max_context) +
                         ", overflow=" + std::to_string(overflow));
                return overflow > 0 ? overflow : -1;
            } catch (const std::exception& e) {
                LOG_DEBUG("Exception parsing vLLM MAX_TOKENS_TOO_HIGH format: " + std::string(e.what()));
            }
        }
    }

    // Try vLLM/modern format: "your request has X input tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t request_pos = error_message.find("your request has ");
        LOG_DEBUG("vLLM format search: max_pos=" + std::to_string(max_pos) + ", request_pos=" + std::to_string(request_pos));
        if (max_pos != std::string::npos && request_pos != std::string::npos) {
            LOG_DEBUG("Found vLLM format markers");
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                max_tokens = std::stoi(error_message.substr(start, end - start));

                start = request_pos + 17;
                end = error_message.find(" ", start);
                actual_tokens = std::stoi(error_message.substr(start, end - start));
                LOG_DEBUG("Parsed vLLM format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens));
            } catch (const std::exception& e) {
                LOG_DEBUG("Exception parsing vLLM format: " + std::string(e.what()));
            }
        }
    }

    // Try OpenAI/vLLM detailed format: "you requested X tokens (Y in the messages, Z in the completion)"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t requested_pos = error_message.find("you requested ");
        LOG_DEBUG("OpenAI detailed format search: max_pos=" + std::to_string(max_pos) + ", requested_pos=" + std::to_string(requested_pos));
        if (max_pos != std::string::npos && requested_pos != std::string::npos) {
            LOG_DEBUG("Found OpenAI detailed format markers");
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                max_tokens = std::stoi(error_message.substr(start, end - start));

                start = requested_pos + 14;
                end = error_message.find(" tokens", start);
                actual_tokens = std::stoi(error_message.substr(start, end - start));
                LOG_DEBUG("Parsed OpenAI detailed format: actual=" + std::to_string(actual_tokens) + ", max=" + std::to_string(max_tokens));
            } catch (const std::exception& e) {
                LOG_DEBUG("Exception parsing OpenAI detailed format: " + std::string(e.what()));
            }
        }
    }

    if (actual_tokens > 0 && max_tokens > 0) {
        int to_evict = actual_tokens - max_tokens;
        LOG_DEBUG("Tokens to evict: " + std::to_string(to_evict));
        return to_evict;
    }

    // Can't parse - return error
    LOG_ERROR("Failed to parse token count from error message: " + error_message);
    return -1;
}

// Implement required ApiBackend pure virtual methods
nlohmann::json OpenAIBackend::build_request_from_session(const Session& session, int max_tokens) {
    nlohmann::json request;
    request["model"] = model_name;

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
        jmsg["content"] = msg.content;

        // Restore tool_calls for assistant messages that made tool calls
        if (msg.type == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = nlohmann::json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to parse stored tool_calls: " + std::string(e.what()));
            }
        }

        if (msg.type == Message::TOOL && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    request["messages"] = messages;

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
    if (max_tokens > 0) {
        // Cap at model's max_output_tokens if specified
        int capped_tokens = max_tokens;
        if (model_config.max_output_tokens > 0 && max_tokens > model_config.max_output_tokens) {
            capped_tokens = model_config.max_output_tokens;
            LOG_DEBUG("Capping max_tokens from " + std::to_string(max_tokens) +
                     " to model's max_output_tokens: " + std::to_string(capped_tokens));
        }
        request[model_config.max_tokens_param_name] = capped_tokens;
    }

    // Add special headers if any
    if (!model_config.special_headers.empty()) {
        LOG_DEBUG("Model has " + std::to_string(model_config.special_headers.size()) + " special headers");
    }

    return request;
}

nlohmann::json OpenAIBackend::build_request(const Session& session,
                                             Message::Type type,
                                             const std::string& content,
                                             const std::string& tool_name,
                                             const std::string& tool_id,
                                             int max_tokens) {
    nlohmann::json request;
    request["model"] = model_name;

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
        jmsg["content"] = msg.content;

        // Restore tool_calls for assistant messages that made tool calls
        if (msg.type == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = nlohmann::json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to parse stored tool_calls: " + std::string(e.what()));
            }
        }

        if (msg.type == Message::TOOL && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    // Add the new message being sent
    nlohmann::json new_msg;
    // Convert Message::Type to role string
    std::string role;
    switch (type) {
        case Message::SYSTEM: role = "system"; break;
        case Message::USER: role = "user"; break;
        case Message::ASSISTANT: role = "assistant"; break;
        case Message::TOOL: role = "tool"; break;
        case Message::FUNCTION: role = "function"; break;
    }
    new_msg["role"] = role;
    new_msg["content"] = content;
    if (!tool_name.empty()) new_msg["name"] = tool_name;
    if (!tool_id.empty()) new_msg["tool_call_id"] = tool_id;
    messages.push_back(new_msg);
    
    request["messages"] = messages;

    LOG_DEBUG("Built request with " + std::to_string(messages.size()) + " messages (session has " +
             std::to_string(session.messages.size()) + " messages)");

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
    if (max_tokens > 0) {
        // Cap at model's max_output_tokens if specified
        int capped_tokens = max_tokens;
        if (model_config.max_output_tokens > 0 && max_tokens > model_config.max_output_tokens) {
            capped_tokens = model_config.max_output_tokens;
            LOG_DEBUG("Capping max_tokens from " + std::to_string(max_tokens) +
                     " to model's max_output_tokens: " + std::to_string(capped_tokens));
        }
        request[model_config.max_tokens_param_name] = capped_tokens;
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
    headers["Authorization"] = "Bearer " + api_key;

    // Add model-specific special headers if any
    for (const auto& [key, value] : model_config.special_headers) {
        headers[key] = value;
        LOG_DEBUG("Adding special header: " + key + " = " + value);
    }
#endif
    return headers;
}

std::string OpenAIBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    return api_endpoint;
#else
    return "";
#endif
}

size_t OpenAIBackend::query_model_context_size(const std::string& model_name) {
    // Check prerequisites for making API call
    if (!http_client || api_key.empty()) {
        LOG_ERROR("HTTP client or API key not available for model query");
        return 0;
    }

    // Make GET request to /models (list endpoint)
    LOG_INFO("Querying model list from /models");
    std::string response = make_get_request("/models");
    LOG_INFO("Model list response (" + std::to_string(response.length()) + " bytes): " +
             (response.length() > 200 ? response.substr(0, 200) + "..." : response));

    // Parse JSON response to find our model and extract context size
    if (!response.empty()) {
        try {
            auto j = json::parse(response);

            // Check if this is a list response with data array (vLLM/OpenAI format)
            if (j.contains("data") && j["data"].is_array() && !j["data"].empty()) {
                // First, try exact match by ID
                for (const auto& model_obj : j["data"]) {
                    if (model_obj.contains("id") && model_obj["id"].get<std::string>() == model_name) {
                        LOG_INFO("Found exact model match in list: " + model_name);

                        // Try max_model_len (vLLM/llama.cpp format)
                        if (model_obj.contains("max_model_len") && model_obj["max_model_len"].is_number()) {
                            size_t context_size = model_obj["max_model_len"].get<size_t>();
                            LOG_INFO("Parsed max_model_len from API: " + std::to_string(context_size));
                            return context_size;
                        }

                        // Try context_window (some OpenAI-compatible APIs)
                        if (model_obj.contains("context_window") && model_obj["context_window"].is_number()) {
                            size_t context_size = model_obj["context_window"].get<size_t>();
                            LOG_INFO("Parsed context_window from API: " + std::to_string(context_size));
                            return context_size;
                        }

                        // Try context_length (official OpenAI format)
                        if (model_obj.contains("context_length") && model_obj["context_length"].is_number()) {
                            size_t context_size = model_obj["context_length"].get<size_t>();
                            LOG_INFO("Parsed context_length from API: " + std::to_string(context_size));
                            return context_size;
                        }

                        // Try meta.n_ctx_train (llama.cpp format)
                        if (model_obj.contains("meta") && model_obj["meta"].is_object()) {
                            if (model_obj["meta"].contains("n_ctx_train") && model_obj["meta"]["n_ctx_train"].is_number()) {
                                size_t context_size = model_obj["meta"]["n_ctx_train"].get<size_t>();
                                LOG_INFO("Parsed n_ctx_train from API: " + std::to_string(context_size));
                                return context_size;
                            }
                        }
                    }
                }

                // No exact match found - use first available model (common for llama.cpp/single-model servers)
                LOG_INFO("No exact match for '" + model_name + "', using first available model from list");
                const auto& model_obj = j["data"][0];

                std::string actual_model_id = model_obj.contains("id") ? model_obj["id"].get<std::string>() : "unknown";
                LOG_INFO("Using model: " + actual_model_id);

                // Try max_model_len (vLLM/llama.cpp format)
                if (model_obj.contains("max_model_len") && model_obj["max_model_len"].is_number()) {
                    size_t context_size = model_obj["max_model_len"].get<size_t>();
                    LOG_INFO("Parsed max_model_len from API: " + std::to_string(context_size));
                    return context_size;
                }

                // Try context_window (some OpenAI-compatible APIs)
                if (model_obj.contains("context_window") && model_obj["context_window"].is_number()) {
                    size_t context_size = model_obj["context_window"].get<size_t>();
                    LOG_INFO("Parsed context_window from API: " + std::to_string(context_size));
                    return context_size;
                }

                // Try context_length (official OpenAI format)
                if (model_obj.contains("context_length") && model_obj["context_length"].is_number()) {
                    size_t context_size = model_obj["context_length"].get<size_t>();
                    LOG_INFO("Parsed context_length from API: " + std::to_string(context_size));
                    return context_size;
                }

                // Try meta.n_ctx_train (llama.cpp format)
                if (model_obj.contains("meta") && model_obj["meta"].is_object()) {
                    if (model_obj["meta"].contains("n_ctx_train") && model_obj["meta"]["n_ctx_train"].is_number()) {
                        size_t context_size = model_obj["meta"]["n_ctx_train"].get<size_t>();
                        LOG_INFO("Parsed n_ctx_train from API: " + std::to_string(context_size));
                        return context_size;
                    }
                }

                LOG_INFO("Model found but no context size field detected");
            }
        } catch (const json::exception& e) {
            LOG_WARN("Failed to parse model list JSON: " + std::string(e.what()));
        }
    }

    // Unable to query context size from API - return 0 and let caller handle fallback
    LOG_WARN("Could not query context size from API for model: " + model_name);
    return 0;
}
