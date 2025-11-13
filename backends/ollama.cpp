#include "ollama.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

OllamaBackend::OllamaBackend(size_t context_size) : ApiBackend(context_size) {
    // Initialize with config values
    model_name = config->model;

    // Set API endpoint from config (api_base or default)
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        // Ensure it has /api/chat endpoint for native Ollama API
        if (api_endpoint.find("/api/chat") == std::string::npos) {
            if (api_endpoint.back() != '/') {
                api_endpoint += "/";
            }
            api_endpoint += "api/chat";
        }
        LOG_INFO("Using custom Ollama endpoint: " + api_endpoint);
    }
    // else: keep default api_endpoint = "http://localhost:11434/api/chat"

    // Parse backend-specific config if available
    parse_backend_config(config->backend_config("ollama"));
}

OllamaBackend::~OllamaBackend() {
}

void OllamaBackend::initialize(Session& session) {
    LOG_INFO("Initializing Ollama backend...");

    // Auto-detect model if not specified
    if (model_name.empty()) {
        LOG_INFO("No model specified, querying server for available models");
        std::string queried_model = query_available_model();
        if (!queried_model.empty()) {
            model_name = queried_model;
            LOG_INFO("Using model from server: " + model_name);
        } else {
            LOG_WARN("Failed to query server for model, will use first API response to determine");
        }
    }

    // Call base class initialize() which handles context size query and calibration
    ApiBackend::initialize(session);

    // Force auto-eviction for Ollama backend
    // Ollama silently truncates messages when context is full (no error returned),
    // so we MUST proactively evict to prevent data loss
    session.auto_evict = true;
    LOG_INFO("Auto-eviction enabled for Ollama (prevents silent truncation)");

    LOG_INFO("Ollama backend initialized successfully");
}

Response OllamaBackend::parse_http_response(const HttpResponse& http_response) {
    Response resp;

    // Check HTTP status
    if (!http_response.is_success()) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.finish_reason = "error";

        // Try to parse error JSON (Ollama uses OpenAI-compatible format)
        try {
            json error_json = json::parse(http_response.body);
            if (error_json.contains("error")) {
                if (error_json["error"].is_object() && error_json["error"].contains("message")) {
                    resp.error = error_json["error"]["message"].get<std::string>();
                } else if (error_json["error"].is_string()) {
                    resp.error = error_json["error"].get<std::string>();
                }
            }
        } catch (...) {
            resp.error = http_response.error_message.empty() ? "API request failed" : http_response.error_message;
        }

        if (resp.error.empty()) {
            resp.error = http_response.error_message.empty() ? "Unknown API error" : http_response.error_message;
        }

        return resp;
    }

    // Parse successful response
    try {
        json json_resp = json::parse(http_response.body);

        // Check for native Ollama /api/chat format (has "message" directly)
        if (json_resp.contains("message") && json_resp["message"].is_object()) {
            const auto& message = json_resp["message"];

            // Get content
            if (message.contains("content") && !message["content"].is_null()) {
                resp.content = message["content"].get<std::string>();
            }

            // Get finish reason (done_reason in native format)
            if (json_resp.contains("done_reason")) {
                resp.finish_reason = json_resp["done_reason"].get<std::string>();
            }

            // Extract token usage (native Ollama format)
            if (json_resp.contains("prompt_eval_count")) {
                resp.prompt_tokens = json_resp["prompt_eval_count"].get<int>();
            }
            if (json_resp.contains("eval_count")) {
                resp.completion_tokens = json_resp["eval_count"].get<int>();
            }

            // Parse tool calls if present (TODO: check native Ollama tool call format)
            if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
                for (const auto& tc : message["tool_calls"]) {
                    ToolParser::ToolCall tool_call;

                    if (tc.contains("function")) {
                        const auto& func = tc["function"];
                        if (func.contains("name")) {
                            tool_call.name = func["name"].get<std::string>();
                        }
                        if (func.contains("arguments")) {
                            // Handle both string and object arguments
                            if (func["arguments"].is_string()) {
                                tool_call.raw_json = func["arguments"].get<std::string>();
                            } else {
                                tool_call.raw_json = func["arguments"].dump();
                            }

                            // Parse arguments
                            try {
                                json args_json = func["arguments"].is_string()
                                    ? json::parse(func["arguments"].get<std::string>())
                                    : func["arguments"];

                                for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                                    if (it.value().is_string()) {
                                        tool_call.parameters[it.key()] = it.value().get<std::string>();
                                    } else if (it.value().is_number_integer()) {
                                        tool_call.parameters[it.key()] = it.value().get<int>();
                                    } else if (it.value().is_number_float()) {
                                        tool_call.parameters[it.key()] = it.value().get<double>();
                                    } else if (it.value().is_boolean()) {
                                        tool_call.parameters[it.key()] = it.value().get<bool>();
                                    } else {
                                        tool_call.parameters[it.key()] = it.value().dump();
                                    }
                                }
                            } catch (const std::exception& e) {
                                LOG_DEBUG("Failed to parse tool arguments: " + std::string(e.what()));
                            }
                        }
                    }

                    resp.tool_calls.push_back(tool_call);
                }
            }
        }
        // Fall back to OpenAI-compatible format (for backwards compatibility)
        else if (json_resp.contains("choices") && !json_resp["choices"].empty()) {
            const auto& choice = json_resp["choices"][0];

            if (choice.contains("finish_reason")) {
                resp.finish_reason = choice["finish_reason"].get<std::string>();
            }

            if (choice.contains("message")) {
                const auto& message = choice["message"];
                if (message.contains("content") && !message["content"].is_null()) {
                    resp.content = message["content"].get<std::string>();
                }

                // Parse tool calls (OpenAI format)
                if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
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

                                try {
                                    json args_json = json::parse(args_str);
                                    for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                                        if (it.value().is_string()) {
                                            tool_call.parameters[it.key()] = it.value().get<std::string>();
                                        } else if (it.value().is_number_integer()) {
                                            tool_call.parameters[it.key()] = it.value().get<int>();
                                        } else if (it.value().is_number_float()) {
                                            tool_call.parameters[it.key()] = it.value().get<double>();
                                        } else if (it.value().is_boolean()) {
                                            tool_call.parameters[it.key()] = it.value().get<bool>();
                                        } else {
                                            tool_call.parameters[it.key()] = it.value().dump();
                                        }
                                    }
                                } catch (const std::exception& e) {
                                    LOG_DEBUG("Failed to parse tool arguments: " + std::string(e.what()));
                                }
                            }
                        }

                        resp.tool_calls.push_back(tool_call);
                    }
                }
            }

            // Extract token usage (OpenAI format)
            if (json_resp.contains("usage")) {
                const auto& usage = json_resp["usage"];
                if (usage.contains("prompt_tokens")) {
                    resp.prompt_tokens = usage["prompt_tokens"].get<int>();
                }
                if (usage.contains("completion_tokens")) {
                    resp.completion_tokens = usage["completion_tokens"].get<int>();
                }
            }
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

nlohmann::json OllamaBackend::build_request_from_session(const Session& session, int max_tokens) {
    json request;
    request["model"] = model_name;
    request["stream"] = false;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    // Build messages array
    json messages = json::array();

    // Add system message if present
    if (!session.system_message.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", session.system_message}
        });
    }

    // Add conversation messages
    for (const auto& msg : session.messages) {
        json message;

        switch (msg.type) {
            case Message::USER:
                message["role"] = "user";
                message["content"] = msg.content;
                break;

            case Message::ASSISTANT:
                message["role"] = "assistant";
                message["content"] = msg.content;
                break;

            case Message::TOOL:
                message["role"] = "tool";
                message["content"] = msg.content;
                if (!msg.tool_call_id.empty()) {
                    message["tool_call_id"] = msg.tool_call_id;
                }
                if (!msg.tool_name.empty()) {
                    message["name"] = msg.tool_name;
                }
                break;

            default:
                continue;
        }

        messages.push_back(message);
    }

    request["messages"] = messages;

    // Add tools if available (OpenAI format)
    if (!session.tools.empty()) {
        json tools_array = json::array();

        for (const auto& tool_def : session.tools) {
            json tool;
            tool["type"] = "function";

            json function;
            function["name"] = tool_def.name;
            function["description"] = tool_def.description;

            // Parameters are already JSON object
            if (!tool_def.parameters.empty()) {
                function["parameters"] = tool_def.parameters;
            } else {
                function["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tool["function"] = function;
            tools_array.push_back(tool);
        }

        request["tools"] = tools_array;
    }

    // Add options with num_ctx to ensure Ollama uses detected context size
    json options;
    options["num_ctx"] = context_size;

    // Add sampling parameters (Ollama supports: temperature, top_p, top_k, repeat_penalty)
    options["temperature"] = temperature;
    options["top_p"] = top_p;
    if (top_k > 0) {
        options["top_k"] = top_k;
    }
    if (repeat_penalty != 1.0f) {
        options["repeat_penalty"] = repeat_penalty;
    }

    request["options"] = options;

    return request;
}

nlohmann::json OllamaBackend::build_request(const Session& session,
                                              Message::Type type,
                                              const std::string& content,
                                              const std::string& tool_name,
                                              const std::string& tool_id,
                                              int max_tokens) {
    json request;
    request["model"] = model_name;
    request["stream"] = false;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    // Build messages array
    json messages = json::array();

    // Add system message if present
    if (!session.system_message.empty()) {
        messages.push_back({
            {"role", "system"},
            {"content", session.system_message}
        });
    }

    // Add existing messages from session
    for (const auto& msg : session.messages) {
        json message;

        switch (msg.type) {
            case Message::USER:
                message["role"] = "user";
                message["content"] = msg.content;
                break;

            case Message::ASSISTANT:
                message["role"] = "assistant";
                message["content"] = msg.content;
                // Note: Tool calls are embedded in the content as text, not structured
                break;

            case Message::TOOL:
                message["role"] = "tool";
                message["content"] = msg.content;
                if (!msg.tool_call_id.empty()) {
                    message["tool_call_id"] = msg.tool_call_id;
                }
                if (!msg.tool_name.empty()) {
                    message["name"] = msg.tool_name;
                }
                break;

            default:
                continue;
        }

        messages.push_back(message);
    }

    // Add the new message
    json new_message;
    switch (type) {
        case Message::USER:
            new_message["role"] = "user";
            new_message["content"] = content;
            break;

        case Message::TOOL:
            new_message["role"] = "tool";
            new_message["content"] = content;
            if (!tool_id.empty()) {
                new_message["tool_call_id"] = tool_id;
            }
            if (!tool_name.empty()) {
                new_message["name"] = tool_name;
            }
            break;

        default:
            break;
    }

    messages.push_back(new_message);
    request["messages"] = messages;

    // Add tools if available (OpenAI format)
    if (!session.tools.empty()) {
        json tools_array = json::array();

        for (const auto& tool_def : session.tools) {
            json tool;
            tool["type"] = "function";

            json function;
            function["name"] = tool_def.name;
            function["description"] = tool_def.description;

            // Parameters are already JSON object
            if (!tool_def.parameters.empty()) {
                function["parameters"] = tool_def.parameters;
            } else {
                function["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tool["function"] = function;
            tools_array.push_back(tool);
        }

        request["tools"] = tools_array;
    }

    // Add options with num_ctx to ensure Ollama uses detected context size
    json options;
    options["num_ctx"] = context_size;

    // Add sampling parameters (Ollama supports: temperature, top_p, top_k, repeat_penalty)
    options["temperature"] = temperature;
    options["top_p"] = top_p;
    if (top_k > 0) {
        options["top_k"] = top_k;
    }
    if (repeat_penalty != 1.0f) {
        options["repeat_penalty"] = repeat_penalty;
    }

    request["options"] = options;

    return request;
}

std::string OllamaBackend::parse_response(const nlohmann::json& response) {
    // Extract content from OpenAI-compatible response
    if (response.contains("choices") && !response["choices"].empty()) {
        const auto& choice = response["choices"][0];
        if (choice.contains("message") && choice["message"].contains("content")) {
            return choice["message"]["content"].get<std::string>();
        }
    }
    return "";
}

int OllamaBackend::extract_tokens_to_evict(const HttpResponse& response) {
    // Try to extract error message
    std::string error_message = response.error_message;
    if (error_message.empty() && !response.body.empty()) {
        try {
            auto json_body = json::parse(response.body);
            if (json_body.contains("error") && json_body["error"].contains("message")) {
                error_message = json_body["error"]["message"].get<std::string>();
            }
        } catch (...) {
            error_message = response.body;
        }
    }

    // Ollama uses OpenAI-compatible error format
    // "This model's maximum context length is 16385 tokens. However, your messages resulted in 44366 tokens."

    int actual_tokens = -1;
    int max_tokens = -1;

    // Try OpenAI format
    size_t max_pos = error_message.find("maximum context length is ");
    size_t resulted_pos = error_message.find("resulted in ");

    if (max_pos != std::string::npos && resulted_pos != std::string::npos) {
        try {
            size_t start = max_pos + 26;
            size_t end = error_message.find(" tokens", start);
            max_tokens = std::stoi(error_message.substr(start, end - start));

            start = resulted_pos + 12;
            end = error_message.find(" tokens", start);
            actual_tokens = std::stoi(error_message.substr(start, end - start));
        } catch (...) {}
    }

    if (actual_tokens > 0 && max_tokens > 0) {
        return actual_tokens - max_tokens;
    }

    return -1;
}

std::map<std::string, std::string> OllamaBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    // Ollama doesn't require auth, but we can add a dummy header for compatibility
    headers["Authorization"] = "Bearer ollama";
    return headers;
}

std::string OllamaBackend::get_api_endpoint() {
    return api_endpoint;
}

size_t OllamaBackend::query_model_context_size(const std::string& model_name) {
    const size_t DEFAULT_CONTEXT_SIZE = 8192;  // Conservative default for Ollama models

    // Try to query from Ollama's /api/show endpoint
    try {
        if (!http_client) {
            LOG_WARN("HTTP client not initialized for Ollama");
            return DEFAULT_CONTEXT_SIZE;
        }

        json request;
        request["model"] = model_name;
        std::string request_str = request.dump();

        // Build URL for /api/show endpoint (strip /api/chat from endpoint)
        std::string base_url = api_endpoint;
        size_t pos = base_url.find("/api/chat");
        if (pos != std::string::npos) {
            base_url = base_url.substr(0, pos);
        }
        std::string show_url = base_url + "/api/show";

        // Prepare headers
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        // Make POST request to /api/show
        HttpResponse response = http_client->post(show_url, request_str, headers);

        if (response.is_success()) {
            json resp = json::parse(response.body);

            // Parse model info - look for context_length or num_ctx
            if (resp.contains("model_info")) {
                const auto& model_info = resp["model_info"];

                // Search for any field containing "context_length" (e.g., "context_length", "qwen2.context_length")
                for (auto& [key, value] : model_info.items()) {
                    if (key.find("context_length") != std::string::npos) {
                        if (value.is_number()) {
                            size_t ctx_size = value.get<size_t>();
                            LOG_DEBUG("Found context_length in model_info[\"" + key + "\"]: " + std::to_string(ctx_size));
                            return ctx_size;
                        }
                    }
                }
            }

            // Try alternate fields
            if (resp.contains("num_ctx")) {
                return resp["num_ctx"].get<size_t>();
            }

            // Try parameters
            if (resp.contains("parameters")) {
                const auto& params = resp["parameters"];
                if (params.contains("num_ctx")) {
                    return params["num_ctx"].get<size_t>();
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_DEBUG("Failed to query Ollama model context size: " + std::string(e.what()));
    }

    LOG_WARN("Could not query context size for model " + model_name);
    return 0;  // Let base class try probing
}

std::string OllamaBackend::query_available_model() {
    try {
        if (!http_client) {
            LOG_WARN("HTTP client not initialized for Ollama");
            return "";
        }

        // Build URL for /api/tags endpoint (strip /api/chat from endpoint)
        std::string base_url = api_endpoint;
        size_t pos = base_url.find("/api/chat");
        if (pos != std::string::npos) {
            base_url = base_url.substr(0, pos);
        }
        std::string tags_url = base_url + "/api/tags";

        // Prepare headers
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        // Make GET request to /api/tags
        LOG_DEBUG("Querying Ollama for available models: " + tags_url);
        HttpResponse response = http_client->get(tags_url, headers);

        if (response.is_success()) {
            json resp = json::parse(response.body);

            // Parse models list - look for first available model
            if (resp.contains("models") && resp["models"].is_array() && !resp["models"].empty()) {
                const auto& models = resp["models"];
                for (const auto& model : models) {
                    if (model.contains("name") && model["name"].is_string()) {
                        std::string name = model["name"].get<std::string>();
                        LOG_INFO("Found Ollama model: " + name);
                        return name;
                    }
                }
            }
        } else {
            LOG_DEBUG("Failed to query Ollama models: " + response.error_message);
        }
    } catch (const std::exception& e) {
        LOG_DEBUG("Failed to query Ollama available models: " + std::string(e.what()));
    }

    LOG_WARN("Could not query available models from Ollama");
    return "";
}