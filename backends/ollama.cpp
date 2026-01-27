#include "ollama.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"
#include "../tools/utf8_sanitizer.h"

using json = nlohmann::json;

OllamaBackend::OllamaBackend(size_t context_size, Session& session, EventCallback callback)
    : ApiBackend(context_size, session, callback) {
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
        dout(1) << "Using custom Ollama endpoint: " + api_endpoint << std::endl;
    }
    // else: keep default api_endpoint = "http://localhost:11434/api/chat"

    // Parse backend-specific config if available
    parse_backend_config();

    // --- Initialization ---

    // Auto-detect model if not specified
    if (model_name.empty()) {
        dout(1) << "No model specified, querying server for available models" << std::endl;
        auto models = get_models();
        if (!models.empty()) {
            model_name = models[0];
            dout(1) << "Using model from server: " + model_name << std::endl;
        } else {
            dout(1) << std::string("WARNING: ") +"Failed to query server for model, will use first API response to determine" << std::endl;
        }
    }

    // Query context size if not set
    if (this->context_size == 0) {
        size_t api_context_size = query_model_context_size(model_name);
        if (api_context_size > 0) {
            this->context_size = api_context_size;
            dout(1) << "Using API's context size: " + std::to_string(this->context_size) << std::endl;
        }
    }

    // NOTE: Context safety margin disabled - ollama already sends proper sampling params (top_p, top_k)
    // // Apply safety margin to auto-detected context size
    // // This accounts for server-side caching and other overhead
    // // Reduce by 10% or 4096 tokens, whichever is smaller
    // if (this->context_size > 0) {
    //     size_t original_context = this->context_size;
    //     size_t margin = std::min(this->context_size / 10, (size_t)4096);
    //     if (margin > 0 && this->context_size > margin) {
    //         this->context_size -= margin;
    //         dout(1) << "Applied context safety margin: " + std::to_string(original_context) +
    //                   " -> " + std::to_string(this->context_size) +
    //                   " (reduced by " + std::to_string(margin) + " tokens)" << std::endl;
    //     }
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

    // Force auto-eviction for Ollama backend
    // Ollama silently truncates messages when context is full (no error returned),
    // so we MUST proactively evict to prevent data loss
    session.auto_evict = true;
    dout(1) << "Auto-eviction enabled for Ollama (prevents silent truncation)" << std::endl;

    dout(1) << "Ollama backend initialized successfully" << std::endl;
}

OllamaBackend::~OllamaBackend() {
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
        std::string sanitized_body = utf8_sanitizer::sanitize_utf8(http_response.body);
        json json_resp = json::parse(sanitized_body);

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
                                dout(1) << "Failed to parse tool arguments: " + std::string(e.what()) << std::endl;
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
                                    dout(1) << "Failed to parse tool arguments: " + std::string(e.what()) << std::endl;
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

        switch (msg.role) {
            case Message::USER:
                message["role"] = "user";
                message["content"] = msg.content;
                break;

            case Message::ASSISTANT:
                message["role"] = "assistant";
                message["content"] = msg.content;
                break;

            case Message::TOOL_RESPONSE:
            case Message::FUNCTION:
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

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Ollama supports: temperature, top_p, top_k, repeat_penalty
        options["temperature"] = temperature;
        options["top_p"] = top_p;
        if (top_k > 0) {
            options["top_k"] = top_k;
        }
        if (repeat_penalty != 1.0f) {
            options["repeat_penalty"] = repeat_penalty;
        }
    }

    request["options"] = options;

    return request;
}

nlohmann::json OllamaBackend::build_request(const Session& session,
                                              Message::Role role,
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

        switch (msg.role) {
            case Message::USER:
                message["role"] = "user";
                message["content"] = msg.content;
                break;

            case Message::ASSISTANT:
                message["role"] = "assistant";
                message["content"] = msg.content;
                break;

            case Message::TOOL_RESPONSE:
            case Message::FUNCTION:
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
    switch (role) {
        case Message::USER:
            new_message["role"] = "user";
            new_message["content"] = content;
            break;

        case Message::ASSISTANT:
            new_message["role"] = "assistant";
            new_message["content"] = content;
            break;

        case Message::TOOL_RESPONSE:
        case Message::FUNCTION:
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

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Ollama supports: temperature, top_p, top_k, repeat_penalty
        options["temperature"] = temperature;
        options["top_p"] = top_p;
        if (top_k > 0) {
            options["top_k"] = top_k;
        }
        if (repeat_penalty != 1.0f) {
            options["repeat_penalty"] = repeat_penalty;
        }
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

// NOTE: add_message() removed - use Frontend::add_message_to_session() + generate_response() instead
#if 0
void OllamaBackend::add_message_REMOVED(Session& session,
                                    Message::Role role,
                                    const std::string& content,
                                    const std::string& tool_name,
                                    const std::string& tool_id,

                                    int max_tokens) {
    // If streaming disabled, use base class non-streaming implementation
    if (!config->streaming) {
        /* removed */ return;
    }

    reset_output_state();

    dout(1) << "OllamaBackend::add_message (streaming): max_tokens=" + std::to_string(max_tokens) << std::endl;

    const int MAX_RETRIES = 3;
    int retry = 0;

    while (retry < MAX_RETRIES) {
        // Build request with entire session + new message
        nlohmann::json request = build_request(session, role, content, tool_name, tool_id, max_tokens);

        // Ensure streaming is enabled (default for Ollama)
        request["stream"] = true;

        dout(1) << "Sending streaming request to Ollama API" << std::endl;

        // Get headers and endpoint
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

        // Enforce rate limits
        enforce_rate_limits();

        // Fire USER event callback before sending to provider
        // This displays the user prompt when accepted
        if (role == Message::USER && !content.empty()) {
            callback(CallbackEvent::USER_PROMPT, content, "", "");
        }

        // Streaming state
        Response accumulated_resp;
        accumulated_resp.success = true;
        accumulated_resp.code = Response::SUCCESS;
        std::string accumulated_content;
        bool stream_complete = false;
        std::string line_buffer;

        // Streaming callback to process NDJSON chunks
        auto stream_handler = [&](const std::string& chunk, void* user_data) -> bool {
            // Ollama uses NDJSON - newline-delimited JSON
            // Accumulate data and process complete lines
            line_buffer += chunk;

            size_t pos;
            while ((pos = line_buffer.find('\n')) != std::string::npos) {
                std::string line = line_buffer.substr(0, pos);
                line_buffer = line_buffer.substr(pos + 1);

                if (line.empty()) continue;

                try {
                    json delta_json = json::parse(line);

                    // Ollama format: message.content contains the delta
                    if (delta_json.contains("message") && delta_json["message"].contains("content")) {
                        std::string delta_text = delta_json["message"]["content"].get<std::string>();
                        if (!delta_text.empty()) {
                            accumulated_content += delta_text;
                            accumulated_resp.content = accumulated_content;

                            // Route through output() for filtering (backticks, buffering)
                            if (!output(delta_text)) {
                                stream_complete = true;
                                return true;
                            }
                        }

                        // Handle tool calls
                        if (delta_json["message"].contains("tool_calls") &&
                            delta_json["message"]["tool_calls"].is_array()) {
                            for (const auto& tc : delta_json["message"]["tool_calls"]) {
                                ToolParser::ToolCall tool_call;
                                if (tc.contains("function")) {
                                    tool_call.name = tc["function"].value("name", "");
                                    if (tc["function"].contains("arguments")) {
                                        tool_call.raw_json = tc["function"]["arguments"].dump();
                                    }
                                }
                                tool_call.tool_call_id = "ollama_" + std::to_string(accumulated_resp.tool_calls.size());
                                accumulated_resp.tool_calls.push_back(tool_call);
                            }
                        }
                    }

                    // Check for completion
                    if (delta_json.contains("done") && delta_json["done"].get<bool>()) {
                        stream_complete = true;

                        // Get finish reason
                        if (delta_json.contains("done_reason")) {
                            std::string reason = delta_json["done_reason"].get<std::string>();
                            if (reason == "stop") {
                                accumulated_resp.finish_reason = "stop";
                            } else if (reason == "length") {
                                accumulated_resp.finish_reason = "length";
                            } else {
                                accumulated_resp.finish_reason = reason;
                            }
                        }

                        // Extract token counts
                        if (delta_json.contains("prompt_eval_count")) {
                            accumulated_resp.prompt_tokens = delta_json["prompt_eval_count"].get<int>();
                        }
                        if (delta_json.contains("eval_count")) {
                            accumulated_resp.completion_tokens = delta_json["eval_count"].get<int>();
                        }
                    }

                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") +"Failed to parse Ollama NDJSON line: " + std::string(e.what()) << std::endl;
                }
            }

            // Always return true to curl to avoid errors
            return true;
        };

        // Make streaming HTTP call
        HttpResponse http_response = http_client->post_stream_cancellable(endpoint, request.dump(), headers,
                                                                           stream_handler, nullptr);

        // Check for HTTP errors
        if (!http_response.is_success()) {
            accumulated_resp.success = false;
            accumulated_resp.code = Response::ERROR;
            accumulated_resp.finish_reason = "error";

            // Extract error message from response body (JSON) or error_message
            std::string error_msg = http_response.error_message;
            if (error_msg.empty() && !http_response.body.empty()) {
                try {
                    auto json_body = nlohmann::json::parse(http_response.body);
                    if (json_body.contains("error") && json_body["error"].contains("message")) {
                        error_msg = json_body["error"]["message"].get<std::string>();
                    } else if (json_body.contains("error") && json_body["error"].is_string()) {
                        error_msg = json_body["error"].get<std::string>();
                    }
                } catch (...) {
                    error_msg = http_response.body.substr(0, 200);
                }
            }
            if (error_msg.empty()) {
                error_msg = "HTTP error " + std::to_string(http_response.status_code);
            }
            accumulated_resp.error = error_msg;

            // Check for context overflow
            int tokens_to_evict = extract_tokens_to_evict(http_response);

            if (tokens_to_evict > 0) {
                auto ranges = session.calculate_messages_to_evict(tokens_to_evict);
                if (ranges.empty()) {
                    accumulated_resp.code = Response::CONTEXT_FULL;
                    accumulated_resp.error = "Context full, cannot evict enough messages";
                    if (role == Message::TOOL_RESPONSE) {
                        add_tool_response(session, content, tool_name, tool_id);
                    }
                    callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return;
                }

                if (!session.evict_messages(ranges)) {
                    accumulated_resp.error = "Failed to evict messages";
                    if (role == Message::TOOL_RESPONSE) {
                        add_tool_response(session, content, tool_name, tool_id);
                    }
                    callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return;
                }

                retry++;
                dout(1) << "Evicted messages, retrying (attempt " + std::to_string(retry + 1) + "/" +
                        std::to_string(MAX_RETRIES) + "" << std::endl;

                continue;
            }

            // Non-context error - add TOOL_RESPONSE for session consistency
            if (role == Message::TOOL_RESPONSE) {
                add_tool_response(session, content, tool_name, tool_id);
            }
            callback(CallbackEvent::ERROR, accumulated_resp.error, "api_error", "");
            callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return;
        }

        // Flush any remaining output from the filter
        flush_output();

        // Success - update session with messages
        if (stream_complete && accumulated_resp.success) {
            // Estimate token counts
            int new_message_tokens = estimate_message_tokens(content);

            // Add user message to session
            Message user_msg(role, content, new_message_tokens);
            user_msg.tool_name = tool_name;
            user_msg.tool_call_id = tool_id;
            session.messages.push_back(user_msg);
            session.total_tokens += new_message_tokens;

            if (role == Message::USER) {
                session.last_user_message_index = session.messages.size() - 1;
                session.last_user_message_tokens = new_message_tokens;
            }

            // Add assistant response to session
            int asst_tokens = accumulated_resp.completion_tokens > 0 ?
                             accumulated_resp.completion_tokens :
                             estimate_message_tokens(accumulated_resp.content);

            Message asst_msg(Message::ASSISTANT, accumulated_resp.content, asst_tokens);
            session.messages.push_back(asst_msg);
            session.total_tokens += asst_tokens;
            session.last_assistant_message_index = session.messages.size() - 1;
            session.last_assistant_message_tokens = asst_tokens;
            session.last_prompt_tokens = accumulated_resp.prompt_tokens;

            accumulated_resp.was_streamed = true;

            // Set finish reason based on tool calls
            if (!accumulated_resp.tool_calls.empty()) {
                accumulated_resp.finish_reason = "tool_calls";
            } else if (accumulated_resp.finish_reason.empty()) {
                accumulated_resp.finish_reason = "stop";
            }
        }

        dout(1) << "add_message complete: prompt_tokens=" + std::to_string(accumulated_resp.prompt_tokens) +
                  ", completion_tokens=" + std::to_string(accumulated_resp.completion_tokens) +
                  ", finish_reason=" + accumulated_resp.finish_reason << std::endl;

        callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return;
    }

    // Max retries reached
    callback(CallbackEvent::ERROR, "Max retries reached", "error", "");
    callback(CallbackEvent::STOP, "error", "", "");
}
#endif

size_t OllamaBackend::query_model_context_size(const std::string& model_name) {
    const size_t DEFAULT_CONTEXT_SIZE = 8192;  // Conservative default for Ollama models

    // Try to query from Ollama's /api/show endpoint
    try {
        if (!http_client) {
            dout(1) << std::string("WARNING: ") +"HTTP client not initialized for Ollama" << std::endl;
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
                            dout(1) << "Found context_length in model_info[\"" + key + "\"]: " + std::to_string(ctx_size) << std::endl;
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
        dout(1) << "Failed to query Ollama model context size: " + std::string(e.what()) << std::endl;
    }

    dout(1) << std::string("WARNING: ") +"Could not query context size for model " + model_name << std::endl;
    return 0;  // Let base class try probing
}

std::vector<std::string> OllamaBackend::fetch_models() {
    std::vector<std::string> result;

    try {
        if (!http_client) {
            return result;
        }

        // Build URL for /api/tags endpoint
        std::string base_url = api_endpoint;
        size_t pos = base_url.find("/api/chat");
        if (pos != std::string::npos) {
            base_url = base_url.substr(0, pos);
        }
        std::string tags_url = base_url + "/api/tags";

        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        HttpResponse response = http_client->get(tags_url, headers);

        if (response.is_success()) {
            json resp = json::parse(response.body);
            if (resp.contains("models") && resp["models"].is_array()) {
                for (const auto& model : resp["models"]) {
                    if (model.contains("name") && model["name"].is_string()) {
                        result.push_back(model["name"].get<std::string>());
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        dout(1) << "Failed to query Ollama models: " + std::string(e.what()) << std::endl;
    }

    return result;
}