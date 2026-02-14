#include "gemini.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"
#include "sse_parser.h"
#include "../tools/utf8_sanitizer.h"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// GeminiBackend implementation
GeminiBackend::GeminiBackend(size_t context_size, Session& session, EventCallback callback)
    : ApiBackend(context_size, session, callback) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Detect model configuration from Models database
    model_config = Models::detect_from_api_model("gemini", model_name);
    max_output_tokens = model_config.max_output_tokens;
    dout(1) << "Detected model config: context=" + std::to_string(model_config.context_window) +
              ", max_output=" + std::to_string(model_config.max_output_tokens) +
              ", param_name=" + model_config.max_tokens_param_name << std::endl;

    // Set API endpoint from config if provided
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        dout(1) << "Using custom Gemini endpoint: " + api_endpoint << std::endl;
    }

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config();

    // --- Initialization ---

    // Validate API key
    if (api_key.empty()) {
        std::cerr << "Gemini API key is required" << std::endl;
        throw std::runtime_error("Gemini API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        model_name = "gemini-2.0-flash-exp";
        dout(1) << "No model specified, using default: " + model_name << std::endl;
        model_config = Models::detect_from_api_model("gemini", model_name);
        max_output_tokens = model_config.max_output_tokens;
    }

    // Set context size from model config if not already set
    if (this->context_size == 0 && model_config.context_window > 0) {
        this->context_size = model_config.context_window;
        dout(1) << "Using model's context size: " + std::to_string(this->context_size) << std::endl;
    }

    // If context_size is still 0, try to query it from the API
    if (this->context_size == 0) {
        size_t api_context_size = query_model_context_size(model_name);
        if (api_context_size > 0) {
            this->context_size = api_context_size;
            dout(1) << "Using API's context size: " + std::to_string(this->context_size) << std::endl;
        }
    }

    // Calibrate token counts (if enabled in config)
    if (config->calibration) {
        dout(1) << "Calibrating token counts..." << std::endl;
        calibrate_token_counts(session);
    } else {
        dout(1) << "Calibration disabled, using default estimates" << std::endl;
        session.system_message_tokens = estimate_message_tokens(session.system_message);
        session.last_prompt_tokens = session.system_message_tokens;
    }

    dout(1) << "Gemini backend initialized successfully" << std::endl;
}

GeminiBackend::~GeminiBackend() {
}

size_t GeminiBackend::query_model_context_size(const std::string& model_name) {
    // Gemini doesn't have a /v1/models endpoint to query
    // Return 0 to let models database handle known models
    return 0;
}

std::string GeminiBackend::make_get_request(const std::string& url) {
    if (!http_client) {
        dout(1) << "HTTP client not initialized" << std::endl;
        return "";
    }

    // Make GET request with API key in header
    std::map<std::string, std::string> headers = get_api_headers();
    HttpResponse response = http_client->get(url, headers);

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
            dout(1) << "Authentication failed: " + error_msg << std::endl;
            throw BackendError("Authentication failed: " + error_msg);
        }

        dout(1) << "GET request to " + url + " failed: " + response.error_message << std::endl;
        return "";
    }

    return response.body;
}

std::vector<std::string> GeminiBackend::fetch_models() {
    std::vector<std::string> result;

    if (!http_client || api_key.empty()) {
        return result;
    }

    // Gemini uses query parameter for API key, endpoint is /v1beta/models
    std::string url = "https://generativelanguage.googleapis.com/v1beta/models?key=" + api_key;

    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";

    HttpResponse response = http_client->get(url, headers);

    if (response.is_success()) {
        try {
            auto j = json::parse(response.body);
            if (j.contains("models") && j["models"].is_array()) {
                for (const auto& model : j["models"]) {
                    if (model.contains("name") && model["name"].is_string()) {
                        // Name is like "models/gemini-pro", extract just "gemini-pro"
                        std::string name = model["name"].get<std::string>();
                        size_t pos = name.find("models/");
                        if (pos != std::string::npos) {
                            name = name.substr(pos + 7);
                        }
                        result.push_back(name);
                    }
                }
            }
        } catch (const json::exception& e) {
            dout(1) << "Failed to parse Gemini /models response: " + std::string(e.what()) << std::endl;
        }
    }

    return result;
}

Response GeminiBackend::parse_http_response(const HttpResponse& http_response) {
    Response resp;

    // Check HTTP status
    if (!http_response.is_success()) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.finish_reason = "error";

        // Try to parse error JSON
        try {
            json error_json = json::parse(http_response.body);
            if (error_json.contains("error") && error_json["error"].contains("message")) {
                resp.error = error_json["error"]["message"].get<std::string>();
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

        // Extract token usage
        if (json_resp.contains("usageMetadata")) {
            const auto& usage = json_resp["usageMetadata"];
            if (usage.contains("promptTokenCount")) {
                resp.prompt_tokens = usage["promptTokenCount"].get<int>();
            }
            if (usage.contains("candidatesTokenCount")) {
                resp.completion_tokens = usage["candidatesTokenCount"].get<int>();
            }
        }

        // Extract content from candidates
        if (json_resp.contains("candidates") && json_resp["candidates"].is_array() && !json_resp["candidates"].empty()) {
            const auto& candidate = json_resp["candidates"][0];

            // Get finish reason
            if (candidate.contains("finishReason")) {
                resp.finish_reason = candidate["finishReason"].get<std::string>();
            }

            // Get content from parts
            if (candidate.contains("content") && candidate["content"].contains("parts")) {
                const auto& parts = candidate["content"]["parts"];
                std::string response_text;
                bool has_function_call = false;

                for (const auto& part : parts) {
                    if (part.contains("text")) {
                        response_text += part["text"].get<std::string>();
                    } else if (part.contains("functionCall")) {
                        has_function_call = true;
                        // Parse function call into ToolParser::ToolCall
                        ToolParser::ToolCall tool_call;

                        const auto& func_call = part["functionCall"];
                        if (func_call.contains("name")) {
                            tool_call.name = func_call["name"].get<std::string>();
                        }

                        // Generate ID for this tool call
                        tool_call.tool_call_id = "call_" + std::to_string(resp.tool_calls.size());

                        if (func_call.contains("args")) {
                            tool_call.raw_json = func_call["args"].dump();

                            // Parse parameters
                            try {
                                const auto& args = func_call["args"];
                                for (auto it = args.begin(); it != args.end(); ++it) {
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
                                dout(1) << std::string("WARNING: ") +"Failed to parse tool call parameters: " + std::string(e.what()) << std::endl;
                            }
                        }

                        resp.tool_calls.push_back(tool_call);
                    }
                }

                // If there are functionCall parts, store the full parts array as JSON
                // so it can be reconstructed when building future requests
                if (has_function_call) {
                    resp.tool_calls_json = parts.dump();
                }
                resp.content = response_text;
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

nlohmann::json GeminiBackend::build_request_from_session(const Session& session, int max_tokens) {
    json request;

    // Set generation config
    json gen_config;
    // max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    if (max_tokens > 0) {
        gen_config["maxOutputTokens"] = max_tokens;
    }
    if (!gen_config.empty()) {
        request["generationConfig"] = gen_config;
    }

    // System instruction (if present)
    if (!session.system_message.empty()) {
        request["systemInstruction"] = {
            {"parts", json::array({
                {{"text", session.system_message}}
            })}
        };
    }

    // Build contents array (user/model messages)
    json contents = json::array();

    for (const auto& msg : session.messages) {
        json content;

        if (msg.role == Message::USER) {
            content["role"] = "user";
            content["parts"] = json::array({{{"text", msg.content}}});
        } else if (msg.role == Message::ASSISTANT) {
            content["role"] = "model";
            // Check if this message has functionCall parts (stored as JSON)
            if (!msg.tool_calls_json.empty()) {
                try {
                    json tool_calls = json::parse(msg.tool_calls_json);
                    // Check if it's OpenAI format (array with "function" objects)
                    // vs Gemini format (array with "functionCall" parts)
                    if (tool_calls.is_array() && !tool_calls.empty() &&
                        tool_calls[0].contains("function")) {
                        // Convert OpenAI format to Gemini format
                        json parts = json::array();
                        for (const auto& tc : tool_calls) {
                            if (tc.contains("function")) {
                                json func_call;
                                func_call["functionCall"]["name"] = tc["function"]["name"];
                                if (tc["function"].contains("arguments")) {
                                    std::string args_str = tc["function"]["arguments"].get<std::string>();
                                    try {
                                        func_call["functionCall"]["args"] = json::parse(args_str);
                                    } catch (...) {
                                        func_call["functionCall"]["args"] = json::object();
                                    }
                                }
                                parts.push_back(func_call);
                            }
                        }
                        content["parts"] = parts;
                    } else {
                        // Already in Gemini format
                        content["parts"] = tool_calls;
                    }
                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") +"Failed to parse stored parts: " + std::string(e.what()) << std::endl;
                    content["parts"] = json::array({{{"text", msg.content}}});
                }
            } else {
                content["parts"] = json::array({{{"text", msg.content}}});
            }
        } else if (msg.role == Message::TOOL_RESPONSE) {
            // Tool results in Gemini format
            content["role"] = "user";
            content["parts"] = json::array({
                {
                    {"functionResponse", {
                        {"name", msg.tool_name},
                        {"response", {
                            {"content", msg.content}
                        }}
                    }}
                }
            });
        }

        if (!content.empty()) {
            contents.push_back(content);
        }
    }

    request["contents"] = contents;

    // Add tools if present
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json function_decl;
            function_decl["name"] = tool.name;
            function_decl["description"] = tool.description;

            if (!tool.parameters.empty()) {
                // Gemini requires array properties to have 'items' field
                // Fix up the schema if needed
                json params = tool.parameters;
                if (params.contains("properties") && params["properties"].is_object()) {
                    for (auto& [key, prop] : params["properties"].items()) {
                        if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                            // Add default items schema
                            prop["items"] = {{"type", "object"}};
                        }
                    }
                }
                function_decl["parameters"] = params;
            }

            tools.push_back({{"functionDeclarations", json::array({function_decl})}});
        }
        request["tools"] = tools;
    }

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Gemini uses generationConfig object
        json generation_config;
        generation_config["temperature"] = temperature;
        generation_config["topP"] = top_p;
        if (top_k > 0) {
            generation_config["topK"] = top_k;
        }
        request["generationConfig"] = generation_config;
    }

    return request;
}

nlohmann::json GeminiBackend::build_request(const Session& session,
                                              Message::Role role,
                                              const std::string& content,
                                              const std::string& tool_name,
                                              const std::string& tool_id,
                                              int max_tokens) {
    json request;

    // Set generation config
    json gen_config;
    // max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    if (max_tokens > 0) {
        gen_config["maxOutputTokens"] = max_tokens;
    }
    if (!gen_config.empty()) {
        request["generationConfig"] = gen_config;
    }

    // System instruction
    if (!session.system_message.empty()) {
        request["systemInstruction"] = {
            {"parts", json::array({
                {{"text", session.system_message}}
            })}
        };
    }

    // Build contents array
    json contents = json::array();

    // Add existing messages
    for (const auto& msg : session.messages) {
        json jcontent;

        if (msg.role == Message::USER) {
            jcontent["role"] = "user";
            jcontent["parts"] = json::array({{{"text", msg.content}}});
        } else if (msg.role == Message::ASSISTANT) {
            jcontent["role"] = "model";
            // Check if this message has functionCall parts (stored as JSON)
            if (!msg.tool_calls_json.empty()) {
                try {
                    jcontent["parts"] = json::parse(msg.tool_calls_json);
                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") +"Failed to parse stored parts: " + std::string(e.what()) << std::endl;
                    jcontent["parts"] = json::array({{{"text", msg.content}}});
                }
            } else {
                jcontent["parts"] = json::array({{{"text", msg.content}}});
            }
        } else if (msg.role == Message::TOOL_RESPONSE) {
            jcontent["role"] = "user";
            jcontent["parts"] = json::array({
                {
                    {"functionResponse", {
                        {"name", msg.tool_name},
                        {"response", {
                            {"content", msg.content}
                        }}
                    }}
                }
            });
        }

        if (!jcontent.empty()) {
            contents.push_back(jcontent);
        }
    }

    // Add the new message
    json new_content;
    if (role == Message::USER) {
        new_content["role"] = "user";
        new_content["parts"] = json::array({{{"text", content}}});
    } else if (role == Message::TOOL_RESPONSE) {
        new_content["role"] = "user";
        new_content["parts"] = json::array({
            {
                {"functionResponse", {
                    {"name", tool_name},
                    {"response", {
                        {"content", content}
                    }}
                }}
            }
        });
    }

    if (!new_content.empty()) {
        contents.push_back(new_content);
    }

    request["contents"] = contents;

    // Add tools if present
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json function_decl;
            function_decl["name"] = tool.name;
            function_decl["description"] = tool.description;

            if (!tool.parameters.empty()) {
                // Gemini requires array properties to have 'items' field
                // Fix up the schema if needed
                json params = tool.parameters;
                if (params.contains("properties") && params["properties"].is_object()) {
                    for (auto& [key, prop] : params["properties"].items()) {
                        if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                            // Add default items schema
                            prop["items"] = {{"type", "object"}};
                        }
                    }
                }
                function_decl["parameters"] = params;
            }

            tools.push_back({{"functionDeclarations", json::array({function_decl})}});
        }
        request["tools"] = tools;
    }

    // Add sampling parameters (only if sampling is enabled)
    if (sampling) {
        // Gemini uses generationConfig object
        json generation_config;
        generation_config["temperature"] = temperature;
        generation_config["topP"] = top_p;
        if (top_k > 0) {
            generation_config["topK"] = top_k;
        }
        request["generationConfig"] = generation_config;
    }

    return request;
}

std::string GeminiBackend::parse_response(const nlohmann::json& response) {
    // Extract text content from Gemini response
    if (response.contains("candidates") && !response["candidates"].empty()) {
        const auto& candidate = response["candidates"][0];
        if (candidate.contains("content") && candidate["content"].contains("parts")) {
            std::string result;
            for (const auto& part : candidate["content"]["parts"]) {
                if (part.contains("text")) {
                    result += part["text"].get<std::string>();
                }
            }
            return result;
        }
    }
    return "";
}

int GeminiBackend::extract_tokens_to_evict(const HttpResponse& response) {
    // Extract error message from HTTP response
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

    // Gemini error parsing for context overflow
    // TODO: Add Gemini-specific error message parsing when we encounter real errors
    // For now, return -1 (not a context error)
    return -1;
}

std::map<std::string, std::string> GeminiBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    // Gemini uses x-goog-api-key header
    headers["x-goog-api-key"] = api_key;
    return headers;
}

std::string GeminiBackend::get_api_endpoint() {
    // Gemini endpoint format: base_url/{model}:generateContent
    return api_endpoint + model_name + ":generateContent";
}

std::string GeminiBackend::get_streaming_endpoint() {
    // Gemini streaming endpoint format: base_url/{model}:streamGenerateContent?alt=sse
    return api_endpoint + model_name + ":streamGenerateContent?alt=sse";
}


void GeminiBackend::generate_from_session(Session& session, int max_tokens) {
    // Always use streaming for generate_from_session - the callback mechanism
    // works the same whether the API server client requested stream or not
    reset_output_state();
    clear_tool_calls();  // Clear any previous tool calls

    dout(1) << "GeminiBackend::generate_from_session (streaming): max_tokens=" + std::to_string(max_tokens) << std::endl;

    // Build request from session
    nlohmann::json request = build_request_from_session(session, max_tokens);

    dout(1) << "Sending streaming request to Gemini API (generate_from_session)" << std::endl;

    // Get headers and streaming endpoint
    auto headers = get_api_headers();
    std::string endpoint = get_streaming_endpoint();

    // Streaming state
    std::string accumulated_content;
    SSEParser sse_parser;
    bool stream_complete = false;
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int completion_tokens = 0;

    // Streaming callback to process SSE chunks
    auto stream_handler = [&](const std::string& chunk, void* user_data) -> bool {
        sse_parser.process_chunk(chunk,
            [&](const std::string& event, const std::string& data, const std::string& id) -> bool {
                try {
                    json delta_json = json::parse(data);

                    // Gemini format: candidates[0].content.parts[0].text
                    if (delta_json.contains("candidates") && !delta_json["candidates"].empty()) {
                        const auto& candidate = delta_json["candidates"][0];

                        // Get finish reason if present
                        if (candidate.contains("finishReason") && !candidate["finishReason"].is_null()) {
                            std::string reason = candidate["finishReason"].get<std::string>();
                            if (reason == "STOP") {
                                finish_reason = "stop";
                            } else if (reason == "MAX_TOKENS") {
                                finish_reason = "length";
                            } else {
                                finish_reason = reason;
                            }
                        }

                        // Get content
                        if (candidate.contains("content") && candidate["content"].contains("parts")) {
                            const auto& parts = candidate["content"]["parts"];
                            if (!parts.empty() && parts[0].contains("text")) {
                                std::string delta_text = parts[0]["text"].get<std::string>();
                                accumulated_content += delta_text;

                                // Route through output() for filtering (backticks, buffering)
                                if (!output(delta_text)) {
                                    stream_complete = true;
                                    return false;
                                }
                            }

                            // Handle function calls
                            if (!parts.empty() && parts[0].contains("functionCall")) {
                                const auto& fc = parts[0]["functionCall"];
                                std::string tool_name = fc.value("name", "");
                                std::string params_json = fc.contains("args") ? fc["args"].dump() : "{}";
                                static int gemini_tool_counter = 0;
                                std::string tool_id = "gemini_" + std::to_string(++gemini_tool_counter);

                                // Record for emission after STOP (don't emit here)
                                record_tool_call(tool_name, params_json, tool_id);

                                dout(1) << "generate_from_session: got functionCall name=" + tool_name << std::endl;
                            }
                        }
                    }

                    // Extract usage metadata
                    if (delta_json.contains("usageMetadata")) {
                        const auto& usage = delta_json["usageMetadata"];
                        prompt_tokens = usage.value("promptTokenCount", 0);
                        completion_tokens = usage.value("candidatesTokenCount", 0);
                        stream_complete = true;
                    }

                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") + "Failed to parse Gemini SSE data: " + std::string(e.what()) << std::endl;
                }

                return true;
            });

        return true;
    };

    // Make streaming HTTP call
    HttpResponse http_response = http_client->post_stream_cancellable(endpoint, request.dump(), headers,
                                                                       stream_handler, nullptr);

    // Check for HTTP errors
    if (!http_response.is_success() && !stream_complete) {
        std::string error_msg = http_response.error_message.empty() ? "API request failed" : http_response.error_message;

        // Check if this is a context-full error - throw ContextFullException
        // so the frontend (session owner) can handle eviction
        int tokens_to_evict = extract_tokens_to_evict(http_response);
        if (tokens_to_evict > 0) {
            throw ContextFullException(error_msg);
        }

        callback(CallbackEvent::ERROR, error_msg, "error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return;
    }

    // Flush any remaining output from the filter
    flush_output();

    // Update session token counts using delta tracking
    update_session_tokens(session, prompt_tokens, completion_tokens);

    // Adjust finish reason for tool calls
    if (!accumulated_tool_calls.empty()) {
        finish_reason = "tool_calls";
    }

    callback(CallbackEvent::STOP, finish_reason, "", "");

    // Emit tool calls AFTER STOP - frontend handles immediately
    for (const auto& tc : accumulated_tool_calls) {
        std::string name = tc["function"]["name"];
        std::string args = tc["function"]["arguments"];
        std::string id = tc["id"];
        callback(CallbackEvent::TOOL_CALL, args, name, id);
    }
}
