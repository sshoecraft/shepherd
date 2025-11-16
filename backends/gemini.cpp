#include "gemini.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// GeminiBackend implementation
GeminiBackend::GeminiBackend(size_t context_size)
    : ApiBackend(context_size) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Detect model configuration from Models database
    model_config = Models::detect_from_api_model("gemini", model_name);
    max_output_tokens = model_config.max_output_tokens;
    LOG_DEBUG("Detected model config: context=" + std::to_string(model_config.context_window) +
              ", max_output=" + std::to_string(model_config.max_output_tokens) +
              ", param_name=" + model_config.max_tokens_param_name);

    // Set API endpoint from config if provided
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        LOG_INFO("Using custom Gemini endpoint: " + api_endpoint);
    }

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config();

    LOG_DEBUG("GeminiBackend created");
}

GeminiBackend::~GeminiBackend() {
}

void GeminiBackend::initialize(Session& session) {
    LOG_INFO("Initializing Gemini backend...");

    // Validate API key
    if (api_key.empty()) {
        LOG_ERROR("Gemini API key is required");
        throw std::runtime_error("Gemini API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        model_name = "gemini-2.0-flash-exp";
        LOG_INFO("No model specified, using default: " + model_name);
        model_config = Models::detect_from_api_model("gemini", model_name);
    max_output_tokens = model_config.max_output_tokens;
    }

    // Set context size from model config if not already set
    if (context_size == 0 && model_config.context_window > 0) {
        context_size = model_config.context_window;
        LOG_INFO("Using model's context size: " + std::to_string(context_size));
    }

    // Call base class initialize() which handles calibration
    ApiBackend::initialize(session);

    LOG_INFO("Gemini backend initialized successfully");
}

size_t GeminiBackend::query_model_context_size(const std::string& model_name) {
    // Gemini doesn't have a /v1/models endpoint to query
    // Return 0 to let models database handle known models
    return 0;
}

std::string GeminiBackend::make_get_request(const std::string& url) {
    if (!http_client) {
        LOG_ERROR("HTTP client not initialized");
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
            LOG_ERROR("Authentication failed: " + error_msg);
            throw BackendError("Authentication failed: " + error_msg);
        }

        LOG_DEBUG("GET request to " + url + " failed: " + response.error_message);
        return "";
    }

    return response.body;
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
        json json_resp = json::parse(http_response.body);

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
                                LOG_WARN("Failed to parse tool call parameters: " + std::string(e.what()));
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

        if (msg.type == Message::USER) {
            content["role"] = "user";
            content["parts"] = json::array({{{"text", msg.content}}});
        } else if (msg.type == Message::ASSISTANT) {
            content["role"] = "model";
            // Check if this message has functionCall parts (stored as JSON)
            if (!msg.tool_calls_json.empty()) {
                try {
                    content["parts"] = json::parse(msg.tool_calls_json);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to parse stored parts: " + std::string(e.what()));
                    content["parts"] = json::array({{{"text", msg.content}}});
                }
            } else {
                content["parts"] = json::array({{{"text", msg.content}}});
            }
        } else if (msg.type == Message::TOOL) {
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

    // Add sampling parameters (Gemini uses generationConfig object)
    json generation_config;
    generation_config["temperature"] = temperature;
    generation_config["topP"] = top_p;
    if (top_k > 0) {
        generation_config["topK"] = top_k;
    }
    request["generationConfig"] = generation_config;

    return request;
}

nlohmann::json GeminiBackend::build_request(const Session& session,
                                              Message::Type type,
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

        if (msg.type == Message::USER) {
            jcontent["role"] = "user";
            jcontent["parts"] = json::array({{{"text", msg.content}}});
        } else if (msg.type == Message::ASSISTANT) {
            jcontent["role"] = "model";
            // Check if this message has functionCall parts (stored as JSON)
            if (!msg.tool_calls_json.empty()) {
                try {
                    jcontent["parts"] = json::parse(msg.tool_calls_json);
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to parse stored parts: " + std::string(e.what()));
                    jcontent["parts"] = json::array({{{"text", msg.content}}});
                }
            } else {
                jcontent["parts"] = json::array({{{"text", msg.content}}});
            }
        } else if (msg.type == Message::TOOL) {
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
    if (type == Message::USER) {
        new_content["role"] = "user";
        new_content["parts"] = json::array({{{"text", content}}});
    } else if (type == Message::TOOL) {
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

    // Add sampling parameters (Gemini uses generationConfig object)
    json generation_config;
    generation_config["temperature"] = temperature;
    generation_config["topP"] = top_p;
    if (top_k > 0) {
        generation_config["topK"] = top_k;
    }
    request["generationConfig"] = generation_config;

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
