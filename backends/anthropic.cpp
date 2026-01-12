#include "anthropic.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"
#include "sse_parser.h"
#include "../tools/utf8_sanitizer.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cctype>

using json = nlohmann::json;

// AnthropicBackend implementation
AnthropicBackend::AnthropicBackend(size_t context_size, Session& session, EventCallback callback)
    : ApiBackend(context_size, session, callback) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Detect model configuration from Models database
    model_config = Models::detect_from_api_model("anthropic", model_name);
    max_output_tokens = model_config.max_output_tokens;
    dout(1) << "Detected model config: context=" + std::to_string(model_config.context_window) +
              ", max_output=" + std::to_string(model_config.max_output_tokens) +
              ", param_name=" + model_config.max_tokens_param_name << std::endl;

    // Set API endpoint from config if provided
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        dout(1) << "Using custom Anthropic endpoint: " + api_endpoint << std::endl;
    }

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config();

    // --- Initialization ---

    // Validate API key
    if (api_key.empty()) {
        std::cerr << "Anthropic API key is required" << std::endl;
        throw std::runtime_error("Anthropic API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        model_name = "claude-sonnet-4-5";
        dout(1) << "No model specified, using default: " + model_name << std::endl;
        model_config = Models::detect_from_api_model("anthropic", model_name);
        max_output_tokens = model_config.max_output_tokens;
    }

    // Set API version
    if (api_version.empty()) {
        api_version = "2023-06-01";
        dout(1) << "Using Anthropic API version: " + api_version << std::endl;
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

    dout(1) << "Anthropic backend initialized successfully" << std::endl;
}

AnthropicBackend::~AnthropicBackend() {
}



// ========== ApiBackend Required Methods ==========

Response AnthropicBackend::parse_http_response(const HttpResponse& http_response) {
    Response resp;

    // Check HTTP status
    if (!http_response.is_success()) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.finish_reason = "error";

        // Try to parse error JSON
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

        // Extract usage data
        if (json_resp.contains("usage") && json_resp["usage"].is_object()) {
            resp.prompt_tokens = json_resp["usage"].value("input_tokens", 0);
            resp.completion_tokens = json_resp["usage"].value("output_tokens", 0);
        }

        // Extract finish reason
        if (json_resp.contains("stop_reason")) {
            resp.finish_reason = json_resp["stop_reason"].get<std::string>();
        }

        // Extract content - handle both text and tool_use blocks
        if (json_resp.contains("content") && json_resp["content"].is_array() && !json_resp["content"].empty()) {
            std::string response_text;
            bool has_tool_use = false;

            // Process all content blocks
            for (const auto& content_block : json_resp["content"]) {
                if (!content_block.contains("type")) continue;

                std::string block_type = content_block["type"];

                if (block_type == "text" && content_block.contains("text")) {
                    // Append text content
                    response_text += content_block["text"].get<std::string>();
                } else if (block_type == "tool_use") {
                    has_tool_use = true;

                    // Parse tool_use block into ToolParser::ToolCall
                    ToolParser::ToolCall tool_call;

                    if (content_block.contains("id")) {
                        tool_call.tool_call_id = content_block["id"].get<std::string>();
                    }

                    if (content_block.contains("name")) {
                        tool_call.name = content_block["name"].get<std::string>();
                    }

                    if (content_block.contains("input")) {
                        tool_call.raw_json = content_block["input"].dump();

                        // Parse parameters
                        try {
                            const auto& input = content_block["input"];
                            for (auto it = input.begin(); it != input.end(); ++it) {
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

            // If there are tool_use blocks, store the full content array as JSON
            // so it can be reconstructed when building future requests
            if (has_tool_use) {
                resp.content = json_resp["content"].dump();
            } else {
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

nlohmann::json AnthropicBackend::build_request_from_session(const Session& session, int max_tokens) {
    json request;
    request["model"] = model_name;

    // Cap max_tokens at model's max_output_tokens limit (Anthropic-specific constraint)
    // max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    int actual_max_tokens = max_tokens;
    if (actual_max_tokens <= 0) {
        actual_max_tokens = 1024;  // Sensible default
    }

    request["max_tokens"] = actual_max_tokens;

    // Build messages array
    json messages = json::array();

    // Add all messages from session
    for (const auto& msg : session.messages) {
        json jmsg;

        if (msg.role == Message::USER) {
            jmsg["role"] = "user";
            jmsg["content"] = msg.content;
        } else if (msg.role == Message::ASSISTANT) {
            jmsg["role"] = "assistant";

            // Check if this message has tool_calls (from OpenAI format)
            if (!msg.tool_calls_json.empty()) {
                try {
                    json tool_calls = json::parse(msg.tool_calls_json);
                    json content_array = json::array();

                    // Add text content first if present
                    // But skip if content is a JSON array (native Anthropic format with tool_use blocks)
                    if (!msg.content.empty()) {
                        bool is_json_array = false;
                        try {
                            json content_test = json::parse(msg.content);
                            is_json_array = content_test.is_array();
                        } catch (...) {}

                        if (!is_json_array) {
                            content_array.push_back({{"type", "text"}, {"text", msg.content}});
                        }
                    }

                    // Convert OpenAI tool_calls to Anthropic tool_use blocks
                    if (tool_calls.is_array()) {
                        for (const auto& tc : tool_calls) {
                            if (tc.contains("function")) {
                                json tool_use;
                                tool_use["type"] = "tool_use";
                                tool_use["id"] = tc.value("id", "");
                                tool_use["name"] = tc["function"].value("name", "");

                                // Parse arguments string to JSON object
                                if (tc["function"].contains("arguments")) {
                                    std::string args_str = tc["function"]["arguments"].get<std::string>();
                                    try {
                                        tool_use["input"] = json::parse(args_str);
                                    } catch (...) {
                                        tool_use["input"] = json::object();
                                    }
                                } else {
                                    tool_use["input"] = json::object();
                                }
                                content_array.push_back(tool_use);
                            }
                        }
                    }

                    jmsg["content"] = content_array;
                } catch (const std::exception& e) {
                    dout(1) << "Failed to parse tool_calls_json: " << e.what() << std::endl;
                    jmsg["content"] = msg.content;
                }
            } else {
                // Check if content is JSON array (from native tool_use blocks)
                try {
                    json content_json = json::parse(msg.content);
                    if (content_json.is_array()) {
                        jmsg["content"] = content_json;
                    } else {
                        jmsg["content"] = msg.content;
                    }
                } catch (...) {
                    jmsg["content"] = msg.content;
                }
            }
        } else if (msg.role == Message::TOOL_RESPONSE) {
            // Tool results are user messages with tool_result content blocks
            // Anthropic requires all tool_results to be in a single user message
            json tool_result_block = {
                {"type", "tool_result"},
                {"tool_use_id", msg.tool_call_id},
                {"content", msg.content}
            };

            // Check if last message is already a user message with tool_result content
            // If so, append to it instead of creating a new message
            if (!messages.empty() && messages.back()["role"] == "user" &&
                messages.back()["content"].is_array() &&
                !messages.back()["content"].empty() &&
                messages.back()["content"][0].contains("type") &&
                messages.back()["content"][0]["type"] == "tool_result") {
                // Append to existing tool_result message
                messages.back()["content"].push_back(tool_result_block);
                continue;  // Skip adding jmsg below
            }

            jmsg["role"] = "user";
            jmsg["content"] = json::array({tool_result_block});
        }

        if (!jmsg.empty()) {
            messages.push_back(jmsg);
        }
    }

    request["messages"] = messages;

    // Add system message as separate field
    if (!session.system_message.empty()) {
        request["system"] = session.system_message;
    }

    // Add tools if present
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json jtool;
            jtool["name"] = tool.name;
            jtool["description"] = tool.description;

            if (!tool.parameters.empty()) {
                jtool["input_schema"] = tool.parameters;
            } else {
                jtool["input_schema"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add sampling parameters (Anthropic supports: temperature, top_p, top_k)
    // Note: Anthropic doesn't allow both temperature and top_p to be set
    // Use temperature unless it's at default (0.7) and top_p is not default (1.0)
    bool temp_is_default = (temperature >= 0.69f && temperature <= 0.71f);
    bool top_p_is_default = (top_p >= 0.99f);

    if (temp_is_default && !top_p_is_default) {
        request["top_p"] = top_p;
    } else {
        request["temperature"] = temperature;
    }
    if (top_k > 0) {
        request["top_k"] = top_k;
    }

    return request;
}

nlohmann::json AnthropicBackend::build_request(const Session& session,
                                                Message::Role role,
                                                const std::string& content,
                                                const std::string& tool_name,
                                                const std::string& tool_id,
                                                int max_tokens) {
    json request;
    request["model"] = model_name;

    // Cap max_tokens at model's max_output_tokens limit (Anthropic-specific constraint)
    // max_tokens is already capped by session.cpp's calculate_desired_completion_tokens()
    int actual_max_tokens = max_tokens;
    if (actual_max_tokens <= 0) {
        actual_max_tokens = 1024;  // Sensible default
    }

    request["max_tokens"] = actual_max_tokens;

    // Build messages array
    json messages = json::array();

    // Add system message as separate field
    if (!session.system_message.empty()) {
        request["system"] = session.system_message;
    }

    // Add existing messages
    for (const auto& msg : session.messages) {
        json jmsg;

        if (msg.role == Message::USER) {
            jmsg["role"] = "user";
            jmsg["content"] = msg.content;
        } else if (msg.role == Message::ASSISTANT) {
            jmsg["role"] = "assistant";

            dout(1) << "Processing assistant message, tool_calls_json.empty()=" << (msg.tool_calls_json.empty() ? "true" : "false") << std::endl;
            if (!msg.tool_calls_json.empty()) {
                dout(1) << "tool_calls_json=" << msg.tool_calls_json << std::endl;
            }

            // Check if this message has tool_calls (from OpenAI format)
            if (!msg.tool_calls_json.empty()) {
                try {
                    json tool_calls = json::parse(msg.tool_calls_json);
                    json content_array = json::array();

                    // Add text content first if present
                    // But skip if content is a JSON array (native Anthropic format with tool_use blocks)
                    if (!msg.content.empty()) {
                        bool is_json_array = false;
                        try {
                            json content_test = json::parse(msg.content);
                            is_json_array = content_test.is_array();
                        } catch (...) {}

                        if (!is_json_array) {
                            content_array.push_back({{"type", "text"}, {"text", msg.content}});
                        }
                    }

                    // Convert OpenAI tool_calls to Anthropic tool_use blocks
                    if (tool_calls.is_array()) {
                        for (const auto& tc : tool_calls) {
                            if (tc.contains("function")) {
                                json tool_use;
                                tool_use["type"] = "tool_use";
                                tool_use["id"] = tc.value("id", "");
                                tool_use["name"] = tc["function"].value("name", "");

                                // Parse arguments string to JSON object
                                if (tc["function"].contains("arguments")) {
                                    std::string args_str = tc["function"]["arguments"].get<std::string>();
                                    try {
                                        tool_use["input"] = json::parse(args_str);
                                    } catch (...) {
                                        tool_use["input"] = json::object();
                                    }
                                } else {
                                    tool_use["input"] = json::object();
                                }
                                content_array.push_back(tool_use);
                            }
                        }
                    }

                    jmsg["content"] = content_array;
                } catch (const std::exception& e) {
                    dout(1) << "Failed to parse tool_calls_json: " << e.what() << std::endl;
                    jmsg["content"] = msg.content;
                }
            } else {
                // Check if content is JSON array (from native tool_use blocks)
                try {
                    json content_json = json::parse(msg.content);
                    if (content_json.is_array()) {
                        jmsg["content"] = content_json;
                    } else {
                        jmsg["content"] = msg.content;
                    }
                } catch (...) {
                    jmsg["content"] = msg.content;
                }
            }
        } else if (msg.role == Message::TOOL_RESPONSE) {
            // Tool results are user messages with tool_result content blocks
            // Anthropic requires all tool_results to be in a single user message
            json tool_result_block = {
                {"type", "tool_result"},
                {"tool_use_id", msg.tool_call_id},
                {"content", msg.content}
            };

            // Check if last message is already a user message with tool_result content
            // If so, append to it instead of creating a new message
            if (!messages.empty() && messages.back()["role"] == "user" &&
                messages.back()["content"].is_array() &&
                !messages.back()["content"].empty() &&
                messages.back()["content"][0].contains("type") &&
                messages.back()["content"][0]["type"] == "tool_result") {
                // Append to existing tool_result message
                messages.back()["content"].push_back(tool_result_block);
                continue;  // Skip adding jmsg below
            }

            jmsg["role"] = "user";
            jmsg["content"] = json::array({tool_result_block});
        }

        if (!jmsg.empty()) {
            messages.push_back(jmsg);
        }
    }

    // Add the new message (the one being added via add_message)
    json new_msg;
    if (role == Message::USER) {
        new_msg["role"] = "user";
        new_msg["content"] = content;
    } else if (role == Message::TOOL_RESPONSE) {
        // Tool result - check if we should merge with previous tool_result message
        json tool_result_block = {
            {"type", "tool_result"},
            {"tool_use_id", tool_id},
            {"content", content}
        };

        if (!messages.empty() && messages.back()["role"] == "user" &&
            messages.back()["content"].is_array() &&
            !messages.back()["content"].empty() &&
            messages.back()["content"][0].contains("type") &&
            messages.back()["content"][0]["type"] == "tool_result") {
            // Append to existing tool_result message
            messages.back()["content"].push_back(tool_result_block);
        } else {
            new_msg["role"] = "user";
            new_msg["content"] = json::array({tool_result_block});
        }
    }

    if (!new_msg.empty()) {
        messages.push_back(new_msg);
    }

    request["messages"] = messages;

    // Add tools if present
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json jtool;
            jtool["name"] = tool.name;
            jtool["description"] = tool.description;

            if (!tool.parameters.empty()) {
                jtool["input_schema"] = tool.parameters;
            } else {
                jtool["input_schema"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add sampling parameters (Anthropic supports: temperature, top_p, top_k)
    // Note: Anthropic doesn't allow both temperature and top_p to be set
    // Use temperature unless it's at default (0.7) and top_p is not default (1.0)
    bool temp_is_default = (temperature >= 0.69f && temperature <= 0.71f);
    bool top_p_is_default = (top_p >= 0.99f);

    if (temp_is_default && !top_p_is_default) {
        request["top_p"] = top_p;
    } else {
        request["temperature"] = temperature;
    }
    if (top_k > 0) {
        request["top_k"] = top_k;
    }

    return request;
}

std::string AnthropicBackend::parse_response(const nlohmann::json& response) {
    // Extract content from Anthropic response
    if (response.contains("content") && response["content"].is_array() && !response["content"].empty()) {
        std::string result;
        for (const auto& content_block : response["content"]) {
            if (content_block.contains("type") && content_block["type"] == "text" &&
                content_block.contains("text")) {
                result += content_block["text"].get<std::string>();
            }
        }
        return result;
    }
    return "";
}

int AnthropicBackend::extract_tokens_to_evict(const HttpResponse& response) {
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

    // Anthropic format: "input length and max_tokens exceed context limit: 187254 + 20000 > 204798"
    size_t colon_pos = error_message.find("context limit: ");
    if (colon_pos != std::string::npos) {
        try {
            // Parse "187254 + 20000 > 204798"
            size_t start = colon_pos + 15;
            std::string numbers = error_message.substr(start);

            // Extract input tokens
            size_t plus_pos = numbers.find(" + ");
            int input_tokens = std::stoi(numbers.substr(0, plus_pos));

            // Extract max_tokens
            size_t gt_pos = numbers.find(" > ");
            size_t max_start = plus_pos + 3;
            int max_output = std::stoi(numbers.substr(max_start, gt_pos - max_start));

            // Extract limit
            int limit = std::stoi(numbers.substr(gt_pos + 3));

            // Total = input + max_output, need to evict: total - limit
            int to_evict = (input_tokens + max_output) - limit;
            return to_evict > 0 ? to_evict : -1;
        } catch (...) {}
    }

    return -1;
}

std::map<std::string, std::string> AnthropicBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    headers["x-api-key"] = api_key;
    headers["anthropic-version"] = api_version;

    // Add model-specific special headers if any
    for (const auto& [key, value] : model_config.special_headers) {
        headers[key] = value;
        dout(1) << "Adding special header: " + key + " = " + value << std::endl;
    }

    return headers;
}

std::string AnthropicBackend::get_api_endpoint() {
    return api_endpoint;
}

void AnthropicBackend::set_model(const std::string& model) {
    model_name = model;
    model_config = Models::detect_from_api_model("anthropic", model_name);
    max_output_tokens = model_config.max_output_tokens;
    dout(1) << "Model switched to: " + model_name +
               ", max_output_tokens=" + std::to_string(max_output_tokens) << std::endl;
}

void AnthropicBackend::add_message(Session& session,
                                       Message::Role role,
                                       const std::string& content,
                                       const std::string& tool_name,
                                       const std::string& tool_id,
                                       
                                       int max_tokens) {
    // If streaming disabled, use base class non-streaming implementation
    if (!config->streaming) {
        ApiBackend::add_message(session, role, content, tool_name, tool_id, max_tokens); return;
    }

    reset_output_state();

    dout(1) << "AnthropicBackend::add_message (streaming): max_tokens=" + std::to_string(max_tokens) << std::endl;

    const int MAX_RETRIES = 3;
    int retry = 0;

    while (retry < MAX_RETRIES) {
        // Build request with entire session + new message + max_tokens
        nlohmann::json request = build_request(session, role, content, tool_name, tool_id, max_tokens);

        // Add streaming flag
        request["stream"] = true;

        dout(1) << "Sending streaming request to Anthropic API with max_tokens=" + std::to_string(max_tokens) << std::endl;

        // Get headers and endpoint
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

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
        std::string raw_response_data;  // Accumulate raw response for error parsing
        SSEParser sse_parser;
        bool stream_complete = false;
        bool message_started = false;

        // Tool use streaming state
        struct ToolUseBlock {
            std::string id;
            std::string name;
            std::string partial_json;
        };
        std::optional<ToolUseBlock> current_tool_block;
        nlohmann::json content_blocks = nlohmann::json::array();  // Accumulate all content blocks

        // Streaming callback to process SSE chunks
        auto stream_handler = [&](const std::string& chunk, void* user_data) -> bool {
            // Accumulate raw response for error parsing (limit to 4KB)
            if (raw_response_data.length() < 4096) {
                raw_response_data += chunk;
            }

            // Process SSE events from chunk
            // We ignore the return value - always return true to curl to avoid "Failed writing" error
            // We track completion via stream_complete flag instead
            sse_parser.process_chunk(chunk,
                [&](const std::string& event_type, const std::string& data, const std::string& id) -> bool {
                    // Parse JSON data
                    try {
                        json event_json = json::parse(data);

                        // Handle different event types
                        if (event_type == "message_start") {
                            message_started = true;
                            // Extract usage from message object
                            if (event_json.contains("message")) {
                                const auto& message = event_json["message"];
                                if (message.contains("usage")) {
                                    accumulated_resp.prompt_tokens = message["usage"].value("input_tokens", 0);
                                }
                            }
                        }
                        else if (event_type == "content_block_start") {
                            // Content block starting - could be text or tool_use
                            if (event_json.contains("content_block")) {
                                const auto& block = event_json["content_block"];
                                std::string block_type = block.value("type", "");

                                if (block_type == "tool_use") {
                                    // Tool use block starting - capture name and id
                                    current_tool_block = ToolUseBlock{
                                        block.value("id", ""),
                                        block.value("name", ""),
                                        ""
                                    };
                                    dout(1) << "Tool use block starting: " + current_tool_block->name << std::endl;
                                }
                            }
                        }
                        else if (event_type == "content_block_delta") {
                            // Content delta - text chunk
                            if (event_json.contains("delta")) {
                                const auto& delta = event_json["delta"];

                                if (delta.contains("text")) {
                                    std::string delta_text = delta["text"].get<std::string>();
                                    accumulated_content += delta_text;
                                    accumulated_resp.content = accumulated_content;

                                    // Route through output() for filtering (backticks, buffering)
                                    if (!output(delta_text)) {
                                        return false;
                                    }
                                }
                                else if (delta.contains("partial_json")) {
                                    // Tool use parameters being streamed
                                    if (current_tool_block) {
                                        current_tool_block->partial_json += delta["partial_json"].get<std::string>();
                                    }
                                }
                            }
                        }
                        else if (event_type == "content_block_stop") {
                            // Content block finished
                            if (current_tool_block) {
                                // Parse accumulated JSON and create tool call
                                try {
                                    nlohmann::json input = current_tool_block->partial_json.empty() ?
                                        nlohmann::json::object() :
                                        nlohmann::json::parse(current_tool_block->partial_json);

                                    // Add to content blocks for response.content
                                    content_blocks.push_back({
                                        {"type", "tool_use"},
                                        {"id", current_tool_block->id},
                                        {"name", current_tool_block->name},
                                        {"input", input}
                                    });

                                    // Create tool call
                                    ToolParser::ToolCall tool_call;
                                    tool_call.name = current_tool_block->name;
                                    tool_call.tool_call_id = current_tool_block->id;
                                    tool_call.raw_json = input.dump();

                                    // Parse parameters like non-streaming code does
                                    for (auto it = input.begin(); it != input.end(); ++it) {
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

                                    accumulated_resp.tool_calls.push_back(tool_call);

                                    dout(1) << "Tool use block complete: " + current_tool_block->name << std::endl;
                                } catch (const std::exception& e) {
                                    dout(1) << std::string("WARNING: ") +"Failed to parse tool use JSON: " + std::string(e.what()) << std::endl;
                                }
                                current_tool_block.reset();
                            } else if (!accumulated_content.empty()) {
                                // Text block finished - add to content blocks
                                content_blocks.push_back({
                                    {"type", "text"},
                                    {"text", accumulated_content}
                                });
                            }
                        }
                        else if (event_type == "message_delta") {
                            // Message metadata updates
                            if (event_json.contains("delta")) {
                                const auto& delta = event_json["delta"];
                                if (delta.contains("stop_reason")) {
                                    accumulated_resp.finish_reason = delta["stop_reason"].get<std::string>();
                                }
                            }
                            // Usage updates
                            if (event_json.contains("usage")) {
                                accumulated_resp.completion_tokens = event_json["usage"].value("output_tokens", 0);
                            }
                        }
                        else if (event_type == "message_stop") {
                            // Message complete
                            stream_complete = true;
                            // If we have tool calls, set content as JSON array
                            if (!accumulated_resp.tool_calls.empty()) {
                                accumulated_resp.content = content_blocks.dump();
                            }
                            return false; // Stop processing
                        }
                        else if (event_type == "error") {
                            // Error event
                            accumulated_resp.success = false;
                            accumulated_resp.code = Response::ERROR;
                            accumulated_resp.finish_reason = "error";
                            if (event_json.contains("error")) {
                                const auto& error = event_json["error"];
                                accumulated_resp.error = error.value("message", "Unknown error");
                            }
                            return false; // Stop processing
                        }

                    } catch (const std::exception& e) {
                        dout(1) << std::string("WARNING: ") +"Failed to parse Anthropic SSE data: " + std::string(e.what()) << std::endl;
                    }

                    return true; // Continue processing
                });

            // Always return true to curl so it doesn't report errors
            // We handle stream completion via stream_complete flag
            return true;
        };

        // Make streaming HTTP call (cancellable via escape key)
        HttpResponse http_response = http_client->post_stream_cancellable(endpoint, request.dump(), headers,
                                                                           stream_handler, nullptr);

        // Check for HTTP errors
        if (!http_response.is_success()) {
            accumulated_resp.success = false;
            accumulated_resp.code = Response::ERROR;
            accumulated_resp.finish_reason = "error";

            // Try to extract error message from raw response data (API errors come as JSON)
            std::string error_msg = http_response.error_message;
            if (!raw_response_data.empty()) {
                try {
                    auto error_json = json::parse(raw_response_data);
                    if (error_json.contains("error") && error_json["error"].contains("message")) {
                        error_msg = error_json["error"]["message"].get<std::string>();
                    }
                } catch (...) {
                    // Not JSON, use raw data if we don't have an error message
                    if (error_msg.empty()) {
                        error_msg = raw_response_data.substr(0, 200);
                    }
                }
            }
            accumulated_resp.error = error_msg;
            callback(CallbackEvent::ERROR, error_msg, "api_error", "");

            // Check if context length error
            int tokens_to_evict = extract_tokens_to_evict(http_response);

            if (tokens_to_evict > 0) {
                // Calculate which messages to evict
                auto ranges = session.calculate_messages_to_evict(tokens_to_evict);

                if (ranges.empty()) {
                    accumulated_resp.code = Response::CONTEXT_FULL;
                    accumulated_resp.error = "Context full, cannot evict enough messages";
                    if (role == Message::TOOL_RESPONSE) {
                        add_tool_response(session, content, tool_name, tool_id);
                    }
                    callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return;
                }

                // Evict messages
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

                continue; // Retry
            }

            // Non-context error - add TOOL_RESPONSE for session consistency
            if (role == Message::TOOL_RESPONSE) {
                add_tool_response(session, content, tool_name, tool_id);
            }
            callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", ""); return; // Return error
        }

        flush_output();

        // Success - update session with messages
        if (stream_complete && accumulated_resp.success) {
            // Calculate token counts
            int new_message_tokens;
            if (accumulated_resp.prompt_tokens > 0 && session.last_prompt_tokens > 0) {
                new_message_tokens = accumulated_resp.prompt_tokens - session.last_prompt_tokens;
                if (new_message_tokens < 0) new_message_tokens = 1;
            } else {
                new_message_tokens = estimate_message_tokens(content);
            }

            // Update EMA ratio
            if (new_message_tokens > 0 && content.length() > 0) {
                float actual_ratio = (float)content.length() / new_message_tokens;
                float deviation_ratio = actual_ratio / chars_per_token;

                if (deviation_ratio >= 0.5f && deviation_ratio <= 2.0f) {
                    const float alpha = 0.2f;
                    chars_per_token = (1.0f - alpha) * chars_per_token + alpha * actual_ratio;
                    if (chars_per_token < 2.0f) chars_per_token = 2.0f;
                    if (chars_per_token > 5.0f) chars_per_token = 5.0f;
                }
            }

            // Add messages to session
            Message user_msg(role, content, new_message_tokens);
            user_msg.tool_name = tool_name;
            user_msg.tool_call_id = tool_id;
            session.messages.push_back(user_msg);

            if (role == Message::USER) {
                session.last_user_message_index = session.messages.size() - 1;
                session.last_user_message_tokens = new_message_tokens;
            }

            // Update baseline
            if (accumulated_resp.prompt_tokens > 0) {
                session.last_prompt_tokens = accumulated_resp.prompt_tokens;
            }

            // Add assistant response
            int assistant_tokens = (accumulated_resp.completion_tokens > 0) ?
                                 accumulated_resp.completion_tokens :
                                 estimate_message_tokens(accumulated_resp.content);
            Message assistant_msg(Message::ASSISTANT, accumulated_resp.content, assistant_tokens);
            assistant_msg.tool_calls_json = accumulated_resp.tool_calls_json;
            session.messages.push_back(assistant_msg);

            session.last_assistant_message_index = session.messages.size() - 1;
            session.last_assistant_message_tokens = assistant_tokens;

            if (accumulated_resp.prompt_tokens > 0 && assistant_tokens > 0) {
                session.last_prompt_tokens = accumulated_resp.prompt_tokens + assistant_tokens;
            }

            // Update total tokens
            if (accumulated_resp.prompt_tokens > 0) {
                // Server provided token count - use it
                session.total_tokens = accumulated_resp.prompt_tokens + assistant_tokens;
            } else {
                // Server didn't provide tokens - calculate from session messages
                int total = 0;
                for (const auto& msg : session.messages) {
                    total += msg.tokens;
                }
                session.total_tokens = total;
            }
        }

        callback(CallbackEvent::STOP, accumulated_resp.finish_reason, "", "");

        // Send tool calls AFTER STOP - frontend handles immediately
        for (const auto& tc : accumulated_resp.tool_calls) {
            callback(CallbackEvent::TOOL_CALL, tc.raw_json, tc.name, tc.tool_call_id);
        }
        return;
    }

    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.finish_reason = "error";
    err_resp.error = "Max retries exceeded trying to fit context";
    callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
}

size_t AnthropicBackend::query_model_context_size(const std::string& model_name) {
    // Anthropic doesn't have a /v1/models endpoint to query
    // Return 0 to let models database handle known models
    return 0;
}

std::vector<std::string> AnthropicBackend::fetch_models() {
    std::vector<std::string> result;

    if (!http_client || api_key.empty()) {
        return result;
    }

    // Build request to /v1/models (base URL, not chat endpoint)
    std::map<std::string, std::string> headers;
    headers["x-api-key"] = api_key;
    headers["anthropic-version"] = "2023-06-01";
    headers["Content-Type"] = "application/json";

    HttpResponse response = http_client->get("https://api.anthropic.com/v1/models", headers);

    if (response.is_success()) {
        try {
            auto j = json::parse(response.body);
            if (j.contains("data") && j["data"].is_array()) {
                for (const auto& model : j["data"]) {
                    if (model.contains("id") && model["id"].is_string()) {
                        result.push_back(model["id"].get<std::string>());
                    }
                }
            }
        } catch (const json::exception& e) {
            dout(1) << "Failed to parse Anthropic /models response: " + std::string(e.what()) << std::endl;
        }
    }

    return result;
}

void AnthropicBackend::generate_from_session(Session& session, int max_tokens) {
    // If streaming disabled, use base class non-streaming implementation
    if (!config->streaming) {
        ApiBackend::generate_from_session(session, max_tokens);
        return;
    }

    reset_output_state();

    dout(1) << "AnthropicBackend::generate_from_session (streaming): max_tokens=" + std::to_string(max_tokens) << std::endl;

    // Build request from session
    nlohmann::json request = build_request_from_session(session, max_tokens);

    // Add streaming flag
    request["stream"] = true;

    dout(1) << "Sending streaming request to Anthropic API (generate_from_session)" << std::endl;
    dout(1) << "Request messages: " << request["messages"].dump() << std::endl;

    // Get headers and endpoint
    auto headers = get_api_headers();
    std::string endpoint = get_api_endpoint();

    // Streaming state
    std::string accumulated_content;
    SSEParser sse_parser;
    bool stream_complete = false;
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int completion_tokens = 0;

    // Tool use streaming state
    struct ToolUseBlock {
        std::string id;
        std::string name;
        std::string partial_json;
    };
    std::optional<ToolUseBlock> current_tool_block;
    clear_tool_calls();  // Clear any previous tool calls

    // Streaming callback to process SSE chunks
    auto stream_handler = [&](const std::string& chunk, void* user_data) -> bool {
        sse_parser.process_chunk(chunk,
            [&](const std::string& event_type, const std::string& data, const std::string& id) -> bool {
                try {
                    json event_json = json::parse(data);

                    if (event_type == "message_start") {
                        if (event_json.contains("message")) {
                            const auto& message = event_json["message"];
                            if (message.contains("usage")) {
                                prompt_tokens = message["usage"].value("input_tokens", 0);
                            }
                        }
                    }
                    else if (event_type == "content_block_start") {
                        if (event_json.contains("content_block")) {
                            const auto& block = event_json["content_block"];
                            std::string block_type = block.value("type", "");

                            if (block_type == "tool_use") {
                                current_tool_block = ToolUseBlock{
                                    block.value("id", ""),
                                    block.value("name", ""),
                                    ""
                                };
                                dout(1) << "Tool use block starting: " + current_tool_block->name << std::endl;
                            }
                        }
                    }
                    else if (event_type == "content_block_delta") {
                        if (event_json.contains("delta")) {
                            const auto& delta = event_json["delta"];

                            if (delta.contains("text")) {
                                std::string delta_text = delta["text"].get<std::string>();
                                accumulated_content += delta_text;

                                // Route through output() for filtering (backticks, buffering)
                                if (!output(delta_text)) {
                                    return false;
                                }
                            }
                            else if (delta.contains("partial_json")) {
                                if (current_tool_block) {
                                    current_tool_block->partial_json += delta["partial_json"].get<std::string>();
                                }
                            }
                        }
                    }
                    else if (event_type == "content_block_stop") {
                        if (current_tool_block) {
                            try {
                                nlohmann::json input = current_tool_block->partial_json.empty() ?
                                    nlohmann::json::object() :
                                    nlohmann::json::parse(current_tool_block->partial_json);

                                std::string tool_name = current_tool_block->name;
                                std::string tool_id = current_tool_block->id;
                                std::string params_json = input.dump();

                                // Record for emission after STOP (don't emit here)
                                record_tool_call(tool_name, params_json, tool_id);

                                dout(1) << "Tool use block complete: " + tool_name << std::endl;
                            } catch (const std::exception& e) {
                                dout(1) << std::string("WARNING: ") + "Failed to parse tool use JSON: " + std::string(e.what()) << std::endl;
                            }
                            current_tool_block.reset();
                        }
                    }
                    else if (event_type == "message_delta") {
                        if (event_json.contains("delta")) {
                            const auto& delta = event_json["delta"];
                            if (delta.contains("stop_reason")) {
                                finish_reason = delta["stop_reason"].get<std::string>();
                            }
                        }
                        if (event_json.contains("usage")) {
                            completion_tokens = event_json["usage"].value("output_tokens", 0);
                        }
                    }
                    else if (event_type == "message_stop") {
                        stream_complete = true;
                        return false;
                    }
                    else if (event_type == "error") {
                        if (event_json.contains("error")) {
                            const auto& error = event_json["error"];
                            std::string error_msg = error.value("message", "Unknown error");
                            callback(CallbackEvent::ERROR, error_msg, "error", "");
                        }
                        return false;
                    }

                } catch (const std::exception& e) {
                    dout(1) << std::string("WARNING: ") + "Failed to parse Anthropic SSE data: " + std::string(e.what()) << std::endl;
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
        callback(CallbackEvent::ERROR, error_msg, "error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return;
    }

    flush_output();

    // Update session token counts (session is source of truth)
    if (prompt_tokens > 0 || completion_tokens > 0) {
        session.total_tokens = prompt_tokens + completion_tokens;
        session.last_prompt_tokens = prompt_tokens;
        session.last_assistant_message_tokens = completion_tokens;
    }

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
