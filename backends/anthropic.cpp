#include "anthropic.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"
#include "sse_parser.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cctype>

using json = nlohmann::json;

// AnthropicBackend implementation
AnthropicBackend::AnthropicBackend(size_t context_size)
    : ApiBackend(context_size) {
    // Initialize with config values
    model_name = config->model;
    api_key = config->key;

    // Don't assume streaming - will test during initialize()

    // Detect model configuration from Models database
    model_config = Models::detect_from_api_model("anthropic", model_name);
    max_output_tokens = model_config.max_output_tokens;
    LOG_DEBUG("Detected model config: context=" + std::to_string(model_config.context_window) +
              ", max_output=" + std::to_string(model_config.max_output_tokens) +
              ", param_name=" + model_config.max_tokens_param_name);

    // Set API endpoint from config if provided
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        LOG_INFO("Using custom Anthropic endpoint: " + api_endpoint);
    }

    // http_client is inherited from ApiBackend and already initialized

    // Parse backend-specific config if available
    parse_backend_config();

    LOG_DEBUG("AnthropicBackend created");
}

AnthropicBackend::~AnthropicBackend() {
}

void AnthropicBackend::initialize(Session& session) {
    LOG_INFO("Initializing Anthropic backend...");

    // Validate API key
    if (api_key.empty()) {
        LOG_ERROR("Anthropic API key is required");
        throw std::runtime_error("Anthropic API key not configured");
    }

    // Auto-detect model if not specified
    if (model_name.empty()) {
        model_name = "claude-sonnet-4-5";
        LOG_INFO("No model specified, using default: " + model_name);
        model_config = Models::detect_from_api_model("anthropic", model_name);
    max_output_tokens = model_config.max_output_tokens;
    }

    // Set API version
    if (api_version.empty()) {
        api_version = "2023-06-01";
        LOG_INFO("Using Anthropic API version: " + api_version);
    }

    // Set context size from model config if not already set
    if (context_size == 0 && model_config.context_window > 0) {
        context_size = model_config.context_window;
        LOG_INFO("Using model's context size: " + std::to_string(context_size));
    }

    // Call base class initialize() which handles calibration
    ApiBackend::initialize(session);

    LOG_INFO("Anthropic backend initialized successfully");
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
        json json_resp = json::parse(http_response.body);

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
                            LOG_WARN("Failed to parse tool call parameters: " + std::string(e.what()));
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

        if (msg.type == Message::USER) {
            jmsg["role"] = "user";
            jmsg["content"] = msg.content;
        } else if (msg.type == Message::ASSISTANT) {
            jmsg["role"] = "assistant";
            // Check if content is JSON array (from tool_use blocks)
            try {
                json content_json = json::parse(msg.content);
                if (content_json.is_array()) {
                    // Use the parsed JSON array directly
                    jmsg["content"] = content_json;
                } else {
                    // Regular text content
                    jmsg["content"] = msg.content;
                }
            } catch (...) {
                // Not JSON, treat as regular text
                jmsg["content"] = msg.content;
            }
        } else if (msg.type == Message::TOOL) {
            // Tool results are user messages with tool_result content blocks
            jmsg["role"] = "user";
            jmsg["content"] = json::array({
                {
                    {"type", "tool_result"},
                    {"tool_use_id", msg.tool_call_id},
                    {"content", msg.content}
                }
            });
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
                                                Message::Type type,
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

        if (msg.type == Message::USER) {
            jmsg["role"] = "user";
            jmsg["content"] = msg.content;
        } else if (msg.type == Message::ASSISTANT) {
            jmsg["role"] = "assistant";
            // Check if content is JSON array (from tool_use blocks)
            try {
                json content_json = json::parse(msg.content);
                if (content_json.is_array()) {
                    // Use the parsed JSON array directly
                    jmsg["content"] = content_json;
                } else {
                    // Regular text content
                    jmsg["content"] = msg.content;
                }
            } catch (...) {
                // Not JSON, treat as regular text
                jmsg["content"] = msg.content;
            }
        } else if (msg.type == Message::TOOL) {
            jmsg["role"] = "user";
            jmsg["content"] = json::array({
                {
                    {"type", "tool_result"},
                    {"tool_use_id", msg.tool_call_id},
                    {"content", msg.content}
                }
            });
        }

        if (!jmsg.empty()) {
            messages.push_back(jmsg);
        }
    }

    // Add the new message
    json new_msg;
    if (type == Message::USER) {
        new_msg["role"] = "user";
        new_msg["content"] = content;
    } else if (type == Message::TOOL) {
        new_msg["role"] = "user";
        new_msg["content"] = json::array({
            {
                {"type", "tool_result"},
                {"tool_use_id", tool_id},
                {"content", content}
            }
        });
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
        LOG_DEBUG("Adding special header: " + key + " = " + value);
    }

    return headers;
}

std::string AnthropicBackend::get_api_endpoint() {
    return api_endpoint;
}
Response AnthropicBackend::add_message_stream(Session& session,
                                             Message::Type type,
                                             const std::string& content,
                                             StreamCallback callback,
                                             const std::string& tool_name,
                                             const std::string& tool_id,
                                             int prompt_tokens,
                                             int max_tokens) {
    LOG_DEBUG("AnthropicBackend::add_message_stream: prompt_tokens=" + std::to_string(prompt_tokens) +
             ", max_tokens=" + std::to_string(max_tokens));

    const int MAX_RETRIES = 3;
    int retry = 0;

    while (retry < MAX_RETRIES) {
        // Build request with entire session + new message + max_tokens
        nlohmann::json request = build_request(session, type, content, tool_name, tool_id, max_tokens);

        // Add streaming flag
        request["stream"] = true;

        LOG_DEBUG("Sending streaming request to Anthropic API with max_tokens=" + std::to_string(max_tokens));

        // Get headers and endpoint
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

        // Streaming state
        Response accumulated_resp;
        accumulated_resp.success = true;
        accumulated_resp.code = Response::SUCCESS;
        std::string accumulated_content;
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
                                    LOG_DEBUG("Tool use block starting: " + current_tool_block->name);
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

                                    // Invoke user callback with delta
                                    if (callback) {
                                        if (!callback(delta_text, accumulated_content, accumulated_resp)) {
                                            // User requested stop
                                            return false;
                                        }
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

                                    LOG_DEBUG("Tool use block complete: " + current_tool_block->name);
                                } catch (const std::exception& e) {
                                    LOG_WARN("Failed to parse tool use JSON: " + std::string(e.what()));
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
                        LOG_WARN("Failed to parse Anthropic SSE data: " + std::string(e.what()));
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
            accumulated_resp.error = http_response.error_message;

            // Check if context length error
            int tokens_to_evict = extract_tokens_to_evict(http_response);

            if (tokens_to_evict > 0) {
                // Calculate which messages to evict
                auto ranges = session.calculate_messages_to_evict(tokens_to_evict);

                if (ranges.empty()) {
                    accumulated_resp.code = Response::CONTEXT_FULL;
                    accumulated_resp.error = "Context full, cannot evict enough messages";
                    return accumulated_resp;
                }

                // Evict messages
                if (!session.evict_messages(ranges)) {
                    accumulated_resp.error = "Failed to evict messages";
                    return accumulated_resp;
                }

                retry++;
                LOG_INFO("Evicted messages, retrying (attempt " + std::to_string(retry + 1) + "/" +
                        std::to_string(MAX_RETRIES) + ")");
                continue; // Retry
            }

            return accumulated_resp; // Return error
        }

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
            Message user_msg(type, content, new_message_tokens);
            user_msg.tool_name = tool_name;
            user_msg.tool_call_id = tool_id;
            session.messages.push_back(user_msg);

            if (type == Message::USER) {
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

            session.total_tokens = accumulated_resp.prompt_tokens + assistant_tokens;
        }

        return accumulated_resp;
    }

    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.finish_reason = "error";
    err_resp.error = "Max retries exceeded trying to fit context";
    return err_resp;
}

size_t AnthropicBackend::query_model_context_size(const std::string& model_name) {
    // Anthropic doesn't have a /v1/models endpoint to query
    // Return 0 to let models database handle known models
    return 0;
}
