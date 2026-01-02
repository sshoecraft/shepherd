#include "shepherd.h"
#include "api_server.h"
#include "../session.h"
#include "backend.h"
#include "../backends/chat_template.h"
#include "../tools/tool_parser.h"
#include "../tools/utf8_sanitizer.h"
#include "../config.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>

using json = nlohmann::json;

// External config
extern std::unique_ptr<Config> config;

// APIServer class implementation
APIServer::APIServer(const std::string& host, int port)
    : Server(host, port, "api") {
    // Set up the event callback - routes to request_handler when set
    callback = [this](CallbackEvent event, const std::string& content,
                      const std::string& name, const std::string& id) -> bool {
        if (request_handler) {
            return request_handler(event, content, name, id);
        }
        return true;
    };
}

APIServer::~APIServer() {
}

// Sanitize string to valid UTF-8 by replacing invalid sequences with replacement char
static std::string sanitize_utf8(const std::string& input) {
    std::string output;
    output.reserve(input.size());

    for (size_t i = 0; i < input.size(); ) {
        unsigned char c = input[i];

        if (c < 0x80) {
            // ASCII
            output += c;
            i++;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < input.size() &&
                   (input[i+1] & 0xC0) == 0x80) {
            // 2-byte sequence
            output += input[i];
            output += input[i+1];
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < input.size() &&
                   (input[i+1] & 0xC0) == 0x80 &&
                   (input[i+2] & 0xC0) == 0x80) {
            // 3-byte sequence
            output += input[i];
            output += input[i+1];
            output += input[i+2];
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < input.size() &&
                   (input[i+1] & 0xC0) == 0x80 &&
                   (input[i+2] & 0xC0) == 0x80 &&
                   (input[i+3] & 0xC0) == 0x80) {
            // 4-byte sequence
            output += input[i];
            output += input[i+1];
            output += input[i+2];
            output += input[i+3];
            i += 4;
        } else {
            // Invalid UTF-8, replace with replacement character
            output += "\xEF\xBF\xBD";
            i++;
        }
    }
    return output;
}

// Strip thinking blocks from content and optionally extract reasoning
// Returns: {content without thinking, reasoning content}
static std::pair<std::string, std::string> strip_thinking_blocks(
    const std::string& content,
    const std::vector<std::string>& start_markers,
    const std::vector<std::string>& end_markers) {

    if (start_markers.empty() || end_markers.empty()) {
        return {content, ""};
    }

    std::string result = content;
    std::string reasoning;

    // Try each start marker
    for (const auto& start_marker : start_markers) {
        size_t start_pos = 0;
        while ((start_pos = result.find(start_marker, start_pos)) != std::string::npos) {
            // Find matching end marker
            size_t end_pos = std::string::npos;
            for (const auto& end_marker : end_markers) {
                size_t pos = result.find(end_marker, start_pos + start_marker.length());
                if (pos != std::string::npos && (end_pos == std::string::npos || pos < end_pos)) {
                    end_pos = pos;
                }
            }

            if (end_pos != std::string::npos) {
                // Extract reasoning content (without tags)
                size_t content_start = start_pos + start_marker.length();
                std::string think_content = result.substr(content_start, end_pos - content_start);

                // Trim whitespace from reasoning
                size_t first = think_content.find_first_not_of(" \t\n\r");
                size_t last = think_content.find_last_not_of(" \t\n\r");
                if (first != std::string::npos && last != std::string::npos) {
                    think_content = think_content.substr(first, last - first + 1);
                }

                if (!reasoning.empty()) {
                    reasoning += "\n\n";
                }
                reasoning += think_content;

                // Find end of end marker
                size_t remove_end = end_pos;
                for (const auto& end_marker : end_markers) {
                    if (result.compare(end_pos, end_marker.length(), end_marker) == 0) {
                        remove_end = end_pos + end_marker.length();
                        break;
                    }
                }

                // Remove the thinking block from result
                result.erase(start_pos, remove_end - start_pos);
                // Don't increment start_pos since we erased content
            } else {
                // No end marker found, skip this occurrence
                start_pos += start_marker.length();
            }
        }
    }

    // Trim leading/trailing whitespace from result
    size_t first = result.find_first_not_of(" \t\n\r");
    size_t last = result.find_last_not_of(" \t\n\r");
    if (first != std::string::npos && last != std::string::npos) {
        result = result.substr(first, last - first + 1);
    } else if (first == std::string::npos) {
        result = "";
    }

    return {result, reasoning};
}

// Generate random ID for chat completion
static std::string generate_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::string id = "chatcmpl-";
    for (int i = 0; i < 8; i++) {
        id += hex[dis(gen)];
    }
    return id;
}

// Extract tool call from response
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    if (resp.content.empty()) {
        return std::nullopt;
    }
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

// Create OpenAI-compatible error response
static json create_error_response(int status_code, const std::string& message) {
    std::string error_type;
    switch (status_code) {
        case 400:
            error_type = "invalid_request_error";
            break;
        case 401:
            error_type = "authentication_error";
            break;
        case 429:
            error_type = "rate_limit_error";
            break;
        case 500:
            error_type = "server_error";
            break;
        default:
            error_type = "api_error";
    }

    return json{
        {"error", {
            {"message", message},
            {"type", error_type},
            {"code", std::to_string(status_code)}
        }}
    };
}

// Check if error message is a context limit error (400 vs 500)
static bool is_context_limit_error(const std::string& error_msg) {
    std::string lower_msg = error_msg;
    std::transform(lower_msg.begin(), lower_msg.end(), lower_msg.begin(), ::tolower);
    return (lower_msg.find("context limit") != std::string::npos ||
            lower_msg.find("context window") != std::string::npos ||
            lower_msg.find("maximum context") != std::string::npos ||
            lower_msg.find("context length") != std::string::npos ||
            lower_msg.find("exceeded") != std::string::npos);
}

void APIServer::register_endpoints() {
    // POST /v1/chat/completions - OpenAI-compatible chat completions
    tcp_server.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        dout(1) << "POST /v1/chat/completions - acquiring lock" << std::endl;
        // Lock to serialize requests (single-threaded processing)
        std::lock_guard<std::mutex> lock(backend_mutex);
        dout(1) << "POST /v1/chat/completions - lock acquired" << std::endl;

        // Increment request counter
        requests_processed++;

        try {
            // Parse request body as JSON
            json request;
            try {
                std::string sanitized_body = utf8_sanitizer::sanitize_utf8(req.body);
                request = json::parse(sanitized_body);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(create_error_response(400, "Invalid JSON").dump(), "application/json");
                return;
            }

            // Create session for this request
            Session request_session;

            bool stream = request.value("stream", false);

            // Parse tools from request (if provided)
            request_session.tools.clear();
            if (request.contains("tools") && request["tools"].is_array()) {
                for (const auto& tool : request["tools"]) {
                    Session::Tool st;

                    // OpenAI format: tool["function"]["name"]
                    if (tool.contains("function") && tool["function"].is_object()) {
                        st.name = tool["function"].value("name", "");
                        st.description = tool["function"].value("description", "");
                        if (tool["function"].contains("parameters")) {
                            st.parameters = tool["function"]["parameters"];
                        } else {
                            st.parameters = json::object();
                        }
                    } else {
                        // Fallback: direct format
                        st.name = tool.value("name", "");
                        st.description = tool.value("description", "");
                        if (tool.contains("parameters")) {
                            st.parameters = tool["parameters"];
                        } else {
                            st.parameters = json::object();
                        }
                    }
                    request_session.tools.push_back(st);
                }
                dout(1) << "Parsed " + std::to_string(request_session.tools.size()) + " tools from request" << std::endl;
            }

            // Parse messages from request into session
            // OpenAI protocol sends FULL conversation history each time, so REPLACE not append
            request_session.messages.clear();
            for (const auto& msg : request["messages"]) {
                std::string role = msg.value("role", "user");

                // Content can be a string or an array (for multi-modal messages)
                std::string content;
                if (msg.contains("content") && !msg["content"].is_null()) {
                    if (msg["content"].is_string()) {
                        content = msg["content"].get<std::string>();
                    } else if (msg["content"].is_array()) {
                        // Extract text from content array
                        for (const auto& part : msg["content"]) {
                            if (part.contains("type") && part["type"] == "text" && part.contains("text")) {
                                if (!content.empty()) content += "\n";
                                content += part["text"].get<std::string>();
                            }
                        }
                    }
                }

                // Convert role string to Message::Role
                Message::Role msg_role;
                if (role == "system") {
                    // System messages handled separately
                    request_session.system_message = content;
                    continue;
                } else if (role == "user") {
                    msg_role = Message::USER;
                } else if (role == "assistant") {
                    msg_role = Message::ASSISTANT;
                } else if (role == "tool") {
                    msg_role = Message::TOOL_RESPONSE;
                } else {
                    msg_role = Message::USER; // fallback
                }

                // Create Message with estimated tokens
                Message m(msg_role, content, content.length() / 4);
                if (msg.contains("tool_call_id")) {
                    m.tool_call_id = msg["tool_call_id"];
                }
                if (msg.contains("name")) {
                    m.tool_name = msg["name"];
                }
                if (msg.contains("tool_calls")) {
                    m.tool_calls_json = msg["tool_calls"].dump();
                    dout(1) << "Parsed tool_calls for assistant message: " << m.tool_calls_json << std::endl;
                }
                dout(2) << "Parsed message: role=" << role << " content_len=" << content.length()
                        << " tool_calls_json_len=" << m.tool_calls_json.length() << std::endl;
                request_session.messages.push_back(m);

                // Track last user/assistant messages for context preservation
                if (msg_role == Message::USER) {
                    request_session.last_user_message_index = request_session.messages.size() - 1;
                    request_session.last_user_message_tokens = m.tokens;
                } else if (msg_role == Message::ASSISTANT) {
                    request_session.last_assistant_message_index = request_session.messages.size() - 1;
                    request_session.last_assistant_message_tokens = m.tokens;
                }
            }

            // Parse parameters
            int max_tokens = 0;
            if (request.contains("max_tokens")) {
                max_tokens = request["max_tokens"];
            }

            // Parse sampling parameters from request (OpenAI-compatible)
            if (request.contains("temperature")) {
                request_session.temperature = request["temperature"].get<float>();
            }
            if (request.contains("top_p")) {
                request_session.top_p = request["top_p"].get<float>();
            }
            if (request.contains("top_k")) {
                request_session.top_k = request["top_k"].get<int>();
            }
            if (request.contains("min_p")) {
                request_session.min_p = request["min_p"].get<float>();
            }
            if (request.contains("repetition_penalty")) {
                request_session.repetition_penalty = request["repetition_penalty"].get<float>();
            }
            if (request.contains("presence_penalty")) {
                request_session.presence_penalty = request["presence_penalty"].get<float>();
            }
            if (request.contains("frequency_penalty")) {
                request_session.frequency_penalty = request["frequency_penalty"].get<float>();
            }

            dout(1) << "Session sampling params: temp=" + std::to_string(request_session.temperature) +
                     " top_p=" + std::to_string(request_session.top_p) +
                     " rep_penalty=" + std::to_string(request_session.repetition_penalty) +
                     " freq_penalty=" + std::to_string(request_session.frequency_penalty) +
                     " pres_penalty=" + std::to_string(request_session.presence_penalty) << std::endl;

            dout(1) << "Calling generate_from_session with " + std::to_string(request_session.messages.size()) +
                     " messages and " + std::to_string(request_session.tools.size()) + " tools (stream=" +
                     (stream ? "true" : "false") + ")" << std::endl;


            // Handle streaming vs non-streaming
            if (stream) {
                // True streaming using set_content_provider
                std::string request_id = generate_id();
                std::string model_name = request.value("model", "shepherd");

                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                // Get thinking markers for streaming filter (for non-channel models)
                auto thinking_start = backend->get_thinking_start_markers();
                auto thinking_end = backend->get_thinking_end_markers();
                bool filter_thinking = !config->thinking && !thinking_start.empty();

                // Channel parsing is now handled by the backend's process_output()
                // No need for a separate channel parser here - content arrives already filtered
                const auto* caps = backend->get_chat_template_caps();
                bool has_channels = caps && caps->has_channels;
                if (has_channels) {
                    dout(1) << "Streaming with backend channel parsing, thinking=" << (config->thinking ? "true" : "false") << std::endl;
                }

                // Capture backend pointer and mutex for lambda
                Backend* backend_ptr = backend.get();
                std::mutex* mutex_ptr = &backend_mutex;

                // Use content provider for true token-by-token streaming
                res.set_content_provider(
                    "text/event-stream",
                    [this, backend_ptr, mutex_ptr, request_session, max_tokens, request_id, model_name, thinking_start, thinking_end, filter_thinking, has_channels](size_t offset, httplib::DataSink& sink) mutable {
                        // Acquire lock for backend access (serialize with other requests)
                        std::lock_guard<std::mutex> stream_lock(*mutex_ptr);

                        // Send initial chunk with role (vLLM/OpenAI compatible)
                        json initial_chunk = {
                            {"id", request_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::time(nullptr)},
                            {"model", model_name},
                            {"choices", json::array({{
                                {"index", 0},
                                {"delta", {{"role", "assistant"}, {"content", ""}}},
                                {"logprobs", nullptr},
                                {"finish_reason", nullptr}
                            }})}
                        };
                        std::string initial_data = "data: " + initial_chunk.dump() + "\n\n";
                        sink.write(initial_data.c_str(), initial_data.size());

                        // State for thinking block filtering during streaming (non-channel models)
                        // Note: For channel-based models (has_channels=true), the backend's
                        // process_output() handles channel parsing before calling this callback
                        bool in_thinking = false;
                        std::string pending_buffer;

                        // Helper to send an SSE content chunk
                        auto send_content_chunk = [&](const std::string& text) -> bool {
                            if (text.empty()) return true;
                            json delta_chunk = {
                                {"id", request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", std::time(nullptr)},
                                {"model", model_name},
                                {"choices", json::array({{
                                    {"index", 0},
                                    {"delta", {{"content", sanitize_utf8(text)}}},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr}
                                }})}
                            };
                            std::string chunk_data = "data: " + delta_chunk.dump() + "\n\n";
                            return sink.write(chunk_data.c_str(), chunk_data.size());
                        };

                        // Set request handler for this streaming request
                        request_handler = [&](CallbackEvent type,
                                              const std::string& content,
                                              const std::string& tool_name_arg,
                                              const std::string& tool_call_id) -> bool {
                            // Handle STOP - signals completion, don't treat finish_reason as content
                            // Note: Channel parser flush is now handled by backend's flush_output()
                            if (type == CallbackEvent::STOP) {
                                return true;  // Final chunk with finish_reason sent after this
                            }
                            // Handle ERROR
                            if (type == CallbackEvent::ERROR) {
                                return true;  // Errors handled via response status
                            }
                            // Handle TOOL_CALL (queue for later)
                            if (type == CallbackEvent::TOOL_CALL) {
                                return true;  // Tool calls returned in response
                            }
                            // Only process content-type events
                            if (type != CallbackEvent::CONTENT && type != CallbackEvent::THINKING && type != CallbackEvent::CODEBLOCK) {
                                return true;
                            }

                            dout(2) << "API callback received: type=" << static_cast<int>(type) << " content=[" << content.substr(0, 50) << "] len=" << content.length() << std::endl;

                            // For channel-based models: backend's process_output() already filtered
                            // the content through ChannelParser, so we receive clean content here
                            if (has_channels) {
                                return send_content_chunk(content);
                            }

                            // For non-channel models: use existing thinking block filter
                            std::string output_delta = content;
                            const std::string& delta = content;

                            if (filter_thinking) {
                                pending_buffer += delta;

                                // Check for thinking start markers
                                if (!in_thinking) {
                                    for (const auto& marker : thinking_start) {
                                        size_t pos = pending_buffer.find(marker);
                                        if (pos != std::string::npos) {
                                            output_delta = pending_buffer.substr(0, pos);
                                            pending_buffer = pending_buffer.substr(pos + marker.length());
                                            in_thinking = true;
                                            break;
                                        }
                                    }
                                    if (!in_thinking) {
                                        bool could_be_partial = false;
                                        for (const auto& marker : thinking_start) {
                                            for (size_t len = 1; len < marker.length() && len <= pending_buffer.length(); ++len) {
                                                if (pending_buffer.compare(pending_buffer.length() - len, len, marker, 0, len) == 0) {
                                                    could_be_partial = true;
                                                    break;
                                                }
                                            }
                                            if (could_be_partial) break;
                                        }
                                        if (could_be_partial) {
                                            output_delta = "";
                                        } else {
                                            output_delta = pending_buffer;
                                            pending_buffer.clear();
                                        }
                                    }
                                }

                                // Check for thinking end markers
                                if (in_thinking) {
                                    for (const auto& marker : thinking_end) {
                                        size_t pos = pending_buffer.find(marker);
                                        if (pos != std::string::npos) {
                                            pending_buffer = pending_buffer.substr(pos + marker.length());
                                            in_thinking = false;
                                            output_delta = "";
                                            break;
                                        }
                                    }
                                    if (in_thinking) {
                                        output_delta = "";
                                    }
                                }
                            }

                            // Send chunk
                            dout(2) << "API stream: output_delta=[" << output_delta.substr(0, 100) << "] len=" << output_delta.length() << std::endl;
                            return send_content_chunk(output_delta);
                        };

                        std::string finish_reason = "stop";
                        backend_ptr->generate_from_session(request_session, max_tokens);
                        request_handler = nullptr;  // Clear after request

                        // Check for pending tool calls (for channel-based models like GPT-OSS)
                        // These are captured by the channel parser during streaming
                        if (!backend_ptr->pending_tool_calls.empty()) {
                            const auto& tc = backend_ptr->pending_tool_calls[0];
                            json tool_chunk = {
                                {"id", request_id},
                                {"object", "chat.completion.chunk"},
                                {"created", std::time(nullptr)},
                                {"model", model_name},
                                {"choices", json::array({{
                                    {"index", 0},
                                    {"delta", {
                                        {"tool_calls", json::array({{
                                            {"index", 0},
                                            {"id", tc.tool_call_id.empty() ? "call_" + std::to_string(std::time(nullptr)) : tc.tool_call_id},
                                            {"type", "function"},
                                            {"function", {
                                                {"name", tc.name},
                                                {"arguments", tc.raw_json}
                                            }}
                                        }})}
                                    }},
                                    {"logprobs", nullptr},
                                    {"finish_reason", nullptr}
                                }})}
                            };
                            std::string tool_data = "data: " + tool_chunk.dump() + "\n\n";
                            sink.write(tool_data.c_str(), tool_data.size());
                            finish_reason = "tool_calls";
                            dout(1) << "Streaming: sent pending_tool_call: " << tc.name << std::endl;
                        }

                        // Send final chunk with finish_reason (vLLM compatible)
                        json final_chunk = {
                            {"id", request_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::time(nullptr)},
                            {"model", model_name},
                            {"choices", json::array({{
                                {"index", 0},
                                {"delta", {{"content", ""}}},
                                {"logprobs", nullptr},
                                {"finish_reason", finish_reason}
                            }})}
                        };
                        std::string final_data = "data: " + final_chunk.dump() + "\n\n";
                        sink.write(final_data.c_str(), final_data.size());

                        // Send usage chunk (OpenAI stream_options.include_usage=true compatible)
                        // Token counts are updated by backend in generate_from_session()
                        int prompt_tokens = request_session.last_prompt_tokens;
                        int completion_tokens = request_session.last_assistant_message_tokens;
                        dout(1) << "API server streaming usage: prompt_tokens=" << prompt_tokens
                                << ", completion_tokens=" << completion_tokens
                                << ", total_tokens=" << request_session.total_tokens << std::endl;
                        json usage_chunk = {
                            {"id", request_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::time(nullptr)},
                            {"model", model_name},
                            {"choices", json::array()},
                            {"usage", {
                                {"prompt_tokens", prompt_tokens},
                                {"completion_tokens", completion_tokens},
                                {"total_tokens", prompt_tokens + completion_tokens}
                            }}
                        };
                        std::string usage_data = "data: " + usage_chunk.dump() + "\n\n";
                        sink.write(usage_data.c_str(), usage_data.size());

                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.c_str(), done.size());
                        sink.done();
                        return false;  // Done - no more data
                    }
                );
                return;
            }

            // Non-streaming response
            // Set up request handler to accumulate response content
            std::string accumulated_content;
            std::string error_message;
            std::string finish_reason = "stop";

            request_handler = [&accumulated_content, &error_message, &finish_reason](
                CallbackEvent event, const std::string& data,
                const std::string& type, const std::string& id) -> bool {
                switch (event) {
                    case CallbackEvent::CONTENT:
                        accumulated_content += data;
                        break;
                    case CallbackEvent::ERROR:
                        error_message = data;
                        break;
                    case CallbackEvent::STOP:
                        if (!data.empty()) finish_reason = data;
                        break;
                    default:
                        break;
                }
                return true;
            };

            backend->generate_from_session(request_session, max_tokens);
            request_handler = nullptr;  // Clear after request

            // Build Response from accumulated content
            Response resp;
            bool has_content = !accumulated_content.empty();
            bool has_tool_calls = !backend->pending_tool_calls.empty();
            if (error_message.empty() && (has_content || has_tool_calls)) {
                resp.success = true;
                resp.content = accumulated_content;
                resp.finish_reason = has_tool_calls ? "tool_calls" : finish_reason;
                // Get token counts from session (updated by backend during generate_from_session)
                resp.prompt_tokens = request_session.last_prompt_tokens;
                resp.completion_tokens = request_session.last_assistant_message_tokens;
            } else if (!error_message.empty()) {
                resp.success = false;
                resp.error = error_message;
            } else {
                resp.success = false;
                resp.error = "No assistant response generated";
            }

            // Check for errors
            if (!resp.success) {
                // Check if this is a context limit error (400 vs 500)
                if (is_context_limit_error(resp.error)) {
                    res.status = 400;
                    json error_resp = {
                        {"error", {
                            {"message", resp.error},
                            {"type", "invalid_request_error"},
                            {"code", "context_length_exceeded"}
                        }}
                    };
                    res.set_content(error_resp.dump(), "application/json");
                } else {
                    res.status = 500;
                    res.set_content(create_error_response(500, resp.error).dump(), "application/json");
                }
                return;
            }

            // Strip thinking blocks if thinking mode is disabled
            std::string response_content = resp.content;
            std::string reasoning_content;
            if (!config->thinking) {
                auto thinking_start = backend->get_thinking_start_markers();
                auto thinking_end = backend->get_thinking_end_markers();
                auto [content, reasoning] = strip_thinking_blocks(response_content, thinking_start, thinking_end);
                response_content = content;
                reasoning_content = reasoning;
            }

            // Check for tool calls (use stripped content)
            Response stripped_resp = resp;
            stripped_resp.content = response_content;
            auto tool_call_opt = extract_tool_call(stripped_resp, backend.get());

            // Also check backend's pending_tool_calls (for channel-based models like GPT-OSS)
            // These are captured by the channel parser during streaming
            if (!tool_call_opt.has_value() && !backend->pending_tool_calls.empty()) {
                const auto& tc = backend->pending_tool_calls[0];
                ToolParser::ToolCall parsed_tc;
                parsed_tc.name = tc.name;
                parsed_tc.tool_call_id = tc.tool_call_id;
                parsed_tc.raw_json = tc.raw_json;
                // Parse raw_json into parameters
                try {
                    auto args_json = nlohmann::json::parse(tc.raw_json);
                    for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                        if (it.value().is_string()) {
                            parsed_tc.parameters[it.key()] = it.value().get<std::string>();
                        } else if (it.value().is_number_integer()) {
                            parsed_tc.parameters[it.key()] = it.value().get<int>();
                        } else if (it.value().is_number_float()) {
                            parsed_tc.parameters[it.key()] = it.value().get<double>();
                        } else if (it.value().is_boolean()) {
                            parsed_tc.parameters[it.key()] = it.value().get<bool>();
                        } else {
                            parsed_tc.parameters[it.key()] = it.value().dump();
                        }
                    }
                } catch (...) {
                    // If parsing fails, leave parameters empty
                }
                tool_call_opt = parsed_tc;
                dout(1) << "Using pending_tool_call from channel parser: " << tc.name << std::endl;
            }

            // Build OpenAI-compatible response
            json choice = {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", ""}
                }},
                {"finish_reason", "stop"}
            };

            // Add reasoning_content if we have it (like llama-server does)
            if (!reasoning_content.empty()) {
                choice["message"]["reasoning_content"] = sanitize_utf8(reasoning_content);
            }

            if (tool_call_opt.has_value()) {
                auto tool_call = tool_call_opt.value();
                choice["message"]["content"] = sanitize_utf8(tool_call.content);
                choice["finish_reason"] = "tool_calls";

                json tc;
                tc["id"] = tool_call.tool_call_id.empty() ?
                          "call_" + std::to_string(std::time(nullptr)) :
                          tool_call.tool_call_id;
                tc["type"] = "function";

                // Build function object with name and arguments
                json function_obj;
                function_obj["name"] = tool_call.name;

                // Convert parameters to JSON object
                json parameters = json::object();
                for (const auto& [key, value] : tool_call.parameters) {
                    if (value.type() == typeid(std::string)) {
                        parameters[key] = std::any_cast<std::string>(value);
                    } else if (value.type() == typeid(int)) {
                        parameters[key] = std::any_cast<int>(value);
                    } else if (value.type() == typeid(double)) {
                        parameters[key] = std::any_cast<double>(value);
                    } else if (value.type() == typeid(bool)) {
                        parameters[key] = std::any_cast<bool>(value);
                    }
                }

                // OpenAI expects arguments as a JSON string, not an object
                function_obj["arguments"] = parameters.dump();

                tc["function"] = function_obj;
                choice["message"]["tool_calls"] = json::array({tc});
            } else {
                // No tool call - regular response
                choice["message"]["content"] = sanitize_utf8(response_content);
                choice["finish_reason"] = resp.finish_reason.empty() ? "stop" : resp.finish_reason;
            }

            // Get usage statistics from session (updated by backend during generate_from_session)
            int prompt_tokens = request_session.last_prompt_tokens;
            int completion_tokens = request_session.last_assistant_message_tokens;

            dout(1) << "API server token response: prompt_tokens=" << prompt_tokens
                    << ", completion_tokens=" << completion_tokens
                    << ", total_tokens=" << request_session.total_tokens << std::endl;

            // If backend didn't provide token counts, estimate them
            if (prompt_tokens == 0 && completion_tokens == 0 && !accumulated_content.empty()) {
                completion_tokens = backend->count_message_tokens(Message::ASSISTANT, accumulated_content, "", "");
                dout(1) << "API server estimated completion_tokens=" << completion_tokens << std::endl;
            }

            json response_body = {
                {"id", generate_id()},
                {"object", "chat.completion"},
                {"created", std::time(nullptr)},
                {"model", request.value("model", "shepherd")},
                {"choices", json::array({choice})},
                {"usage", {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens}
                }}
            };

            res.set_content(response_body.dump(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/chat/completions: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/models - List available models
    tcp_server.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            json model_info = {
                {"id", backend->model_name},
                {"object", "model"},
                {"created", std::time(nullptr)},
                {"owned_by", "shepherd"},
                {"max_model_len", backend->context_size}
            };

            json response = {
                {"object", "list"},
                {"data", json::array({model_info})}
            };

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/models: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/models/{model_name} - Get specific model info
    tcp_server.Get("/v1/models/:model_name", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            json response = {
                {"id", backend->model_name},
                {"object", "model"},
                {"created", std::time(nullptr)},
                {"owned_by", "shepherd"},
                {"context_window", backend->context_size},
                {"backend", backend->backend_name},
                {"model_name", backend->model_name}
            };

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/models/:model_name: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });
}
