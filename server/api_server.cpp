#include "api_server.h"
#include "../logger.h"
#include "../session.h"
#include "../backends/backend.h"
#include "../tools/tool_parser.h"
#include "../http_client.h"
#include "../config.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <atomic>

using json = nlohmann::json;

// External config
extern std::unique_ptr<Config> config;

// APIServer class implementation
APIServer::APIServer(const std::string& host, int port)
    : Server(host, port) {
}

APIServer::~APIServer() {
}

int APIServer::run(Session& session) {
    return run_api_server(backend.get(), host, port);
}

// Global mutex to serialize backend requests (single-threaded processing)
static std::mutex backend_mutex;

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

// Extract tool call from response (same as server.cpp)
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

int run_api_server(Backend* backend, const std::string& host, int port) {
    httplib::Server svr;

    LOG_INFO("Starting API server on " + host + ":" + std::to_string(port));

    // POST /v1/chat/completions - OpenAI-compatible chat completions
    svr.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        LOG_DEBUG("POST /v1/chat/completions - acquiring lock");
        // Lock to serialize requests (single-threaded processing)
        std::lock_guard<std::mutex> lock(backend_mutex);
        LOG_DEBUG("POST /v1/chat/completions - lock acquired");

        try {
            // Parse request body as JSON
            json request;
            try {
                request = json::parse(req.body);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(create_error_response(400, "Invalid JSON").dump(), "application/json");
                return;
            }

            // Create session for this request
            Session session;

            bool stream = request.value("stream", false);

            // Parse tools from request (if provided)
            session.tools.clear();
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
                    session.tools.push_back(st);
                }
                LOG_DEBUG("Parsed " + std::to_string(session.tools.size()) + " tools from request");
            }

            // Parse messages from request into session
            // OpenAI protocol sends FULL conversation history each time, so REPLACE not append
            session.messages.clear();
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

                // Convert role to Message::Type
                Message::Type type;
                if (role == "system") {
                    // System messages handled separately
                    session.system_message = content;
                    continue;
                } else if (role == "user") {
                    type = Message::USER;
                } else if (role == "assistant") {
                    type = Message::ASSISTANT;
                } else if (role == "tool") {
                    type = Message::TOOL;
                } else {
                    type = Message::USER; // fallback
                }

                // Create Message with estimated tokens
                Message m(type, content, content.length() / 4);
                if (msg.contains("tool_call_id")) {
                    m.tool_call_id = msg["tool_call_id"];
                }
                if (msg.contains("name")) {
                    m.tool_name = msg["name"];
                }
                if (msg.contains("tool_calls")) {
                    m.tool_calls_json = msg["tool_calls"].dump();
                }
                session.messages.push_back(m);

                // Track last user/assistant messages for context preservation
                if (type == Message::USER) {
                    session.last_user_message_index = session.messages.size() - 1;
                    session.last_user_message_tokens = m.tokens;
                } else if (type == Message::ASSISTANT) {
                    session.last_assistant_message_index = session.messages.size() - 1;
                    session.last_assistant_message_tokens = m.tokens;
                }
            }

            // Parse parameters
            int max_tokens = 0;
            if (request.contains("max_tokens")) {
                max_tokens = request["max_tokens"];
            }

            // Parse sampling parameters from request (OpenAI-compatible)
            if (request.contains("temperature")) {
                session.temperature = request["temperature"].get<float>();
            }
            if (request.contains("top_p")) {
                session.top_p = request["top_p"].get<float>();
            }
            if (request.contains("top_k")) {
                session.top_k = request["top_k"].get<int>();
            }
            if (request.contains("min_p")) {
                session.min_p = request["min_p"].get<float>();
            }
            if (request.contains("repetition_penalty")) {
                session.repetition_penalty = request["repetition_penalty"].get<float>();
            }
            if (request.contains("presence_penalty")) {
                session.presence_penalty = request["presence_penalty"].get<float>();
            }
            if (request.contains("frequency_penalty")) {
                session.frequency_penalty = request["frequency_penalty"].get<float>();
            }

            LOG_DEBUG("Session sampling params: temp=" + std::to_string(session.temperature) +
                     " top_p=" + std::to_string(session.top_p) +
                     " freq_penalty=" + std::to_string(session.frequency_penalty));

            LOG_DEBUG("Calling generate_from_session with " + std::to_string(session.messages.size()) +
                     " messages and " + std::to_string(session.tools.size()) + " tools (stream=" +
                     (stream ? "true" : "false") + ")");

            // Handle streaming vs non-streaming
            if (stream) {
                // True streaming using set_content_provider
                std::string request_id = generate_id();
                std::string model_name = request.value("model", "shepherd");

                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                // Get thinking markers for streaming filter
                auto thinking_start = backend->get_thinking_start_markers();
                auto thinking_end = backend->get_thinking_end_markers();
                bool filter_thinking = !config->thinking && !thinking_start.empty();

                // Use content provider for true token-by-token streaming
                res.set_content_provider(
                    "text/event-stream",
                    [backend, session, max_tokens, request_id, model_name, thinking_start, thinking_end, filter_thinking](size_t offset, httplib::DataSink& sink) mutable {
                        // State for thinking block filtering during streaming
                        bool in_thinking = false;
                        std::string pending_buffer;

                        auto stream_callback = [&](const std::string& delta,
                                                   const std::string& accumulated,
                                                   const Response& partial_response) -> bool {
                            std::string output_delta = delta;

                            // Filter thinking blocks if needed
                            if (filter_thinking) {
                                pending_buffer += delta;

                                // Check for thinking start markers
                                if (!in_thinking) {
                                    for (const auto& marker : thinking_start) {
                                        size_t pos = pending_buffer.find(marker);
                                        if (pos != std::string::npos) {
                                            // Output content before the marker
                                            output_delta = pending_buffer.substr(0, pos);
                                            pending_buffer = pending_buffer.substr(pos + marker.length());
                                            in_thinking = true;
                                            break;
                                        }
                                    }
                                    // Check if we might be in the middle of a marker
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
                                            // Hold back potential partial marker
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
                                        output_delta = "";  // Suppress content inside thinking block
                                    }
                                }
                            }

                            // Only send chunk if there's content
                            if (!output_delta.empty()) {
                                json delta_chunk = {
                                    {"id", request_id},
                                    {"object", "chat.completion.chunk"},
                                    {"created", std::time(nullptr)},
                                    {"model", model_name},
                                    {"choices", json::array({{
                                        {"index", 0},
                                        {"delta", {{"content", sanitize_utf8(output_delta)}}},
                                        {"finish_reason", nullptr}
                                    }})}
                                };
                                std::string chunk_data = "data: " + delta_chunk.dump() + "\n\n";
                                return sink.write(chunk_data.c_str(), chunk_data.size());
                            }
                            return true;  // Continue even if we didn't output anything
                        };

                        // Generate tokens with streaming callback
                        Response resp = backend->generate_from_session(
                            session, max_tokens, stream_callback
                        );

                        // Send final chunk
                        json final_chunk = {
                            {"id", request_id},
                            {"object", "chat.completion.chunk"},
                            {"created", std::time(nullptr)},
                            {"model", model_name},
                            {"choices", json::array({{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", resp.success ? "stop" : "error"}
                            }})}
                        };
                        std::string final_data = "data: " + final_chunk.dump() + "\n\n";
                        sink.write(final_data.c_str(), final_data.size());

                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.c_str(), done.size());
                        sink.done();
                        return false;  // Done - no more data
                    }
                );
                return;
            }

            // Non-streaming response
            Response resp = backend->generate_from_session(session, max_tokens);

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
            auto tool_call_opt = extract_tool_call(stripped_resp, backend);

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

            // Get usage statistics
            int prompt_tokens = resp.prompt_tokens;
            int completion_tokens = resp.completion_tokens;

            // Fall back to backend tracking if Response doesn't have counts
            if (prompt_tokens == 0 && completion_tokens == 0) {
                prompt_tokens = backend->last_prompt_tokens;
                completion_tokens = backend->last_completion_tokens;
            }

            // Final fallback for local backends
            if (prompt_tokens == 0 && completion_tokens == 0) {
                prompt_tokens = backend->context_token_count;
                completion_tokens = 0;
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
            LOG_ERROR(std::string("Exception in /v1/chat/completions: ") + e.what());
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/models - List available models
    svr.Get("/v1/models", [&](const httplib::Request& req, httplib::Response& res) {
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
            LOG_ERROR(std::string("Exception in /v1/models: ") + e.what());
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/models/{model_name} - Get specific model info
    svr.Get("/v1/models/:model_name", [&](const httplib::Request& req, httplib::Response& res) {
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
            LOG_ERROR(std::string("Exception in /v1/models/:model_name: ") + e.what());
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /health - Health check
    svr.Get("/health", [&](const httplib::Request& req, httplib::Response& res) {
        json response = {
            {"status", "ok"},
            {"backend_connected", backend != nullptr}
        };
        res.set_content(response.dump(), "application/json");
    });

    // Start server
    LOG_INFO("API server ready");
    bool success = svr.listen(host.c_str(), port);

    if (!success) {
        LOG_ERROR("Failed to start API server");
        return 1;
    }

    return 0;
}
