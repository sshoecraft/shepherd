#include "shepherd.h"
#include "cli_server.h"
#include "../shepherd.h"
#include "../tools/tool.h"
#include "../tools/tool_parser.h"
#include "../tools/utf8_sanitizer.h"
#include "../mcp/mcp.h"
#include "../rag.h"
#include "../config.h"
#include <algorithm>
#include <thread>
#include <chrono>

using json = nlohmann::json;

extern std::unique_ptr<Config> config;

// CliServerState methods
void CliServerState::add_input(const std::string& prompt) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        input_queue.push_back(prompt);
    }
    queue_cv.notify_one();
}

size_t CliServerState::queue_size() {
    std::lock_guard<std::mutex> lock(queue_mutex);
    return input_queue.size();
}

std::string CliServerState::get_next_input() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_cv.wait(lock, [this] { return !input_queue.empty() || !running; });
    if (!running && input_queue.empty()) {
        return "";
    }
    std::string prompt = input_queue.front();
    input_queue.pop_front();
    return prompt;
}

void CliServerState::broadcast_event(const std::string& event_type, const nlohmann::json& data) {
    std::lock_guard<std::mutex> lock(sse_mutex);

    nlohmann::json event;
    event["type"] = event_type;
    event["data"] = data;
    std::string sse_data = "data: " + event.dump() + "\n\n";

    dprintf(3, "Broadcasting SSE event: %s to %zu clients", event_type.c_str(), sse_clients.size());

    // Send to all connected SSE clients
    for (auto it = sse_clients.begin(); it != sse_clients.end(); ) {
        SSEClient* client = *it;
        if (client && client->sink) {
            if (!client->sink->write(sse_data.c_str(), sse_data.size())) {
                // Client disconnected, remove from list
                dprintf(1, "SSE client disconnected during broadcast");
                it = sse_clients.erase(it);
                delete client;
                continue;
            }
        }
        ++it;
    }
}

// CLIServer class implementation
CLIServer::CLIServer(const std::string& host, int port)
    : Server(host, port, "cli") {
}

CLIServer::~CLIServer() {
}

void CLIServer::init(bool no_mcp, bool no_tools) {
    // Use common tool initialization from Frontend base class
    Frontend::init_tools(session, tools, no_mcp, no_tools);
}

// Helper: Extract tool call from Response
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    if (!resp.tool_calls.empty()) {
        return resp.tool_calls[0];
    }
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

// Process a single request with tool execution loop
static json process_request(CliServerState& state, const std::string& prompt) {
    json result;
    result["success"] = true;

    std::string accumulated_response;

    // Add user message and generate response
    // Output flows through backend callback; response retrieved from session messages
    state.session->add_message(Message::USER, prompt);

    // Broadcast user message to all SSE clients
    // Find the user message we just added
    for (auto it = state.session->messages.rbegin(); it != state.session->messages.rend(); ++it) {
        if (it->role == Message::USER) {
            json msg_json;
            msg_json["role"] = it->get_role();
            msg_json["content"] = utf8_sanitizer::sanitize_utf8(it->content);
            msg_json["tokens"] = it->tokens;
            state.broadcast_event("message_added", msg_json);
            break;
        }
    }

    // Find the assistant response
    Response resp;
    resp.success = true;
    if (!state.session->messages.empty() &&
        state.session->messages.back().role == Message::ASSISTANT) {
        resp.content = state.session->messages.back().content;
    }

    accumulated_response = resp.content;

    // Broadcast initial assistant response to all SSE clients
    if (!resp.content.empty() && !state.session->messages.empty()) {
        const auto& assistant_msg = state.session->messages.back();
        if (assistant_msg.get_role() == "assistant") {
            json msg_json;
            msg_json["role"] = assistant_msg.get_role();
            msg_json["content"] = utf8_sanitizer::sanitize_utf8(assistant_msg.content);
            msg_json["tokens"] = assistant_msg.tokens;
            state.broadcast_event("message_added", msg_json);
        }
    }

    // Tool execution loop
    const int max_tool_iterations = 50;
    int iteration = 0;

    while (iteration < max_tool_iterations) {
        iteration++;

        // Check for tool calls
        auto tool_call_opt = extract_tool_call(resp, state.backend);

        if (!tool_call_opt) {
            // No more tool calls - done
            break;
        }

        auto& tool_call = *tool_call_opt;
        std::string tool_name = tool_call.name;

        // Execute tool (handles truncation)
        ToolResult tool_result = state.server->execute_tool(*state.tools, tool_name, tool_call.parameters, tool_call.tool_call_id);

        // Send tool result to model and get next response
        state.session->add_message(Message::TOOL_RESPONSE, tool_result.content, tool_name, tool_call.tool_call_id);

        // Build Response from session's last message
        if (!state.session->messages.empty() &&
            state.session->messages.back().role == Message::ASSISTANT) {
            resp.success = true;
            resp.content = state.session->messages.back().content;
        }

        // Broadcast tool message to all SSE clients
        if (!state.session->messages.empty()) {
            const auto& tool_msg = state.session->messages.back();
            json msg_json;
            msg_json["role"] = tool_msg.get_role();
            msg_json["content"] = utf8_sanitizer::sanitize_utf8(tool_msg.content);
            msg_json["tokens"] = tool_msg.tokens;
            if (!tool_msg.tool_name.empty()) {
                msg_json["tool_name"] = tool_msg.tool_name;
            }
            if (!tool_msg.tool_call_id.empty()) {
                msg_json["tool_call_id"] = tool_msg.tool_call_id;
            }
            state.broadcast_event("message_added", msg_json);
        }

        // Broadcast assistant response to all SSE clients
        if (!resp.content.empty() && !state.session->messages.empty()) {
            const auto& assistant_msg = state.session->messages.back();
            if (assistant_msg.get_role() == "assistant") {
                json msg_json;
                msg_json["role"] = assistant_msg.get_role();
                msg_json["content"] = utf8_sanitizer::sanitize_utf8(assistant_msg.content);
                msg_json["tokens"] = assistant_msg.tokens;
                state.broadcast_event("message_added", msg_json);
            }
        }

        if (!resp.success) {
            result["success"] = false;
            result["error"] = resp.error;
            return result;
        }

        accumulated_response = resp.content;
    }

    result["response"] = accumulated_response;
    return result;
}

void CLIServer::add_status_info(nlohmann::json& status) {
    status["processing"] = state.processing.load();
    status["queue_depth"] = static_cast<int>(state.queue_size());
    status["messages"] = state.session ? state.session->messages.size() : 0;

    if (state.processing.load() && !state.current_request.empty()) {
        // Truncate for display
        std::string truncated = state.current_request;
        if (truncated.length() > 100) {
            truncated = truncated.substr(0, 100) + "...";
        }
        status["current_request"] = truncated;
    }
}

void CLIServer::on_server_start() {
    // Start background processing thread for async requests
    processor_thread = std::thread([this]() {
        dout(1) << "CLI server processor thread started" << std::endl;
        while (state.running) {
            std::string prompt = state.get_next_input();
            if (prompt.empty()) {
                continue;  // Shutdown or spurious wakeup
            }

            state.processing = true;
            state.current_request = prompt;

            dout(1) << "Processing async request: " + prompt.substr(0, 50) + "..." << std::endl;
            json result = process_request(state, prompt);

            // Increment request counter
            requests_processed++;

            // For async requests, we just log the result (no one is waiting)
            if (result.value("success", false)) {
                dout(1) << "Async request completed successfully" << std::endl;
            } else {
                dout(1) << "Async request failed: " + result.value("error", "unknown") << std::endl;
            }

            state.current_request.clear();
            state.processing = false;
        }
        dout(1) << "CLI server processor thread stopped" << std::endl;
    });
}

void CLIServer::on_shutdown() {
    // Signal SSE threads and processor to exit BEFORE tcp_server.stop()
    state.running = false;
    state.queue_cv.notify_all();
}

void CLIServer::on_server_stop() {

    // Signal any in-flight generation to cancel
    g_generation_cancelled = true;

    // Wait for processor thread to finish
    if (processor_thread.joinable()) {
        processor_thread.join();
    }
}

void CLIServer::register_endpoints() {
    // Initialize state with backend, session, tools, and server pointer
    state.server = this;
    state.backend = backend.get();
    state.session = &session;
    state.tools = &tools;

    // POST /request - Main request endpoint
    tcp_server.Post("/request", [this](const httplib::Request& req, httplib::Response& res) {
        // Parse request
        json request_json;
        std::string prompt;
        bool stream = false;
        bool async_mode = false;

        if (!req.body.empty()) {
            try {
                std::string sanitized_body = utf8_sanitizer::sanitize_utf8(req.body);
                request_json = json::parse(sanitized_body);
                if (request_json.contains("prompt")) {
                    prompt = request_json["prompt"].get<std::string>();
                }
                stream = request_json.value("stream", false);
                async_mode = request_json.value("async", false);
            } catch (const json::exception&) {
                // Not JSON, treat body as raw prompt
                prompt = req.body;
            }
        }

        if (prompt.empty()) {
            json error_response;
            error_response["success"] = false;
            error_response["error"] = "Missing prompt";
            res.status = 400;
            res.set_content(error_response.dump(), "application/json");
            return;
        }

        // Async mode: queue and return immediately
        if (async_mode && !stream) {
            state.add_input(prompt);
            json response;
            response["success"] = true;
            response["queued"] = true;
            response["queue_position"] = static_cast<int>(state.queue_size());
            res.set_content(response.dump(), "application/json");
            return;
        }

        // Acquire request mutex - blocks until previous request completes
        std::unique_lock<std::mutex> request_lock(state.request_mutex);

        // Check if we're shutting down
        if (!state.running) {
            json error_response;
            error_response["success"] = false;
            error_response["error"] = "Server is shutting down";
            res.status = 503;
            res.set_content(error_response.dump(), "application/json");
            return;
        }

        // Log request with client info (at debug level)
        std::string client_info = req.remote_addr;
        if (!req.get_header_value("X-Client-ID").empty()) {
            client_info += " (" + req.get_header_value("X-Client-ID") + ")";
        }
        dout(1) << "Request from " + client_info + ": " + prompt.substr(0, 100) + (prompt.length() > 100 ? "..." : "") << std::endl;

        // User message will be broadcast after add_message() below

        // Synchronous processing (with optional streaming)
        state.processing = true;
        state.current_request = prompt;

        // Increment request counter
        requests_processed++;

        try {
            if (stream) {
                // Streaming response using SSE
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                // Capture state pointer for lambda
                CliServerState* state_ptr = &state;

                res.set_content_provider(
                    "text/event-stream",
                    [state_ptr, prompt](size_t offset, httplib::DataSink& sink) mutable {
                        std::string accumulated_response;

                        // Broadcast user message to all SSE clients
                        json user_msg_event;
                        user_msg_event["role"] = "user";
                        user_msg_event["content"] = prompt;
                        state_ptr->broadcast_event("message_added", user_msg_event);

                        // Tool call filtering state
                        std::vector<std::string> tool_markers = state_ptr->backend->get_tool_call_markers();
                        std::string pending_buffer;
                        bool in_tool_call = false;

                        // Helper to check if buffer ends with partial marker
                        auto ends_with_partial_marker = [&tool_markers](const std::string& buf) -> bool {
                            // Check if buffer ends with start of any marker
                            for (const auto& marker : tool_markers) {
                                for (size_t len = 1; len < marker.size() && len <= buf.size(); len++) {
                                    if (buf.substr(buf.size() - len) == marker.substr(0, len)) {
                                        return true;
                                    }
                                }
                            }
                            // Check for partial JSON tool call pattern at end
                            std::string json_pattern = "{\"name\"";
                            for (size_t len = 1; len < json_pattern.size() && len <= buf.size(); len++) {
                                if (buf.substr(buf.size() - len) == json_pattern.substr(0, len)) {
                                    return true;
                                }
                            }
                            return false;
                        };

                        // Helper to check if buffer contains any marker
                        auto matches_marker = [&tool_markers](const std::string& buf) -> bool {
                            for (const auto& marker : tool_markers) {
                                if (buf.find(marker) != std::string::npos) {
                                    return true;
                                }
                            }
                            // Check for raw JSON tool call
                            if (buf.find("{\"name\"") != std::string::npos) {
                                return true;
                            }
                            return false;
                        };

                        // Find position where tool call starts in buffer
                        auto find_tool_call_start = [&tool_markers](const std::string& buf) -> size_t {
                            size_t earliest = std::string::npos;
                            for (const auto& marker : tool_markers) {
                                size_t pos = buf.find(marker);
                                if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
                                    earliest = pos;
                                }
                            }
                            size_t json_pos = buf.find("{\"name\"");
                            if (json_pos != std::string::npos && (earliest == std::string::npos || json_pos < earliest)) {
                                earliest = json_pos;
                            }
                            return earliest;
                        };

                        // Callback streams to SSE sink with tool call filtering
                        auto callback = [&](CallbackEvent type,
                                                   const std::string& content,
                                                   const std::string& tool_name_arg,
                                                   const std::string& tool_call_id) -> bool {
                            // Handle STOP - signals completion, content is finish_reason
                            if (type == CallbackEvent::STOP) {
                                // Flush any pending content before signaling completion
                                if (!pending_buffer.empty()) {
                                    std::string sanitized = utf8_sanitizer::sanitize_utf8(pending_buffer);
                                    json chunk;
                                    chunk["delta"] = sanitized;
                                    std::string data = "data: " + chunk.dump() + "\n\n";
                                    state_ptr->broadcast_event("delta", {{"delta", sanitized}});
                                    sink.write(data.c_str(), data.size());
                                    pending_buffer.clear();
                                }
                                return true;
                            }

                            // Handle ERROR
                            if (type == CallbackEvent::ERROR) {
                                json error_chunk;
                                error_chunk["error"] = content;
                                std::string data = "data: " + error_chunk.dump() + "\n\n";
                                sink.write(data.c_str(), data.size());
                                return true;
                            }

                            // Handle TOOL_CALL
                            if (type == CallbackEvent::TOOL_CALL) {
                                // Tool calls are handled via pending_tool_calls queue
                                return true;
                            }

                            if (content.empty()) return true;

                            // Handle USER message echo specially
                            if (type == CallbackEvent::USER_PROMPT) {
                                json user_chunk;
                                user_chunk["user_echo"] = content;
                                std::string data = "data: " + user_chunk.dump() + "\n\n";
                                state_ptr->broadcast_event("user_echo", user_chunk);
                                sink.write(data.c_str(), data.size());
                                return true;
                            }

                            const std::string& delta = content;  // Alias for compatibility

                            // Add delta to pending buffer
                            pending_buffer += delta;

                            // If we're already in a tool call, just buffer it
                            if (in_tool_call) {
                                return true;
                            }

                            // Check if buffer contains a tool call marker
                            if (matches_marker(pending_buffer)) {
                                // Found tool call - output everything before it, then buffer the rest
                                size_t tool_start = find_tool_call_start(pending_buffer);
                                if (tool_start > 0) {
                                    // Output text before the tool call
                                    std::string before_tool = pending_buffer.substr(0, tool_start);
                                    std::string sanitized = utf8_sanitizer::sanitize_utf8(before_tool);

                                    json chunk;
                                    chunk["delta"] = sanitized;
                                    std::string data = "data: " + chunk.dump() + "\n\n";

                                    json delta_event;
                                    delta_event["delta"] = sanitized;
                                    state_ptr->broadcast_event("delta", delta_event);

                                    sink.write(data.c_str(), data.size());

                                    // Keep only the tool call part
                                    pending_buffer = pending_buffer.substr(tool_start);
                                }
                                in_tool_call = true;
                                return true;
                            }

                            // Check if buffer ends with partial marker - keep buffering
                            if (ends_with_partial_marker(pending_buffer)) {
                                return true;
                            }

                            // Safe to output the pending buffer
                            if (!pending_buffer.empty()) {
                                std::string sanitized = utf8_sanitizer::sanitize_utf8(pending_buffer);

                                json chunk;
                                chunk["delta"] = sanitized;
                                std::string data = "data: " + chunk.dump() + "\n\n";

                                json delta_event;
                                delta_event["delta"] = sanitized;
                                state_ptr->broadcast_event("delta", delta_event);

                                pending_buffer.clear();
                                if (!sink.write(data.c_str(), data.size())) {
                                    return false;
                                }
                            }
                            return true;
                        };

                        // Set callback and add user message (triggers generation)
                        state_ptr->backend->callback = callback;
                        state_ptr->session->add_message(Message::USER, prompt);

                        // Build Response from session's last message
                        Response resp;
                        resp.success = true;
                        if (!state_ptr->session->messages.empty() &&
                            state_ptr->session->messages.back().role == Message::ASSISTANT) {
                            resp.content = state_ptr->session->messages.back().content;
                        }

                        accumulated_response = resp.content;

                        // Tool execution loop
                        const int max_tool_iterations = 50;
                        int iteration = 0;

                        while (iteration < max_tool_iterations) {
                            iteration++;

                            auto tool_call_opt = extract_tool_call(resp, state_ptr->backend);
                            if (!tool_call_opt) {
                                break;
                            }

                            auto& tool_call = *tool_call_opt;
                            std::string tool_name = tool_call.name;

                            // Send tool call notification with parameters
                            json tool_chunk;
                            tool_chunk["tool_call"] = tool_name;
                            // Convert parameters map to JSON
                            json params_json = json::object();
                            for (const auto& [key, val] : tool_call.parameters) {
                                if (val.type() == typeid(std::string)) {
                                    params_json[key] = std::any_cast<std::string>(val);
                                } else if (val.type() == typeid(int)) {
                                    params_json[key] = std::any_cast<int>(val);
                                } else if (val.type() == typeid(double)) {
                                    params_json[key] = std::any_cast<double>(val);
                                } else if (val.type() == typeid(bool)) {
                                    params_json[key] = std::any_cast<bool>(val);
                                } else {
                                    params_json[key] = "[complex]";
                                }
                            }
                            tool_chunk["parameters"] = params_json;
                            std::string tool_data = "data: " + tool_chunk.dump() + "\n\n";
                            sink.write(tool_data.c_str(), tool_data.size());

                            // Broadcast to SSE clients
                            state_ptr->broadcast_event("tool_call", tool_chunk);

                            // Execute tool (handles truncation)
                            ToolResult tool_result = state_ptr->server->execute_tool(*state_ptr->tools, tool_name, tool_call.parameters, tool_call.tool_call_id);

                            // Broadcast tool result to SSE clients
                            json result_event;
                            result_event["tool_name"] = tool_name;
                            result_event["success"] = tool_result.success;
                            if (!tool_result.success) {
                                result_event["error"] = tool_result.error;
                            }
                            state_ptr->broadcast_event("tool_result", result_event);

                            // Reset filtering state for next generation
                            in_tool_call = false;
                            pending_buffer.clear();

                            // Send tool result to model
                            state_ptr->session->add_message(
                                Message::TOOL_RESPONSE, tool_result.content, tool_name, tool_call.tool_call_id);

                            // Build Response from session's last message
                            if (!state_ptr->session->messages.empty() &&
                                state_ptr->session->messages.back().role == Message::ASSISTANT) {
                                resp.success = true;
                                resp.content = state_ptr->session->messages.back().content;
                            }

                            accumulated_response = resp.content;
                        }

                        // Broadcast response complete to SSE clients
                        json complete_event;
                        complete_event["response"] = accumulated_response;
                        state_ptr->broadcast_event("response_complete", complete_event);

                        // Send final done message
                        json done_chunk;
                        done_chunk["done"] = true;
                        done_chunk["response"] = accumulated_response;
                        std::string done_data = "data: " + done_chunk.dump() + "\n\n";
                        sink.write(done_data.c_str(), done_data.size());
                        sink.done();
                        return false;  // Signal no more content
                    }
                );
            } else {
                // Non-streaming response
                json response = process_request(state, prompt);
                res.set_content(response.dump(), "application/json");
            }

        } catch (const std::exception& e) {
            json error_response;
            error_response["success"] = false;
            error_response["error"] = e.what();
            res.status = 500;
            res.set_content(error_response.dump(), "application/json");
        }

        state.current_request.clear();
        state.processing = false;
    });

    // POST /clear - Reset conversation
    tcp_server.Post("/clear", [this](const httplib::Request&, httplib::Response& res) {
        state.session->messages.clear();
        state.session->last_prompt_tokens = 0;
        state.session->total_tokens = 0;
        state.session->last_user_message_index = -1;
        state.session->last_user_message_tokens = 0;
        state.session->last_assistant_message_index = -1;
        state.session->last_assistant_message_tokens = 0;

        json response;
        response["success"] = true;
        response["message"] = "Conversation cleared";
        res.set_content(response.dump(), "application/json");
    });

    // GET /session - Get full session state
    tcp_server.Get("/session", [this](const httplib::Request&, httplib::Response& res) {
        json response;
        response["success"] = true;

        // Context info from backend
        if (state.backend) {
            response["context_size"] = state.backend->context_size;
            response["model"] = state.backend->model_name;
        }

        // Token counts
        response["total_tokens"] = state.session->total_tokens;
        response["system_tokens"] = state.session->system_message_tokens;
        response["desired_completion_tokens"] = state.session->desired_completion_tokens;

        // Messages in OpenAI format
        json messages = json::array();
        for (const auto& msg : state.session->messages) {
            json m;
            m["role"] = msg.get_role();
            m["content"] = msg.content;
            m["tokens"] = msg.tokens;

            if (!msg.tool_name.empty()) {
                m["tool_name"] = msg.tool_name;
            }
            if (!msg.tool_call_id.empty()) {
                m["tool_call_id"] = msg.tool_call_id;
            }
            if (!msg.tool_calls_json.empty()) {
                try {
                    m["tool_calls"] = json::parse(msg.tool_calls_json);
                } catch (...) {
                    m["tool_calls_raw"] = msg.tool_calls_json;
                }
            }

            messages.push_back(m);
        }
        response["messages"] = messages;

        // System message if set
        if (!state.session->system_message.empty()) {
            response["system_message"] = state.session->system_message;
        }

        res.set_content(response.dump(), "application/json");
    });

    // GET /updates - SSE endpoint for live session updates
    tcp_server.Get("/updates", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");

        // Get client identifier from header or IP
        std::string client_id = req.get_header_value("X-Client-ID");
        if (client_id.empty()) {
            client_id = req.remote_addr;
        }

        std::cout << "Client connected: " << req.remote_addr << std::endl;

        res.set_content_provider(
            "text/event-stream",
            [this, client_id](size_t offset, httplib::DataSink& sink) mutable {
                // Create client entry
                SSEClient* client = new SSEClient();
                client->sink = &sink;
                client->client_id = client_id;

                // Register this client
                {
                    std::lock_guard<std::mutex> lock(state.sse_mutex);
                    state.sse_clients.push_back(client);
                }

                // Send initial connected event
                json connected;
                connected["type"] = "connected";
                connected["data"]["client_id"] = client_id;
                std::string initial = "data: " + connected.dump() + "\n\n";
                sink.write(initial.c_str(), initial.size());

                // Keep connection open - the broadcast_event function will send data
                // This provider will block until the connection is closed
                while (state.running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // Check if sink is still valid by trying to send a keep-alive
                    // (empty comment line is valid SSE)
                    const char* keepalive = ": keepalive\n\n";
                    if (!sink.write(keepalive, strlen(keepalive))) {
                        break;  // Connection closed
                    }
                }

                // Cleanup: remove this client from the list
                {
                    std::lock_guard<std::mutex> lock(state.sse_mutex);
                    auto it = std::find(state.sse_clients.begin(), state.sse_clients.end(), client);
                    if (it != state.sse_clients.end()) {
                        state.sse_clients.erase(it);
                    }
                }
                delete client;

                dout(1) << "SSE client disconnected: " + client_id << std::endl;
                sink.done();
                return false;
            }
        );
    });

    dout(1) << "CLI server endpoints: /health, /status, /request, /clear, /session, /updates" << std::endl;
}
