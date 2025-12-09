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

    LOG_DEBUG("Broadcasting SSE event: " + event_type + " to " + std::to_string(sse_clients.size()) + " clients");

    // Send to all connected SSE clients
    for (auto it = sse_clients.begin(); it != sse_clients.end(); ) {
        SSEClient* client = *it;
        if (client && client->sink) {
            if (!client->sink->write(sse_data.c_str(), sse_data.size())) {
                // Client disconnected, remove from list
                LOG_DEBUG("SSE client disconnected during broadcast");
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

void CLIServer::init(Session& session, bool no_mcp, bool no_tools, const std::string& provider_name) {
    // Initialize RAG system using global config
    std::string db_path = config->memory_database;
    if (db_path.empty()) {
        try {
            db_path = Config::get_default_memory_db_path();
        } catch (const ConfigError& e) {
            LOG_ERROR("Failed to determine memory database path: " + std::string(e.what()));
            throw;
        }
    } else if (db_path[0] == '~') {
        // Expand ~ if present
        db_path = Config::get_home_directory() + db_path.substr(1);
    }

    if (!RAGManager::initialize(db_path, config->max_db_size)) {
        throw std::runtime_error("Failed to initialize RAG system");
    }
    LOG_INFO("RAG initialized with database: " + db_path);

    if (no_tools) {
        LOG_INFO("Tools disabled for CLI server");
        return;
    }

    LOG_INFO("Initializing tools for CLI server...");

    // Register all native tools
    register_filesystem_tools(tools);
    register_command_tools(tools);
    register_json_tools(tools);
    register_http_tools(tools);
    register_memory_tools(tools);
    register_mcp_resource_tools(tools);
    register_core_tools(tools);

    // Initialize MCP servers
    if (!no_mcp) {
        auto& mcp = MCP::instance();
        mcp.initialize(tools);
        LOG_INFO("MCP initialized with " + std::to_string(mcp.get_tool_count()) + " tools");
    } else {
        LOG_INFO("MCP system disabled");
    }

    // Build the combined tool list
    tools.build_all_tools();

    LOG_INFO("CLI server tools initialized: " + std::to_string(tools.all_tools.size()) + " total");

    // Populate session.tools from Tools instance
    if (!no_tools) {
        tools.populate_session_tools(session);
        LOG_DEBUG("Session initialized with " + std::to_string(session.tools.size()) + " tools");
    }
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

    // Add user message and generate initial response
    Response resp = state.session->add_message(Message::USER, prompt);

    if (!resp.success) {
        result["success"] = false;
        result["error"] = resp.error;
        return result;
    }

    accumulated_response = resp.content;

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

        // Execute the tool
        ToolResult tool_result = state.tools->execute(tool_name, tool_call.parameters);

        if (tool_result.success) {
            // Sanitize UTF-8 in result
            std::string sanitized = utf8_sanitizer::sanitize_utf8(tool_result.content);

            // Send tool result to model and get next response
            resp = state.session->add_message(Message::TOOL, sanitized, tool_name, tool_call.tool_call_id);
        } else {
            // Send error to model
            std::string error_msg = "Error: " + tool_result.error;
            resp = state.session->add_message(Message::TOOL, error_msg, tool_name, tool_call.tool_call_id);
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
        LOG_DEBUG("CLI server processor thread started");
        while (state.running) {
            std::string prompt = state.get_next_input();
            if (prompt.empty()) {
                continue;  // Shutdown or spurious wakeup
            }

            state.processing = true;
            state.current_request = prompt;

            LOG_DEBUG("Processing async request: " + prompt.substr(0, 50) + "...");
            json result = process_request(state, prompt);

            // Increment request counter
            requests_processed++;

            // For async requests, we just log the result (no one is waiting)
            if (result.value("success", false)) {
                LOG_DEBUG("Async request completed successfully");
            } else {
                LOG_DEBUG("Async request failed: " + result.value("error", "unknown"));
            }

            state.current_request.clear();
            state.processing = false;
        }
        LOG_DEBUG("CLI server processor thread stopped");
    });
}

void CLIServer::on_server_stop() {
    // Signal everything to stop
    state.running = false;
    state.queue_cv.notify_all();

    // Signal any in-flight generation to cancel
    g_generation_cancelled = true;

    // Wait for processor thread to finish
    if (processor_thread.joinable()) {
        processor_thread.join();
    }
}

void CLIServer::register_endpoints(Session& session) {
    // Initialize state with backend, session, and tools
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
                request_json = json::parse(req.body);
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
        LOG_DEBUG("Request from " + client_info + ": " + prompt.substr(0, 100) + (prompt.length() > 100 ? "..." : ""));

        // Broadcast that a new request is starting
        json request_event;
        request_event["prompt"] = prompt;
        request_event["client"] = client_info;
        state.broadcast_event("request_start", request_event);

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

                        // Tool call filtering state
                        std::vector<std::string> tool_markers = state_ptr->backend->get_tool_call_markers();
                        std::string pending_buffer;
                        bool in_tool_call = false;

                        // Helper to check if buffer could be start of any marker
                        auto could_be_marker_start = [&tool_markers](const std::string& buf) -> bool {
                            for (const auto& marker : tool_markers) {
                                if (marker.substr(0, buf.size()) == buf) {
                                    return true;
                                }
                            }
                            // Also check for raw JSON tool call pattern
                            if (buf.size() > 0 && buf[0] == '{') {
                                std::string json_pattern = "{\"name\"";
                                if (json_pattern.substr(0, buf.size()) == buf) {
                                    return true;
                                }
                            }
                            return false;
                        };

                        // Helper to check if buffer matches any marker
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

                        // Streaming callback with tool call filtering
                        auto stream_callback = [&](const std::string& delta,
                                                   const std::string& accumulated,
                                                   const Response& partial) -> bool {
                            if (delta.empty()) return true;

                            // Add delta to pending buffer
                            pending_buffer += delta;

                            // If we're in a tool call, just buffer it
                            if (in_tool_call) {
                                return true;
                            }

                            // Check if this could be the start of a tool call
                            if (matches_marker(pending_buffer)) {
                                // We've hit a tool call marker - stop streaming content
                                in_tool_call = true;
                                return true;
                            }

                            // Check if we might be starting a tool call
                            if (could_be_marker_start(pending_buffer)) {
                                // Keep buffering - might be a tool call
                                return true;
                            }

                            // Safe to output the pending buffer
                            if (!pending_buffer.empty()) {
                                json chunk;
                                chunk["delta"] = pending_buffer;
                                std::string data = "data: " + chunk.dump() + "\n\n";

                                // Broadcast delta to SSE clients
                                json delta_event;
                                delta_event["delta"] = pending_buffer;
                                state_ptr->broadcast_event("delta", delta_event);

                                pending_buffer.clear();
                                if (!sink.write(data.c_str(), data.size())) {
                                    return false;
                                }
                            }
                            return true;
                        };

                        // Add user message with streaming
                        Response resp = state_ptr->session->add_message_stream(
                            Message::USER, prompt, stream_callback);

                        if (!resp.success) {
                            json error_chunk;
                            error_chunk["error"] = resp.error;
                            error_chunk["done"] = true;
                            std::string data = "data: " + error_chunk.dump() + "\n\n";
                            sink.write(data.c_str(), data.size());
                            sink.done();
                            return true;
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

                            // Send tool call notification
                            json tool_chunk;
                            tool_chunk["tool_call"] = tool_name;
                            std::string tool_data = "data: " + tool_chunk.dump() + "\n\n";
                            sink.write(tool_data.c_str(), tool_data.size());

                            // Execute the tool
                            ToolResult tool_result = state_ptr->tools->execute(tool_name, tool_call.parameters);

                            // Reset filtering state for next generation
                            in_tool_call = false;
                            pending_buffer.clear();

                            if (tool_result.success) {
                                std::string sanitized = utf8_sanitizer::sanitize_utf8(tool_result.content);
                                resp = state_ptr->session->add_message_stream(
                                    Message::TOOL, sanitized, stream_callback, tool_name, tool_call.tool_call_id);
                            } else {
                                std::string error_msg = "Error: " + tool_result.error;
                                resp = state_ptr->session->add_message_stream(
                                    Message::TOOL, error_msg, stream_callback, tool_name, tool_call.tool_call_id);
                            }

                            if (!resp.success) {
                                json error_chunk;
                                error_chunk["error"] = resp.error;
                                error_chunk["done"] = true;
                                std::string data = "data: " + error_chunk.dump() + "\n\n";
                                sink.write(data.c_str(), data.size());
                                sink.done();
                                return true;
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

        LOG_INFO("SSE client connected for updates: " + client_id);

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

                LOG_INFO("SSE client disconnected: " + client_id);
                sink.done();
                return false;
            }
        );
    });

    LOG_INFO("CLI server endpoints: /health, /status, /request, /clear, /session, /updates");
}
