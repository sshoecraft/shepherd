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
    // Use timed wait to allow periodic scheduler polling
    bool got_input = queue_cv.wait_for(lock, std::chrono::seconds(1),
        [this] { return !input_queue.empty() || !running; });
    if (!got_input || (!running && input_queue.empty())) {
        return "";
    }
    std::string prompt = input_queue.front();
    input_queue.pop_front();
    return prompt;
}

void CliServerState::send_to_observers(const std::function<void(ClientOutputs::ClientOutput&)>& action) {
    std::lock_guard<std::mutex> lock(observers_mutex);

    // Send to all connected observers, remove disconnected ones
    for (auto it = observers.begin(); it != observers.end(); ) {
        ClientOutputs::StreamingOutput* observer = *it;
        if (observer && observer->is_connected()) {
            action(*observer);
            ++it;
        } else {
            dprintf(1, "Observer disconnected during send");
            it = observers.erase(it);
            // Note: observer is owned by the /updates content provider, not deleted here
        }
    }
}

void CliServerState::register_observer(ClientOutputs::StreamingOutput* observer) {
    std::lock_guard<std::mutex> lock(observers_mutex);
    observers.push_back(observer);
    dprintf(2, "Observer registered, total: %zu", observers.size());
}

void CliServerState::unregister_observer(ClientOutputs::StreamingOutput* observer) {
    std::lock_guard<std::mutex> lock(observers_mutex);
    auto it = std::find(observers.begin(), observers.end(), observer);
    if (it != observers.end()) {
        observers.erase(it);
        dprintf(2, "Observer unregistered, remaining: %zu", observers.size());
    }
}

// CLIServer class implementation
CLIServer::CLIServer(const std::string& host, int port)
    : Server(host, port, "cli") {
}

CLIServer::~CLIServer() {
}

void CLIServer::init(bool no_mcp_flag, bool no_tools_flag, bool no_rag_flag) {
    // Store flags for later use (e.g., fallback to local tools in register_endpoints)
    no_mcp = no_mcp_flag;
    no_tools = no_tools_flag;
    no_rag = no_rag_flag;

    // Use common tool initialization from Frontend base class
    init_tools(no_mcp, no_tools, false, no_rag);
}


// Unified generation function - sends to requester AND all observers
static void do_generation(CliServerState& state,
                          ClientOutputs::ClientOutput* requester,
                          const std::string& prompt,
                          int max_tokens = 0) {
    std::string accumulated_response;

    // Send user prompt to requester and observers
    requester->on_user_prompt(prompt);
    state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
        obs.on_user_prompt(prompt);
    });

    // Tool call filtering state
    std::vector<std::string> tool_markers = state.backend->get_tool_call_markers();
    std::string pending_buffer;
    bool in_tool_call = false;

    // Helper to check if buffer ends with partial marker
    auto ends_with_partial_marker = [&tool_markers](const std::string& buf) -> bool {
        for (const auto& marker : tool_markers) {
            for (size_t len = 1; len < marker.size() && len <= buf.size(); len++) {
                if (buf.substr(buf.size() - len) == marker.substr(0, len)) {
                    return true;
                }
            }
        }
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

    // Helper to flush pending buffer as delta
    auto flush_pending = [&]() {
        if (!pending_buffer.empty()) {
            requester->on_delta(pending_buffer);
            state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                obs.on_delta(pending_buffer);
            });
            pending_buffer.clear();
        }
    };

    // Callback streams to requester AND all observers
    auto callback = [&](CallbackEvent type,
                        const std::string& content,
                        const std::string& tool_name_arg,
                        const std::string& tool_call_id) -> bool {
        // Handle STOP
        if (type == CallbackEvent::STOP) {
            // Only flush if not in a tool call (tool call content should not be shown)
            if (!in_tool_call) {
                flush_pending();
            }
            return true;
        }

        // Handle ERROR
        if (type == CallbackEvent::ERROR) {
            requester->on_error(content);
            state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                obs.on_error(content);
            });
            return true;
        }

        // Handle TOOL_CALL - fires after STOP, execute immediately
        if (type == CallbackEvent::TOOL_CALL) {
            // Parse params for display
            json params_json;
            try {
                params_json = json::parse(content);
            } catch (...) {
                params_json = json::object();
            }

            // Send tool call to clients
            requester->on_tool_call(tool_name_arg, params_json, tool_call_id);
            state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                obs.on_tool_call(tool_name_arg, params_json, tool_call_id);
            });

            // Execute tool
            ToolResult tool_result = state.server->execute_tool(
                *state.tools, tool_name_arg, content, tool_call_id, state.server->session.user_id);

            // Send tool result to clients
            requester->on_tool_result(tool_name_arg, tool_result.success, tool_result.error);
            state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                obs.on_tool_result(tool_name_arg, tool_result.success, tool_result.error);
            });

            // Add tool result to session and generate next response
            {
                auto lock = state.backend->acquire_lock();
                state.server->add_message_to_session(
                    Message::TOOL_RESPONSE, tool_result.content, tool_name_arg, tool_call_id);
                state.server->generate_response(max_tokens);
            }

            // Queue memory extraction after tool-response generation
            state.server->queue_memory_extraction();

            return true;
        }

        // Skip STATS - these are local display only, not sent to clients
        if (type == CallbackEvent::STATS) {
            return true;
        }

        if (content.empty()) return true;

        // Handle USER_PROMPT echo (already handled above)
        if (type == CallbackEvent::USER_PROMPT) {
            return true;
        }

        // Handle CODEBLOCK - send as separate event type
        if (type == CallbackEvent::CODEBLOCK) {
            requester->on_codeblock(content);
            state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                obs.on_codeblock(content);
            });
            return true;
        }

        const std::string& delta = content;
        pending_buffer += delta;

        if (in_tool_call) {
            return true;
        }

        if (matches_marker(pending_buffer)) {
            size_t tool_start = find_tool_call_start(pending_buffer);
            if (tool_start > 0) {
                std::string before_tool = pending_buffer.substr(0, tool_start);
                requester->on_delta(before_tool);
                state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
                    obs.on_delta(before_tool);
                });
                pending_buffer = pending_buffer.substr(tool_start);
            }
            in_tool_call = true;
            return true;
        }

        if (ends_with_partial_marker(pending_buffer)) {
            return true;
        }

        // Safe to output
        flush_pending();
        return true;
    };

    // Set callback, add user message and generate
    state.backend->callback = callback;
    {
        auto lock = state.backend->acquire_lock();
        state.server->add_message_to_session(Message::USER, prompt);
        state.server->enrich_with_rag_context(state.server->session);
        state.server->generate_response(max_tokens);
    }

    // Queue memory extraction after generation
    state.server->queue_memory_extraction();

    // Build Response from session's last message
    Response resp;
    resp.success = true;
    // Tool calls are handled recursively in callback
    // Just get final response content
    if (!state.session->messages.empty() &&
        state.session->messages.back().role == Message::ASSISTANT) {
        resp.content = state.session->messages.back().content;
    }
    accumulated_response = resp.content;

    // Send completion to requester and observers
    requester->on_complete(accumulated_response);
    state.send_to_observers([&](ClientOutputs::ClientOutput& obs) {
        obs.on_complete(accumulated_response);
    });
}

// Process a single request with tool execution loop (for async requests)
static json process_request(CliServerState& state, const std::string& prompt) {
    // Create a batched output that just collects the response
    // (no HTTP response to write to - this is for async processing)
    std::string accumulated_response;
    bool has_error = false;
    std::string error_message;

    // Simple output collector (not a real HTTP response)
    struct AsyncCollector : public ClientOutputs::ClientOutput {
        std::string& accumulated;
        bool& has_error;
        std::string& error_msg;

        AsyncCollector(std::string& acc, bool& err, std::string& errmsg)
            : accumulated(acc), has_error(err), error_msg(errmsg) {}

        void on_delta(const std::string& delta) override {
            accumulated += delta;
        }
        void on_codeblock(const std::string& content) override {
            accumulated += "```\n" + content;  // Wrap in markdown for non-streaming
        }
        void on_user_prompt(const std::string&) override {}
        void on_message_added(const std::string&, const std::string&, int) override {}
        void on_tool_call(const std::string&, const nlohmann::json&, const std::string&) override {}
        void on_tool_result(const std::string&, bool, const std::string&) override {}
        void on_complete(const std::string& full_response) override {
            accumulated = full_response;
        }
        void on_error(const std::string& error) override {
            has_error = true;
            error_msg = error;
        }
        void flush() override {}
        bool is_connected() const override { return true; }
    };

    AsyncCollector collector(accumulated_response, has_error, error_message);
    do_generation(state, &collector, prompt);

    json result;
    if (has_error) {
        result["success"] = false;
        result["error"] = error_message;
    } else {
        result["success"] = true;
        result["response"] = accumulated_response;
    }
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
    // Initialize scheduler (unless disabled)
    if (!g_disable_scheduler) {
        scheduler.name = config->scheduler_name;
        scheduler.load();
        scheduler.set_fire_callback([this](const std::string& prompt) {
            state.add_input(prompt);
        });
        scheduler.start();
        dout(1) << "CLI server scheduler initialized with " << scheduler.list().size() << " schedules" << std::endl;
    }

    // Start background processing thread for async requests
    processor_thread = std::thread([this]() {
        dout(1) << "CLI server processor thread started" << std::endl;
        while (state.running) {
            // Poll scheduler for timed prompts
            if (!g_disable_scheduler) {
                scheduler.poll();
            }

            std::string prompt = state.get_next_input();
            if (prompt.empty()) {
                continue;  // Timeout, shutdown, or spurious wakeup
            }

            state.processing = true;
            state.current_request = prompt;

            dout(1) << "Processing async request: " + prompt.substr(0, 50) + "..." << std::endl;

            // Acquire request mutex to prevent concurrent backend access
            std::unique_lock<std::mutex> request_lock(state.request_mutex);
            json result = process_request(state, prompt);
            request_lock.unlock();

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
    // Stop scheduler
    if (!g_disable_scheduler) {
        scheduler.stop();
    }

    // Signal any in-flight generation to cancel
    g_generation_cancelled = true;

    // Wait for processor thread to finish
    if (processor_thread.joinable()) {
        processor_thread.join();
    }
}

void CLIServer::register_endpoints() {
    // If server_tools mode, fetch tools from server or fall back to local
    if (config->server_tools && !no_tools) {
        Provider* p = get_provider(current_provider);
        if (p && !p->base_url.empty()) {
            init_remote_tools(p->base_url, p->api_key);
        } else {
            std::cerr << "Warning: --server-tools requires an API provider with base_url, falling back to local tools" << std::endl;
            init_tools(no_mcp, no_tools, true, no_rag);  // force_local = true
        }
    }

    // Initialize state with backend, session, tools, and server pointer
    state.server = this;
    state.backend = backend.get();
    state.session = &session;
    state.tools = &tools;

    // Copy tool names to backend for output filtering
    // This enables the backend to detect tool calls and emit TOOL_CALL events
    for (const auto& tool : session.tools) {
        backend->valid_tool_names.insert(tool.name);
    }

    // POST /request - Main request endpoint
    tcp_server.Post("/request", [this](const httplib::Request& req, httplib::Response& res) {
        // Parse request
        json request_json;
        std::string prompt;
        bool stream = false;
        bool async_mode = false;
        int max_tokens = 0;

        if (!req.body.empty()) {
            try {
                std::string sanitized_body = utf8_sanitizer::sanitize_utf8(req.body);
                request_json = json::parse(sanitized_body);
                if (request_json.contains("prompt")) {
                    prompt = request_json["prompt"].get<std::string>();
                }
                stream = request_json.value("stream", false);
                async_mode = request_json.value("async", false);
                max_tokens = request_json.value("max_tokens", 0);
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

                CliServerState* state_ptr = &state;

                res.set_content_provider(
                    "text/event-stream",
                    [state_ptr, prompt, client_info, max_tokens](size_t offset, httplib::DataSink& sink) mutable {
                        // Create streaming output for this request
                        ClientOutputs::StreamingOutput requester(&sink, client_info);

                        // Run unified generation
                        do_generation(*state_ptr, &requester, prompt, max_tokens);

                        // Flush (sends done marker) and close
                        requester.flush();
                        sink.done();
                        return false;
                    }
                );
            } else {
                // Non-streaming response using BatchedOutput
                ClientOutputs::BatchedOutput requester(&res);
                do_generation(state, &requester, prompt, max_tokens);
                requester.flush();
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
            m["content"] = utf8_sanitizer::sanitize_utf8(msg.content);
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
                    m["tool_calls_raw"] = utf8_sanitizer::sanitize_utf8(msg.tool_calls_json);
                }
            }

            messages.push_back(m);
        }
        response["messages"] = messages;

        // System message if set
        if (!state.session->system_message.empty()) {
            response["system_message"] = utf8_sanitizer::sanitize_utf8(state.session->system_message);
        }

        res.set_content(response.dump(), "application/json");
    });

    // GET /updates - SSE endpoint for live session updates (observers)
    tcp_server.Get("/updates", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");

        // Get client identifier from header or IP
        std::string client_id = req.get_header_value("X-Client-ID");
        if (client_id.empty()) {
            client_id = req.remote_addr;
        }

        std::cout << "Observer connected: " << req.remote_addr << std::endl;

        res.set_content_provider(
            "text/event-stream",
            [this, client_id](size_t offset, httplib::DataSink& sink) mutable {
                // Create streaming output for this observer
                ClientOutputs::StreamingOutput observer(&sink, client_id);

                // Register this observer
                state.register_observer(&observer);

                // Send initial connected event
                json connected;
                connected["type"] = "connected";
                connected["data"]["client_id"] = client_id;
                std::string initial = "data: " + connected.dump() + "\n\n";
                sink.write(initial.c_str(), initial.size());

                // Keep connection open - events are sent via send_to_observers()
                while (state.running && observer.is_connected()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // Send keep-alive to detect disconnection
                    if (!observer.send_keepalive()) {
                        break;
                    }
                }

                // Cleanup: unregister this observer
                state.unregister_observer(&observer);

                dout(1) << "Observer disconnected: " + client_id << std::endl;
                sink.done();
                return false;
            }
        );
    });

    dout(1) << "CLI server endpoints: /health, /status, /request, /clear, /session, /updates" << std::endl;
}
