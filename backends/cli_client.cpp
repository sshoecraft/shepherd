#include "cli_client.h"
#include "shepherd.h"
#include "logger.h"
#include "terminal_io.h"
#include "output_queue.h"
#include "sse_parser.h"
#include "nlohmann/json.hpp"

extern std::unique_ptr<Config> config;
extern TerminalIO tio;
extern ThreadQueue<std::string> g_output_queue;

CLIClientBackend::CLIClientBackend(const std::string& url)
    : Backend(0), base_url(url) {
    is_local = false;
    sse_handles_output = true;
    http_client = std::make_unique<HttpClient>();
}

CLIClientBackend::~CLIClientBackend() {
    // Stop SSE listener thread
    sse_running = false;
    if (sse_thread.joinable()) {
        sse_thread.join();
    }
}

void CLIClientBackend::parse_backend_config() {
    if (config && !config->api_base.empty()) {
        base_url = config->api_base;
    }
}

void CLIClientBackend::initialize(Session& session) {
    parse_backend_config();
    LOG_INFO("CLI client connecting to: " + base_url);

    // Fetch session from server
    std::string endpoint = base_url + "/session";
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";

    HttpResponse http_resp = http_client->get(endpoint, headers);

    if (!http_resp.is_success()) {
        LOG_WARN("Failed to fetch session from server: " + http_resp.error_message);
        return;
    }

    try {
        nlohmann::json session_data = nlohmann::json::parse(http_resp.body);

        if (!session_data.value("success", false)) {
            LOG_WARN("Server returned error for /session");
            return;
        }

        // Update backend/session info from server
        if (session_data.contains("context_size")) {
            context_size = session_data["context_size"].get<int>();
        }
        if (session_data.contains("model")) {
            model_name = session_data["model"].get<std::string>();
        }
        if (session_data.contains("total_tokens")) {
            session.total_tokens = session_data["total_tokens"].get<int>();
        }

        // Display session history
        if (session_data.contains("messages") && session_data["messages"].is_array()) {

            for (const auto& msg : session_data["messages"]) {
                std::string role = msg.value("role", "unknown");
                std::string content = msg.value("content", "");
                std::string output;
                Color color = Color::DEFAULT;

                if (role == "user") {
                    output = "> " + content + "\n";
                    color = Color::GREEN;
                } else if (role == "assistant") {
                    output = content + "\n";
                    color = Color::DEFAULT;
                } else if (role == "tool") {
                    std::string tool_name = msg.value("tool_name", "tool");
                    output = "[" + tool_name + "] " + content + "\n";
                    color = Color::CYAN;
                }
                // Skip system messages - they're huge and not useful to display

                if (!output.empty()) {
                    tio.write(output.c_str(), output.size(), color);
                }
            }

        }

        LOG_INFO("Session loaded: " + std::to_string(session_data["messages"].size()) + " messages, " +
                 std::to_string(context_size) + " context size");

    } catch (const std::exception& e) {
        LOG_WARN("Failed to parse session response: " + std::string(e.what()));
    }

    // Start SSE listener thread for live updates
    sse_running = true;
    sse_thread = std::thread(&CLIClientBackend::sse_listener_thread, this);
}

void CLIClientBackend::sse_listener_thread() {
    LOG_INFO("SSE listener thread started");

    std::string endpoint = base_url + "/updates";
    std::map<std::string, std::string> headers;
    headers["Accept"] = "text/event-stream";

    // Reconnection loop - keep trying while sse_running is true
    while (sse_running) {
        HttpClient sse_client;
        SSEParser sse_parser;

        auto stream_handler = [this, &sse_parser](const std::string& chunk, void* userdata) -> bool {
            if (!sse_running) return false;

            sse_parser.process_chunk(chunk, [this](const std::string& event, const std::string& data, const std::string& id) -> bool {
                try {
                    nlohmann::json json_data = nlohmann::json::parse(data);
                    std::string event_type = json_data.value("type", "");
                    auto event_data = json_data.value("data", nlohmann::json::object());

                    if (event_type == "message_added") {
                        // Render message by role (same logic as session load)
                        std::string role = event_data.value("role", "unknown");
                        std::string content = event_data.value("content", "");
                        std::string output;

                        if (role == "user") {
                            output = "\033[32m> " + content + "\033[0m\n";
                        } else if (role == "assistant") {
                            output = content + "\n";
                        } else if (role == "tool") {
                            std::string tool_name = event_data.value("tool_name", "tool");
                            output = "\033[36m[" + tool_name + "] " + content + "\033[0m\n";
                        }

                        if (!output.empty()) {
                            // Thread-safe: Use output queue instead of tio.write() from SSE thread
                            g_output_queue.push(output);
                        }
                    }
                } catch (...) {}
                return sse_running;
            });

            return sse_running;
        };

        HttpResponse http_resp = sse_client.get_stream(endpoint, headers, stream_handler, nullptr);

        if (!sse_running) break;  // Clean shutdown requested

        if (!http_resp.is_success()) {
            LOG_WARN("SSE connection lost, reconnecting in 2s...");
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    LOG_INFO("SSE listener thread stopped");
}

Response CLIClientBackend::send_request(const std::string& prompt, StreamCallback callback) {
    Response resp;
    std::string endpoint = base_url + "/request";

    nlohmann::json request;
    request["prompt"] = prompt;
    request["stream"] = true;  // Server streams, SSE handles display

    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";

    // Just POST and wait for completion - SSE handles all display
    std::string accumulated;
    bool stream_complete = false;

    auto stream_handler = [&](const std::string& chunk, void* userdata) -> bool {
        if (stream_complete) return true;

        std::istringstream stream(chunk);
        std::string line;

        while (std::getline(stream, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            if (line.substr(0, 6) == "data: ") {
                try {
                    nlohmann::json event = nlohmann::json::parse(line.substr(6));

                    if (event.contains("delta")) {
                        accumulated += event["delta"].get<std::string>();
                    }
                    if (event.contains("done") && event["done"].get<bool>()) {
                        resp.content = event.value("response", accumulated);
                        resp.success = true;
                        resp.code = Response::SUCCESS;
                        stream_complete = true;
                    }
                    if (event.contains("error")) {
                        resp.success = false;
                        resp.code = Response::ERROR;
                        resp.error = event["error"].get<std::string>();
                        stream_complete = true;
                    }
                } catch (...) {}
            }
        }
        return true;
    };

    HttpResponse http_resp = http_client->post_stream_cancellable(endpoint, request.dump(), headers, stream_handler, nullptr);

    if (!http_resp.is_success() && !resp.success) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.error = http_resp.error_message.empty() ? "HTTP request failed" : http_resp.error_message;
    }

    return resp;
}

Response CLIClientBackend::add_message(Session& session,
                                       Message::Type type,
                                       const std::string& content,
                                       const std::string& tool_name,
                                       const std::string& tool_id,
                                       int prompt_tokens,
                                       int max_tokens) {
    // Only handle USER messages - CLI server manages its own conversation
    if (type != Message::USER) {
        Response resp;
        resp.success = true;
        resp.code = Response::SUCCESS;
        return resp;
    }

    // Use streaming if enabled in config
    if (config && config->streaming) {
        tio.begin_response();

        Response resp = send_request(content,
            [](const std::string& delta, const std::string& accumulated, const Response& partial) -> bool {
                g_output_queue.push(delta);
                return true;
            });

        tio.end_response();
        resp.was_streamed = true;
        return resp;
    }

    return send_request(content, nullptr);
}

Response CLIClientBackend::add_message_stream(Session& session,
                                              Message::Type type,
                                              const std::string& content,
                                              StreamCallback callback,
                                              const std::string& tool_name,
                                              const std::string& tool_id,
                                              int prompt_tokens,
                                              int max_tokens) {
    // Only handle USER messages - CLI server manages its own conversation
    if (type != Message::USER) {
        Response resp;
        resp.success = true;
        resp.code = Response::SUCCESS;
        return resp;
    }

    return send_request(content, callback);
}

Response CLIClientBackend::generate_from_session(const Session& session, int max_tokens, StreamCallback callback) {
    // Find the last user message
    std::string prompt;
    for (auto it = session.messages.rbegin(); it != session.messages.rend(); ++it) {
        if (it->type == Message::USER) {
            prompt = it->content;
            break;
        }
    }

    if (prompt.empty()) {
        Response resp;
        resp.success = false;
        resp.code = Response::ERROR;
        resp.error = "No user message found";
        return resp;
    }

    return send_request(prompt, callback);
}

int CLIClientBackend::count_message_tokens(Message::Type type,
                                           const std::string& content,
                                           const std::string& tool_name,
                                           const std::string& tool_id) {
    // CLI server manages tokens - just estimate
    return content.length() / 4;
}
