#include "cli_client.h"
#include "shepherd.h"
#include "sse_parser.h"
#include "nlohmann/json.hpp"
#include <sstream>

extern std::unique_ptr<Config> config;

CLIClientBackend::CLIClientBackend(const std::string& url, Session& session, EventCallback callback)
    : Backend(0, session, callback), base_url(url) {
    is_local = false;
    sse_handles_output = true;
    http_client = std::make_unique<HttpClient>();

    // --- Initialization ---
    parse_backend_config();
    dout(1) << "CLI client connecting to: " + base_url << std::endl;

    // Fetch session from server
    std::string endpoint = base_url + "/session";
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";

    HttpResponse http_resp = http_client->get(endpoint, headers);

    if (!http_resp.is_success()) {
        dout(1) << std::string("WARNING: ") + "Failed to fetch session from server: " + http_resp.error_message << std::endl;
        return;
    }

    try {
        nlohmann::json session_data = nlohmann::json::parse(http_resp.body);

        if (!session_data.value("success", false)) {
            dout(1) << std::string("WARNING: ") + "Server returned error for /session" << std::endl;
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
                std::string msg_output;
                Message::Role msg_type = Message::ASSISTANT;

                if (role == "user") {
                    msg_output = "> " + content + "\n";
                    msg_type = Message::USER;
                } else if (role == "assistant") {
                    // Check if this is a tool call message (OpenAI format)
                    if (msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty()) {
                        // Display tool calls in format: * tool_name(param=value, ...)
                        for (const auto& tc : msg["tool_calls"]) {
                            std::string tool_name = "tool";
                            std::string tool_args_str;
                            if (tc.contains("function")) {
                                tool_name = tc["function"].value("name", "tool");
                                if (tc["function"].contains("arguments")) {
                                    // Parse arguments and format as param=value
                                    try {
                                        nlohmann::json args_json = tc["function"]["arguments"];
                                        if (args_json.is_string()) {
                                            args_json = nlohmann::json::parse(args_json.get<std::string>());
                                        }
                                        std::vector<std::string> parts;
                                        for (auto it = args_json.begin(); it != args_json.end(); ++it) {
                                            std::string val_str;
                                            if (it.value().is_string()) {
                                                val_str = it.value().get<std::string>();
                                            } else {
                                                val_str = it.value().dump();
                                            }
                                            // Truncate long values
                                            if (val_str.length() > 50) {
                                                val_str = val_str.substr(0, 47) + "...";
                                            }
                                            parts.push_back(it.key() + "=" + val_str);
                                        }
                                        for (size_t i = 0; i < parts.size(); i++) {
                                            if (i > 0) tool_args_str += ", ";
                                            tool_args_str += parts[i];
                                        }
                                    } catch (...) {
                                        tool_args_str = tc["function"]["arguments"].dump();
                                    }
                                }
                            }
                            std::cout << "  * " + tool_name + "(" + tool_args_str + ")\n";
                        }
                        continue;  // Don't output content for tool call messages
                    }
                    // Check if content is a JSON tool call (local model format)
                    if (!content.empty() && content[0] == '{') {
                        try {
                            nlohmann::json tc = nlohmann::json::parse(content);
                            if (tc.contains("name") && tc.contains("parameters")) {
                                std::string tool_name = tc.value("name", "tool");
                                std::string tool_args_str;
                                std::vector<std::string> parts;
                                for (auto it = tc["parameters"].begin(); it != tc["parameters"].end(); ++it) {
                                    std::string val_str;
                                    if (it.value().is_string()) {
                                        val_str = it.value().get<std::string>();
                                    } else {
                                        val_str = it.value().dump();
                                    }
                                    if (val_str.length() > 50) {
                                        val_str = val_str.substr(0, 47) + "...";
                                    }
                                    parts.push_back(it.key() + "=" + val_str);
                                }
                                for (size_t i = 0; i < parts.size(); i++) {
                                    if (i > 0) tool_args_str += ", ";
                                    tool_args_str += parts[i];
                                }
                                std::cout << "  * " + tool_name + "(" + tool_args_str + ")\n";
                                continue;
                            }
                        } catch (...) {
                            // Not valid JSON, treat as normal content
                        }
                    }
                    msg_output = "  " + content + "\n";  // Indent assistant
                    msg_type = Message::ASSISTANT;
                } else if (role == "tool") {
                    // Tool result - show error message or success checkmark
                    if (content.rfind("Error: ", 0) == 0) {
                        msg_output = "    " + content + "\n";
                    } else {
                        msg_output = "    ✓ Success\n";
                    }
                    msg_type = Message::TOOL_RESPONSE;
                }
                // Skip system messages - they're huge and not useful to display

                if (!msg_output.empty()) {
                    std::cout << msg_output;
                }
            }

        }

        dout(1) << "Session loaded: " + std::to_string(session_data["messages"].size()) + " messages, " +
                 std::to_string(context_size) + " context size" << std::endl;

    } catch (const std::exception& e) {
        dout(1) << std::string("WARNING: ") + "Failed to parse session response: " + std::string(e.what()) << std::endl;
    }

    // Start SSE listener thread for live updates
    sse_running = true;
    sse_thread = std::thread(&CLIClientBackend::sse_listener_thread, this);
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

void CLIClientBackend::sse_listener_thread() {
    dout(1) << "SSE listener thread started" << std::endl;

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
                        // Render message by role - output handler decides formatting
                        std::string role = event_data.value("role", "unknown");
                        std::string content = event_data.value("content", "");

                        Message::Role msg_type = Message::ASSISTANT;
                        std::string display_content = content;
                        if (role == "user") {
                            msg_type = Message::USER;
                            display_content = "> " + content;
                        } else if (role == "tool") {
                            msg_type = Message::TOOL_RESPONSE;
                        }
                        std::cout << display_content + "\n";
                    } else if (event_type == "tool_call") {
                        // Tool call notification - format as * tool_name(param=value, ...)
                        std::string tool_name = event_data.value("tool_call", "tool");
                        std::string args_str;
                        if (event_data.contains("parameters")) {
                            try {
                                std::vector<std::string> parts;
                                for (auto& [key, val] : event_data["parameters"].items()) {
                                    std::string val_str = val.is_string() ? val.get<std::string>() : val.dump();
                                    if (val_str.length() > 50) {
                                        val_str = val_str.substr(0, 47) + "...";
                                    }
                                    parts.push_back(key + "=" + val_str);
                                }
                                for (size_t i = 0; i < parts.size(); i++) {
                                    if (i > 0) args_str += ", ";
                                    args_str += parts[i];
                                }
                            } catch (...) {}
                        }
                        std::cout << "  * " + tool_name + "(" + args_str + ")\n";
                    } else if (event_type == "tool_result") {
                        // Tool execution result
                        bool success = event_data.value("success", true);
                        if (success) {
                            std::cout << std::string("    ✓ Success\n");
                        } else {
                            std::string error = event_data.value("error", "Unknown error");
                            std::cout << std::string("    Error: ") + error + "\n";
                        }
                    } else if (event_type == "user_echo") {
                        // User prompt echo from server
                        std::string content = event_data.value("user_echo", "");
                        if (!content.empty()) {
                            std::cout << content;
                        }
                    } else if (event_type == "delta") {
                        // Streaming delta from server
                        std::string delta = event_data.value("delta", "");
                        if (!delta.empty()) {
                            std::cout << delta;
                        }
                    } else if (event_type == "response_complete") {
                        // End of response - add newline
                        std::cout << std::string("\n");
                    }
                } catch (...) {}
                return sse_running;
            });

            return sse_running;
        };

        HttpResponse http_resp = sse_client.get_stream(endpoint, headers, stream_handler, nullptr);

        if (!sse_running) break;  // Clean shutdown requested

        if (!http_resp.is_success()) {
            dout(1) << std::string("WARNING: ") +"SSE connection lost, reconnecting in 2s..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }

    dout(1) << "SSE listener thread stopped" << std::endl;
}

Response CLIClientBackend::send_request(const std::string& prompt, EventCallback callback) {
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

                    if (event.contains("user_echo")) {
                        // User prompt echo - fire callback with USER type
                        if (config->streaming) {
                            callback(CallbackEvent::USER_PROMPT, event["user_echo"].get<std::string>(), "", "");
                        }
                    }
                    if (event.contains("delta")) {
                        std::string delta = event["delta"].get<std::string>();
                        accumulated += delta;
                        // Fire callback with ASSISTANT type for streaming
                        if (config->streaming) {
                            callback(CallbackEvent::CONTENT, delta, "", "");
                        }
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

void CLIClientBackend::add_message(Session& session,
                                       Message::Role role,
                                       const std::string& content,
                                       const std::string& tool_name,
                                       const std::string& tool_id,
                                       int max_tokens) {
    // Only handle USER messages - CLI server manages its own conversation
    if (role != Message::USER) {
        callback(CallbackEvent::STOP, "stop", "", "");
        return;
    }

    // Send request - uses constructor callback for streaming
    send_request(content, callback);
}

void CLIClientBackend::generate_from_session(Session& session, int max_tokens) {
    // Find the last user message
    std::string prompt;
    for (auto it = session.messages.rbegin(); it != session.messages.rend(); ++it) {
        if (it->role == Message::USER) {
            prompt = it->content;
            break;
        }
    }

    if (prompt.empty()) {
        callback(CallbackEvent::ERROR, "No user message found", "error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return;
    }

    send_request(prompt, callback);
}

int CLIClientBackend::count_message_tokens(Message::Role role,
                                           const std::string& content,
                                           const std::string& tool_name,
                                           const std::string& tool_id) {
    // CLI server manages tokens - just estimate
    return content.length() / 4;
}
