#include "cli_client.h"
#include "shepherd.h"
#include "sse_parser.h"
#include "nlohmann/json.hpp"
#include "ansi.h"
#include <sstream>

extern std::unique_ptr<Config> config;

CLIClientBackend::CLIClientBackend(const std::string& url, Session& session, EventCallback callback)
    : Backend(0, session, callback), base_url(url) {
    is_gpu = false;  // API backend (CLI client to remote server)
    sse_handles_output = true;
    http_client = std::make_unique<HttpClient>();

    // --- Initialization ---
    parse_backend_config();
    dout(1) << "CLI client connecting to: " + base_url << std::endl;

    // Fetch session from server
    std::string endpoint = base_url + "/session";
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    if (!api_key.empty()) {
        headers["Authorization"] = "Bearer " + api_key;
    }

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
                    // Use ANSI green for user prompts (CLI ignores USER_PROMPT callback)
                    std::cout << ANSI_FG_GREEN << "> " << content << ANSI_RESET << std::endl;
                    continue;
                } else if (role == "assistant") {
                    // Check if this is a tool call message (OpenAI format)
                    if (msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty()) {
                        // Display tool calls via callback for consistent formatting
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
                            callback(CallbackEvent::TOOL_DISP, tool_args_str, tool_name, "");
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
                                callback(CallbackEvent::TOOL_DISP, tool_args_str, tool_name, "");
                                continue;
                            }
                        } catch (...) {
                            // Not valid JSON, treat as normal content
                        }
                    }
                    // Route through filter() to handle thinking blocks
                    reset_output_state();
                    filter(content.c_str(), content.length());
                    flush_output();
                    callback(CallbackEvent::STOP, "", "", "");  // Signal end of this message
                    continue;  // Skip the msg_output path
                } else if (role == "tool") {
                    // Tool result - use callback for consistent formatting
                    if (content.rfind("Error: ", 0) == 0) {
                        callback(CallbackEvent::RESULT_DISP, content, "error", "");
                    } else {
                        callback(CallbackEvent::RESULT_DISP, "Success", "", "");
                    }
                    continue;
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
    // Get API key from config (used for server authentication)
    if (config && !config->key.empty() && config->key != "none") {
        api_key = config->key;
    }
}

void CLIClientBackend::sse_listener_thread() {
    dout(1) << "SSE listener thread started" << std::endl;

    std::string endpoint = base_url + "/updates";
    std::map<std::string, std::string> headers;
    headers["Accept"] = "text/event-stream";
    if (!api_key.empty()) {
        headers["Authorization"] = "Bearer " + api_key;
    }

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
                        std::string role = event_data.value("role", "unknown");
                        std::string content = event_data.value("content", "");

                        if (role == "user") {
                            // Skip our own user messages (frontend already echoed)
                            // Show other clients' messages when we're not in a request
                            if (!request_in_progress) {
                                callback(CallbackEvent::USER_PROMPT, "> " + content + "\n", "", "");
                            }
                        }
                        // Tool and assistant messages handled by other events
                    } else if (event_type == "tool_call") {
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
                        // Display-only tool call (server executes it, not us)
                        callback(CallbackEvent::TOOL_DISP, args_str, tool_name, "");
                    } else if (event_type == "tool_result") {
                        bool success = event_data.value("success", true);
                        std::string result_str = success ? "Success" : ("Error: " + event_data.value("error", "Unknown"));
                        // Display-only tool result
                        callback(CallbackEvent::RESULT_DISP, result_str, event_data.value("tool_name", ""), "");
                    } else if (event_type == "delta") {
                        std::string delta = event_data.value("delta", "");
                        if (!delta.empty()) {
                            callback(CallbackEvent::CONTENT, delta, "", "");
                        }
                    } else if (event_type == "codeblock") {
                        std::string content = event_data.value("content", "");
                        if (!content.empty()) {
                            callback(CallbackEvent::CODEBLOCK, content, "", "");
                        }
                    } else if (event_type == "response_complete") {
                        callback(CallbackEvent::STOP, "stop", "", "");
                    } else if (event_type == "error") {
                        std::string error = event_data.value("error", "Unknown error");
                        callback(CallbackEvent::ERROR, error, "server_error", "");
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

Response CLIClientBackend::send_request(const std::string& prompt, EventCallback cb) {
    Response resp;
    std::string endpoint = base_url + "/request";

    nlohmann::json request;
    request["prompt"] = prompt;
    request["stream"] = false;  // Non-streaming - SSE /updates handles all display

    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    if (!api_key.empty()) {
        headers["Authorization"] = "Bearer " + api_key;
    }

    // Mark request in progress (SSE listener will skip our own user messages)
    request_in_progress = true;

    // Simple synchronous POST - SSE listener handles real-time display
    HttpResponse http_resp = http_client->post(endpoint, request.dump(), headers);

    if (!http_resp.is_success()) {
        request_in_progress = false;
        resp.success = false;
        resp.code = Response::ERROR;
        resp.error = http_resp.error_message.empty() ? "HTTP request failed" : http_resp.error_message;
        return resp;
    }

    try {
        nlohmann::json response_json = nlohmann::json::parse(http_resp.body);

        if (response_json.value("success", false)) {
            resp.success = true;
            resp.code = Response::SUCCESS;
            resp.content = response_json.value("response", "");
        } else {
            resp.success = false;
            resp.code = Response::ERROR;
            resp.error = response_json.value("error", "Unknown error");
        }
    } catch (const std::exception& e) {
        resp.success = false;
        resp.code = Response::ERROR;
        resp.error = std::string("Failed to parse response: ") + e.what();
    }

    request_in_progress = false;
    return resp;
}

// NOTE: add_message() removed - use Frontend::add_message_to_session() + generate_response() instead

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
