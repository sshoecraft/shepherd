#include "cli_client.h"
#include "shepherd.h"
#include "logger.h"
#include "terminal_io.h"
#include "nlohmann/json.hpp"

extern std::unique_ptr<Config> config;
extern TerminalIO tio;

CLIClientBackend::CLIClientBackend(const std::string& url)
    : Backend(0), base_url(url) {
    is_local = false;
    http_client = std::make_unique<HttpClient>();
}

void CLIClientBackend::parse_backend_config() {
    if (config && !config->api_base.empty()) {
        base_url = config->api_base;
    }
}

void CLIClientBackend::initialize(Session& session) {
    parse_backend_config();
    LOG_INFO("CLI client connecting to: " + base_url);
}

Response CLIClientBackend::send_request(const std::string& prompt, StreamCallback callback) {
    Response resp;
    std::string endpoint = base_url + "/request";

    nlohmann::json request;
    request["prompt"] = prompt;
    request["stream"] = (callback != nullptr);

    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";

    if (callback) {
        // Streaming request
        std::string accumulated;
        bool stream_complete = false;

        auto stream_handler = [&](const std::string& chunk, void* userdata) -> bool {
            // If stream is complete, stop processing
            if (stream_complete) {
                return true;  // Return true to avoid curl error, but we won't process more
            }

            // Parse SSE data lines
            std::istringstream stream(chunk);
            std::string line;

            while (std::getline(stream, line)) {
                // Remove \r if present
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }

                // Skip empty lines
                if (line.empty()) continue;

                // Parse "data: {...}" format
                if (line.substr(0, 6) == "data: ") {
                    std::string json_str = line.substr(6);
                    try {
                        nlohmann::json event = nlohmann::json::parse(json_str);

                        if (event.contains("delta")) {
                            std::string delta = event["delta"].get<std::string>();
                            accumulated += delta;
                            // Call the callback with delta
                            if (!callback(delta, accumulated, resp)) {
                                stream_complete = true;
                            }
                        }

                        if (event.contains("done") && event["done"].get<bool>()) {
                            if (event.contains("response")) {
                                resp.content = event["response"].get<std::string>();
                            } else {
                                resp.content = accumulated;
                            }
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

                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to parse SSE event: " + std::string(e.what()));
                    }
                }
            }
            // Always return true to curl to avoid "Failed writing" error
            return true;
        };

        HttpResponse http_resp = http_client->post_stream_cancellable(endpoint, request.dump(), headers, stream_handler, nullptr);

        if (!http_resp.is_success() && !resp.success) {
            resp.success = false;
            resp.code = Response::ERROR;
            resp.error = http_resp.error_message.empty() ? "HTTP request failed" : http_resp.error_message;
        }

    } else {
        // Non-streaming request
        HttpResponse http_resp = http_client->post(endpoint, request.dump(), headers);

        if (!http_resp.is_success()) {
            resp.success = false;
            resp.code = Response::ERROR;
            resp.error = http_resp.error_message.empty() ? "HTTP request failed" : http_resp.error_message;
            return resp;
        }

        try {
            nlohmann::json json_resp = nlohmann::json::parse(http_resp.body);

            if (json_resp.contains("success") && !json_resp["success"].get<bool>()) {
                resp.success = false;
                resp.code = Response::ERROR;
                resp.error = json_resp.value("error", "Unknown error");
                return resp;
            }

            resp.success = true;
            resp.code = Response::SUCCESS;
            resp.content = json_resp.value("response", "");

        } catch (const std::exception& e) {
            resp.success = false;
            resp.code = Response::ERROR;
            resp.error = std::string("JSON parse error: ") + e.what();
        }
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
                tio.write(delta.c_str(), delta.length());
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
