#include "shepherd.h"
#include "anthropic_server.h"
#include "../version.h"
#include "../session.h"
#include "backend.h"
#include "../backends/api.h"
#include "../tools/utf8_sanitizer.h"
#include "../config.h"
#include "../provider.h"
#include "../http_client.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>

using json = nlohmann::json;

// External config
extern std::unique_ptr<Config> config;

// Generate random ID for messages
static std::string generate_msg_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::string id = "msg_";
    for (int i = 0; i < 24; i++) {
        id += hex[dis(gen)];
    }
    return id;
}

// AnthropicServer implementation
AnthropicServer::AnthropicServer(const std::string& host, int port,
                                 const std::string& ssl_cert, const std::string& ssl_key,
                                 bool passthrough)
    : Server(host, port, "anthropic", ssl_cert, ssl_key), passthrough_mode(passthrough) {
    callback = [this](CallbackEvent event, const std::string& content,
                      const std::string& name, const std::string& id) -> bool {
        if (request_handler) {
            return request_handler(event, content, name, id);
        }
        return true;
    };
    if (passthrough_mode) {
        dout(1) << "Anthropic server running in passthrough mode" << std::endl;
    }
}

AnthropicServer::~AnthropicServer() {
}

void AnthropicServer::init(const FrontendFlags& flags) {
    init_tools(flags, true);
}

std::string AnthropicServer::extract_api_key(const httplib::Request& req) const {
    // Check x-api-key header first (Anthropic style)
    auto it = req.headers.find("x-api-key");
    if (it != req.headers.end()) {
        return it->second;
    }

    // Fall back to Authorization: Bearer (also accepted)
    it = req.headers.find("Authorization");
    if (it != req.headers.end()) {
        const std::string& auth = it->second;
        const std::string bearer_prefix = "Bearer ";
        if (auth.size() > bearer_prefix.size() &&
            auth.substr(0, bearer_prefix.size()) == bearer_prefix) {
            return auth.substr(bearer_prefix.size());
        }
    }
    return "";
}

nlohmann::json AnthropicServer::convert_to_anthropic_response(
    const std::string& content,
    const std::string& model,
    const std::string& stop_reason,
    int input_tokens,
    int output_tokens) {

    json response = {
        {"id", generate_msg_id()},
        {"type", "message"},
        {"role", "assistant"},
        {"model", model},
        {"content", json::array({{
            {"type", "text"},
            {"text", content}
        }})},
        {"stop_reason", stop_reason},
        {"stop_sequence", nullptr},
        {"usage", {
            {"input_tokens", input_tokens},
            {"output_tokens", output_tokens}
        }}
    };
    return response;
}

void AnthropicServer::register_endpoints() {
    // Check if current provider is an API type
    Provider* provider = get_provider(current_provider);
    if (provider && provider->is_api()) {
        is_api_provider = true;
        shared_oauth_cache = std::make_shared<SharedOAuthCache>();
        dout(1) << "API provider detected - using per-request backends" << std::endl;
    }

    // POST /v1/messages - Anthropic Messages API
    tcp_server->Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        dout(1) << "POST /v1/messages" << std::endl;

        // Per-request backend for API providers
        std::unique_ptr<Backend> request_backend;
        Backend* active_backend = nullptr;

        if (is_api_provider) {
            Provider* provider = get_provider(current_provider);
            if (provider) {
                Session temp_session;
                request_backend = provider->connect(temp_session, callback);
                if (request_backend) {
                    auto* api_backend = dynamic_cast<ApiBackend*>(request_backend.get());
                    if (api_backend && shared_oauth_cache) {
                        api_backend->set_shared_oauth_cache(shared_oauth_cache);
                    }
                    active_backend = request_backend.get();
                }
            }
            if (!active_backend) {
                res.status = 500;
                json error = {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", "Failed to create backend"}}}};
                res.set_content(error.dump(), "application/json");
                return;
            }
        } else {
            active_backend = backend.get();
        }

        requests_processed++;

        try {
            // Passthrough mode - proxy request directly to upstream API
            if (passthrough_mode) {
                // Get endpoint from backend
                std::string endpoint = active_backend->get_api_endpoint();
                if (endpoint.empty()) {
                    endpoint = "https://api.anthropic.com/v1/messages";
                }

                // Get headers from backend (includes OAuth/API key auth)
                std::map<std::string, std::string> headers = active_backend->get_api_headers();
                headers["Content-Type"] = "application/json";

                dout(1) << "Passthrough: forwarding to " << endpoint << std::endl;

                // Check if streaming
                json request_json;
                try {
                    request_json = json::parse(req.body);
                } catch (...) {
                    res.status = 400;
                    json error = {{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", "Invalid JSON"}}}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }

                bool stream = request_json.value("stream", false);

                if (stream) {
                    // Streaming passthrough - use shared_ptr so HttpClient outlives the lambda
                    auto http_client = std::make_shared<HttpClient>();
                    http_client->set_ssl_verify(false);
                    std::string body_copy = req.body;  // Copy body since req may go out of scope

                    res.set_header("Content-Type", "text/event-stream");
                    res.set_header("Cache-Control", "no-cache");
                    res.set_header("Connection", "keep-alive");

                    res.set_content_provider(
                        "text/event-stream",
                        [http_client, endpoint, body_copy, headers](size_t, httplib::DataSink& sink) {
                            HttpResponse upstream = http_client->post_stream(
                                endpoint, body_copy, headers,
                                [&sink](const std::string& chunk, void*) -> bool {
                                    return sink.write(chunk.c_str(), chunk.size());
                                },
                                nullptr
                            );
                            sink.done();
                            return true;
                        }
                    );
                } else {
                    // Non-streaming passthrough
                    HttpClient http_client;
                    http_client.set_ssl_verify(false);
                    HttpResponse upstream = http_client.post(endpoint, req.body, headers);
                    res.status = upstream.status_code;
                    res.set_content(upstream.body, "application/json");
                }
                return;
            }

            // Parse request
            json request;
            try {
                request = json::parse(utf8_sanitizer::sanitize_utf8(req.body));
                dout(5) << "Anthropic request: " << request.dump(2) << std::endl;
            } catch (const std::exception& e) {
                res.status = 400;
                json error = {{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", "Invalid JSON"}}}};
                res.set_content(error.dump(), "application/json");
                return;
            }

            // Extract parameters
            std::string model = request.value("model", backend->model_name);
            int max_tokens = request.value("max_tokens", 4096);
            bool stream = request.value("stream", false);

            // Build session from Anthropic format
            Session request_session;

            // Handle tools from request - pass through to upstream
            if (request.contains("tools") && request["tools"].is_array()) {
                for (const auto& tool : request["tools"]) {
                    Session::Tool t;
                    t.name = tool.value("name", "");
                    t.description = tool.value("description", "");
                    if (tool.contains("input_schema")) {
                        t.parameters = tool["input_schema"];
                    }
                    if (!t.name.empty()) {
                        request_session.tools.push_back(t);
                    }
                }
                dout(1) << "Loaded " << request_session.tools.size() << " tools from request" << std::endl;
            }

            // Handle system message
            if (request.contains("system")) {
                if (request["system"].is_string()) {
                    request_session.system_message = request["system"].get<std::string>();
                } else if (request["system"].is_array()) {
                    // Array of content blocks
                    std::string system_text;
                    for (const auto& block : request["system"]) {
                        if (block.contains("text")) {
                            if (!system_text.empty()) system_text += "\n";
                            system_text += block["text"].get<std::string>();
                        }
                    }
                    request_session.system_message = system_text;
                }
            }

            // Convert Anthropic messages to internal format
            if (request.contains("messages") && request["messages"].is_array()) {
                for (const auto& msg : request["messages"]) {
                    std::string role = msg.value("role", "user");
                    std::string text_content;
                    std::string tool_calls_json;
                    std::vector<std::pair<std::string, std::string>> tool_results; // id, content

                    // Handle content - can be string or array of content blocks
                    if (msg.contains("content")) {
                        if (msg["content"].is_string()) {
                            text_content = msg["content"].get<std::string>();
                        } else if (msg["content"].is_array()) {
                            json tool_uses = json::array();
                            for (const auto& block : msg["content"]) {
                                std::string block_type = block.value("type", "");
                                if (block_type == "text" && block.contains("text")) {
                                    if (!text_content.empty()) text_content += "\n";
                                    text_content += block["text"].get<std::string>();
                                } else if (block_type == "tool_use") {
                                    // Assistant message with tool_use - store for upstream
                                    json tc = {
                                        {"id", block.value("id", "")},
                                        {"function", {
                                            {"name", block.value("name", "")},
                                            {"arguments", block.contains("input") ? block["input"].dump() : "{}"}
                                        }}
                                    };
                                    tool_uses.push_back(tc);
                                } else if (block_type == "tool_result") {
                                    // User message with tool_result - collect all of them
                                    std::string tr_id = block.value("tool_use_id", "");
                                    std::string tr_content;
                                    if (block.contains("content")) {
                                        if (block["content"].is_string()) {
                                            tr_content = block["content"].get<std::string>();
                                        } else if (block["content"].is_array()) {
                                            for (const auto& inner : block["content"]) {
                                                if (inner.value("type", "") == "text") {
                                                    if (!tr_content.empty()) tr_content += "\n";
                                                    tr_content += inner.value("text", "");
                                                }
                                            }
                                        }
                                    }
                                    tool_results.push_back({tr_id, tr_content});
                                }
                            }
                            if (!tool_uses.empty()) {
                                tool_calls_json = tool_uses.dump();
                            }
                        }
                    }

                    if (role == "user") {
                        if (!tool_results.empty()) {
                            // Add a TOOL_RESPONSE message for each tool_result
                            for (const auto& [tr_id, tr_content] : tool_results) {
                                Message m(Message::TOOL_RESPONSE, tr_content);
                                m.tool_call_id = tr_id;
                                request_session.messages.push_back(m);
                            }
                        } else {
                            request_session.messages.push_back(Message(Message::USER, text_content));
                        }
                    } else if (role == "assistant") {
                        Message m(Message::ASSISTANT, text_content);
                        m.tool_calls_json = tool_calls_json;
                        request_session.messages.push_back(m);
                    }
                }
            }

            // Log request
            if (!request_session.messages.empty()) {
                std::cout << "[anthropic] " << req.remote_addr << " "
                          << request_session.messages.back().content.substr(0, 100) << std::endl;
            }

            // Handle streaming
            if (stream) {
                std::shared_ptr<Backend> stream_backend;
                if (is_api_provider && request_backend) {
                    stream_backend = std::move(request_backend);
                } else {
                    stream_backend = std::shared_ptr<Backend>(backend.get(), [](Backend*){});
                }

                auto backend_lock = active_backend->acquire_lock();

                // Prefill
                try {
                    active_backend->prefill_session(request_session);
                } catch (const std::exception& e) {
                    res.status = 400;
                    json error = {{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", e.what()}}}};
                    res.set_content(error.dump(), "application/json");
                    return;
                }

                // Set up streaming response
                res.set_header("Content-Type", "text/event-stream; charset=utf-8");
                res.set_header("Cache-Control", "no-cache");

                std::string msg_id = generate_msg_id();

                // Use shared_ptr for state that needs to survive the lambda
                auto output_tokens = std::make_shared<int>(0);

                res.set_content_provider(
                    "text/event-stream",
                    [this, stream_backend, request_session, max_tokens, msg_id, model, output_tokens, backend_lock](size_t offset, httplib::DataSink& sink) mutable {
                        // Send message_start event
                        json msg_start = {
                            {"type", "message_start"},
                            {"message", {
                                {"id", msg_id},
                                {"type", "message"},
                                {"role", "assistant"},
                                {"model", model},
                                {"content", json::array()},
                                {"stop_reason", nullptr},
                                {"stop_sequence", nullptr},
                                {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}
                            }}
                        };
                        std::string event = "event: message_start\ndata: " + msg_start.dump() + "\n\n";
                        sink.write(event.c_str(), event.size());

                        // Track state across streaming
                        auto content_block_index = std::make_shared<int>(0);
                        auto text_started = std::make_shared<bool>(false);
                        auto stop_reason = std::make_shared<std::string>("end_turn");

                        // Stream handler - emits SSE events for content and tool_use
                        stream_backend->callback = [&sink, output_tokens, content_block_index, text_started, stop_reason](
                            CallbackEvent type, const std::string& content,
                            const std::string& name, const std::string& id) -> bool {

                            if (type == CallbackEvent::CONTENT || type == CallbackEvent::CODEBLOCK) {
                                // Start text block if not started
                                if (!*text_started) {
                                    *text_started = true;
                                    json block_start = {
                                        {"type", "content_block_start"},
                                        {"index", *content_block_index},
                                        {"content_block", {{"type", "text"}, {"text", ""}}}
                                    };
                                    std::string event = "event: content_block_start\ndata: " + block_start.dump() + "\n\n";
                                    sink.write(event.c_str(), event.size());
                                }

                                (*output_tokens)++;
                                json delta = {
                                    {"type", "content_block_delta"},
                                    {"index", *content_block_index},
                                    {"delta", {{"type", "text_delta"}, {"text", content}}}
                                };
                                std::string delta_event = "event: content_block_delta\ndata: " + delta.dump() + "\n\n";
                                return sink.write(delta_event.c_str(), delta_event.size());

                            } else if (type == CallbackEvent::STOP) {
                                // content contains the finish_reason
                                if (content == "tool_calls" || content == "tool_use") {
                                    *stop_reason = "tool_use";
                                } else if (!content.empty()) {
                                    *stop_reason = content;
                                }
                            }
                            return true;
                        };

                        // Generate
                        Session mutable_session = request_session;
                        stream_backend->generate_from_session(mutable_session, max_tokens);

                        // Close text block if we started one
                        if (*text_started) {
                            json block_stop = {{"type", "content_block_stop"}, {"index", *content_block_index}};
                            event = "event: content_block_stop\ndata: " + block_stop.dump() + "\n\n";
                            sink.write(event.c_str(), event.size());
                            (*content_block_index)++;
                        }

                        // Emit tool_use blocks from accumulated_tool_calls
                        if (!stream_backend->accumulated_tool_calls.empty()) {
                            *stop_reason = "tool_use";
                            for (const auto& tc : stream_backend->accumulated_tool_calls) {
                                std::string tool_id = tc.value("id", "");
                                std::string tool_name = tc["function"].value("name", "");
                                json tool_input = json::object();
                                if (tc.contains("function") && tc["function"].contains("arguments")) {
                                    try {
                                        tool_input = json::parse(tc["function"]["arguments"].get<std::string>());
                                    } catch (...) {}
                                }

                                // Send content_block_start for tool_use
                                json tool_block_start = {
                                    {"type", "content_block_start"},
                                    {"index", *content_block_index},
                                    {"content_block", {
                                        {"type", "tool_use"},
                                        {"id", tool_id},
                                        {"name", tool_name},
                                        {"input", json::object()}
                                    }}
                                };
                                event = "event: content_block_start\ndata: " + tool_block_start.dump() + "\n\n";
                                sink.write(event.c_str(), event.size());

                                // Send input_json_delta with the full input
                                json input_delta = {
                                    {"type", "content_block_delta"},
                                    {"index", *content_block_index},
                                    {"delta", {
                                        {"type", "input_json_delta"},
                                        {"partial_json", tool_input.dump()}
                                    }}
                                };
                                event = "event: content_block_delta\ndata: " + input_delta.dump() + "\n\n";
                                sink.write(event.c_str(), event.size());

                                // Send content_block_stop
                                json tool_block_stop = {{"type", "content_block_stop"}, {"index", *content_block_index}};
                                event = "event: content_block_stop\ndata: " + tool_block_stop.dump() + "\n\n";
                                sink.write(event.c_str(), event.size());

                                (*content_block_index)++;
                                (*output_tokens)++;
                            }
                        }

                        // Send message_delta with stop_reason
                        json msg_delta = {
                            {"type", "message_delta"},
                            {"delta", {{"stop_reason", *stop_reason}, {"stop_sequence", nullptr}}},
                            {"usage", {{"output_tokens", *output_tokens}}}
                        };
                        event = "event: message_delta\ndata: " + msg_delta.dump() + "\n\n";
                        sink.write(event.c_str(), event.size());

                        // Send message_stop
                        json msg_stop = {{"type", "message_stop"}};
                        event = "event: message_stop\ndata: " + msg_stop.dump() + "\n\n";
                        sink.write(event.c_str(), event.size());

                        sink.done();
                        return false;
                    },
                    [](bool) {}
                );
            } else {
                // Non-streaming response
                auto backend_lock = active_backend->acquire_lock();

                std::string response_content;
                std::string stop_reason = "end_turn";
                int output_tokens = 0;
                int input_tokens = 0;

                auto handler = [&](CallbackEvent type, const std::string& content,
                                   const std::string&, const std::string&) -> bool {
                    if (type == CallbackEvent::CONTENT || type == CallbackEvent::CODEBLOCK) {
                        response_content += content;
                        output_tokens++;
                    } else if (type == CallbackEvent::STOP) {
                        // content contains the finish_reason
                        if (content == "tool_calls" || content == "tool_use") {
                            stop_reason = "tool_use";
                        } else if (!content.empty()) {
                            stop_reason = content;
                        }
                    }
                    return true;
                };

                if (is_api_provider && request_backend) {
                    request_backend->callback = handler;
                } else {
                    request_handler = handler;
                }

                active_backend->generate_from_session(request_session, max_tokens);

                // Capture token counts before clearing handler
                input_tokens = request_session.last_prompt_tokens;

                // Build content array - include tool_use blocks if present
                json content_array = json::array();

                // Add text content if present
                if (!response_content.empty()) {
                    content_array.push_back({{"type", "text"}, {"text", response_content}});
                }

                // Add tool_use blocks from accumulated_tool_calls
                if (!active_backend->accumulated_tool_calls.empty()) {
                    stop_reason = "tool_use";
                    for (const auto& tc : active_backend->accumulated_tool_calls) {
                        json tool_use = {
                            {"type", "tool_use"},
                            {"id", tc.value("id", "")},
                            {"name", tc["function"].value("name", "")},
                            {"input", json::object()}
                        };
                        // Parse input from function arguments
                        if (tc.contains("function") && tc["function"].contains("arguments")) {
                            std::string args = tc["function"]["arguments"].get<std::string>();
                            try {
                                tool_use["input"] = json::parse(args);
                            } catch (...) {
                                tool_use["input"] = json::object();
                            }
                        }
                        content_array.push_back(tool_use);
                    }
                    output_tokens += active_backend->accumulated_tool_calls.size();
                }

                // Clear handler before backend goes out of scope
                if (is_api_provider && request_backend) {
                    request_backend->callback = [](CallbackEvent, const std::string&,
                        const std::string&, const std::string&) { return true; };
                } else {
                    request_handler = nullptr;
                }

                // Build response
                json response = {
                    {"id", generate_msg_id()},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", model},
                    {"content", content_array},
                    {"stop_reason", stop_reason},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"input_tokens", input_tokens},
                        {"output_tokens", output_tokens}
                    }}
                };

                res.set_content(response.dump(), "application/json");

                dout(1) << "[anthropic] Response sent successfully" << std::endl;
            }

        } catch (const std::exception& e) {
            dout(1) << "[anthropic] ERROR: " << e.what() << std::endl;
            res.status = 500;
            json error = {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", e.what()}}}};
            res.set_content(error.dump(), "application/json");
        }
    });

    // GET /v1/models - List models (for compatibility)
    tcp_server->Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        std::string model_id = backend->display_name.empty() ? backend->model_name : backend->display_name;

        json response = {
            {"object", "list"},
            {"data", json::array({{
                {"id", model_id},
                {"object", "model"},
                {"created", std::time(nullptr)},
                {"owned_by", "shepherd"}
            }})}
        };

        res.set_content(response.dump(), "application/json");
    });
}
