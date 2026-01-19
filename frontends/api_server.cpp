#include "shepherd.h"
#include "api_server.h"
#include "../session.h"
#include "backend.h"
#include "../backends/chat_template.h"
#include "../backends/api.h"
#include "../tools/tool_parser.h"
#include "../tools/utf8_sanitizer.h"
#include "../config.h"
#include "../provider.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>

using json = nlohmann::json;

// External config
extern std::unique_ptr<Config> config;

// APIServer class implementation
APIServer::APIServer(const std::string& host, int port, const std::string& auth_mode,
                     bool, bool)
    : Server(host, port, "api", auth_mode) {
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

void APIServer::init(bool no_mcp, bool no_tools) {
    init_tools(no_mcp, no_tools);
}

std::string APIServer::extract_bearer_token(const httplib::Request& req) const {
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return "";
    }

    const std::string& auth = it->second;
    const std::string bearer_prefix = "Bearer ";
    if (auth.size() > bearer_prefix.size() &&
        auth.substr(0, bearer_prefix.size()) == bearer_prefix) {
        return auth.substr(bearer_prefix.size());
    }
    return "";
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
    // Check if current provider is an API type (supports per-request backends)
    Provider* provider = get_provider(current_provider);
    if (provider && provider->is_api()) {
        is_api_provider_ = true;
        shared_oauth_cache_ = std::make_shared<SharedOAuthCache>();
        dout(1) << "API provider detected - using per-request backends with shared OAuth cache" << std::endl;
    }

    // POST /v1/chat/completions - OpenAI-compatible chat completions
    tcp_server.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        auto request_start_time = std::chrono::high_resolution_clock::now();

        // Per-request backend for API providers (nullptr for GPU providers)
        std::unique_ptr<Backend> request_backend;
        Backend* active_backend = nullptr;

        // Optional lock - only needed for GPU backends (shared backend)
        std::unique_ptr<std::lock_guard<std::mutex>> lock_guard;

        if (is_api_provider_) {
            // API provider - create per-request backend (no mutex needed)
            dout(1) << "POST /v1/chat/completions - creating per-request backend" << std::endl;
            Provider* provider = get_provider(current_provider);
            if (provider) {
                // Create a temporary session for backend creation
                Session temp_session;
                request_backend = provider->connect(temp_session, callback);
                if (request_backend) {
                    // Set shared OAuth cache on API backends
                    auto* api_backend = dynamic_cast<ApiBackend*>(request_backend.get());
                    if (api_backend && shared_oauth_cache_) {
                        api_backend->set_shared_oauth_cache(shared_oauth_cache_);
                    }
                    active_backend = request_backend.get();
                }
            }
            if (!active_backend) {
                res.status = 500;
                res.set_content(create_error_response(500, "Failed to create backend").dump(), "application/json");
                return;
            }
        } else {
            // GPU provider - use shared backend with mutex
            dout(1) << "POST /v1/chat/completions - acquiring lock for GPU backend" << std::endl;
            lock_guard = std::make_unique<std::lock_guard<std::mutex>>(backend_mutex);
            dout(1) << "POST /v1/chat/completions - lock acquired" << std::endl;
            active_backend = backend.get();
        }

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

            // Check authentication and route based on permissions
            std::string api_key = extract_bearer_token(req);
            bool auth_required = key_store && key_store->is_enabled();

            if (auth_required) {
                // Auth mode is json - require valid API key
                if (api_key.empty()) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "API key required").dump(), "application/json");
                    return;
                }
                if (!key_store->validate_key(api_key)) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "Invalid API key").dump(), "application/json");
                    return;
                }
            }

            // Standard OpenAI behavior - stateless request handling

            // Ensure tools array exists - some models require it for proper chat template formatting
            if (!request.contains("tools")) {
                request["tools"] = json::array();
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

                // Create Message - backend will set actual token count during prefill
                Message m(msg_role, content, 0);
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

                // Track last user/assistant message indices - token counts set by backend during prefill
                if (msg_role == Message::USER) {
                    request_session.last_user_message_index = request_session.messages.size() - 1;
                } else if (msg_role == Message::ASSISTANT) {
                    request_session.last_assistant_message_index = request_session.messages.size() - 1;
                }
            }

            // Parse parameters
            int max_tokens = 0;
            if (request.contains("max_tokens")) {
                max_tokens = request["max_tokens"];
            }

            // Parse sampling parameters from request (OpenAI-compatible + llama.cpp extensions)
            auto& sp = request_session.sampling;
            if (request.contains("temperature")) sp.temperature = request["temperature"].get<float>();
            if (request.contains("top_p")) sp.top_p = request["top_p"].get<float>();
            if (request.contains("top_k")) sp.top_k = request["top_k"].get<int>();
            if (request.contains("min_p")) sp.min_p = request["min_p"].get<float>();
            if (request.contains("typical_p")) sp.typ_p = request["typical_p"].get<float>();
            if (request.contains("top_n_sigma")) sp.top_n_sigma = request["top_n_sigma"].get<float>();
            if (request.contains("repetition_penalty")) sp.repetition_penalty = request["repetition_penalty"].get<float>();
            if (request.contains("presence_penalty")) sp.presence_penalty = request["presence_penalty"].get<float>();
            if (request.contains("frequency_penalty")) sp.frequency_penalty = request["frequency_penalty"].get<float>();
            if (request.contains("penalty_last_n")) sp.penalty_last_n = request["penalty_last_n"].get<int>();
            if (request.contains("dynatemp_range")) sp.dynatemp_range = request["dynatemp_range"].get<float>();
            if (request.contains("dynatemp_exponent")) sp.dynatemp_exponent = request["dynatemp_exponent"].get<float>();
            if (request.contains("dry_multiplier")) sp.dry_multiplier = request["dry_multiplier"].get<float>();
            if (request.contains("dry_base")) sp.dry_base = request["dry_base"].get<float>();
            if (request.contains("dry_allowed_length")) sp.dry_allowed_length = request["dry_allowed_length"].get<int>();
            if (request.contains("dry_penalty_last_n")) sp.dry_penalty_last_n = request["dry_penalty_last_n"].get<int>();
            if (request.contains("xtc_probability")) sp.xtc_probability = request["xtc_probability"].get<float>();
            if (request.contains("xtc_threshold")) sp.xtc_threshold = request["xtc_threshold"].get<float>();
            if (request.contains("mirostat")) sp.mirostat = request["mirostat"].get<int>();
            if (request.contains("mirostat_tau")) sp.mirostat_tau = request["mirostat_tau"].get<float>();
            if (request.contains("mirostat_eta")) sp.mirostat_eta = request["mirostat_eta"].get<float>();
            if (request.contains("seed")) sp.seed = request["seed"].get<uint32_t>();
            if (request.contains("min_keep")) sp.min_keep = request["min_keep"].get<int>();

            dout(1) << "Session sampling params: temp=" + std::to_string(sp.temperature) +
                     " top_p=" + std::to_string(sp.top_p) +
                     " rep_penalty=" + std::to_string(sp.repetition_penalty) +
                     " freq_penalty=" + std::to_string(sp.frequency_penalty) +
                     " pres_penalty=" + std::to_string(sp.presence_penalty) << std::endl;

            dout(1) << "Calling generate_from_session with " + std::to_string(request_session.messages.size()) +
                     " messages and " + std::to_string(request_session.tools.size()) + " tools (stream=" +
                     (stream ? "true" : "false") + ")" << std::endl;

            // Log incoming request details
            std::cout << "[api] " << request.value("model", "shepherd")
                      << " msgs=" << request_session.messages.size()
                      << ", tools=" << request_session.tools.size()
                      << ", stream=" << (stream ? "true" : "false")
                      << std::endl;

            // Handle streaming vs non-streaming
            if (stream) {
                // True streaming using set_content_provider
                std::string request_id = generate_id();
                std::string model_name = request.value("model", "shepherd");

                // Get thinking markers for streaming filter (for non-channel models)
                auto thinking_start = active_backend->get_thinking_start_markers();
                auto thinking_end = active_backend->get_thinking_end_markers();
                bool filter_thinking = !config->thinking && !thinking_start.empty();

                // Channel parsing is now handled by the backend's process_output()
                // No need for a separate channel parser here - content arrives already filtered
                const auto* caps = active_backend->get_chat_template_caps();
                bool has_channels = caps && caps->has_channels;
                if (has_channels) {
                    dout(1) << "Streaming with backend channel parsing, thinking=" << (config->thinking ? "true" : "false") << std::endl;
                }

                // For streaming, we need to keep the backend alive for the duration of the stream
                // Convert unique_ptr to shared_ptr for API providers (so lambda can own it)
                // For GPU providers, use a non-owning shared_ptr to the shared backend
                std::shared_ptr<Backend> stream_backend;
                std::mutex* mutex_ptr = nullptr;

                if (is_api_provider_ && request_backend) {
                    // API provider - transfer ownership to shared_ptr for lambda capture
                    stream_backend = std::move(request_backend);
                    // No mutex needed - each request has its own backend
                } else {
                    // GPU provider - use shared backend with mutex
                    stream_backend = std::shared_ptr<Backend>(backend.get(), [](Backend*){});  // Non-owning
                    mutex_ptr = &backend_mutex;
                }

                // Phase 1: Prefill BEFORE committing to streaming
                // This allows us to return HTTP 400 for context overflow instead of 200 + SSE error
                // Set up handler to capture prefill stats
                request_handler = [this](CallbackEvent type, const std::string& content,
                                         const std::string&, const std::string&) -> bool {
                    if (type == CallbackEvent::STATS && config->stats) {
                        std::cout << "\033[90m" << content << "\033[0m" << std::flush;
                    }
                    return true;
                };
                try {
                    active_backend->prefill_session(request_session);
                } catch (const ContextFullException& e) {
                    request_handler = nullptr;
                    // Context overflow - return proper HTTP 400 (headers not sent yet!)
                    std::cout << "[api] ERROR: Context overflow during prefill - " << e.what() << std::endl;
                    res.status = 400;
                    json error_resp = {
                        {"error", {
                            {"message", e.what()},
                            {"type", "invalid_request_error"},
                            {"code", "context_length_exceeded"}
                        }}
                    };
                    res.set_content(error_resp.dump(), "application/json");
                    return;
                } catch (const std::exception& e) {
                    request_handler = nullptr;
                    // Other errors during prefill
                    std::cout << "[api] ERROR: Prefill failed - " << e.what() << std::endl;
                    res.status = 500;
                    json error_resp = {
                        {"error", {
                            {"message", e.what()},
                            {"type", "server_error"},
                            {"code", "500"}
                        }}
                    };
                    res.set_content(error_resp.dump(), "application/json");
                    return;
                }
                request_handler = nullptr;

                // Phase 2: Prefill succeeded - now safe to commit to streaming
                res.set_header("Content-Type", "text/event-stream; charset=utf-8");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                // Use content provider for true token-by-token streaming
                res.set_content_provider(
                    "text/event-stream",
                    [this, stream_backend, mutex_ptr, request_session, max_tokens, request_id, model_name, thinking_start, thinking_end, filter_thinking, has_channels, request_start_time](size_t offset, httplib::DataSink& sink) mutable {
                        // Acquire lock for backend access only for GPU backends (shared backend)
                        // API backends have their own per-request backend, no mutex needed
                        std::unique_ptr<std::lock_guard<std::mutex>> stream_lock;
                        if (mutex_ptr) {
                            stream_lock = std::make_unique<std::lock_guard<std::mutex>>(*mutex_ptr);
                        }

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
                        bool in_code_block_for_sse = false;  // Track code block state for SSE output
                        bool in_thinking_block = false;  // Track <think> block state for channel-based models
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
                            // Handle STATS - print to server console if enabled
                            if (type == CallbackEvent::STATS) {
                                if (config->stats) {
                                    std::cout << "\033[90m" << content << "\033[0m" << std::flush;
                                }
                                return true;
                            }
                            // Only process content-type events
                            if (type != CallbackEvent::CONTENT && type != CallbackEvent::THINKING && type != CallbackEvent::CODEBLOCK) {
                                return true;
                            }

                            dout(3) << "API callback received: type=" << static_cast<int>(type) << " content=[" << content.substr(0, 50) << "] len=" << content.length() << std::endl;

                            // For channel-based models: backend's HarmonyParser separates
                            // reasoning (THINKING) from content (CONTENT). Backend only sends
                            // THINKING events when show_thinking is enabled.
                            // Wrap THINKING in <think></think> blocks so clients can parse them.
                            if (has_channels) {
                                // Handle THINKING events - wrap in <think></think>
                                if (type == CallbackEvent::THINKING) {
                                    if (!in_thinking_block) {
                                        in_thinking_block = true;
                                        if (!send_content_chunk("<think>\n")) return false;
                                    }
                                    return send_content_chunk(content);
                                }

                                // Close thinking block when switching to other event types
                                if (in_thinking_block) {
                                    in_thinking_block = false;
                                    if (!send_content_chunk("</think>\n")) return false;
                                }

                                // CODEBLOCK events need to be wrapped in ``` for SSE clients
                                if (type == CallbackEvent::CODEBLOCK) {
                                    // Track code block state to emit opening/closing ```
                                    if (!in_code_block_for_sse) {
                                        in_code_block_for_sse = true;
                                        if (!send_content_chunk("```\n")) return false;
                                    }
                                    return send_content_chunk(content);
                                } else {
                                    // Close code block if we were in one
                                    if (in_code_block_for_sse) {
                                        in_code_block_for_sse = false;
                                        if (!send_content_chunk("```\n")) return false;
                                    }
                                    return send_content_chunk(content);
                                }
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
                            dout(3) << "API stream: output_delta=[" << output_delta.substr(0, 100) << "] len=" << output_delta.length() << std::endl;
                            return send_content_chunk(output_delta);
                        };

                        std::string finish_reason = "stop";
                        try {
                            // Use generate_from_prefilled since prefill_session() was called before streaming
                            stream_backend->generate_from_prefilled(request_session, max_tokens);
                        } catch (const std::exception& e) {
                            // Errors during generation (prefill already succeeded)
                            // Note: ContextFullException should have been caught during prefill phase
                            json error_chunk = {
                                {"error", {
                                    {"message", e.what()},
                                    {"type", "server_error"},
                                    {"code", "500"}
                                }}
                            };
                            std::string error_data = "data: " + error_chunk.dump() + "\n\ndata: [DONE]\n\n";
                            sink.write(error_data.c_str(), error_data.size());
                            sink.done();
                            request_handler = nullptr;
                            return false;
                        }
                        request_handler = nullptr;  // Clear after request

                        // Check for accumulated tool calls (for channel-based models like GPT-OSS)
                        // These are captured by the channel parser during streaming
                        if (!stream_backend->accumulated_tool_calls.empty()) {
                            const auto& tc = stream_backend->accumulated_tool_calls[0];
                            std::string tc_id = tc.value("id", "call_" + std::to_string(std::time(nullptr)));
                            std::string tc_name = tc["function"].value("name", "");
                            std::string tc_args = tc["function"].value("arguments", "{}");

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
                                            {"id", tc_id},
                                            {"type", "function"},
                                            {"function", {
                                                {"name", tc_name},
                                                {"arguments", tc_args}
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
                            dout(1) << "Streaming: sent tool_call: " << tc_name << std::endl;
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

                        // Log streaming completion details
                        auto stream_end_time = std::chrono::high_resolution_clock::now();
                        auto stream_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stream_end_time - request_start_time).count();
                        std::cout << "[api] " << model_name
                                  << " stream completed: " << prompt_tokens << " prompt + " << completion_tokens << " completion = "
                                  << (prompt_tokens + completion_tokens) << " tokens"
                                  << " (" << stream_duration_ms << "ms)"
                                  << std::endl;

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
                    case CallbackEvent::CODEBLOCK:
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

            try {
                active_backend->generate_from_session(request_session, max_tokens);
            } catch (const ContextFullException& e) {
                request_handler = nullptr;
                res.status = 400;
                json error_resp = {
                    {"error", {
                        {"message", e.what()},
                        {"type", "invalid_request_error"},
                        {"code", "context_length_exceeded"}
                    }}
                };
                res.set_content(error_resp.dump(), "application/json");
                return;
            }
            request_handler = nullptr;  // Clear after request

            // Build Response from accumulated content
            Response resp;
            bool has_content = !accumulated_content.empty();
            bool has_tool_calls = !active_backend->accumulated_tool_calls.empty();
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
                auto thinking_start = active_backend->get_thinking_start_markers();
                auto thinking_end = active_backend->get_thinking_end_markers();
                auto [content, reasoning] = strip_thinking_blocks(response_content, thinking_start, thinking_end);
                response_content = content;
                reasoning_content = reasoning;
            }

            // Check for tool calls (use stripped content)
            Response stripped_resp = resp;
            stripped_resp.content = response_content;
            auto tool_call_opt = extract_tool_call(stripped_resp, active_backend);

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

            // Check for tool calls from text parsing or accumulated_tool_calls
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
            } else if (!active_backend->accumulated_tool_calls.empty()) {
                // Use accumulated_tool_calls from channel parser (already in OpenAI format)
                choice["message"]["tool_calls"] = active_backend->accumulated_tool_calls;
                choice["finish_reason"] = "tool_calls";
                dout(1) << "Using accumulated_tool_calls from channel parser" << std::endl;
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
                completion_tokens = active_backend->count_message_tokens(Message::ASSISTANT, accumulated_content, "", "");
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

            // Log completion details
            auto request_end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(request_end_time - request_start_time).count();
            std::cout << "[api] " << request.value("model", "shepherd")
                      << " completed: " << prompt_tokens << " prompt + " << completion_tokens << " completion = "
                      << (prompt_tokens + completion_tokens) << " tokens"
                      << " (" << duration_ms << "ms)"
                      << std::endl;

            res.set_content(response_body.dump(), "application/json");

        } catch (const ContextFullException& ctx_e) {
            std::cout << "[api] ERROR: Context overflow - " << ctx_e.what() << std::endl;
            res.status = 400;
            res.set_content(create_error_response(400, ctx_e.what()).dump(), "application/json");
        } catch (const std::exception& e) {
            std::cout << "[api] ERROR: " << e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/models - List available models
    tcp_server.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // Use display_name if set, otherwise fall back to model_name
            std::string model_id = backend->display_name.empty() ? backend->model_name : backend->display_name;

            json model_info = {
                {"id", model_id},
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
            // Use display_name if set, otherwise fall back to model_name
            std::string model_id = backend->display_name.empty() ? backend->model_name : backend->display_name;

            json response = {
                {"id", model_id},
                {"object", "model"},
                {"created", std::time(nullptr)},
                {"owned_by", "shepherd"},
                {"context_window", backend->context_size},
                {"backend", backend->backend_name},
                {"model_path", backend->model_name}
            };

            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/models/:model_name: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // GET /v1/tools - List available tools for MCP proxy
    tcp_server.Get("/v1/tools", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // Check authentication if enabled
            bool auth_required = key_store && key_store->is_enabled();
            if (auth_required) {
                std::string api_key = extract_bearer_token(req);
                if (api_key.empty()) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "API key required").dump(), "application/json");
                    return;
                }
                if (!key_store->validate_key(api_key)) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "Invalid API key").dump(), "application/json");
                    return;
                }
            }

            if (tools.all_tools.empty()) {
                json response = {{"tools", json::array()}};
                res.set_content(response.dump(), "application/json");
                return;
            }

            // Build tools list
            json tools_array = json::array();
            for (const auto& name : tools.list()) {
                Tool* tool = tools.get(name);
                if (tool && tools.is_enabled(name)) {
                    // Build parameters JSON schema
                    auto params = tool->get_parameters_schema();
                    json schema;
                    schema["type"] = "object";
                    schema["properties"] = json::object();
                    json required_arr = json::array();

                    for (const auto& param : params) {
                        json param_schema;
                        param_schema["type"] = param.type;
                        if (!param.description.empty()) {
                            param_schema["description"] = param.description;
                        }
                        schema["properties"][param.name] = param_schema;
                        if (param.required) {
                            required_arr.push_back(param.name);
                        }
                    }
                    if (!required_arr.empty()) {
                        schema["required"] = required_arr;
                    }

                    json tool_json = {
                        {"name", tool->name()},
                        {"description", tool->description()},
                        {"parameters", schema}
                    };
                    tools_array.push_back(tool_json);
                }
            }

            json response = {{"tools", tools_array}};
            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/tools: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });

    // POST /v1/tools/execute - Execute a tool for MCP proxy
    tcp_server.Post("/v1/tools/execute", [this](const httplib::Request& req, httplib::Response& res) {
        try {
            // Check authentication if enabled
            bool auth_required = key_store && key_store->is_enabled();
            if (auth_required) {
                std::string api_key = extract_bearer_token(req);
                if (api_key.empty()) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "API key required").dump(), "application/json");
                    return;
                }
                if (!key_store->validate_key(api_key)) {
                    res.status = 401;
                    res.set_content(create_error_response(401, "Invalid API key").dump(), "application/json");
                    return;
                }
            }

            if (tools.all_tools.empty()) {
                res.status = 400;
                res.set_content(create_error_response(400, "Tools not available").dump(), "application/json");
                return;
            }

            // Parse request
            json request;
            try {
                request = json::parse(req.body);
            } catch (...) {
                res.status = 400;
                res.set_content(create_error_response(400, "Invalid JSON").dump(), "application/json");
                return;
            }

            std::string tool_name = request.value("name", "");
            std::string tool_call_id = request.value("tool_call_id", "");

            if (tool_name.empty()) {
                res.status = 400;
                res.set_content(create_error_response(400, "Missing tool name").dump(), "application/json");
                return;
            }

            // Get arguments as JSON string
            std::string arguments_json;
            if (request.contains("arguments")) {
                if (request["arguments"].is_string()) {
                    arguments_json = request["arguments"].get<std::string>();
                } else {
                    arguments_json = request["arguments"].dump();
                }
            } else {
                arguments_json = "{}";
            }

            // Truncate params for logging
            std::string params_log = arguments_json;
            if (params_log.length() > 100) {
                params_log = params_log.substr(0, 100) + "...";
            }

            // Check if tool exists
            Tool* tool = tools.get(tool_name);
            if (!tool) {
                std::cout << "[tool] " << tool_name << ": NOT FOUND (params: " << params_log << ")" << std::endl;
                res.status = 404;
                json response = {
                    {"tool_call_id", tool_call_id},
                    {"success", false},
                    {"content", ""},
                    {"error", "Tool not found: " + tool_name}
                };
                res.set_content(response.dump(), "application/json");
                return;
            }

            // Log tool call with parameters
            std::cout << "[tool] " << tool_name << ": calling with " << params_log << std::endl;

            // Execute the tool (uses Frontend::execute_tool for truncation)
            ToolResult result = execute_tool(tools, tool_name, arguments_json, tool_call_id);

            // Log result - use summary if available, otherwise truncated content
            std::string log_summary;
            if (!result.summary.empty()) {
                log_summary = result.summary;
            } else {
                log_summary = result.content;
                if (log_summary.length() > 150) {
                    log_summary = log_summary.substr(0, 150) + "...";
                }
            }
            // Replace newlines with spaces for single-line logging
            std::replace(log_summary.begin(), log_summary.end(), '\n', ' ');

            if (result.success) {
                std::cout << "[tool] " << tool_name << ": OK - " << log_summary << std::endl;
            } else {
                std::cout << "[tool] " << tool_name << ": FAILED - " << result.error << std::endl;
            }

            // Build response
            json response = {
                {"tool_call_id", tool_call_id},
                {"success", result.success},
                {"content", result.content}
            };
            if (!result.error.empty()) {
                response["error"] = result.error;
            }

            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            std::cerr << std::string("Exception in /v1/tools/execute: ") + e.what() << std::endl;
            res.status = 500;
            res.set_content(create_error_response(500, e.what()).dump(), "application/json");
        }
    });
}
