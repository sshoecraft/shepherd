#include "ollama.h"
#include "shepherd.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

OllamaBackend::OllamaBackend(size_t context_size)
    : ApiBackend(context_size) {
    // Initialize with config values
    model_name = config->model.empty() ? "llama3.1:8b" : config->model;

    // Set API endpoint from config (api_base or default)
    if (!config->api_base.empty()) {
        api_endpoint = config->api_base;
        // Ensure it has /v1/chat/completions endpoint for OpenAI compatibility
        if (api_endpoint.find("/v1/chat/completions") == std::string::npos &&
            api_endpoint.find("/api/chat") == std::string::npos) {
            if (api_endpoint.back() != '/') {
                api_endpoint += "/";
            }
            api_endpoint += "v1/chat/completions";
        }
        LOG_INFO("Using custom Ollama endpoint: " + api_endpoint);
    }
    // else: keep default api_endpoint = "http://localhost:11434/v1/chat/completions"

    // Parse backend-specific config if available
    parse_backend_config(config->backend_config("ollama"));
}

OllamaBackend::~OllamaBackend() {
}

void OllamaBackend::initialize(Session& session) {
    // Query model context size if context_size is 0
    if (context_size == 0) {
        size_t api_context_size = query_model_context_size(model_name);
        if (api_context_size > 0) {
            context_size = api_context_size;
            LOG_INFO("Ollama model " + model_name + " context size: " + std::to_string(context_size));
        } else {
            context_size = 8192;  // Conservative default for Ollama models
            LOG_WARN("Could not query context size, using default: " + std::to_string(context_size));
        }
    }

    // Call base class initialization (calibration)
    ApiBackend::initialize(session);
}

std::string OllamaBackend::get_model_name() const {
    return model_name_;
}

size_t OllamaBackend::get_context_size() const {
    return context_size_;
}

bool OllamaBackend::is_ready() const {
    return initialized_;
}

void OllamaBackend::shutdown() {
    if (!initialized_) {
        return;
    }

    // HttpClient destructor will handle cleanup
    http_client_.reset();

    initialized_ = false;
    LOG_DEBUG("OllamaBackend shutdown complete");
}

std::string OllamaBackend::make_api_request(const std::string& json_payload) {
    if (!http_client_.get()) {
        LOG_ERROR("HTTP client not initialized");
        return "";
    }

    // Prepare headers
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    if (!api_key_.empty() && api_key_ != "dummy") {
        headers["Authorization"] = "Bearer " + api_key_;
    }

    // Make POST request
    HttpResponse response = http_client_->post(api_endpoint, json_payload, headers);

    if (!response.is_success()) {
        // Try to parse error message from response body
        std::string error_msg = "API request failed with status " + std::to_string(response.status_code);

        if (!response.body.empty()) {
            try {
                auto error_json = json::parse(response.body);
                if (error_json.contains("error")) {
                    if (error_json["error"].is_object() && error_json["error"].contains("message")) {
                        error_msg = error_json["error"]["message"].get<std::string>();
                    } else if (error_json["error"].is_string()) {
                        error_msg = error_json["error"].get<std::string>();
                    }
                }
            } catch (const json::exception&) {
                // If JSON parsing fails, use the raw response body
                error_msg += ": " + response.body.substr(0, 200);
            }
        }

        LOG_ERROR("Ollama API error (HTTP " + std::to_string(response.status_code) + "): " + error_msg);
        throw BackendManagerError(error_msg);
    }

    return response.body;
}

std::string OllamaBackend::make_get_request(const std::string& endpoint) {
    if (!http_client_.get()) {
        LOG_ERROR("HTTP client not initialized");
        return "";
    }

    // Build full URL
    std::string base_url = api_endpoint;
    // Extract base (remove /v1/chat/completions if present)
    size_t pos = base_url.find("/v1/chat/completions");
    if (pos != std::string::npos) {
        base_url = base_url.substr(0, pos);
    }
    std::string full_url = base_url + endpoint;

    // Prepare headers
    std::map<std::string, std::string> headers;
    if (!api_key_.empty() && api_key_ != "dummy") {
        headers["Authorization"] = "Bearer " + api_key_;
    }

    // Make GET request
    HttpResponse response = http_client_->get(full_url, headers);

    if (!response.is_success()) {
        LOG_ERROR("Ollama GET request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message);
        return "";
    }

    return response.body;
}

size_t OllamaBackend::query_model_context_size(const std::string& model_name) {
    const size_t DEFAULT_CONTEXT_SIZE = 32768;  // Conservative default for modern models

    // Try to query from Ollama API: POST /api/show
    try {
        if (!http_client_.get()) {
            LOG_WARN("HTTP client not initialized for Ollama, using default context size: " + std::to_string(DEFAULT_CONTEXT_SIZE));
            return DEFAULT_CONTEXT_SIZE;
        }

        json request;
        request["model"] = model_name;
        std::string request_str = request.dump();

        // Build full URL for /api/show endpoint (needs POST)
        std::string base_url = api_endpoint;
        size_t pos = base_url.find("/v1/chat/completions");
        if (pos != std::string::npos) {
            base_url = base_url.substr(0, pos);
        }
        std::string show_url = base_url + "/api/show";

        // Prepare headers
        std::map<std::string, std::string> headers;
        headers["Content-Type"] = "application/json";

        HttpResponse response = http_client_->post(show_url, request_str, headers);

        if (!response.is_success()) {
            LOG_WARN("Failed to query model info from Ollama API, using default context size: " + std::to_string(DEFAULT_CONTEXT_SIZE));
            return DEFAULT_CONTEXT_SIZE;
        }

        // Parse response to extract context length from model_info
        json model_data = json::parse(response.body);

        // Try to get context length from model_info (field name varies by model family)
        if (model_data.contains("model_info") && model_data["model_info"].is_object()) {
            auto& info = model_data["model_info"];

            // Try different possible field names for context length
            std::vector<std::string> context_fields = {
                "llama.context_length",
                "qwen2.context_length",
                "mistral.context_length",
                "gemma.context_length",
                "phi3.context_length"
            };

            for (const auto& field : context_fields) {
                if (info.contains(field) && info[field].is_number()) {
                    size_t context_len = info[field].get<size_t>();
                    LOG_INFO("Retrieved context size from Ollama API (" + field + "): " + std::to_string(context_len));
                    return context_len;
                }
            }
        }

        LOG_WARN("Context length not found in Ollama API response, using default: " + std::to_string(DEFAULT_CONTEXT_SIZE));
        return DEFAULT_CONTEXT_SIZE;

    } catch (const std::exception& e) {
        LOG_WARN("Error querying Ollama model info: " + std::string(e.what()) + ", using default context size: " + std::to_string(DEFAULT_CONTEXT_SIZE));
        return DEFAULT_CONTEXT_SIZE;
    }
}

std::string OllamaBackend::parse_ollama_response(const std::string& response_json) {
    // Parse Ollama/OpenAI compatible response format
    // Response format: {"choices":[{"message":{"content":"...", "tool_calls":[...]}}]}

    try {
        auto j = json::parse(response_json);

        // Extract usage data from API response and store for server to return
        if (j.contains("usage") && j["usage"].is_object()) {
            last_prompt_tokens_ = j["usage"].value("prompt_tokens", 0);
            last_completion_tokens_ = j["usage"].value("completion_tokens", 0);
            LOG_DEBUG("Usage from API: prompt=" + std::to_string(last_prompt_tokens_) +
                      " completion=" + std::to_string(last_completion_tokens_) +
                      " total=" + std::to_string(last_prompt_tokens_ + last_completion_tokens_));

            // Update message token counts and EMA ratio
            update_message_tokens_from_api(last_prompt_tokens_, last_completion_tokens_);
        } else {
            // Reset to 0 if no usage data
            last_prompt_tokens_ = 0;
            last_completion_tokens_ = 0;
        }

        // Validate response structure
        if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
            LOG_ERROR("Invalid Ollama response: missing choices array");
            return "";
        }

        auto message = j["choices"][0]["message"];
        std::string response_text;

        // Check for tool_calls first (takes priority when present)
        if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
            !message["tool_calls"].empty()) {

            LOG_DEBUG("Ollama response contains " + std::to_string(message["tool_calls"].size()) + " tool call(s)");

            // Process tool calls
            for (const auto& tool_call : message["tool_calls"]) {
                if (tool_call["type"] == "function") {
                    // Convert from OpenAI format to internal format expected by ToolParser
                    json internal_format;
                    internal_format["name"] = tool_call["function"]["name"];

                    // Parse arguments string to JSON object
                    std::string args_str = tool_call["function"]["arguments"];
                    internal_format["parameters"] = json::parse(args_str);

                    // Add tool call ID if present
                    if (tool_call.contains("id")) {
                        internal_format["id"] = tool_call["id"];
                    }

                    // Add on its own line (for ToolParser detection)
                    response_text += "\n" + internal_format.dump() + "\n";

                    LOG_DEBUG("Converted tool call: " + internal_format.dump());
                }
            }
        }

        // Extract text content (may be present alongside tool_calls or on its own)
        if (message.contains("content") && !message["content"].is_null()) {
            std::string content = message["content"].get<std::string>();
            if (!content.empty()) {
                response_text += content;
            }
        }

        return response_text;

    } catch (const json::exception& e) {
        LOG_ERROR("JSON parse error in Ollama response: " + std::string(e.what()));
        return "";
    }
}

// Old generate_from_session removed - now using base class implementation

// ========== New Architecture Methods ==========
// Ollama uses OpenAI-compatible format

std::string OllamaBackend::format_api_request(const SessionContext& session, int max_tokens) {
#ifdef ENABLE_API_BACKENDS
    json request;
    request["model"] = model_name_;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    // Format messages for OpenAI-compatible API
    request["messages"] = json::array();

    // Add system message first if present
    if (!session.system_prompt.empty()) {
        json system_msg;
        system_msg["role"] = "system";
        system_msg["content"] = session.system_prompt;
        request["messages"].push_back(system_msg);
    }

    // Add conversation messages
    for (const auto& msg : session.messages) {
        json message_obj;
        message_obj["role"] = msg.role;
        message_obj["content"] = msg.content;

        // Add optional fields
        if (!msg.name.empty()) {
            message_obj["name"] = msg.name;
        }
        if (!msg.tool_call_id.empty()) {
            message_obj["tool_call_id"] = msg.tool_call_id;
        }

        request["messages"].push_back(message_obj);
    }

    // Add tools if available (OpenAI format)
    if (!session.tools.empty()) {
        json tools_array = json::array();

        for (const auto& tool_def : session.tools) {
            json tool;
            tool["type"] = "function";

            json function;
            function["name"] = tool_def.name;
            function["description"] = tool_def.description;

            // Parameters are already JSON object
            if (!tool_def.parameters.empty()) {
                function["parameters"] = tool_def.parameters;
            } else {
                function["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tool["function"] = function;
            tools_array.push_back(tool);
        }

        request["tools"] = tools_array;
    }

    return request.dump();
#else
    return "";
#endif
}

int OllamaBackend::extract_tokens_to_evict(const std::string& error_message) {
    // Ollama uses OpenAI-compatible format:
    // "This model's maximum context length is 16385 tokens. However, your messages resulted in 44366 tokens."
    // Or shepherd server format: "would need 54721 tokens but limit is 32768 tokens"

    int actual_tokens = -1;
    int max_tokens = -1;

    // Try shepherd server format first: "would need X tokens but limit is Y tokens"
    size_t need_pos = error_message.find("would need ");
    size_t limit_pos = error_message.find("but limit is ");
    if (need_pos != std::string::npos && limit_pos != std::string::npos) {
        try {
            size_t start = need_pos + 11;
            size_t end = error_message.find(" tokens", start);
            actual_tokens = std::stoi(error_message.substr(start, end - start));

            start = limit_pos + 13;
            end = error_message.find(" tokens", start);
            max_tokens = std::stoi(error_message.substr(start, end - start));
        } catch (...) {}
    }

    // Try OpenAI format: "maximum context length is X tokens. However, your messages resulted in Y tokens"
    if (actual_tokens == -1 || max_tokens == -1) {
        size_t max_pos = error_message.find("maximum context length is ");
        size_t resulted_pos = error_message.find("resulted in ");
        if (max_pos != std::string::npos && resulted_pos != std::string::npos) {
            try {
                size_t start = max_pos + 26;
                size_t end = error_message.find(" tokens", start);
                max_tokens = std::stoi(error_message.substr(start, end - start));

                start = resulted_pos + 12;
                end = error_message.find(" tokens", start);
                actual_tokens = std::stoi(error_message.substr(start, end - start));
            } catch (...) {}
        }
    }

    if (actual_tokens > 0 && max_tokens > 0) {
        // Return EXACTLY how much to evict
        return actual_tokens - max_tokens;
    }

    // Can't parse - return error
    return -1;
}

ApiResponse OllamaBackend::parse_api_response(const HttpResponse& http_response) {
    ApiResponse result;
    result.raw_response = http_response.body;

#ifdef ENABLE_API_BACKENDS
    if (http_response.is_success()) {
        try {
            json j = json::parse(http_response.body);

            // Extract content (OpenAI-compatible format)
            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                auto& choice = j["choices"][0];
                if (choice.contains("message") && choice["message"].contains("content")) {
                    result.content = choice["message"]["content"];
                }

                // Check for tool calls
                if (choice["message"].contains("tool_calls")) {
                    result.tool_calls_json = choice["message"]["tool_calls"].dump();
                }
            }

            // Extract token counts
            if (j.contains("usage")) {
                auto& usage = j["usage"];
                if (usage.contains("prompt_tokens")) {
                    result.prompt_tokens = usage["prompt_tokens"];
                }
                if (usage.contains("completion_tokens")) {
                    result.completion_tokens = usage["completion_tokens"];
                }
            }

            result.is_error = false;
        } catch (const json::exception& e) {
            result.is_error = true;
            result.error_code = 500;
            result.error_message = "Failed to parse Ollama response: " + std::string(e.what());
        }
    } else {
        result.is_error = true;
        result.error_code = http_response.status_code;

        // Parse error message from response body
        try {
            json j = json::parse(http_response.body);
            if (j.contains("error")) {
                if (j["error"].is_object() && j["error"].contains("message")) {
                    result.error_message = j["error"]["message"];
                } else if (j["error"].is_string()) {
                    result.error_message = j["error"];
                }
            }
        } catch (...) {
            result.error_message = http_response.body.substr(0, 500);
        }

        // Classify error type
        if (result.error_message.find("context") != std::string::npos &&
            (result.error_message.find("limit") != std::string::npos ||
             result.error_message.find("exceed") != std::string::npos ||
             result.error_message.find("length") != std::string::npos)) {
            result.error_type = "context_overflow";
        } else if (http_response.status_code == 429) {
            result.error_type = "rate_limit";
        } else if (http_response.status_code == 401) {
            result.error_type = "auth";
        } else {
            result.error_type = "api_error";
        }
    }
#endif

    return result;
}

std::map<std::string, std::string> OllamaBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
#ifdef ENABLE_API_BACKENDS
    headers["Content-Type"] = "application/json";
    // Ollama typically doesn't require auth for local instances
    if (!api_key_.empty()) {
        headers["Authorization"] = "Bearer " + api_key_;
    }
#endif
    return headers;
}

std::string OllamaBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    return api_endpoint;
#else
    return "";
#endif
}

void OllamaBackend::parse_specific_config(const std::string& json) {
    if (json.empty() || json == "{}") {
        return;
    }

    try {
        auto j = nlohmann::json::parse(json);

        if (j.contains("api_endpoint")) {
            api_endpoint = j["api_endpoint"].get<std::string>();
            LOG_DEBUG("Ollama: Set api_endpoint = " + api_endpoint);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse Ollama-specific config: " + std::string(e.what()));
    }
}
