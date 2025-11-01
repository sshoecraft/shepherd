#include "grok.h"
#include "../logger.h"
#include "../shepherd.h"
#include "../nlohmann/json.hpp"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// GrokBackend implementation
GrokBackend::GrokBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    LOG_DEBUG("GrokBackend created");

    // Parse backend-specific config
    std::string backend_cfg = config.backend_config(get_backend_name());
    parse_backend_config(backend_cfg);
}

GrokBackend::~GrokBackend() {
    shutdown();
}

bool GrokBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("GrokBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("Grok API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "grok-1" : model_name;
    api_key_ = api_key;

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();

    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for Grok backend");
        return false;
    }

    // Query the API's context size for this model
    size_t api_context_size = query_model_context_size(model_name_);
    if (api_context_size == 0) {
        LOG_ERROR("Failed to query context size for " + model_name_);
        return false;
    }
    LOG_INFO("Grok model " + model_name_ + " API context size: " + std::to_string(api_context_size));

    // Determine final context size and auto_evict flag
    bool auto_evict;
    if (context_size_ == 0) {
        // User didn't specify - use API's limit
        context_size_ = api_context_size;
        auto_evict = false; // Rely on API 400 errors
        LOG_INFO("Using API's context size: " + std::to_string(context_size_) + " (auto_evict=false)");
    } else if (context_size_ > api_context_size) {
        // User requested more than API supports - cap at API limit
        LOG_WARN("Requested context size " + std::to_string(context_size_) +
                 " exceeds API limit " + std::to_string(api_context_size) +
                 ", capping at API limit");
        context_size_ = api_context_size;
        auto_evict = false; // Rely on API 400 errors
    } else if (context_size_ < api_context_size) {
        // User requested less than API supports - need proactive eviction
        auto_evict = true;
        LOG_INFO("Using user's context size: " + std::to_string(context_size_) +
                 " (smaller than API limit " + std::to_string(api_context_size) + ", auto_evict=true)");
    } else {
        // User's limit equals API limit - rely on API errors
        auto_evict = false;
        LOG_INFO("Using context size: " + std::to_string(context_size_) + " (matches API limit, auto_evict=false)");
    }

    // Create the shared context manager with final context size
    context_manager_ = std::make_unique<ApiContextManager>(context_size_);
    context_manager_->auto_evict = auto_evict;
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(context_size_) + " tokens (auto_evict=" +
              std::string(auto_evict ? "true" : "false") + ")");

    LOG_INFO("GrokBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string GrokBackend::generate(int max_tokens) {
    // Use new architecture: build SessionContext and call base class
    SessionContext session;
    build_session_from_context(session);
    return generate_from_session(session, max_tokens);
}

std::string GrokBackend::get_backend_name() const {
    return "grok";
}

std::string GrokBackend::get_model_name() const {
    return model_name_;
}

size_t GrokBackend::get_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return context_size_;
#else
    return 4096;
#endif
}

bool GrokBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && curl_ && !api_key_.empty();
#else
    return false;
#endif
}

void GrokBackend::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef ENABLE_API_BACKENDS
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    curl_global_cleanup();
#endif

    initialized_ = false;
    LOG_DEBUG("GrokBackend shutdown complete");
}

std::string GrokBackend::make_api_request(const std::string& json_payload) {
    // TODO: Implement actual HTTP request with CURL
    // Set headers (Authorization: Bearer api_key_, Content-Type: application/json)
    // POST to api_endpoint_
    // Return response body
    return "TODO: Implement HTTP request";
}

std::string GrokBackend::make_get_request(const std::string& endpoint) {
    // TODO: Implement actual HTTP GET request with CURL
    // Set headers (Authorization: Bearer api_key_)
    // GET to https://api.x.ai/v1 + endpoint
    // Return response body
    return "TODO: Implement HTTP GET request";
}

size_t GrokBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    if (!is_ready()) {
        LOG_ERROR("Grok backend not ready for model query");
        return 0;
    }

    // TODO: Query Grok API for model info (OpenAI-compatible /models endpoint)
    // For now, return known context sizes based on model name
    if (model_name.find("grok-1") != std::string::npos) {
        return 131072; // Grok-1 (128k tokens)
    } else if (model_name.find("grok-2") != std::string::npos) {
        return 131072; // Grok-2 (128k tokens)
    } else if (model_name.find("grok") != std::string::npos) {
        return 131072; // Default Grok model
    }

    // Default fallback
    LOG_WARN("Unknown Grok model: " + model_name + ", using default context size");
    return 131072;
#else
    return 131072;
#endif
}

std::string GrokBackend::parse_grok_response(const std::string& response_json) {
    // TODO: Parse Grok response format (OpenAI-compatible)
    // Extract choices[0].message.content
    // TODO: Also parse usage.prompt_tokens and usage.completion_tokens (OpenAI format)
    // Call update_message_tokens_from_api(prompt_tokens, completion_tokens);
    return "TODO: Parse response";
}

// Old generate_from_session removed - now using base class implementation

// ========== New Architecture Methods ==========
// Grok uses OpenAI-compatible format

std::string GrokBackend::format_api_request(const SessionContext& session, int max_tokens) {
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

int GrokBackend::extract_tokens_to_evict(const std::string& error_message) {
    // Grok uses OpenAI-compatible format:
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

ApiResponse GrokBackend::parse_api_response(const HttpResponse& http_response) {
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
            result.error_message = "Failed to parse Grok response: " + std::string(e.what());
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

std::map<std::string, std::string> GrokBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
#ifdef ENABLE_API_BACKENDS
    headers["Content-Type"] = "application/json";
    headers["Authorization"] = "Bearer " + api_key_;
#endif
    return headers;
}

std::string GrokBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    return api_endpoint;
#else
    return "";
#endif
}

void GrokBackend::parse_specific_config(const std::string& json) {
    if (json.empty() || json == "{}") {
        return;
    }

    try {
        auto j = nlohmann::json::parse(json);

        if (j.contains("api_endpoint")) {
            api_endpoint = j["api_endpoint"].get<std::string>();
            LOG_DEBUG("Grok: Set api_endpoint = " + api_endpoint);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse Grok-specific config: " + std::string(e.what()));
    }
}
