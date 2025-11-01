#include "gemini.h"
#include "../logger.h"
#include "../shepherd.h"
#include "../nlohmann/json.hpp"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// GeminiBackend implementation
GeminiBackend::GeminiBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    LOG_DEBUG("GeminiBackend created");

    // Parse backend-specific config
    std::string backend_cfg = config.backend_config(get_backend_name());
    parse_backend_config(backend_cfg);
}

GeminiBackend::~GeminiBackend() {
    shutdown();
}

bool GeminiBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("GeminiBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("Gemini API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "gemini-pro" : model_name;
    api_key_ = api_key;

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();

    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for Gemini backend");
        return false;
    }

    // Query actual API context size
    size_t api_context_size = query_model_context_size(model_name_);
    if (api_context_size == 0) {
        // Fallback to default if query fails
        api_context_size = 32000;
        LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(api_context_size));
    } else {
        LOG_INFO("Gemini model " + model_name_ + " API context size: " + std::to_string(api_context_size));
    }

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

    LOG_INFO("GeminiBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string GeminiBackend::generate(int max_tokens) {
    // Use new architecture: build SessionContext and call base class
    SessionContext session;
    build_session_from_context(session);
    return generate_from_session(session, max_tokens);
}

std::string GeminiBackend::get_backend_name() const {
    return "gemini";
}

std::string GeminiBackend::get_model_name() const {
    return model_name_;
}

size_t GeminiBackend::get_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return context_size_;
#else
    return 4096;
#endif
}

bool GeminiBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && curl_ && !api_key_.empty();
#else
    return false;
#endif
}

void GeminiBackend::shutdown() {
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
    LOG_DEBUG("GeminiBackend shutdown complete");
}

std::string GeminiBackend::make_api_request(const std::string& json_payload) {
    // TODO: Implement actual HTTP request with CURL
    // Set headers (x-goog-api-key: api_key_, Content-Type: application/json)
    // POST to api_endpoint_/model_name_:generateContent
    // Return response body
    return "TODO: Implement HTTP request";
}

std::string GeminiBackend::make_get_request(const std::string& endpoint) {
    // TODO: Implement actual HTTP GET request with CURL
    // Set headers (x-goog-api-key: api_key_)
    // GET to https://generativelanguage.googleapis.com/v1beta + endpoint
    // Return response body
    return "TODO: Implement HTTP GET request";
}

size_t GeminiBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    if (!is_ready()) {
        LOG_ERROR("Gemini backend not ready for model query");
        return 0;
    }

    // TODO: Query Gemini API for model info using /models/{model_name}
    // For now, return known context sizes based on model name
    if (model_name.find("gemini-1.5") != std::string::npos) {
        if (model_name.find("pro") != std::string::npos) {
            return 2000000; // Gemini 1.5 Pro (2M tokens)
        } else if (model_name.find("flash") != std::string::npos) {
            return 1000000; // Gemini 1.5 Flash (1M tokens)
        }
    } else if (model_name.find("gemini-pro") != std::string::npos) {
        return 32000; // Gemini Pro (32k tokens)
    } else if (model_name.find("gemini-2") != std::string::npos) {
        return 1000000; // Gemini 2.0 Flash (1M tokens)
    }

    // Default fallback
    LOG_WARN("Unknown Gemini model: " + model_name + ", using default context size");
    return 32000;
#else
    return 32000;
#endif
}

std::string GeminiBackend::parse_gemini_response(const std::string& response_json) {
    // TODO: Parse Gemini response format
    // Extract candidates[0].content.parts[0].text
    // TODO: Also parse usageMetadata.promptTokenCount and usageMetadata.candidatesTokenCount
    // Call update_message_tokens_from_api(prompt_tokens, completion_tokens);
    return "TODO: Parse response";
}
// Old generate_from_session removed - now using base class implementation

// ========== New Architecture Methods ==========

std::string GeminiBackend::format_api_request(const SessionContext& session, int max_tokens) {
#ifdef ENABLE_API_BACKENDS
    json request;

    // System instruction (if present)
    if (!session.system_prompt.empty()) {
        request["system_instruction"] = {
            {"parts", json::array({
                {{"text", session.system_prompt}}
            })}
        };
    }

    // Build contents array (user/assistant messages)
    request["contents"] = json::array();

    for (const auto& msg : session.messages) {
        if (msg.role == "user") {
            // Check if this is a tool result
            if (!msg.tool_call_id.empty()) {
                // Gemini tool results go in function_response format
                request["contents"].push_back({
                    {"role", "user"},
                    {"parts", json::array({
                        {
                            {"function_response", {
                                {"name", msg.name},
                                {"response", {
                                    {"content", msg.content}
                                }}
                            }}
                        }
                    })}
                });
            } else {
                // Regular user message
                request["contents"].push_back({
                    {"role", "user"},
                    {"parts", json::array({
                        {{"text", msg.content}}
                    })}
                });
            }
        } else if (msg.role == "assistant") {
            // Gemini uses "model" role for assistant
            // Check if this is a tool call
            try {
                json parsed = json::parse(msg.content);
                if (parsed.contains("name") && parsed.contains("parameters")) {
                    // Format as function_call for Gemini
                    request["contents"].push_back({
                        {"role", "model"},
                        {"parts", json::array({
                            {
                                {"function_call", {
                                    {"name", parsed["name"]},
                                    {"args", parsed["parameters"]}
                                }}
                            }
                        })}
                    });
                } else {
                    // Regular text content
                    request["contents"].push_back({
                        {"role", "model"},
                        {"parts", json::array({
                            {{"text", msg.content}}
                        })}
                    });
                }
            } catch (const json::exception&) {
                // Not JSON, treat as regular text
                request["contents"].push_back({
                    {"role", "model"},
                    {"parts", json::array({
                        {{"text", msg.content}}
                    })}
                });
            }
        } else if (msg.role == "tool") {
            // Tool results in Gemini format
            request["contents"].push_back({
                {"role", "user"},
                {"parts", json::array({
                    {
                        {"function_response", {
                            {"name", msg.name},
                            {"response", {
                                {"content", msg.content}
                            }}
                        }}
                    }
                })}
            });
        }
        // Skip system messages (already handled above)
    }

    // Format tools from SessionContext for Gemini API
    if (!session.tools.empty()) {
        json function_declarations = json::array();

        for (const auto& tool_def : session.tools) {
            json function_decl;
            function_decl["name"] = tool_def.name;
            function_decl["description"] = tool_def.description;

            // Parameters are already JSON object
            if (!tool_def.parameters.empty()) {
                function_decl["parameters"] = tool_def.parameters;
            } else {
                function_decl["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            function_declarations.push_back(function_decl);
        }

        request["tools"] = json::array({
            {{"function_declarations", function_declarations}}
        });
    }

    // Generation config
    if (max_tokens > 0) {
        request["generation_config"] = {
            {"max_output_tokens", max_tokens}
        };
    }

    return request.dump();
#else
    return "";
#endif
}

int GeminiBackend::extract_tokens_to_evict(const std::string& error_message) {
    // Gemini format: "The input token count (8122182) exceeds the maximum number of tokens allowed (1048576)."

    size_t count_pos = error_message.find("input token count (");
    size_t allowed_pos = error_message.find("tokens allowed (");

    if (count_pos != std::string::npos && allowed_pos != std::string::npos) {
        try {
            // Extract actual token count
            size_t start = count_pos + 19;
            size_t end = error_message.find(")", start);
            int actual_tokens = std::stoi(error_message.substr(start, end - start));

            // Extract max allowed
            start = allowed_pos + 16;
            end = error_message.find(")", start);
            int max_tokens = std::stoi(error_message.substr(start, end - start));

            // Return EXACTLY how much to evict
            return actual_tokens - max_tokens;
        } catch (...) {}
    }

    // Can't parse - return error
    return -1;
}

ApiResponse GeminiBackend::parse_api_response(const HttpResponse& http_response) {
    ApiResponse result;
    result.raw_response = http_response.body;

#ifdef ENABLE_API_BACKENDS
    if (http_response.is_success()) {
        try {
            json j = json::parse(http_response.body);

            // Extract usage data
            if (j.contains("usageMetadata") && j["usageMetadata"].is_object()) {
                result.prompt_tokens = j["usageMetadata"].value("promptTokenCount", 0);
                result.completion_tokens = j["usageMetadata"].value("candidatesTokenCount", 0);
            }

            // Extract content from candidates[0].content.parts[0].text
            if (j.contains("candidates") && j["candidates"].is_array() && !j["candidates"].empty()) {
                auto& candidate = j["candidates"][0];
                if (candidate.contains("content") && candidate["content"].contains("parts") &&
                    candidate["content"]["parts"].is_array() && !candidate["content"]["parts"].empty()) {

                    std::string response_text;
                    for (const auto& part : candidate["content"]["parts"]) {
                        if (part.contains("text")) {
                            response_text += part["text"].get<std::string>();
                        } else if (part.contains("function_call")) {
                            // Handle tool calls
                            json tool_call;
                            tool_call["name"] = part["function_call"]["name"];
                            tool_call["parameters"] = part["function_call"]["args"];
                            response_text += "\n" + tool_call.dump() + "\n";
                        }
                    }

                    result.content = response_text;
                }
            }

            result.is_error = false;
        } catch (const json::exception& e) {
            result.is_error = true;
            result.error_code = 500;
            result.error_message = "Failed to parse Gemini response: " + std::string(e.what());
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
            // If JSON parsing fails, use raw body
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

std::map<std::string, std::string> GeminiBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
#ifdef ENABLE_API_BACKENDS
    headers["Content-Type"] = "application/json";
    headers["x-goog-api-key"] = api_key_;
#endif
    return headers;
}

std::string GeminiBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    // Gemini endpoint includes the model name: /v1beta/models/{model}:generateContent
    return api_endpoint + model_name_ + ":generateContent";
#else
    return "";
#endif
}

void GeminiBackend::parse_specific_config(const std::string& json) {
    if (json.empty() || json == "{}") {
        return;
    }

    try {
        auto j = nlohmann::json::parse(json);

        if (j.contains("api_endpoint")) {
            api_endpoint = j["api_endpoint"].get<std::string>();
            LOG_DEBUG("Gemini: Set api_endpoint = " + api_endpoint);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse Gemini-specific config: " + std::string(e.what()));
    }
}
