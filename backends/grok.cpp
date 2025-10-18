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

    // Query the actual context size for this model
    size_t actual_context_size = query_model_context_size(model_name_);
    if (actual_context_size > 0) {
        max_context_size_ = actual_context_size;
        LOG_INFO("Grok model " + model_name_ + " context size: " + std::to_string(actual_context_size));
    } else {
        LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(max_context_size_));
    }

    // Create the shared context manager with actual model context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens");

    // Grok silently truncates - always use proactive eviction
    auto_evict_ = true;
    LOG_DEBUG("Grok backend: auto_evict enabled (API does not return context errors)");

    LOG_INFO("GrokBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string GrokBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Grok backend not initialized");
    }

    LOG_DEBUG("Grok generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // Proactive eviction for Grok (since Grok silently truncates instead of returning errors)
    int estimated_tokens = estimate_context_tokens();
    LOG_DEBUG("Estimated context tokens: " + std::to_string(estimated_tokens) + "/" + std::to_string(max_context_size_));

    if (estimated_tokens > static_cast<int>(max_context_size_)) {
        if (g_server_mode) {
            // Server mode: throw exception, let client handle it
            throw ContextManagerError(
                "Context limit exceeded: estimated " +
                std::to_string(estimated_tokens) + " tokens but limit is " +
                std::to_string(max_context_size_) + " tokens. Client must manage context window.");
        } else {
            // CLI mode: proactively evict to make room
            LOG_INFO("Proactively evicting messages (estimated " + std::to_string(estimated_tokens) +
                     " tokens exceeds limit of " + std::to_string(max_context_size_) + ")");
            context_manager_->evict_oldest_messages();
        }
    }

    // Get current context for API call
    std::string context_json = context_manager_->get_context_for_inference();

    // TODO: Implement actual Grok API call
    std::string response = "Grok skeleton response";

    // Return response directly - main will add it to context
    return response;
}

std::string GrokBackend::get_backend_name() const {
    return "grok";
}

std::string GrokBackend::get_model_name() const {
    return model_name_;
}

size_t GrokBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return max_context_size_;
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
    return "TODO: Parse response";
}

std::string GrokBackend::generate_from_session(const SessionContext& session, int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Grok backend not initialized");
    }

    LOG_DEBUG("Grok generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // Build API request JSON from SessionContext
    // Grok uses OpenAI-compatible format
    try {
        json request;
        request["model"] = model_name_;

        if (max_tokens > 0) {
            request["max_tokens"] = max_tokens;
        }

        // Format messages for Grok API from SessionContext (OpenAI-compatible format)
        request["messages"] = json::array();

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

        // Format tools for Grok API from SessionContext (OpenAI-compatible format)
        if (!session.tools.empty()) {
            json tools_array = json::array();

            for (const auto& tool_def : session.tools) {
                json tool;
                tool["type"] = "function";

                json function;
                function["name"] = tool_def.name;
                function["description"] = tool_def.description;

                // Parse parameters JSON or use default schema
                if (!tool_def.parameters_json.empty()) {
                    try {
                        function["parameters"] = json::parse(tool_def.parameters_json);
                    } catch (const json::exception& e) {
                        LOG_DEBUG("Tool " + tool_def.name + " has invalid JSON schema, using empty fallback");
                        function["parameters"] = {
                            {"type", "object"},
                            {"properties", json::object()},
                            {"required", json::array()}
                        };
                    }
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
            LOG_DEBUG("Added " + std::to_string(tools_array.size()) + " tools to Grok request from SessionContext");
        }

        // Convert to string
        std::string request_json = request.dump();

        // Log request for debugging
        if (request_json.length() <= 2000) {
            LOG_DEBUG("Full Grok API request: " + request_json);
        } else {
            LOG_DEBUG("Grok API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
            LOG_DEBUG("Grok API request length: " + std::to_string(request_json.length()) + " bytes");
        }

        // Make API call (currently stubbed)
        std::string response_json = make_api_request(request_json);
        if (response_json.empty()) {
            throw BackendManagerError("Grok API request failed");
        }

        LOG_DEBUG("Grok API response: " + response_json.substr(0, 500));

        // Parse response (currently stubbed, OpenAI-compatible format)
        std::string response_text = parse_grok_response(response_json);
        if (response_text.empty()) {
            throw BackendManagerError("Failed to parse Grok response");
        }

        return response_text;
    } catch (const json::exception& e) {
        throw BackendManagerError("JSON error in generate_from_session: " + std::string(e.what()));
    }
}
