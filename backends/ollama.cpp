#include "ollama.h"
#include "../logger.h"
#include "../shepherd.h"
#include "../nlohmann/json.hpp"
#include "../tools/tool.h"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// Global cancellation flag (defined in main.cpp)
extern bool g_generation_cancelled;

// OllamaTokenizer implementation
OllamaTokenizer::OllamaTokenizer(const std::string& model_name)
    : model_name_(model_name) {
    LOG_DEBUG("Ollama tokenizer initialized for model: " + model_name);
}

int OllamaTokenizer::count_tokens(const std::string& text) {
    // Approximate token count (roughly 4 chars per token)
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> OllamaTokenizer::encode(const std::string& text) {
    // Placeholder implementation
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string OllamaTokenizer::decode(const std::vector<int>& tokens) {
    return "Ollama tokenizer decode not implemented";
}

std::string OllamaTokenizer::get_tokenizer_name() const {
    return "ollama-" + model_name_;
}

// OllamaBackend implementation
OllamaBackend::OllamaBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    tokenizer_ = std::make_unique<OllamaTokenizer>("llama3.1:8b"); // Default model
    LOG_DEBUG("OllamaBackend created");
}

OllamaBackend::~OllamaBackend() {
    shutdown();
}

void OllamaBackend::set_api_base(const std::string& api_base) {
    if (!api_base.empty()) {
        // Ensure it ends with /v1/chat/completions
        if (api_base.find("/v1/chat/completions") == std::string::npos) {
            if (api_base.find("/v1") == std::string::npos) {
                api_endpoint_ = api_base + "/v1/chat/completions";
            } else {
                api_endpoint_ = api_base + "/chat/completions";
            }
        } else {
            api_endpoint_ = api_base;
        }
        LOG_INFO("Ollama API endpoint set to: " + api_endpoint_);
    }
}

bool OllamaBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
    if (initialized_) {
        LOG_WARN("OllamaBackend already initialized");
        return true;
    }

    // Ollama doesn't require an API key, but we accept it for compatibility
    model_name_ = model_name.empty() ? "llama3.1:8b" : model_name;
    api_key_ = api_key.empty() ? "dummy" : api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<OllamaTokenizer>(model_name_);

    // Query the API's context size for this model
    size_t api_context_size = query_model_context_size(model_name_);
    LOG_INFO("Ollama model " + model_name_ + " API context size: " + std::to_string(api_context_size));

    // Determine final context size and auto_evict flag
    bool auto_evict;
    if (max_context_size_ == 0) {
        // User didn't specify - use API's limit
        max_context_size_ = api_context_size;
        auto_evict = false; // Rely on API 400 errors (with truncate=false)
        LOG_INFO("Using API's context size: " + std::to_string(max_context_size_) + " (auto_evict=false)");
    } else if (max_context_size_ > api_context_size) {
        // User requested more than API supports - cap at API limit
        LOG_WARN("Requested context size " + std::to_string(max_context_size_) +
                 " exceeds API limit " + std::to_string(api_context_size) +
                 ", capping at API limit");
        max_context_size_ = api_context_size;
        auto_evict = false; // Rely on API 400 errors
    } else if (max_context_size_ < api_context_size) {
        // User requested less than API supports - need proactive eviction
        auto_evict = true;
        LOG_INFO("Using user's context size: " + std::to_string(max_context_size_) +
                 " (smaller than API limit " + std::to_string(api_context_size) + ", auto_evict=true)");
    } else {
        // User's limit equals API limit - rely on API errors
        auto_evict = false;
        LOG_INFO("Using context size: " + std::to_string(max_context_size_) + " (matches API limit, auto_evict=false)");
    }

    // Create the shared context manager with final context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    context_manager_->auto_evict = auto_evict;
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens (auto_evict=" +
              std::string(auto_evict ? "true" : "false") + ")");

    LOG_INFO("OllamaBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
}

std::string OllamaBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Ollama backend not initialized");
    }

    LOG_DEBUG("Ollama generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // Proactive eviction for Ollama (since Ollama silently truncates instead of returning errors)
    if (context_manager_->auto_evict) {
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
                evict_with_estimation(estimated_tokens);
            }
        }
    }

    // Build API request JSON directly from messages
    json request;
    request["model"] = model_name_;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    request["stream"] = false;
    request["options"]["num_ctx"] = max_context_size_;  // Set context window
    request["options"]["truncate"] = false;  // Disable truncation - return error instead

    // Build tools array once on first call (lazy initialization after tools are registered)
    // MUST happen BEFORE building messages array so tools can be appended to system message
    if (!tools_built_) {
        build_tools_from_registry();
    }

    // Format tools for Ollama API if we have any (uses OpenAI format)
    if (!tools_data_.empty() && tools_json_.empty()) {
        tools_json_ = json::array();

        for (const auto& tool_info : tools_data_) {
            json tool;
            tool["type"] = "function";
            tool["function"] = json::object();
            tool["function"]["name"] = tool_info.name;
            tool["function"]["description"] = tool_info.description;

            // Parse parameters schema (should already be JSON)
            try {
                tool["function"]["parameters"] = json::parse(tool_info.parameters_schema);
            } catch (const json::exception& e) {
                LOG_DEBUG("Tool " + tool_info.name + " has no structured schema, using empty fallback");
                // If parsing fails, create a basic schema
                tool["function"]["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tools_json_.push_back(tool);
        }
        LOG_INFO("Formatted " + std::to_string(tools_json_.size()) + " tools for Ollama API");

        // TESTING: Append tools list to system message (first message in deque)
        std::deque<Message>& messages = context_manager_->get_messages();
        if (!messages.empty() && messages[0].type == Message::SYSTEM) {
            std::string tools_description = "\n\n**Available Tools:**\n";
            for (const auto& tool : tools_json_) {
                std::string name = tool["function"]["name"].get<std::string>();
                std::string desc = tool["function"]["description"].get<std::string>();
                tools_description += "- " + name + ": " + desc + "\n";
            }
            // Modify the system message content directly
            messages[0].content += tools_description;
            LOG_DEBUG("Appended " + std::to_string(tools_json_.size()) + " tools to system message");
            LOG_DEBUG("System message length now: " + std::to_string(messages[0].content.length()));
        }
    }

    // Format messages for Ollama API (AFTER tools are appended to system message)
    request["messages"] = json::array();
    const auto& messages = context_manager_->get_messages();

    for (const auto& msg : messages) {
        json message_obj;

        if (msg.type == Message::SYSTEM) {
            message_obj["role"] = "system";
            message_obj["content"] = msg.content;
        } else if (msg.type == Message::USER) {
            message_obj["role"] = "user";
            message_obj["content"] = msg.content;
        } else if (msg.type == Message::ASSISTANT) {
            message_obj["role"] = "assistant";
            message_obj["content"] = msg.content;
        } else if (msg.type == Message::TOOL) {
            // Ollama/OpenAI compatible: use user role for tool results
            message_obj["role"] = "user";
            message_obj["content"] = msg.content;
        }

        if (!message_obj.empty()) {
            request["messages"].push_back(message_obj);
        }
    }

    // Add tools to request (OpenAI format)
    if (!tools_json_.empty()) {
        request["tools"] = tools_json_;
        LOG_DEBUG("Added " + std::to_string(tools_json_.size()) + " tools to request");
    }

    // Use centralized retry logic with automatic eviction
    return generate_with_retry(
        // Lambda 1: Build request JSON from current context
        [this, max_tokens]() -> json {
            json req;
            req["model"] = model_name_;

            if (max_tokens > 0) {
                req["max_tokens"] = max_tokens;
            }

            req["stream"] = false;
            req["options"]["num_ctx"] = max_context_size_;
            req["options"]["truncate"] = false;

            // Format messages for Ollama API
            req["messages"] = json::array();
            const auto& messages = context_manager_->get_messages();

            for (const auto& msg : messages) {
                json message_obj;

                if (msg.type == Message::SYSTEM) {
                    message_obj["role"] = "system";
                    message_obj["content"] = msg.content;
                } else if (msg.type == Message::USER) {
                    message_obj["role"] = "user";
                    message_obj["content"] = msg.content;
                } else if (msg.type == Message::ASSISTANT) {
                    message_obj["role"] = "assistant";
                    message_obj["content"] = msg.content;
                } else if (msg.type == Message::TOOL) {
                    message_obj["role"] = "user";
                    message_obj["content"] = msg.content;
                }

                if (!message_obj.empty()) {
                    req["messages"].push_back(message_obj);
                }
            }

            // Add tools if available
            if (!tools_json_.empty()) {
                req["tools"] = tools_json_;
            }

            return req;
        },
        // Lambda 2: Execute API request and return response
        [this](const std::string& request_json) -> std::string {
            LOG_DEBUG("=== Full Ollama API Request ===");
            LOG_DEBUG(request_json);
            LOG_DEBUG("=== End Request ===");

            std::string response_body = make_api_request(request_json);

            if (response_body.empty()) {
                throw BackendManagerError("API request failed or returned empty response");
            }

            return parse_ollama_response(response_body);
        }
    );
}

std::string OllamaBackend::get_backend_name() const {
    return "ollama";
}

std::string OllamaBackend::get_model_name() const {
    return model_name_;
}

size_t OllamaBackend::get_max_context_size() const {
    return max_context_size_;
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
    HttpResponse response = http_client_->post(api_endpoint_, json_payload, headers);

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
    std::string base_url = api_endpoint_;
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
        std::string base_url = api_endpoint_;
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

std::string OllamaBackend::generate_from_session(const SessionContext& session, int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Ollama backend not initialized");
    }

    LOG_DEBUG("Ollama generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // Build API request JSON from SessionContext
    // Ollama uses OpenAI-compatible format
    try {
        json request;
        request["model"] = model_name_;

        if (max_tokens > 0) {
            request["max_tokens"] = max_tokens;
        }

        request["stream"] = false;
        request["options"]["num_ctx"] = max_context_size_;  // Set context window
        request["options"]["truncate"] = false;  // Disable truncation - return error instead

        // Format messages for Ollama API from SessionContext (OpenAI-compatible format)
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

        // Format tools for Ollama API from SessionContext (OpenAI-compatible format)
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
            LOG_DEBUG("Added " + std::to_string(tools_array.size()) + " tools to Ollama request from SessionContext");
        }

        // Convert to string
        std::string request_json = request.dump();

        // Log request for debugging
        if (request_json.length() <= 2000) {
            LOG_DEBUG("Full Ollama API request: " + request_json);
        } else {
            LOG_DEBUG("Ollama API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
            LOG_DEBUG("Ollama API request length: " + std::to_string(request_json.length()) + " bytes");
        }

        // Make API call
        std::string response_body = make_api_request(request_json);

        if (response_body.empty()) {
            throw BackendManagerError("Ollama API request failed or returned empty response");
        }

        LOG_DEBUG("Ollama API response: " + response_body.substr(0, 500));

        // Parse response to extract content
        std::string response = parse_ollama_response(response_body);
        if (response.empty()) {
            throw BackendManagerError("Failed to parse Ollama response");
        }

        return response;
    } catch (const json::exception& e) {
        throw BackendManagerError("JSON error in generate_from_session: " + std::string(e.what()));
    }
}
