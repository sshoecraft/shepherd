#include "openai.h"
#include "../logger.h"
#include "../nlohmann/json.hpp"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// OpenAITokenizer implementation
OpenAITokenizer::OpenAITokenizer(const std::string& model_name)
    : model_name_(model_name) {
    LOG_DEBUG("OpenAI tokenizer initialized for model: " + model_name);
}

int OpenAITokenizer::count_tokens(const std::string& text) {
    // TODO: Integrate tiktoken library for accurate token counting
    // For now, use the same approximation as before
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> OpenAITokenizer::encode(const std::string& text) {
    // TODO: Implement tiktoken encoding
    // This is a placeholder implementation
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string OpenAITokenizer::decode(const std::vector<int>& tokens) {
    // TODO: Implement tiktoken decoding
    // This is a placeholder implementation
    return "TODO: Implement tiktoken decode";
}

std::string OpenAITokenizer::get_tokenizer_name() const {
    return "tiktoken-" + model_name_;
}

// OpenAIBackend implementation
OpenAIBackend::OpenAIBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    tokenizer_ = std::make_unique<OpenAITokenizer>("gpt-4"); // Default model
    LOG_DEBUG("OpenAIBackend created");
}

OpenAIBackend::~OpenAIBackend() {
    shutdown();
}

void OpenAIBackend::set_api_base(const std::string& api_base) {
#ifdef ENABLE_API_BACKENDS
    if (!api_base.empty()) {
        // Ensure it ends with /chat/completions
        if (api_base.find("/chat/completions") == std::string::npos) {
            api_endpoint_ = api_base + "/chat/completions";
        } else {
            api_endpoint_ = api_base;
        }
        LOG_INFO("OpenAI API endpoint set to: " + api_endpoint_);
    }
#endif
}

bool OpenAIBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("OpenAIBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("OpenAI API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "gpt-4" : model_name;
    api_key_ = api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<OpenAITokenizer>(model_name_);

    // Only query model's context size if not explicitly set (max_context_size_ == 0 means auto)
    if (max_context_size_ == 0) {
        size_t actual_context_size = query_model_context_size(model_name_);
        if (actual_context_size > 0) {
            max_context_size_ = actual_context_size;
            LOG_INFO("OpenAI model " + model_name_ + " context size: " + std::to_string(actual_context_size));
        } else {
            // Fallback to default if query fails
            max_context_size_ = 128000;
            LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(max_context_size_));
        }
    } else {
        // Context size was explicitly set via command line - respect it
        LOG_INFO("Using explicitly configured context size: " + std::to_string(max_context_size_));
    }

    // Create the shared context manager with final context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens");

    LOG_INFO("OpenAIBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string OpenAIBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("OpenAI backend not initialized");
    }

    LOG_DEBUG("OpenAI generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // Build API request JSON directly from messages
    json request;
    request["model"] = model_name_;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    // Format messages for OpenAI API
    request["messages"] = json::array();
    const auto& messages = context_manager_->get_messages();

    for (const auto& msg : messages) {
        json message_obj;

        if (msg.type == Message::SYSTEM) {
            message_obj["role"] = "system";
            message_obj["content"] = msg.content;
            LOG_DEBUG("System message in request: " + msg.content);
        } else if (msg.type == Message::USER) {
            message_obj["role"] = "user";
            message_obj["content"] = msg.content;
        } else if (msg.type == Message::ASSISTANT) {
            message_obj["role"] = "assistant";
            message_obj["content"] = msg.content;
        } else if (msg.type == Message::TOOL) {
            message_obj["role"] = "tool";
            message_obj["content"] = msg.content;
            if (!msg.tool_call_id.empty()) {
                message_obj["tool_call_id"] = msg.tool_call_id;
            }
        }

        request["messages"].push_back(message_obj);
    }

    // Build and add tools if available
    if (!tools_built_) {
        build_tools_from_registry();
    }

    // Format tools for OpenAI API if we have any
    if (!tools_data_.empty() && tools_json_.empty()) {
        tools_json_ = json::array();

        for (const auto& tool_info : tools_data_) {
            json tool;
            tool["type"] = "function";

            json function;
            function["name"] = tool_info.name;
            function["description"] = tool_info.description;

            // Parse parameters JSON
            try {
                function["parameters"] = json::parse(tool_info.parameters_schema);
            } catch (const json::exception& e) {
                LOG_DEBUG("Tool " + tool_info.name + " has no structured schema, using empty fallback");
                // Provide a basic schema as fallback
                function["parameters"] = {
                    {"type", "object"},
                    {"properties", json::object()},
                    {"required", json::array()}
                };
            }

            tool["function"] = function;
            tools_json_.push_back(tool);
        }

        LOG_INFO("Formatted " + std::to_string(tools_json_.size()) + " tools for OpenAI API");
    }

    // Add tools to request if available
    if (!tools_json_.empty()) {
        request["tools"] = tools_json_;
        LOG_DEBUG("Added " + std::to_string(tools_json_.size()) + " tools to OpenAI request");
    }

    // Make API call
    std::string request_json = request.dump();
    if (request_json.length() <= 2000) {
        LOG_DEBUG("Full OpenAI API request: " + request_json);
    } else {
        LOG_DEBUG("OpenAI API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
        LOG_DEBUG("OpenAI API request length: " + std::to_string(request_json.length()) + " bytes");
    }
    std::string response_body = make_api_request(request_json);

    if (response_body.empty()) {
        throw BackendManagerError("API request failed or returned empty response");
    }

    // Parse response to extract content
    std::string response = parse_openai_response(response_body);

    // Return response directly - main will add it to context
    return response;
}

std::string OpenAIBackend::generate_from_session(const SessionContext& session, int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("OpenAI backend not initialized");
    }

    LOG_DEBUG("OpenAI generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // Build API request JSON from SessionContext
    json request;
    request["model"] = model_name_;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    // Format messages for OpenAI API from SessionContext
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

    // Format tools for OpenAI API from SessionContext
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
        LOG_DEBUG("Added " + std::to_string(tools_array.size()) + " tools to OpenAI request from SessionContext");
    }

    // Make API call
    std::string request_json = request.dump();
    if (request_json.length() <= 2000) {
        LOG_DEBUG("Full OpenAI API request: " + request_json);
    } else {
        LOG_DEBUG("OpenAI API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
        LOG_DEBUG("OpenAI API request length: " + std::to_string(request_json.length()) + " bytes");
    }
    std::string response_body = make_api_request(request_json);

    if (response_body.empty()) {
        throw BackendManagerError("API request failed or returned empty response");
    }

    // Parse response to extract content
    std::string response = parse_openai_response(response_body);

    return response;
}

std::string OpenAIBackend::get_backend_name() const {
    return "openai";
}

std::string OpenAIBackend::get_model_name() const {
    return model_name_;
}

size_t OpenAIBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return max_context_size_;
#else
    return 4096;
#endif
}

bool OpenAIBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && !api_key_.empty();
#else
    return false;
#endif
}

void OpenAIBackend::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef ENABLE_API_BACKENDS
    // HttpClient destructor will handle cleanup
    http_client_.reset();
#endif

    initialized_ = false;
    LOG_DEBUG("OpenAIBackend shutdown complete");
}

std::string OpenAIBackend::make_api_request(const std::string& json_payload) {
#ifdef ENABLE_API_BACKENDS
    if (!http_client_) {
        LOG_ERROR("HTTP client not initialized");
        return "";
    }

    // Prepare headers
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    headers["Authorization"] = "Bearer " + api_key_;

    // Make POST request
    HttpResponse response = http_client_->post(api_endpoint_, json_payload, headers);

    if (!response.is_success()) {
        LOG_ERROR("API request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message);
        return "";
    }

    return response.body;
#else
    return "";
#endif
}

std::string OpenAIBackend::make_get_request(const std::string& endpoint) {
#ifdef ENABLE_API_BACKENDS
    if (!http_client_) {
        LOG_ERROR("HTTP client not initialized");
        return "";
    }

    // Build full URL
    std::string base_url = api_endpoint_;
    // Extract base (remove /chat/completions if present)
    size_t pos = base_url.find("/chat/completions");
    if (pos != std::string::npos) {
        base_url = base_url.substr(0, pos);
    }
    std::string full_url = base_url + endpoint;

    // Prepare headers
    std::map<std::string, std::string> headers;
    headers["Authorization"] = "Bearer " + api_key_;

    // Make GET request
    HttpResponse response = http_client_->get(full_url, headers);

    if (!response.is_success()) {
        LOG_ERROR("GET request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message);
        return "";
    }

    return response.body;
#else
    return "";
#endif
}

size_t OpenAIBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    // Check prerequisites for making API call
    if (!http_client_ || api_key_.empty()) {
        LOG_ERROR("HTTP client or API key not available for model query");
        return 0;
    }

    // Make GET request to /models/{model_name}
    std::string endpoint = "/models/" + model_name;
    LOG_INFO("Querying model info from endpoint: " + endpoint);
    std::string response = make_get_request(endpoint);
    LOG_INFO("Model info response (" + std::to_string(response.length()) + " bytes): " +
             (response.length() > 200 ? response.substr(0, 200) + "..." : response));

    // Parse JSON response to extract context_window or context_length
    if (!response.empty()) {
        try {
            auto j = json::parse(response);

            // Try context_window first (OpenAI-compatible APIs)
            if (j.contains("context_window") && j["context_window"].is_number()) {
                size_t context_size = j["context_window"].get<size_t>();
                LOG_INFO("Parsed context_window from API: " + std::to_string(context_size));
                return context_size;
            }

            // Try context_length as fallback (official OpenAI format)
            if (j.contains("context_length") && j["context_length"].is_number()) {
                size_t context_size = j["context_length"].get<size_t>();
                LOG_INFO("Parsed context_length from API: " + std::to_string(context_size));
                return context_size;
            }

            LOG_WARN("No context_window or context_length field in API response, using fallbacks");
        } catch (const json::exception& e) {
            LOG_WARN("Failed to parse model info JSON: " + std::string(e.what()) + ", using fallbacks");
        }
    }

    // Fallback values based on known models
    if (model_name.find("gpt-4") != std::string::npos) {
        if (model_name.find("turbo") != std::string::npos || model_name.find("1106") != std::string::npos) {
            return 128000; // GPT-4 Turbo
        }
        return 8192; // GPT-4 base
    } else if (model_name.find("gpt-3.5") != std::string::npos) {
        if (model_name.find("16k") != std::string::npos) {
            return 16384; // GPT-3.5 Turbo 16k
        }
        return 4096; // GPT-3.5 Turbo base
    }

    // Default fallback
    size_t default_context = 4096;
    LOG_WARN("Unknown OpenAI model: " + model_name + ", using default context size of " + std::to_string(default_context));
    return default_context;
#else
    return 4096;
#endif
}

std::string OpenAIBackend::parse_openai_response(const std::string& response_json) {
    try {
        auto j = json::parse(response_json);

        // Check for API error
        if (j.contains("error")) {
            std::string error_msg = j["error"].contains("message") ? j["error"]["message"].get<std::string>() : "Unknown error";
            LOG_ERROR("OpenAI API error: " + error_msg);
            return "";
        }

        // Extract message from choices[0].message
        if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) {
            LOG_ERROR("No choices in API response");
            return "";
        }

        auto message = j["choices"][0]["message"];

        // Debug: Log what's in the message
        LOG_DEBUG("OpenAI message keys: " + message.dump());
        if (message.contains("tool_calls")) {
            std::string tool_calls_status = message["tool_calls"].is_null() ? "null" : "present";
            LOG_DEBUG("Message has tool_calls: " + tool_calls_status);
        }
        if (message.contains("content")) {
            std::string content_preview = message["content"].is_null() ? "null" : message["content"].get<std::string>().substr(0, 100);
            LOG_DEBUG("Message content preview: " + content_preview);
        }

        // Check for tool calls first
        if (message.contains("tool_calls") && message["tool_calls"].is_array() && !message["tool_calls"].empty()) {
            // OpenAI can return multiple tool calls, but we'll handle the first one
            auto tool_call = message["tool_calls"][0];

            json tool_call_json;

            // Check if function name exists and is not null
            if (!tool_call.contains("function") || !tool_call["function"].contains("name") ||
                tool_call["function"]["name"].is_null()) {
                LOG_ERROR("Tool call missing function name");
                return "";
            }
            tool_call_json["name"] = tool_call["function"]["name"];

            // Parse arguments string to JSON
            // Handle case where arguments might be null or empty
            if (tool_call["function"].contains("arguments") && !tool_call["function"]["arguments"].is_null()) {
                try {
                    std::string arguments_str = tool_call["function"]["arguments"].get<std::string>();
                    tool_call_json["parameters"] = json::parse(arguments_str);
                } catch (const json::exception& e) {
                    LOG_WARN("Failed to parse tool call arguments: " + std::string(e.what()));
                    tool_call_json["parameters"] = json::object();
                }
            } else {
                // No arguments or null arguments
                tool_call_json["parameters"] = json::object();
            }

            // Add tool call ID if present
            if (tool_call.contains("id")) {
                tool_call_json["id"] = tool_call["id"];
            }

            LOG_DEBUG("Detected tool call: " + tool_call_json["name"].get<std::string>());

            // Return as JSON string for ToolParser to detect
            return tool_call_json.dump();
        }

        // No tool calls, return content
        if (message.contains("content") && !message["content"].is_null()) {
            return message["content"].get<std::string>();
        }

        // If content is null and no tool calls, might be refusal
        if (message.contains("refusal") && !message["refusal"].is_null()) {
            return message["refusal"].get<std::string>();
        }

        LOG_ERROR("No content or tool_calls in API response");
        return "";

    } catch (const json::exception& e) {
        LOG_ERROR("JSON parse error in OpenAI response: " + std::string(e.what()));
        return "";
    }
}