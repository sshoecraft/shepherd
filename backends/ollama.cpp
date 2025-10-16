#include "ollama.h"
#include "../logger.h"
#include "../nlohmann/json.hpp"
#include "../tools/tool.h"
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

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
#ifdef ENABLE_API_BACKENDS
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
#endif
}

bool OllamaBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("OllamaBackend already initialized");
        return true;
    }

    // Ollama doesn't require an API key, but we accept it for compatibility
    model_name_ = model_name.empty() ? "llama3.1:8b" : model_name;
    api_key_ = api_key.empty() ? "dummy" : api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<OllamaTokenizer>(model_name_);

    // Only query model's context size if not explicitly set (max_context_size_ == 0 means auto)
    if (max_context_size_ == 0) {
        size_t actual_context_size = query_model_context_size(model_name_);
        if (actual_context_size > 0) {
            max_context_size_ = actual_context_size;
            LOG_INFO("Ollama model " + model_name_ + " context size: " + std::to_string(actual_context_size));
        } else {
            // Fallback to default if query fails
            max_context_size_ = 8192;
            LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(max_context_size_));
        }
    } else {
        // Context size was explicitly set via command line - respect it
        LOG_INFO("Using explicitly configured context size: " + std::to_string(max_context_size_));
    }

    // Create the shared context manager with final context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens");

    LOG_INFO("OllamaBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string OllamaBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Ollama backend not initialized");
    }

    LOG_DEBUG("Ollama generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // Build API request JSON directly from messages
    json request;
    request["model"] = model_name_;

    if (max_tokens > 0) {
        request["max_tokens"] = max_tokens;
    }

    request["stream"] = false;

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

    // Log full request for debugging
    std::string request_json = request.dump();
    LOG_DEBUG("=== Full Ollama API Request ===");
    LOG_DEBUG(request_json);
    LOG_DEBUG("=== End Request ===");

    // Make API call
    std::string response_body = make_api_request(request_json);

    if (response_body.empty()) {
        throw BackendManagerError("API request failed or returned empty response");
    }

    // Parse response to extract content
    std::string response = parse_ollama_response(response_body);

    // Return response directly - main will add it to context
    return response;
}

std::string OllamaBackend::get_backend_name() const {
    return "ollama";
}

std::string OllamaBackend::get_model_name() const {
    return model_name_;
}

size_t OllamaBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return max_context_size_;
#else
    return 8192;
#endif
}

bool OllamaBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_;
#else
    return false;
#endif
}

void OllamaBackend::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef ENABLE_API_BACKENDS
    // HttpClient destructor will handle cleanup
    http_client_.reset();
#endif

    initialized_ = false;
    LOG_DEBUG("OllamaBackend shutdown complete");
}

std::string OllamaBackend::make_api_request(const std::string& json_payload) {
#ifdef ENABLE_API_BACKENDS
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
        LOG_ERROR("Ollama API request failed with status " + std::to_string(response.status_code) +
                  ": " + response.error_message);
        return "";
    }

    return response.body;
#else
    return "";
#endif
}

std::string OllamaBackend::make_get_request(const std::string& endpoint) {
#ifdef ENABLE_API_BACKENDS
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
#else
    return "";
#endif
}

size_t OllamaBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    // Try to query from Ollama API
    // Ollama endpoint: GET /api/show with model name
    // For now, use reasonable defaults based on model name

    if (model_name.find("llama3.1") != std::string::npos) {
        if (model_name.find("70b") != std::string::npos) {
            return 128000; // Llama 3.1 70B
        }
        return 128000; // Llama 3.1 8B also has 128k context
    } else if (model_name.find("llama3") != std::string::npos) {
        return 8192; // Llama 3 base
    } else if (model_name.find("llama2") != std::string::npos) {
        return 4096; // Llama 2
    } else if (model_name.find("mistral") != std::string::npos) {
        return 8192; // Mistral models
    } else if (model_name.find("mixtral") != std::string::npos) {
        return 32768; // Mixtral
    } else if (model_name.find("codellama") != std::string::npos) {
        if (model_name.find("34b") != std::string::npos) {
            return 16384; // Code Llama 34B
        }
        return 4096; // Code Llama base
    }

    // Default fallback
    LOG_WARN("Unknown Ollama model: " + model_name + ", using default context size");
    return 8192;
#else
    return 8192;
#endif
}

std::string OllamaBackend::parse_ollama_response(const std::string& response_json) {
    // Parse Ollama/OpenAI compatible response format
    // Response format: {"choices":[{"message":{"content":"...", "tool_calls":[...]}}]}

    try {
        auto j = json::parse(response_json);

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
