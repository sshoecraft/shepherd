#include "api_backend.h"
#include "../logger.h"
#include "../tools/tool.h"
#include "../nlohmann/json.hpp"

// Helper function to sanitize strings for JSON by removing invalid UTF-8 bytes
// This prevents JSON parsing errors when tool results contain binary data
static std::string sanitize_for_json(const std::string& input) {
    std::string result;
    result.reserve(input.size());

    // UTF-8 replacement character: U+FFFD (�) = 0xEF 0xBF 0xBD
    const char replacement[] = "\xEF\xBF\xBD";

    for (size_t i = 0; i < input.size(); ) {
        unsigned char c = input[i];

        // ASCII (0x00-0x7F): single byte
        if (c <= 0x7F) {
            result += c;
            i++;
        }
        // 2-byte UTF-8 (0xC0-0xDF)
        else if ((c >= 0xC0 && c <= 0xDF) && (i + 1 < input.size())) {
            unsigned char c2 = input[i + 1];
            if ((c2 & 0xC0) == 0x80) {  // Valid continuation byte
                result += c;
                result += c2;
                i += 2;
            } else {
                result += replacement;
                i++;
            }
        }
        // 3-byte UTF-8 (0xE0-0xEF)
        else if ((c >= 0xE0 && c <= 0xEF) && (i + 2 < input.size())) {
            unsigned char c2 = input[i + 1];
            unsigned char c3 = input[i + 2];
            if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {  // Valid continuation bytes
                result += c;
                result += c2;
                result += c3;
                i += 3;
            } else {
                result += replacement;
                i++;
            }
        }
        // 4-byte UTF-8 (0xF0-0xF7)
        else if ((c >= 0xF0 && c <= 0xF7) && (i + 3 < input.size())) {
            unsigned char c2 = input[i + 1];
            unsigned char c3 = input[i + 2];
            unsigned char c4 = input[i + 3];
            if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80 && (c4 & 0xC0) == 0x80) {
                result += c;
                result += c2;
                result += c3;
                result += c4;
                i += 4;
            } else {
                result += replacement;
                i++;
            }
        }
        // Invalid UTF-8 byte - replace with Unicode replacement character
        else {
            result += replacement;
            i++;
        }
    }

    return result;
}

// ApiContextManager implementation
ApiContextManager::ApiContextManager(size_t max_context_tokens)
    : ContextManager(max_context_tokens) {
    LOG_DEBUG("ApiContextManager initialized");
}

std::string ApiContextManager::get_context_for_inference() {
    // Not used - API backends read messages directly
    return "";
}

int ApiContextManager::count_tokens(const std::string& text) {
    // Not used - token counts come from API responses
    // Messages are stored with token_count=0
    (void)text; // Suppress unused parameter warning
    return 0;
}

int ApiContextManager::calculate_json_overhead() const {
    // Base JSON overhead for API requests
    // This is approximate - actual overhead depends on API format
    int overhead_chars = 100; // Base structure

    for (const auto& msg : messages_) {
        overhead_chars += 50; // Per-message JSON overhead
    }

    return static_cast<int>(overhead_chars / 4.0 + 0.5);
}

// ApiBackend implementation
ApiBackend::ApiBackend(size_t max_context_tokens)
    : BackendManager(max_context_tokens) {
#ifdef ENABLE_API_BACKENDS
    // Create HTTP client for backends that want to use it
    http_client_ = std::make_unique<HttpClient>();
#endif
    // Don't create context_manager_ yet - will be created in initialize() with actual model context size
    LOG_DEBUG("ApiBackend base class constructed");
}

void ApiBackend::add_system_message(const std::string& content) {
    // Token count not needed - actual usage comes from API response
    Message system_msg(Message::SYSTEM, content, 0);
    context_manager_->add_message(system_msg);
    LOG_DEBUG("Added system message to context (API backend)");
}

void ApiBackend::add_user_message(const std::string& content) {
    // Token count not needed - actual usage comes from API response
    Message user_msg(Message::USER, content, 0);
    context_manager_->add_message(user_msg);
    LOG_DEBUG("Added user message to context (API backend)");
}

void ApiBackend::add_assistant_message(const std::string& content) {
    // Token count not needed - actual usage comes from API response
    Message assistant_msg(Message::ASSISTANT, content, 0);
    context_manager_->add_message(assistant_msg);
    LOG_DEBUG("Added assistant message to context (API backend)");
}

void ApiBackend::add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id) {
    // Sanitize content to remove invalid UTF-8 bytes that would break JSON serialization
    std::string sanitized_content = sanitize_for_json(content);

    // Token count not needed - actual usage comes from API response
    Message tool_msg(Message::TOOL, sanitized_content, 0);
    tool_msg.tool_name = tool_name;
    tool_msg.tool_call_id = tool_call_id;
    context_manager_->add_message(tool_msg);
    LOG_DEBUG("Added tool result to context (API backend): " + tool_name +
              (tool_call_id.empty() ? "" : " (id: " + tool_call_id + ")"));
}

uint32_t ApiBackend::evict_to_free_space(uint32_t tokens_needed) {
    // API backends are stateless - no KV cache eviction needed
    // Context management handles message eviction automatically
    return UINT32_MAX;
}

void ApiBackend::set_tools_from_json(const std::string& tools_json) {
    if (tools_json.empty()) {
        LOG_DEBUG("No tools provided");
        tools_built_ = true;
        return;
    }

    try {
        auto tools_array = nlohmann::json::parse(tools_json);
        if (!tools_array.is_array()) {
            LOG_ERROR("Tools JSON is not an array");
            tools_built_ = true;
            return;
        }

        tools_data_.clear();
        tools_data_.reserve(tools_array.size());

        for (const auto& tool : tools_array) {
            if (!tool.contains("function") || !tool["function"].contains("name")) {
                LOG_WARN("Tool missing function.name, skipping");
                continue;
            }

            ToolInfo info;
            info.name = tool["function"]["name"].get<std::string>();
            info.description = tool["function"].value("description", "");

            if (tool["function"].contains("parameters")) {
                info.parameters_schema = tool["function"]["parameters"].dump();
            } else {
                info.parameters_schema = "{}";
            }

            tools_data_.push_back(info);
        }

        tools_built_ = true;
        LOG_INFO("Set tools from JSON: " + std::to_string(tools_data_.size()) + " tools");
    } catch (const nlohmann::json::exception& e) {
        LOG_ERROR("Failed to parse tools JSON: " + std::string(e.what()));
        tools_built_ = true;
    }
}

void ApiBackend::build_tools_from_registry() {
    if (tools_built_) {
        return; // Already built
    }

    auto& registry = ToolRegistry::instance();
    auto tool_names = registry.list_tools();

    if (tool_names.empty()) {
        LOG_DEBUG("No tools registered in ToolRegistry");
        tools_built_ = true;
        return;
    }

    tools_data_.clear();
    tools_data_.reserve(tool_names.size());

    for (const auto& tool_name : tool_names) {
        Tool* tool_ptr = registry.get_tool(tool_name);
        if (!tool_ptr) {
            LOG_WARN("Tool registered but not found: " + tool_name);
            continue;
        }

        ToolInfo info;
        info.name = tool_name;
        info.description = tool_ptr->description();

        // Try to get structured schema first (preferred for API backends)
        auto param_defs = tool_ptr->get_parameters_schema();
        if (!param_defs.empty()) {
            // Build JSON schema from ParameterDef structs
            nlohmann::json schema;
            schema["type"] = "object";
            schema["properties"] = nlohmann::json::object();
            nlohmann::json required_fields = nlohmann::json::array();

            for (const auto& param : param_defs) {
                nlohmann::json prop;
                prop["type"] = param.type;
                if (!param.description.empty()) {
                    prop["description"] = param.description;
                }
                schema["properties"][param.name] = prop;

                if (param.required) {
                    required_fields.push_back(param.name);
                }
            }

            if (!required_fields.empty()) {
                schema["required"] = required_fields;
            }

            info.parameters_schema = schema.dump();
        } else {
            // Fall back to legacy parameters() method (should be JSON for MCP tools now)
            info.parameters_schema = tool_ptr->parameters();
        }

        tools_data_.push_back(info);
    }

    tools_built_ = true;
    LOG_INFO("Built tools data from registry: " + std::to_string(tools_data_.size()) + " tools");
}

int ApiBackend::estimate_context_tokens() const {
    if (!context_manager_) {
        return 0;
    }

    // Calculate total characters in all messages
    int total_chars = 0;
    const auto& messages = context_manager_->get_messages();

    for (const auto& msg : messages) {
        total_chars += msg.content.length();
    }

    // Add JSON overhead (already in tokens from calculate_json_overhead)
    int overhead_tokens = context_manager_->calculate_json_overhead();

    // Estimate message tokens using adaptive ratio
    int message_tokens = static_cast<int>(total_chars / chars_per_token_ + 0.5f);

    return message_tokens + overhead_tokens;
}

void ApiBackend::update_token_ratio(int total_chars, int actual_tokens) {
    if (actual_tokens <= 0 || total_chars <= 0) {
        return; // Invalid data, skip update
    }

    float measured_ratio = static_cast<float>(total_chars) / actual_tokens;

    // Exponential Moving Average: 90% old, 10% new
    // This smooths out noise while adapting to model's actual tokenization
    chars_per_token_ = 0.9f * chars_per_token_ + 0.1f * measured_ratio;

    LOG_DEBUG("Updated chars/token ratio: " + std::to_string(chars_per_token_) +
              " (measured: " + std::to_string(measured_ratio) + ")");
}
