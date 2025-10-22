#include "api_backend.h"
#include "../logger.h"
#include "../tools/tool.h"
#include "../nlohmann/json.hpp"
#include "../shepherd.h"  // For g_server_mode
#include <functional>

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
    // Estimate tokens if auto_evict is enabled (user context < API context, or Grok/Ollama)
    int estimated_tokens = context_manager_->auto_evict ? estimate_message_tokens(content) : 0;

    Message system_msg(Message::SYSTEM, content, estimated_tokens);
    context_manager_->add_message(system_msg);
    LOG_DEBUG("Added system message with " + std::to_string(estimated_tokens) + " tokens");
}

void ApiBackend::add_user_message(const std::string& content) {
    // Estimate tokens if auto_evict is enabled (user context < API context, or Grok/Ollama)
    int estimated_tokens = context_manager_->auto_evict ? estimate_message_tokens(content) : 0;

    Message user_msg(Message::USER, content, estimated_tokens);
    context_manager_->add_message(user_msg);
    LOG_DEBUG("Added user message with " + std::to_string(estimated_tokens) + " tokens");
}

void ApiBackend::add_assistant_message(const std::string& content) {
    // Use actual completion tokens from API response if available
    int token_count = 0;
    if (last_completion_tokens_ > 0) {
        token_count = last_completion_tokens_;
        LOG_DEBUG("Using actual completion tokens from API: " + std::to_string(token_count));
        // Reset so we don't reuse stale values
        last_completion_tokens_ = 0;
    } else if (context_manager_->auto_evict) {
        // Fall back to estimation if auto_evict is enabled and we don't have actual tokens
        token_count = estimate_message_tokens(content);
        LOG_DEBUG("Estimating assistant message tokens: " + std::to_string(token_count));
    }
    // Otherwise token_count stays 0 (no auto_evict, rely on API errors)

    Message assistant_msg(Message::ASSISTANT, content, token_count);
    context_manager_->add_message(assistant_msg);
    LOG_DEBUG("Added assistant message with " + std::to_string(token_count) + " tokens");
}

void ApiBackend::add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id) {
    // Sanitize content to remove invalid UTF-8 bytes that would break JSON serialization
    std::string sanitized_content = sanitize_for_json(content);

    // Estimate tokens if auto_evict is enabled (user context < API context, or Grok/Ollama)
    int estimated_tokens = context_manager_->auto_evict ? estimate_message_tokens(sanitized_content) : 0;

    Message tool_msg(Message::TOOL, sanitized_content, estimated_tokens);
    tool_msg.tool_name = tool_name;
    tool_msg.tool_call_id = tool_call_id;
    context_manager_->add_message(tool_msg);
    LOG_DEBUG("Added tool result with " + std::to_string(estimated_tokens) + " tokens: " + tool_name +
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

int ApiBackend::estimate_message_tokens(const std::string& content) const {
    // Estimate tokens using current EMA ratio
    return static_cast<int>(content.length() / chars_per_token_ + 0.5f);
}

void ApiBackend::update_message_tokens_from_api(int prompt_tokens, int completion_tokens) {
    if (!context_manager_) {
        return;
    }

    auto& messages = context_manager_->get_messages();
    if (messages.empty()) {
        return;
    }

    // Calculate total chars in all messages to update EMA
    int total_chars = 0;
    for (const auto& msg : messages) {
        total_chars += msg.content.length();
    }

    // Update EMA ratio based on actual prompt tokens from API
    if (prompt_tokens > 0 && total_chars > 0) {
        update_token_ratio(total_chars, prompt_tokens);
    }

    // BASELINE/DELTA STRATEGY for accurate token tracking:
    // Turn 1 (warmup): Store baseline (System + warmup message)
    // Turn 2+: Calculate new user message tokens as delta from known messages

    if (first_call_) {
        // First API call - establish baseline
        // This is System + warmup User message, treat as single fixed block
        baseline_tokens_ = prompt_tokens;
        first_call_ = false;
        dprintf(2, "API BASELINE: established baseline_tokens_=%d\n", baseline_tokens_);
        LOG_DEBUG("Established baseline tokens: " + std::to_string(baseline_tokens_) + " (System + warmup message)");

        // Don't try to split System/User here - just use the total as baseline
        // The warmup user message token count stays as estimated (not critical)
    } else {
        // Subsequent calls - use delta calculation
        // Calculate tokens for the new user message using delta

        // Find the last user message (scan backwards)
        int last_user_index = -1;
        for (int i = static_cast<int>(messages.size()) - 1; i >= 0; i--) {
            if (messages[i].type == Message::USER) {
                last_user_index = i;
                break;
            }
        }

        if (last_user_index >= 0) {
            // Sum all known tokens except the last user message
            int known_tokens = baseline_tokens_;  // Start with System + first User (baseline)

            dprintf(5, "API DELTA: baseline=%d, messages.size=%zu, last_user_index=%d\n",
                    known_tokens, messages.size(), last_user_index);

            // Add all messages after the baseline (starting from index 2) except the current user message
            // Baseline = messages[0] (System) + messages[1] (first User)
            for (int j = 2; j < static_cast<int>(messages.size()); j++) {
                if (j != last_user_index) {
                    // Add tokens from Assistant responses and previous User messages
                    dprintf(5, "API DELTA: adding msg[%d] (%s): %d tokens\n",
                            j, messages[j].get_role().c_str(), messages[j].token_count);
                    known_tokens += messages[j].token_count;
                }
            }

            // New user message tokens = total - known
            int new_user_tokens = prompt_tokens - known_tokens;
            messages[last_user_index].token_count = new_user_tokens;

            dprintf(2, "API DELTA: prompt=%d - known=%d = new_user=%d\n",
                    prompt_tokens, known_tokens, new_user_tokens);
            LOG_DEBUG("Delta calculation: prompt_tokens=" + std::to_string(prompt_tokens) +
                      " - known=" + std::to_string(known_tokens) +
                      " = new_user_tokens=" + std::to_string(new_user_tokens));
        }
    }

    // Recalculate total tokens
    context_manager_->recalculate_total_tokens();

    LOG_DEBUG("Updated token counts from API: prompt=" + std::to_string(prompt_tokens) +
              " completion=" + std::to_string(completion_tokens) +
              " total=" + std::to_string(context_manager_->get_total_tokens()));
}

void ApiBackend::evict_with_estimation(int estimated_tokens) {
    // CRITICAL FIX FOR BUG: API backends don't get actual token counts before eviction
    // Messages have token_count=0, so evict_oldest_messages() would abort immediately
    // We need to distribute estimated tokens to messages first

    auto& messages = context_manager_->get_messages();

    // Calculate total characters
    int total_chars = 0;
    for (const auto& msg : messages) {
        total_chars += msg.content.length();
    }

    // Distribute estimated tokens proportionally based on character count
    for (auto& msg : context_manager_->get_messages()) {
        if (total_chars > 0) {
            int est_msg_tokens = static_cast<int>((msg.content.length() * estimated_tokens) / total_chars);
            msg.token_count = est_msg_tokens;
        }
    }

    // Recalculate total so eviction logic sees the correct token count
    context_manager_->recalculate_total_tokens();

    LOG_DEBUG("Updated message token counts for eviction (total: " +
              std::to_string(context_manager_->get_total_tokens()) + ")");

    // Now eviction will work correctly
    context_manager_->evict_oldest_messages();
}

std::string ApiBackend::generate_with_retry(
    std::function<nlohmann::json()> build_request_func,
    std::function<std::string(const std::string&)> execute_request_func,
    int max_retries
) {
    std::string response;
    int retry = 0;

    while (true) {
        try {
            // Build the request JSON using current context
            nlohmann::json request = build_request_func();
            std::string request_json = request.dump();

            // Log request for debugging
            if (request_json.length() <= 2000) {
                LOG_DEBUG("Full API request: " + request_json);
            } else {
                LOG_DEBUG("API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
                LOG_DEBUG("API request length: " + std::to_string(request_json.length()) + " bytes");
            }

            // Execute the API request
            response = execute_request_func(request_json);

            // Success! Break out of retry loop
            break;

        } catch (const BackendManagerError& e) {
            std::string error_msg(e.what());

            // Check if this is a context overflow error
            // 1. Check for HTTP status codes 400 (Bad Request) or 413 (Payload Too Large)
            bool is_http_400_or_413 = (error_msg.find("status 400") != std::string::npos ||
                                       error_msg.find("status 413") != std::string::npos);

            // 2. Check for context-related error message patterns
            bool has_context_keywords = (error_msg.find("context") != std::string::npos &&
                                        (error_msg.find("limit") != std::string::npos ||
                                         error_msg.find("length") != std::string::npos ||
                                         error_msg.find("exceed") != std::string::npos)) ||
                                        error_msg.find("too many tokens") != std::string::npos ||
                                        error_msg.find("maximum context") != std::string::npos ||
                                        error_msg.find("context_length_exceeded") != std::string::npos ||
                                        error_msg.find("prompt is too long") != std::string::npos ||
                                        error_msg.find("requested tokens") != std::string::npos;

            // Context error if: (HTTP 400/413 AND context keywords) OR just strong context keywords
            bool is_context_error = (is_http_400_or_413 && has_context_keywords) || has_context_keywords;

            if (is_context_error) {
                LOG_DEBUG("Detected context overflow error: " + error_msg);
                // In server mode: throw exception immediately so Python can return 400 to client
                // Client is responsible for context management
                if (g_server_mode) {
                    LOG_DEBUG("Context overflow in server mode - throwing exception to client");
                    throw;
                }

                // CLI mode: evict oldest messages and retry
                LOG_INFO("Context overflow detected, evicting oldest messages and retrying (attempt " +
                         std::to_string(retry + 1) + ")");

                // Check if we have any messages left to evict
                if (context_manager_->get_message_count() <= 1) {
                    // Can't evict any more - only system message or empty
                    LOG_ERROR("Context overflow but no more messages to evict");
                    throw BackendManagerError("Context overflow: no more messages available to evict");
                }

                // Evict oldest messages to free up space
                context_manager_->evict_oldest_messages();

                retry++;
                // Continue to next retry iteration
                // build_request_func will be called again with the reduced message list
            } else {
                // Not a context error - rethrow immediately
                throw;
            }
        }
    }

    return response;
}
