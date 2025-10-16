#include "anthropic.h"
#include "../logger.h"
#include "../nlohmann/json.hpp"
#include "../tools/tool.h"
#include <sstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cctype>

using json = nlohmann::json;

// Fallback model info table (used if docs scraping fails)
static const AnthropicModelInfo FALLBACK_MODEL_INFO[] = {
    {"claude-sonnet-4-5-20250929", 200000, 64000},
    {"claude-sonnet-4-20250514", 200000, 64000},
    {"claude-3-7-sonnet-20250219", 200000, 64000},
    {"claude-opus-4-1-20250805", 200000, 32000},
    {"claude-opus-4-20250514", 200000, 32000},
    {"claude-3-5-haiku-20241022", 200000, 8192},
    {"claude-3-haiku-20240307", 200000, 4096},
    {"claude-2", 100000, 4096},
    // Aliases
    {"claude-sonnet-4-5", 200000, 64000},
    {"claude-sonnet-4", 200000, 64000},
    {"claude-3-5-haiku-latest", 200000, 8192},
};

// Discovered models from docs scraping (shared across all functions)
static std::map<std::string, AnthropicModelInfo> g_discovered_models;

// TODO got this from claude: <system_warning>Token usage: 108153/190000; 81847 remaining</system_warning> - need to capture and display this!
/* <system_warning>Long conversation debriefing:

This conversation contains 82 messages, which is higher than average.
When responding in long conversations, Claude should stay helpful and on-topic. This may include continuing to engage on the current topic, suggesting to move in a different direction, offering to start a new conversation, etc.</system_warning> 

<system_warning>Extremely long conversation: 112 total messages (56 user/assistant pairs), well above the 82-message conversation warning.  This conversation is becoming very long. While Claude can continue to be helpful, it should be thoughtful about whether to suggest starting a fresh conversation or wrapping up, especially if: the topic has shifted significantly, the user seems frustrated, or continuing might lead to degraded performance. Claude can naturally offer to start fresh when appropriate.</system_warning> */



// AnthropicTokenizer implementation
AnthropicTokenizer::AnthropicTokenizer(const std::string& model_name)
    : model_name_(model_name) {
    LOG_DEBUG("Anthropic tokenizer initialized for model: " + model_name);
}

int AnthropicTokenizer::count_tokens(const std::string& text) {
    // TODO: Implement custom Anthropic tokenization
    // For now, use approximation (roughly 4 chars per token)
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> AnthropicTokenizer::encode(const std::string& text) {
    // TODO: Implement Anthropic encoding
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string AnthropicTokenizer::decode(const std::vector<int>& tokens) {
    // TODO: Implement Anthropic decoding
    return "TODO: Implement Anthropic decode";
}

std::string AnthropicTokenizer::get_tokenizer_name() const {
    return "anthropic-" + model_name_;
}

// AnthropicBackend implementation
AnthropicBackend::AnthropicBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    tokenizer_ = std::make_unique<AnthropicTokenizer>("claude-3-sonnet"); // Default model
    LOG_DEBUG("AnthropicBackend created");
}

AnthropicBackend::~AnthropicBackend() {
    shutdown();
}

bool AnthropicBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("AnthropicBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("Anthropic API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "claude-3-sonnet" : model_name;
    api_key_ = api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<AnthropicTokenizer>(model_name_);

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();

    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for Anthropic backend");
        return false;
    }

    // Discover API version and context size from docs
    discover_api_metadata();

    if (api_version_.empty()) {
        api_version_ = "2023-06-01";
        LOG_WARN("Could not discover API version, using fallback: " + api_version_);
    } else {
        LOG_INFO("Using Anthropic API version: " + api_version_);
    }

    // Get context size for this model (discovered or fallback)
    size_t context_size = query_model_context_size(model_name_);

    // Allow manual override for testing (use smaller of queried vs requested)
    if (max_context_size_ > 0 && max_context_size_ < context_size) {
        context_size = max_context_size_;
        LOG_WARN("Overriding context size to " + std::to_string(context_size) + " tokens for testing");
    } else {
        LOG_INFO("Anthropic model " + model_name_ + " context size: " + std::to_string(context_size));
    }

    // Create the shared context manager with actual (or overridden) context size
    context_manager_ = std::make_unique<ApiContextManager>(context_size);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(context_size) + " tokens");

    LOG_INFO("AnthropicBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string AnthropicBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Anthropic backend not initialized");
    }

    LOG_DEBUG("Anthropic generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // Calculate available tokens for generation
    int total_context_tokens = context_manager_->get_total_tokens();
    size_t context_window = context_manager_->get_max_context_tokens();
    int available_for_generation = static_cast<int>(context_window) - total_context_tokens - 100; // 100 token buffer

    // Get model-specific max output token limit
    int max_output_limit = query_max_output_tokens(model_name_);

    // Use calculated available space if max_tokens not specified
    int actual_max_tokens = (max_tokens > 0) ? max_tokens : available_for_generation;

    // Cap at both available space AND model output limit
    actual_max_tokens = std::min(actual_max_tokens, available_for_generation);
    actual_max_tokens = std::min(actual_max_tokens, max_output_limit);

    // Anthropic requires at least 1 token
    if (actual_max_tokens < 1) {
        throw BackendManagerError("No tokens available for generation (context full)");
    }

    LOG_DEBUG("Max tokens for generation: " + std::to_string(actual_max_tokens) +
              " (available: " + std::to_string(available_for_generation) +
              ", used: " + std::to_string(total_context_tokens) + "/" + std::to_string(context_window) + ")");

    // Build complete API request JSON directly from messages
    try {
        json request;

        // Required fields
        request["model"] = model_name_;
        request["max_tokens"] = actual_max_tokens;

        // Read messages from context manager and format for Anthropic API
        const auto& messages = context_manager_->get_messages();

        // Anthropic format: separate "system" field from "messages" array
        std::string system_content;
        request["messages"] = json::array();

        for (const auto& msg : messages) {
            if (msg.type == Message::SYSTEM) {
                // Store system message separately
                system_content = msg.content;
            } else if (msg.type == Message::USER) {
                request["messages"].push_back({
                    {"role", "user"},
                    {"content", msg.content}
                });
            } else if (msg.type == Message::ASSISTANT) {
                // Check if this is a tool call (JSON format)
                json assistant_content;
                try {
                    json parsed = json::parse(msg.content);
                    // If it parses as JSON with "name" and "parameters", it's a tool call
                    if (parsed.contains("name") && parsed.contains("parameters")) {
                        // Format as tool_use block for Anthropic
                        assistant_content = json::array({
                            {
                                {"type", "tool_use"},
                                {"id", parsed.value("id", "")},
                                {"name", parsed["name"]},
                                {"input", parsed["parameters"]}
                            }
                        });
                    } else {
                        // Regular text content
                        assistant_content = msg.content;
                    }
                } catch (const json::exception&) {
                    // Not JSON, treat as regular text
                    assistant_content = msg.content;
                }

                request["messages"].push_back({
                    {"role", "assistant"},
                    {"content", assistant_content}
                });
            } else if (msg.type == Message::TOOL) {
                // Tool results for Anthropic must be user messages with tool_result content blocks
                request["messages"].push_back({
                    {"role", "user"},
                    {"content", json::array({
                        {
                            {"type", "tool_result"},
                            {"tool_use_id", msg.tool_call_id},
                            {"content", msg.content}
                        }
                    })}
                });
            }
        }

        // Add system field if we have system content
        if (!system_content.empty()) {
            request["system"] = system_content;
        }

        // Build tools array once on first call (lazy initialization after tools are registered)
        if (!tools_built_) {
            build_tools_from_registry();
        }

        // Format tools for Anthropic API if we have any
        if (!tools_data_.empty() && tools_json_.empty()) {
            tools_json_ = json::array();

            for (const auto& tool_info : tools_data_) {
                json tool;
                tool["name"] = tool_info.name;
                tool["description"] = tool_info.description;

                // Parse parameters schema (should already be JSON)
                try {
                    tool["input_schema"] = json::parse(tool_info.parameters_schema);
                } catch (const json::exception& e) {
                    LOG_DEBUG("Tool " + tool_info.name + " has no structured schema, using empty fallback");
                    // If parsing fails, create a basic schema
                    tool["input_schema"] = {
                        {"type", "object"},
                        {"properties", json::object()},
                        {"required", json::array()}
                    };
                }

                tools_json_.push_back(tool);
            }
            LOG_INFO("Formatted " + std::to_string(tools_json_.size()) + " tools for Anthropic API");
        }

        // Add tools to request
        if (!tools_json_.empty()) {
            request["tools"] = tools_json_;
        }

        // Convert to string
        std::string request_json = request.dump();

        // Log full request for debugging
        LOG_DEBUG("=== Full Anthropic API Request ===");
        LOG_DEBUG(request_json);
        LOG_DEBUG("=== End Request ===");

        // Make API call
        std::string response_json = make_api_request(request_json);
        if (response_json.empty()) {
            throw BackendManagerError("Anthropic API request failed");
        }

        LOG_DEBUG("Anthropic API response: " + response_json.substr(0, 500));

        // Parse response
        std::string response_text = parse_anthropic_response(response_json);
        if (response_text.empty()) {
            throw BackendManagerError("Failed to parse Anthropic response");
        }

        return response_text;
    } catch (const json::exception& e) {
        throw BackendManagerError("JSON error in generate: " + std::string(e.what()));
    }
}

std::string AnthropicBackend::get_backend_name() const {
    return "anthropic";
}

std::string AnthropicBackend::get_model_name() const {
    return model_name_;
}

size_t AnthropicBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return context_manager_ ? context_manager_->get_max_context_tokens() : 200000;
#else
    return 4096;
#endif
}

bool AnthropicBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && curl_ && !api_key_.empty();
#else
    return false;
#endif
}

void AnthropicBackend::shutdown() {
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
    LOG_DEBUG("AnthropicBackend shutdown complete");
}

// CURL write callback to capture response
static size_t anthropic_write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total_size = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// Static helper to get model info from fallback table
const AnthropicModelInfo* AnthropicBackend::get_model_info(const std::string& model_name) {
    size_t table_size = sizeof(FALLBACK_MODEL_INFO) / sizeof(FALLBACK_MODEL_INFO[0]);
    for (size_t i = 0; i < table_size; i++) {
        if (FALLBACK_MODEL_INFO[i].model_name == model_name) {
            return &FALLBACK_MODEL_INFO[i];
        }
    }
    return nullptr;
}

std::string AnthropicBackend::make_api_request(const std::string& json_payload) {
#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        LOG_ERROR("CURL not initialized");
        return "";
    }

    std::string response_body;
    struct curl_slist* headers = nullptr;

    // Set headers
    headers = curl_slist_append(headers, ("x-api-key: " + api_key_).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, ("anthropic-version: " + api_version_).c_str());

    // Configure CURL
    curl_easy_setopt(curl_, CURLOPT_URL, api_endpoint_.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, json_payload.c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, anthropic_write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 120L); // 2 minute timeout

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Cleanup
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        LOG_ERROR("CURL request failed: " + std::string(curl_easy_strerror(res)));
        return "";
    }

    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code != 200) {
        LOG_ERROR("Anthropic API error (HTTP " + std::to_string(http_code) + "): " + response_body);
        return "";
    }

    return response_body;
#else
    LOG_ERROR("API backends not compiled in");
    return "";
#endif
}

std::string AnthropicBackend::make_get_request(const std::string& endpoint) {
#ifdef ENABLE_API_BACKENDS
    if (!curl_) {
        LOG_ERROR("CURL not initialized");
        return "";
    }

    std::string url = "https://api.anthropic.com/v1" + endpoint;
    std::string response_body;
    struct curl_slist* headers = nullptr;

    // Set headers
    headers = curl_slist_append(headers, ("x-api-key: " + api_key_).c_str());
    headers = curl_slist_append(headers, ("anthropic-version: " + api_version_).c_str());

    // Configure CURL for GET
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, anthropic_write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, 30L);

    // Perform request
    CURLcode res = curl_easy_perform(curl_);

    // Cleanup
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        LOG_ERROR("CURL GET request failed: " + std::string(curl_easy_strerror(res)));
        return "";
    }

    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code != 200) {
        LOG_ERROR("Anthropic API GET error (HTTP " + std::to_string(http_code) + "): " + response_body);
        return "";
    }

    return response_body;
#else
    LOG_ERROR("API backends not compiled in");
    return "";
#endif
}

size_t AnthropicBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    // Check discovered models first
    auto it = g_discovered_models.find(model_name);
    if (it != g_discovered_models.end()) {
        return it->second.context_window;
    }

    // Fall back to static table
    const AnthropicModelInfo* info = get_model_info(model_name);
    if (info) {
        return info->context_window;
    }

    // Ultimate fallback
    return 200000;
#else
    return 100000;
#endif
}

int AnthropicBackend::query_max_output_tokens(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    // Check discovered models first
    auto it = g_discovered_models.find(model_name);
    if (it != g_discovered_models.end()) {
        return it->second.max_output_tokens;
    }

    // Fall back to static table
    const AnthropicModelInfo* info = get_model_info(model_name);
    if (info) {
        return info->max_output_tokens;
    }

    // Ultimate fallback
    return 64000;
#else
    return 4096;
#endif
}

std::string AnthropicBackend::parse_anthropic_response(const std::string& response_json) {
    try {
        auto j = json::parse(response_json);

        // Check for API error
        if (j.contains("error")) {
            std::string error_msg = j["error"].contains("message") ? j["error"]["message"].get<std::string>() : "Unknown error";
            LOG_ERROR("Anthropic API error: " + error_msg);
            return "";
        }

        // Extract content - handle both text and tool_use blocks
        if (j.contains("content") && j["content"].is_array() && !j["content"].empty()) {
            std::string response_text;

            // Process all content blocks
            for (const auto& content_block : j["content"]) {
                if (!content_block.contains("type")) continue;

                std::string block_type = content_block["type"];

                if (block_type == "text" && content_block.contains("text")) {
                    // Append text content
                    response_text += content_block["text"].get<std::string>();
                    response_text += "\n";
                } else if (block_type == "tool_use") {
                    // Convert tool_use block to JSON format that ToolParser expects
                    json tool_call;
                    tool_call["name"] = content_block["name"];
                    tool_call["parameters"] = content_block["input"];
                    tool_call["id"] = content_block["id"];  // Anthropic's tool_use_id

                    // Add as JSON on its own line for ToolParser to detect
                    response_text += "\n" + tool_call.dump() + "\n";
                }
            }

            return response_text;
        }

        LOG_ERROR("Failed to extract text from Anthropic response: " + response_json.substr(0, 200));
        return "";
    } catch (const json::exception& e) {
        LOG_ERROR("JSON parse error in Anthropic response: " + std::string(e.what()));
        return "";
    }
}

void AnthropicBackend::discover_api_metadata() {
#ifdef ENABLE_API_BACKENDS
    static bool attempted_discovery = false;

    if (attempted_discovery) {
        return; // Already tried
    }
    attempted_discovery = true;

    CURL* temp_curl = curl_easy_init();
    if (!temp_curl) {
        LOG_DEBUG("Could not initialize CURL for metadata discovery");
        return;
    }

    // Fetch API version
    std::string response;
    curl_easy_setopt(temp_curl, CURLOPT_URL, "https://docs.claude.com/en/api/versioning");
    curl_easy_setopt(temp_curl, CURLOPT_WRITEFUNCTION, anthropic_write_callback);
    curl_easy_setopt(temp_curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(temp_curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(temp_curl, CURLOPT_FOLLOWLOCATION, 1L);

    if (curl_easy_perform(temp_curl) == CURLE_OK) {
        std::vector<std::string> versions;
        size_t pos = 0;
        while ((pos = response.find("anthropic-version", pos)) != std::string::npos) {
            size_t search_end = std::min(pos + 50, response.length());
            for (size_t i = pos; i < search_end - 10; i++) {
                if (response.substr(i, 2) == "20") {
                    std::string candidate = response.substr(i, 10);
                    if (candidate.length() == 10 && candidate[4] == '-' && candidate[7] == '-' &&
                        std::isdigit(candidate[0]) && std::isdigit(candidate[1]) &&
                        std::isdigit(candidate[2]) && std::isdigit(candidate[3]) &&
                        std::isdigit(candidate[5]) && std::isdigit(candidate[6]) &&
                        std::isdigit(candidate[8]) && std::isdigit(candidate[9])) {
                        if (std::find(versions.begin(), versions.end(), candidate) == versions.end()) {
                            versions.push_back(candidate);
                        }
                        break;
                    }
                }
            }
            pos++;
        }
        if (!versions.empty()) {
            std::sort(versions.begin(), versions.end(), std::greater<std::string>());
            api_version_ = versions[0];
            LOG_DEBUG("Discovered API version: " + api_version_);
        }
    }

    // Fetch model info
    response.clear();
    curl_easy_setopt(temp_curl, CURLOPT_URL, "https://docs.claude.com/en/docs/about-claude/models");
    curl_easy_setopt(temp_curl, CURLOPT_WRITEDATA, &response);

    if (curl_easy_perform(temp_curl) == CURLE_OK) {
        // Parse model names, context windows, and max output tokens
        // Look for patterns like: claude-sonnet-4-5-20250929 ... 200K ... 64,000 tokens
        size_t pos = 0;
        while ((pos = response.find("claude-", pos)) != std::string::npos) {
            // Extract model name
            size_t name_end = pos;
            while (name_end < response.length() && response[name_end] != ' ' &&
                   response[name_end] != '"' && response[name_end] != '`' &&
                   response[name_end] != '<' && response[name_end] != '\n') {
                name_end++;
            }
            std::string model_name = response.substr(pos, name_end - pos);

            // Look for context window and max output in next 500 chars
            size_t search_end = std::min(pos + 500, response.length());
            std::string context_str = response.substr(pos, search_end - pos);

            size_t context_window = 0;
            int max_output = 0;

            // Parse context window (look for "200K")
            if (context_str.find("200K") != std::string::npos || context_str.find("200,000") != std::string::npos) {
                context_window = 200000;
            } else if (context_str.find("100K") != std::string::npos || context_str.find("100,000") != std::string::npos) {
                context_window = 100000;
            }

            // Parse max output (look for "64,000 tokens" or "64K")
            if (context_str.find("64,000") != std::string::npos || context_str.find("64K") != std::string::npos) {
                max_output = 64000;
            } else if (context_str.find("32,000") != std::string::npos || context_str.find("32K") != std::string::npos) {
                max_output = 32000;
            } else if (context_str.find("8,192") != std::string::npos) {
                max_output = 8192;
            } else if (context_str.find("4,096") != std::string::npos) {
                max_output = 4096;
            }

            if (context_window > 0 && max_output > 0) {
                g_discovered_models[model_name] = {model_name, context_window, max_output};
            }

            pos = name_end + 1;
        }

        if (!g_discovered_models.empty()) {
            LOG_DEBUG("Discovered " + std::to_string(g_discovered_models.size()) + " models from docs");
        }
    }

    curl_easy_cleanup(temp_curl);
#endif
}

std::string AnthropicBackend::generate_from_session(const SessionContext& session, int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Anthropic backend not initialized");
    }

    LOG_DEBUG("Anthropic generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // Use configured max output limit if max_tokens not specified
    int actual_max_tokens = (max_tokens > 0) ? max_tokens : query_max_output_tokens(model_name_);

    LOG_DEBUG("Max tokens for generation: " + std::to_string(actual_max_tokens));

    // Build complete API request JSON from SessionContext
    try {
        json request;

        // Required fields
        request["model"] = model_name_;
        request["max_tokens"] = actual_max_tokens;

        // Anthropic format: separate "system" field from "messages" array
        request["messages"] = json::array();

        // Process messages from SessionContext
        for (const auto& msg : session.messages) {
            if (msg.role == "system") {
                // Anthropic uses a separate "system" field at the root level
                request["system"] = msg.content;
            } else if (msg.role == "user") {
                // Check if this is a tool result
                if (!msg.tool_call_id.empty()) {
                    // Tool results for Anthropic must be user messages with tool_result content blocks
                    request["messages"].push_back({
                        {"role", "user"},
                        {"content", json::array({
                            {
                                {"type", "tool_result"},
                                {"tool_use_id", msg.tool_call_id},
                                {"content", msg.content}
                            }
                        })}
                    });
                } else {
                    // Regular user message
                    request["messages"].push_back({
                        {"role", "user"},
                        {"content", msg.content}
                    });
                }
            } else if (msg.role == "assistant") {
                // Check if this is a tool call (JSON format)
                json assistant_content;
                try {
                    json parsed = json::parse(msg.content);
                    // If it parses as JSON with "name" and "parameters", it's a tool call
                    if (parsed.contains("name") && parsed.contains("parameters")) {
                        // Format as tool_use block for Anthropic
                        assistant_content = json::array({
                            {
                                {"type", "tool_use"},
                                {"id", parsed.value("id", "")},
                                {"name", parsed["name"]},
                                {"input", parsed["parameters"]}
                            }
                        });
                    } else {
                        // Regular text content
                        assistant_content = msg.content;
                    }
                } catch (const json::exception&) {
                    // Not JSON, treat as regular text
                    assistant_content = msg.content;
                }

                request["messages"].push_back({
                    {"role", "assistant"},
                    {"content", assistant_content}
                });
            } else if (msg.role == "tool") {
                // Tool results for Anthropic must be user messages with tool_result content blocks
                request["messages"].push_back({
                    {"role", "user"},
                    {"content", json::array({
                        {
                            {"type", "tool_result"},
                            {"tool_use_id", msg.tool_call_id},
                            {"content", msg.content}
                        }
                    })}
                });
            }
        }

        // Format tools from SessionContext for Anthropic API
        if (!session.tools.empty()) {
            json tools_array = json::array();

            for (const auto& tool_def : session.tools) {
                json tool;
                tool["name"] = tool_def.name;
                tool["description"] = tool_def.description;

                // Parse parameters JSON or use default schema
                if (!tool_def.parameters_json.empty()) {
                    try {
                        tool["input_schema"] = json::parse(tool_def.parameters_json);
                    } catch (const json::exception& e) {
                        LOG_DEBUG("Tool " + tool_def.name + " has invalid JSON schema, using empty fallback");
                        tool["input_schema"] = {
                            {"type", "object"},
                            {"properties", json::object()},
                            {"required", json::array()}
                        };
                    }
                } else {
                    tool["input_schema"] = {
                        {"type", "object"},
                        {"properties", json::object()},
                        {"required", json::array()}
                    };
                }

                tools_array.push_back(tool);
            }

            request["tools"] = tools_array;
            LOG_DEBUG("Added " + std::to_string(tools_array.size()) + " tools to Anthropic request from SessionContext");
        }

        // Convert to string
        std::string request_json = request.dump();

        // Log request for debugging
        if (request_json.length() <= 2000) {
            LOG_DEBUG("Full Anthropic API request: " + request_json);
        } else {
            LOG_DEBUG("Anthropic API request (first 2000 chars): " + request_json.substr(0, 2000) + "...");
            LOG_DEBUG("Anthropic API request length: " + std::to_string(request_json.length()) + " bytes");
        }

        // Make API call
        std::string response_json = make_api_request(request_json);
        if (response_json.empty()) {
            throw BackendManagerError("Anthropic API request failed");
        }

        LOG_DEBUG("Anthropic API response: " + response_json.substr(0, 500));

        // Parse response
        std::string response_text = parse_anthropic_response(response_json);
        if (response_text.empty()) {
            throw BackendManagerError("Failed to parse Anthropic response");
        }

        return response_text;
    } catch (const json::exception& e) {
        throw BackendManagerError("JSON error in generate_from_session: " + std::string(e.what()));
    }
}
