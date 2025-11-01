#include "anthropic.h"
#include "../logger.h"
#include "../shepherd.h"
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

// AnthropicBackend implementation
AnthropicBackend::AnthropicBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    LOG_DEBUG("AnthropicBackend created");

    // Parse backend-specific config
    std::string backend_cfg = config.backend_config(get_backend_name());
    parse_backend_config(backend_cfg);
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

    // Query actual API context size
    size_t api_context_size = query_model_context_size(model_name_);
    LOG_INFO("Anthropic model " + model_name_ + " API context size: " + std::to_string(api_context_size));

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

    LOG_INFO("AnthropicBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string AnthropicBackend::generate(int max_tokens) {
    // Use new architecture: build SessionContext and call base class
    SessionContext session;
    build_session_from_context(session);
    return generate_from_session(session, max_tokens);
}

std::string AnthropicBackend::get_backend_name() const {
    return "anthropic";
}

std::string AnthropicBackend::get_model_name() const {
    return model_name_;
}

size_t AnthropicBackend::get_context_size() const {
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
    curl_easy_setopt(curl_, CURLOPT_URL, api_endpoint.c_str());
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
        throw BackendManagerError("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }

    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);

    if (http_code != 200) {
        // Try to parse error message from response body
        std::string error_msg = "API request failed with status " + std::to_string(http_code);

        if (!response_body.empty()) {
            try {
                auto error_json = json::parse(response_body);
                if (error_json.contains("error")) {
                    if (error_json["error"].is_object() && error_json["error"].contains("message")) {
                        error_msg = error_json["error"]["message"].get<std::string>();
                    } else if (error_json["error"].is_string()) {
                        error_msg = error_json["error"].get<std::string>();
                    }
                }
            } catch (const json::exception&) {
                // If JSON parsing fails, use the raw response body
                error_msg += ": " + response_body.substr(0, 200);
            }
        }

        LOG_ERROR("Anthropic API error (HTTP " + std::to_string(http_code) + "): " + error_msg);
        throw BackendManagerError(error_msg);
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

        // Extract usage data from API response and store for server to return
        if (j.contains("usage") && j["usage"].is_object()) {
            last_prompt_tokens_ = j["usage"].value("input_tokens", 0);
            last_completion_tokens_ = j["usage"].value("output_tokens", 0);
            LOG_DEBUG("Usage from API: input=" + std::to_string(last_prompt_tokens_) +
                      " output=" + std::to_string(last_completion_tokens_) +
                      " total=" + std::to_string(last_prompt_tokens_ + last_completion_tokens_));

            // Update message token counts and EMA ratio
            update_message_tokens_from_api(last_prompt_tokens_, last_completion_tokens_);
        } else {
            // Reset to 0 if no usage data (shouldn't happen with real APIs)
            last_prompt_tokens_ = 0;
            last_completion_tokens_ = 0;
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

// Old generate_from_session removed - now using base class implementation

// ========== New Architecture Methods ==========

std::string AnthropicBackend::format_api_request(const SessionContext& session, int max_tokens) {
#ifdef ENABLE_API_BACKENDS
    json request;

    // Required fields
    request["model"] = model_name_;

    // Calculate max_tokens if not provided
    int actual_max_tokens = max_tokens;
    if (actual_max_tokens <= 0) {
        // Estimate available tokens
        int total_context_tokens = context_manager_->get_total_tokens();
        size_t context_window = context_manager_->get_max_context_tokens();
        actual_max_tokens = static_cast<int>(context_window) - total_context_tokens - 100;

        // Cap at model's max output limit
        int max_output_limit = query_max_output_tokens(model_name_);
        actual_max_tokens = std::min(actual_max_tokens, max_output_limit);
    }

    // Anthropic requires at least 1 token
    if (actual_max_tokens < 1) {
        actual_max_tokens = 1;
    }

    request["max_tokens"] = actual_max_tokens;

    // Anthropic format: separate "system" field from "messages" array
    request["messages"] = json::array();

    for (const auto& msg : session.messages) {
        if (msg.role == "user") {
            // Check if this is a tool result (has tool_call_id)
            if (!msg.tool_call_id.empty()) {
                // Tool results must be user messages with tool_result content blocks
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
        }
    }

    // Add system field if we have system content
    if (!session.system_prompt.empty()) {
        request["system"] = session.system_prompt;
    }

    // Add tools if available from session
    if (!session.tools.empty()) {
        json tools_array = json::array();

        for (const auto& tool_def : session.tools) {
            json tool;
            tool["name"] = tool_def.name;
            tool["description"] = tool_def.description;

            // Parameters are already JSON object
            if (!tool_def.parameters.empty()) {
                tool["input_schema"] = tool_def.parameters;
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
    }

    return request.dump();
#else
    return "";
#endif
}

int AnthropicBackend::extract_tokens_to_evict(const std::string& error_message) {
    // Anthropic format: "input length and max_tokens exceed context limit: 187254 + 20000 > 204798"

    size_t colon_pos = error_message.find("context limit: ");
    if (colon_pos != std::string::npos) {
        try {
            // Parse "187254 + 20000 > 204798"
            size_t start = colon_pos + 15;
            std::string numbers = error_message.substr(start);

            // Extract input tokens
            size_t plus_pos = numbers.find(" + ");
            int input_tokens = std::stoi(numbers.substr(0, plus_pos));

            // Extract max_tokens
            size_t gt_pos = numbers.find(" > ");
            size_t max_start = plus_pos + 3;
            int max_output = std::stoi(numbers.substr(max_start, gt_pos - max_start));

            // Extract limit
            int limit = std::stoi(numbers.substr(gt_pos + 3));

            // Total = input + max_output, need to evict: total - limit
            return (input_tokens + max_output) - limit;
        } catch (...) {}
    }

    return -1;
}

ApiResponse AnthropicBackend::parse_api_response(const HttpResponse& http_response) {
    ApiResponse result;
    result.raw_response = http_response.body;

#ifdef ENABLE_API_BACKENDS
    if (http_response.is_success()) {
        try {
            json j = json::parse(http_response.body);

            // Extract usage data
            if (j.contains("usage") && j["usage"].is_object()) {
                result.prompt_tokens = j["usage"].value("input_tokens", 0);
                result.completion_tokens = j["usage"].value("output_tokens", 0);
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
                        // Convert tool_use block to JSON format
                        json tool_call;
                        tool_call["name"] = content_block["name"];
                        tool_call["parameters"] = content_block["input"];
                        tool_call["id"] = content_block["id"];

                        // Add as JSON on its own line
                        response_text += "\n" + tool_call.dump() + "\n";
                    }
                }

                result.content = response_text;
            }

            result.is_error = false;
        } catch (const json::exception& e) {
            result.is_error = true;
            result.error_code = 500;
            result.error_message = "Failed to parse Anthropic response: " + std::string(e.what());
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

std::map<std::string, std::string> AnthropicBackend::get_api_headers() {
    std::map<std::string, std::string> headers;
#ifdef ENABLE_API_BACKENDS
    headers["Content-Type"] = "application/json";
    headers["x-api-key"] = api_key_;
    headers["anthropic-version"] = api_version_;
#endif
    return headers;
}

std::string AnthropicBackend::get_api_endpoint() {
#ifdef ENABLE_API_BACKENDS
    return api_endpoint;
#else
    return "";
#endif
}

void AnthropicBackend::parse_specific_config(const std::string& json) {
    if (json.empty() || json == "{}") {
        return;
    }

    try {
        auto j = nlohmann::json::parse(json);

        if (j.contains("api_endpoint")) {
            api_endpoint = j["api_endpoint"].get<std::string>();
            LOG_DEBUG("Anthropic: Set api_endpoint = " + api_endpoint);
        }

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse Anthropic-specific config: " + std::string(e.what()));
    }
}
