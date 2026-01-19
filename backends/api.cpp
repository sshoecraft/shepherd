
#include "shepherd.h"
#include "api.h"
#include "shared_oauth_cache.h"
#include "session.h"
#include "message.h"
#include "sse_parser.h"

// ApiBackend implementation
ApiBackend::ApiBackend(size_t context_size, Session& session, EventCallback callback)
    : Backend(context_size, session, callback) {
	http_client = std::make_unique<HttpClient>();
	http_client->set_timeout(timeout_seconds);

	// Configure SSL settings from config if available
	if (!config->json.is_null() && !config->json.empty()) {
		if (config->json.contains("ssl_verify")) {
			bool ssl_verify = config->json["ssl_verify"].get<bool>();
			http_client->set_ssl_verify(ssl_verify);
			dout(1) << "SSL verification: " + std::string(ssl_verify ? "enabled" : "disabled") << std::endl;
		}
		if (config->json.contains("ca_bundle_path")) {
			std::string ca_bundle = config->json["ca_bundle_path"].get<std::string>();
			http_client->set_ca_bundle(ca_bundle);
			dout(1) << "CA bundle path: " + ca_bundle << std::endl;
		}

		// Load OAuth configuration
		if (config->json.contains("client_id")) {
			oauth_client_id_ = config->json["client_id"].get<std::string>();
			dout(1) << "OAuth client_id loaded" << std::endl;
		}
		if (config->json.contains("client_secret")) {
			oauth_client_secret_ = config->json["client_secret"].get<std::string>();
			dout(1) << "OAuth client_secret loaded (length: " + std::to_string(oauth_client_secret_.length()) + ")" << std::endl;
		}
		if (config->json.contains("token_url")) {
			oauth_token_url_ = config->json["token_url"].get<std::string>();
			dout(1) << "OAuth token_url: " + oauth_token_url_ << std::endl;
		}
		if (config->json.contains("token_scope")) {
			oauth_scope_ = config->json["token_scope"].get<std::string>();
			dout(1) << "OAuth scope: " + oauth_scope_ << std::endl;
		}
	}

	dout(1) << "ApiBackend created" << std::endl;
}

// Output through common filter (backticks, buffering)
// API backends use this for streaming output to get consistent filtering
bool ApiBackend::output(const char* text, size_t len) {
    return filter(text, len);
}

void ApiBackend::parse_backend_config() {
    if (config->json.is_null() || config->json.empty()) {
        dout(1) << "ApiBackend::parse_backend_config: empty config, using defaults" << std::endl;
        return;  // No config, use defaults
    }

    try {
        if (config->json.contains("temperature")) temperature = config->json["temperature"].get<float>();
        if (config->json.contains("top_p")) top_p = config->json["top_p"].get<float>();
        if (config->json.contains("top_k")) top_k = config->json["top_k"].get<int>();
        if (config->json.contains("frequency_penalty")) frequency_penalty = config->json["frequency_penalty"].get<float>();
        if (config->json.contains("presence_penalty")) presence_penalty = config->json["presence_penalty"].get<float>();
        if (config->json.contains("repeat_penalty")) repeat_penalty = config->json["repeat_penalty"].get<float>();
        if (config->json.contains("stop")) {
            stop_sequences.clear();  // Clear defaults when user explicitly sets stop
            if (config->json["stop"].is_array() && !config->json["stop"].empty()) {
                stop_sequences = config->json["stop"].get<std::vector<std::string>>();
            } else if (config->json["stop"].is_string()) {
                stop_sequences.push_back(config->json["stop"].get<std::string>());
            }
            // If stop is empty array [], stop_sequences remains empty (no stopping)
        }
        if (config->json.contains("timeout")) {
            timeout_seconds = config->json["timeout"].get<long>();
            if (http_client) {
                http_client->set_timeout(timeout_seconds);
            }
        }

        dout(1) << "Loaded API backend config: timeout=" + std::to_string(timeout_seconds) +
                  "s, temperature=" + std::to_string(temperature) +
                  ", top_p=" + std::to_string(top_p) +
                  ", top_k=" + std::to_string(top_k) +
                  ", frequency_penalty=" + std::to_string(frequency_penalty) +
                  ", presence_penalty=" + std::to_string(presence_penalty) +
                  ", repeat_penalty=" + std::to_string(repeat_penalty) +
                  ", stop_sequences=" + std::to_string(stop_sequences.size()) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse API backend config: " + std::string(e.what()) << std::endl;
    }
}

void ApiBackend::add_message(Session& session, Message::Role role, const std::string& content, const std::string& tool_name, const std::string& tool_id, int max_tokens) {
    dout(1) << "ApiBackend::add_message: max_tokens=" + std::to_string(max_tokens) +
             ", streaming=" + (config->streaming ? "yes" : "no") << std::endl;

    // Non-streaming implementation (base class)
    // Derived classes (OpenAI, Anthropic, Gemini) override for true streaming
    // All output flows through callback (CONTENT, TOOL_CALL, ERROR, STOP)

    const int MAX_RETRIES = 3;
    int retry = 0;

    while (retry < MAX_RETRIES) {
        // Build request with entire session + new message + max_tokens
        nlohmann::json request = build_request(session, role, content, tool_name, tool_id, max_tokens);

        dout(1) << "Sending to API with max_tokens=" + std::to_string(max_tokens) << std::endl;

        // Enforce rate limits before making request
        enforce_rate_limits();

        // Get headers and endpoint from concrete backend
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

//		std::cout << "request: " << request " << std::endl;
        HttpResponse http_response = http_client->post(endpoint, request.dump(), headers);
//		std::cout << "response: " << http_response.body " << std::endl;

        // Parse response using backend-specific parser
        Response resp = parse_http_response(http_response);

        if (resp.success) {
			// user prompt accepted by the provider, send a user event
	        if (config->streaming && role == Message::USER && !content.empty()) {
				callback(CallbackEvent::USER_PROMPT, content, "", "");
        	}

            // Success! Use delta method to get exact tokens for the new message
            // Delta = current_prompt_tokens - last_prompt_tokens
            int new_message_tokens;
            if (resp.prompt_tokens > 0 && session.last_prompt_tokens > 0) {
                new_message_tokens = resp.prompt_tokens - session.last_prompt_tokens;
                if (new_message_tokens < 0) new_message_tokens = 1; // Safety floor
                dout(1) << "Exact message tokens from delta: " + std::to_string(new_message_tokens) +
                         " (current=" + std::to_string(resp.prompt_tokens) +
                         " - last=" + std::to_string(session.last_prompt_tokens) + "" << std::endl;

            } else {
                // Fallback to estimate if API doesn't provide usage
                new_message_tokens = estimate_message_tokens(content);
                dout(1) << "Estimated message tokens: " + std::to_string(new_message_tokens) << std::endl;
            }

            // Update EMA with actual token ratio from this message
            if (new_message_tokens > 0 && content.length() > 0) {
                float actual_ratio = (float)content.length() / new_message_tokens;

                // Use ratio-based outlier detection to filter cache-induced anomalies
                // Check how much the new measurement deviates from current EMA
                float deviation_ratio = actual_ratio / chars_per_token;
                const float MIN_DEVIATION_RATIO = 0.5f;  // Reject if < 50% of current EMA
                const float MAX_DEVIATION_RATIO = 2.0f;  // Reject if > 200% of current EMA

                if (deviation_ratio >= MIN_DEVIATION_RATIO && deviation_ratio <= MAX_DEVIATION_RATIO) {
                    const float alpha = 0.2f;  // EMA smoothing factor (20% weight on new data)
                    float old_cpt = chars_per_token;
                    chars_per_token = (1.0f - alpha) * chars_per_token + alpha * actual_ratio;

                    // Clamp to absolute reasonable bounds as safety net
                    const float MIN_CPT = 1.0f;  // Very dense tokens (1:1 char:token ratio)
                    const float MAX_CPT = 5.0f;  // Very sparse tokens (English prose)
                    if (chars_per_token < MIN_CPT) chars_per_token = MIN_CPT;
                    if (chars_per_token > MAX_CPT) chars_per_token = MAX_CPT;

                    dout(1) << "Updated EMA chars_per_token: " + std::to_string(old_cpt) + " -> " +
                             std::to_string(chars_per_token) +
                             " (actual: " + std::to_string(actual_ratio) +
                             ", deviation: " + std::to_string(deviation_ratio) + "x" << std::endl;

                } else {
                    dout(1) << "Skipping EMA update - deviation ratio out of bounds: " +
                             std::to_string(deviation_ratio) + "x (actual: " +
                             std::to_string(actual_ratio) + ", likely prompt caching" << std::endl;

                }
            }

            // Add user/tool message to session with exact token count
            Message user_msg(role, content, new_message_tokens);
            user_msg.tool_name = tool_name;
            user_msg.tool_call_id = tool_id;
            session.messages.push_back(user_msg);

            // Track last user message for context preservation
            if (role == Message::USER) {
                session.last_user_message_index = session.messages.size() - 1;
                session.last_user_message_tokens = new_message_tokens;
            }

            // Update baseline for next delta calculation
            if (resp.prompt_tokens > 0) {
                session.last_prompt_tokens = resp.prompt_tokens;
            }

            // Add assistant response with exact completion tokens from API
            int assistant_tokens = (resp.completion_tokens > 0) ? resp.completion_tokens : estimate_message_tokens(resp.content);
            Message assistant_msg(Message::ASSISTANT, resp.content, assistant_tokens);
            assistant_msg.tool_calls_json = resp.tool_calls_json;  // Persist tool calls for conversation history
            session.messages.push_back(assistant_msg);

            // Track last assistant message for context preservation
            session.last_assistant_message_index = session.messages.size() - 1;
            session.last_assistant_message_tokens = assistant_tokens;

            // Update baseline again to include assistant's message
            if (resp.prompt_tokens > 0 && assistant_tokens > 0) {
                session.last_prompt_tokens = resp.prompt_tokens + assistant_tokens;
            }

            // Set total tokens from API (authoritative source of truth)
            session.total_tokens = resp.prompt_tokens + assistant_tokens;

            dout(1) << "Message tokens - new msg: " + std::to_string(new_message_tokens) +
                     ", assistant: " + std::to_string(assistant_tokens) +
                     ", session total: " + std::to_string(session.total_tokens) << std::endl;

            // Route content through unified output filter (handles tool call detection)
            // For structured API responses (like Anthropic), extract text blocks only
            if (!resp.content.empty()) {
                std::string display_content = resp.content;

                // Check if content is a JSON array (structured response with content blocks)
                try {
                    auto content_json = nlohmann::json::parse(resp.content);
                    if (content_json.is_array()) {
                        // Extract text from content blocks, skip tool_use blocks
                        std::string text_only;
                        for (const auto& block : content_json) {
                            if (block.contains("type") && block["type"] == "text" && block.contains("text")) {
                                text_only += block["text"].get<std::string>();
                            }
                        }
                        display_content = text_only;
                    }
                } catch (...) {
                    // Not JSON, use as-is
                }

                if (!display_content.empty()) {
                    // Route through output() for filtering (backticks, buffering)
                    output(display_content);
                    flush_output();
                }
            }

            // Signal completion with finish reason
            callback(CallbackEvent::STOP, resp.finish_reason, "", "");

            // Send tool calls AFTER STOP - frontend handles immediately
            if (!resp.tool_calls.empty()) {
                for (const auto& tc : resp.tool_calls) {
                    callback(CallbackEvent::TOOL_CALL, tc.raw_json, tc.name, tc.tool_call_id);
                }
            }
            return;
        }

        // Check if context length error
        int tokens_to_evict = extract_tokens_to_evict(http_response);

        if (tokens_to_evict > 0) {
            // Calculate which messages to evict using Session's eviction logic
            auto ranges = session.calculate_messages_to_evict(tokens_to_evict);

            if (ranges.empty()) {
                callback(CallbackEvent::ERROR, "Context full, cannot evict enough messages", "context_full", "");
                callback(CallbackEvent::STOP, "error", "", "");
                return;
            }

            // Evict messages using Session's eviction method
            if (!session.evict_messages(ranges)) {
                callback(CallbackEvent::ERROR, "Failed to evict messages", "eviction_failed", "");
                callback(CallbackEvent::STOP, "error", "", "");
                return;
            }

            retry++;
            dout(1) << "Evicted messages, retrying (attempt " + std::to_string(retry + 1) + "/" +
                    std::to_string(MAX_RETRIES) + "" << std::endl;


        } else {
            // Not a context error - add TOOL_RESPONSE for session consistency, then signal error
            if (role == Message::TOOL_RESPONSE) {
                add_tool_response(session, content, tool_name, tool_id);
            }
            callback(CallbackEvent::ERROR, resp.error, "api_error", "");
            callback(CallbackEvent::STOP, "error", "", "");
            return;
        }
    }

    callback(CallbackEvent::ERROR, "Max retries exceeded trying to fit context", "max_retries", "");
    callback(CallbackEvent::STOP, "error", "", "");
}

void ApiBackend::generate_from_session(Session& session, int max_tokens) {
    // Build request using backend-specific format
    nlohmann::json request = build_request_from_session(session, max_tokens);

    // Get headers and endpoint from backend
    auto headers = get_api_headers();
    std::string endpoint = get_api_endpoint();

    // Make HTTP call
    HttpResponse http_response = http_client->post(endpoint, request.dump(), headers);

    // Parse response using backend-specific parser
    Response resp = parse_http_response(http_response);

    // Update session token counts from API response (session is source of truth)
    if (resp.prompt_tokens > 0 || resp.completion_tokens > 0) {
        session.total_tokens = resp.prompt_tokens + resp.completion_tokens;
        session.last_prompt_tokens = resp.prompt_tokens;
        session.last_assistant_message_tokens = resp.completion_tokens;
    }

    if (resp.success) {
        // If we have structured tool calls from API, use those exclusively
        // API backends get structured tool_calls from API response - no content filtering needed
        if (!resp.tool_calls.empty()) {
            // Output any text content that's NOT the tool call JSON
            // For Anthropic, resp.content is JSON array - extract text portions only
            try {
                auto content_json = nlohmann::json::parse(resp.content);
                if (content_json.is_array()) {
                    for (const auto& block : content_json) {
                        if (block.contains("type") && block["type"] == "text" && block.contains("text")) {
                            output(block["text"].get<std::string>());
                        }
                    }
                }
            } catch (...) {
                // Not JSON array, output as-is (but this shouldn't happen with tool calls)
                output(resp.content);
            }
            flush_output();

            // Tool calls will be sent AFTER STOP below
        } else if (!resp.content.empty()) {
            // No structured tool calls - output content through filter
            output(resp.content);
            flush_output();
        }

        // Store assistant message in session (critical for tool call flows)
        int assistant_tokens = resp.completion_tokens > 0 ? resp.completion_tokens : estimate_message_tokens(resp.content);
        Message assistant_msg(Message::ASSISTANT, resp.content, assistant_tokens);

        // Build tool_calls_json if we have tool calls
        if (!resp.tool_calls.empty()) {
            nlohmann::json tool_calls_array = nlohmann::json::array();
            for (const auto& tc : resp.tool_calls) {
                nlohmann::json tc_json;
                tc_json["id"] = tc.tool_call_id;
                tc_json["type"] = "function";
                tc_json["function"]["name"] = tc.name;
                tc_json["function"]["arguments"] = tc.raw_json;
                tool_calls_array.push_back(tc_json);
            }
            assistant_msg.tool_calls_json = tool_calls_array.dump();
        }

        session.messages.push_back(assistant_msg);
        session.last_assistant_message_index = session.messages.size() - 1;
        session.last_assistant_message_tokens = assistant_tokens;

        callback(CallbackEvent::STOP, resp.finish_reason, "", "");

        // Send tool calls AFTER STOP - frontend handles immediately
        for (const auto& tc : resp.tool_calls) {
            callback(CallbackEvent::TOOL_CALL, tc.raw_json, tc.name, tc.tool_call_id);
        }
    } else {
        callback(CallbackEvent::ERROR, resp.error, "api_error", "");
        callback(CallbackEvent::STOP, "error", "", "");
    }
}

int ApiBackend::estimate_message_tokens(const std::string& content) const {
    // Estimate tokens using adaptive chars/token ratio
    return static_cast<int>(content.length() / chars_per_token);
}

int ApiBackend::count_message_tokens(Message::Role role,
                                     const std::string& content,
                                     const std::string& tool_name,
                                     const std::string& tool_id) {
    // Build the full JSON request exactly as add_message() would
    // This captures all JSON overhead, role labels, formatting, etc.
    Session temp_session;
    temp_session.system_message = ""; // Empty for counting just this message
    temp_session.tools = {}; // No tools for counting just this message

    nlohmann::json request = build_request(temp_session, role, content, tool_name, tool_id);

    // Get formatted JSON string length
    std::string json_str = request.dump();
    int json_length = json_str.length();

    // Apply EMA to estimate tokens
    int estimated_tokens = static_cast<int>(json_length / chars_per_token);

    dout(1) << "count_message_tokens: JSON length=" + std::to_string(json_length) +
             ", estimated tokens=" + std::to_string(estimated_tokens) +
             " (chars_per_token=" + std::to_string(chars_per_token) + "" << std::endl;


    return estimated_tokens;
}

void ApiBackend::calibrate_token_counts(Session& session) {
    dout(1) << "Calibrating token counts with single probe message..." << std::endl;

    try {
        // Create probe session with system + tools + "."
        Session probe_session;
        probe_session.system_message = session.system_message;
        probe_session.tools = session.tools;

        // Build probe request: system + tools + "."
        nlohmann::json request = build_request(probe_session, Message::USER, ".", "", "");

        // Get headers and endpoint from derived backend
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

        // Send probe
        dout(1) << "Sending calibration probe (system + tools + dot)..." << std::endl;
        HttpResponse response = http_client->post(endpoint, request.dump(), headers);

        // Parse response using backend-specific parser
        Response parsed = parse_http_response(response);

        if (!parsed.success) {
            // Check for critical errors that should fail immediately
            // 1. Authentication errors (401, 403, etc.)
            if (response.status_code == 401 || response.status_code == 403 ||
                parsed.error.find("authentication") != std::string::npos ||
                parsed.error.find("invalid x-api-key") != std::string::npos ||
                parsed.error.find("unauthorized") != std::string::npos) {
                std::cerr << "Authentication failed: " + parsed.error << std::endl;
                throw BackendError("Authentication failed: " + parsed.error);
            }

            // 2. Connection errors (status_code 0 means connection failure)
            if (response.status_code == 0 ||
                parsed.error.find("Could not connect") != std::string::npos ||
                parsed.error.find("connect to server") != std::string::npos) {
                std::cerr << "Connection failed: " + parsed.error << std::endl;
                throw BackendError("Connection failed: " + parsed.error);
            }

            dout(1) << std::string("WARNING: ") +"Calibration probe failed: " + parsed.error + ", using estimates" << std::endl;
            throw std::runtime_error("Calibration probe failed");
        }

        // Use prompt_tokens from parsed response (works for all backends)
        int probe_tokens = parsed.prompt_tokens;

        // Set baseline: probe includes system + tools + ".", but we don't store the dot or response
        // So baseline should be system + tools only (minus the dot)
        session.last_prompt_tokens = probe_tokens - 1;
        dout(1) << "Calibration baseline set: " + std::to_string(session.last_prompt_tokens) + " tokens (system + tools, excluding probe '.')" << std::endl;

        // For display/debugging purposes, calculate approximate breakdown
        // (Not used for actual token counting - we use delta method)
        session.system_message_tokens = probe_tokens - 1; // Rough estimate: everything minus the dot (includes tools)

        dout(1) << "Calibration complete - baseline: " + std::to_string(session.last_prompt_tokens) + " tokens" << std::endl;

    } catch (const BackendError& e) {
        // Re-throw authentication and other critical errors - don't catch these
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Calibration failed: " + std::string(e.what()) << std::endl;
        // Fall back to estimates for non-critical errors
        session.system_message_tokens = estimate_message_tokens(session.system_message);
        // Note: system_message_tokens includes tools estimate
        session.last_prompt_tokens = session.system_message_tokens;

        // Use conservative default ratio for API models (optimized for code-heavy content)
        chars_per_token = 1.75f;
    }
}

// Rate limiting enforcement
void ApiBackend::enforce_rate_limits() {
    dout(1) << "enforce_rate_limits() called" << std::endl;
    std::lock_guard<std::mutex> lock(rate_limit_mutex_);

    auto now = std::chrono::steady_clock::now();

    // Get rate limits from provider config
    int requests_per_second = 0;
    int requests_per_minute = 0;

    if (!config->json.is_null() && !config->json.empty()) {
        requests_per_second = config->json.value("requests_per_second", 0);
        requests_per_minute = config->json.value("requests_per_minute", 0);
    }

    dout(1) << "Rate limit check: requests_per_second=" + std::to_string(requests_per_second) +
              ", requests_per_minute=" + std::to_string(requests_per_minute) << std::endl;

    // If no rate limits configured, return immediately
    if (requests_per_second == 0 && requests_per_minute == 0) {
        dout(1) << "No rate limits configured, skipping" << std::endl;
        return;
    }

    dout(1) << "Rate limits are configured, proceeding with enforcement" << std::endl;

    // Remove timestamps older than 1 minute (we only need to track up to 1 minute)
    auto one_minute_ago = now - std::chrono::minutes(1);
    while (!request_timestamps_.empty() && request_timestamps_.front() < one_minute_ago) {
        request_timestamps_.pop_front();
    }

    // Check requests_per_second limit
    if (requests_per_second > 0) {
        auto one_second_ago = now - std::chrono::seconds(1);
        int requests_last_second = 0;
        for (auto it = request_timestamps_.rbegin(); it != request_timestamps_.rend(); ++it) {
            if (*it >= one_second_ago) {
                requests_last_second++;
            } else {
                break;
            }
        }

        if (requests_last_second >= requests_per_second) {
            // Calculate how long to sleep
            auto oldest_in_window = request_timestamps_[request_timestamps_.size() - requests_last_second];
            auto wait_until = oldest_in_window + std::chrono::seconds(1);
            auto sleep_duration = wait_until - now;

            if (sleep_duration.count() > 0) {
                dout(1) << "Rate limit: sleeping " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(sleep_duration).count()) +
                         "ms (requests_per_second=" + std::to_string(requests_per_second) + "" << std::endl;

                std::this_thread::sleep_for(sleep_duration);
                now = std::chrono::steady_clock::now();
            }
        }
    }

    dout(1) << "About to check requests_per_minute limit" << std::endl;

    // Check requests_per_minute limit
    if (requests_per_minute > 0) {
        dout(1) << "Checking requests_per_minute: " + std::to_string(request_timestamps_.size()) + " requests in last minute, limit=" + std::to_string(requests_per_minute) << std::endl;
        if ((int)request_timestamps_.size() >= requests_per_minute) {
            // Calculate how long to sleep
            auto oldest_in_window = request_timestamps_.front();
            auto wait_until = oldest_in_window + std::chrono::minutes(1);
            auto sleep_duration = wait_until - now;

            if (sleep_duration.count() > 0) {
                dout(1) << "Rate limit: sleeping " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(sleep_duration).count()) +
                         "ms (requests_per_minute=" + std::to_string(requests_per_minute) + "" << std::endl;

                std::this_thread::sleep_for(sleep_duration);
                now = std::chrono::steady_clock::now();
            }
        }
    } else {
        dout(1) << "requests_per_minute is NOT > 0, it is: " + std::to_string(requests_per_minute) << std::endl;
    }

    // Record this request timestamp
    request_timestamps_.push_back(now);
}

// OAuth 2.0 token acquisition using client credentials grant
ApiBackend::OAuthToken ApiBackend::acquire_oauth_token(const std::string& client_id,
                                                        const std::string& client_secret,
                                                        const std::string& token_url,
                                                        const std::string& scope) {
    dout(1) << "Acquiring OAuth token from: " + token_url << std::endl;

    OAuthToken token;

    try {
        // Build form-urlencoded request body
        std::string body = "grant_type=client_credentials&client_id=" + client_id +
                          "&client_secret=" + client_secret;
        if (!scope.empty()) {
            body += "&scope=" + scope;
        }

        // Set headers for OAuth token request
        std::map<std::string, std::string> headers = {
            {"Content-Type", "application/x-www-form-urlencoded"},
            {"Accept", "application/json"}
        };

        // Make POST request
        HttpResponse response = http_client->post(token_url, body, headers);

        if (!response.is_success()) {
            std::cerr << "OAuth token request failed with status " + std::to_string(response.status_code) << std::endl;
            std::cerr << "Response body: " + response.body << std::endl;
            return token;  // Return empty token
        }

        // Parse JSON response
        nlohmann::json response_json = nlohmann::json::parse(response.body);

        token.access_token = response_json.value("access_token", "");
        token.token_type = response_json.value("token_type", "Bearer");
        int expires_in = response_json.value("expires_in", 3600);

        // Set expiry time (current time + expires_in, minus 60 seconds buffer)
        token.expires_at = time(nullptr) + expires_in - 60;

        dout(1) << "OAuth token acquired successfully (expires in " + std::to_string(expires_in) + " seconds)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to acquire OAuth token: " + std::string(e.what()) << std::endl;
    }

    return token;
}

// Ensure we have a valid OAuth token, refreshing if needed
bool ApiBackend::ensure_valid_oauth_token() {
    // If OAuth is not configured, return true (use regular API key auth)
    if (oauth_client_id_.empty() || oauth_client_secret_.empty() || oauth_token_url_.empty()) {
        return true;
    }

    // Use shared OAuth cache if available (for per-request backend mode)
    if (shared_oauth_cache_) {
        auto cached_token = shared_oauth_cache_->get_token(
            oauth_client_id_, oauth_client_secret_, oauth_token_url_, oauth_scope_);
        if (!cached_token.access_token.empty()) {
            // Copy token to local storage for get_api_headers() to use
            oauth_token_.access_token = cached_token.access_token;
            oauth_token_.token_type = cached_token.token_type;
            oauth_token_.expires_at = cached_token.expires_at;
            return true;
        }
        std::cerr << "Failed to acquire OAuth token from shared cache" << std::endl;
        return false;
    }

    // Fall back to per-backend token management
    // Check if current token is valid
    if (oauth_token_.is_valid()) {
        dout(1) << "OAuth token is valid" << std::endl;
        return true;
    }

    // Token is expired or missing, acquire new one
    dout(1) << "OAuth token expired or missing, acquiring new token..." << std::endl;
    oauth_token_ = acquire_oauth_token(oauth_client_id_, oauth_client_secret_,
                                       oauth_token_url_, oauth_scope_);

    if (!oauth_token_.access_token.empty()) {
        dout(1) << "OAuth token refreshed successfully" << std::endl;
        return true;
    }

    std::cerr << "Failed to acquire OAuth token" << std::endl;
    return false;
}

void ApiBackend::set_shared_oauth_cache(std::shared_ptr<SharedOAuthCache> cache) {
    shared_oauth_cache_ = cache;
}

void ApiBackend::add_tool_response(Session& session,
                                   const std::string& content,
                                   const std::string& tool_name,
                                   const std::string& tool_id) {
    int tokens = estimate_message_tokens(content);
    Message tool_msg(Message::TOOL_RESPONSE, content, tokens);
    tool_msg.tool_name = tool_name;
    tool_msg.tool_call_id = tool_id;
    session.messages.push_back(tool_msg);
    dout(1) << "Added TOOL_RESPONSE to session: " + tool_name << std::endl;
}
