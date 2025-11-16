
#include "shepherd.h"
#include "api.h"
#include "session.h"
#include "message.h"
#include "terminal_io.h"
#include "sse_parser.h"

// ApiBackend implementation
ApiBackend::ApiBackend(size_t context_size) : Backend(context_size) {
	http_client = std::make_unique<HttpClient>();
	http_client->set_timeout(timeout_seconds);
	LOG_DEBUG("ApiBackend created");
}

void ApiBackend::parse_backend_config() {
    if (config->json.is_null() || config->json.empty()) {
        LOG_DEBUG("ApiBackend::parse_backend_config: empty config, using defaults");
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

        LOG_DEBUG("Loaded API backend config: temperature=" + std::to_string(temperature) +
                  ", top_p=" + std::to_string(top_p) +
                  ", top_k=" + std::to_string(top_k) +
                  ", frequency_penalty=" + std::to_string(frequency_penalty) +
                  ", presence_penalty=" + std::to_string(presence_penalty) +
                  ", repeat_penalty=" + std::to_string(repeat_penalty) +
                  ", stop_sequences=" + std::to_string(stop_sequences.size()));
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse API backend config: " + std::string(e.what()));
    }
}

void ApiBackend::initialize(Session& session) {
    LOG_INFO("Initializing API backend...");

    // If context_size is 0, query it from the API
    if (context_size == 0) {
        size_t api_context_size = query_model_context_size(model_name);
        if (api_context_size > 0) {
            context_size = api_context_size;
            LOG_INFO("Using API's context size: " + std::to_string(context_size));
        } else {
            LOG_ERROR("No context size specified and could not detect from API");
            LOG_ERROR("Please specify context size using --context-size argument");
            throw BackendError("Cannot determine context size: API detection failed and none was specified. Use --context-size to set manually.");
        }
    }

    // Calibrate token counts (if enabled in config)
    if (config->calibration) {
        LOG_INFO("Calibrating token counts...");
        calibrate_token_counts(session);
    } else {
        LOG_INFO("Calibration disabled, using default estimates");
        // Use default chars_per_token (already set to 2.5f in ApiBackend constructor)
        session.system_message_tokens = estimate_message_tokens(session.system_message);
        session.last_prompt_tokens = session.system_message_tokens;
    }

    LOG_INFO("API backend initialization complete");
}

Response ApiBackend::add_message(Session& session, Message::Type type, const std::string& content, const std::string& tool_name, const std::string& tool_id, int prompt_tokens, int max_tokens) {
    LOG_DEBUG("ApiBackend::add_message: prompt_tokens=" + std::to_string(prompt_tokens) +
             ", max_tokens=" + std::to_string(max_tokens));

    // On first message, check config and try streaming if enabled
    if (!streaming_tested) {
        streaming_tested = true;
        if (!config->streaming) {
            streaming_enabled = false;
            LOG_INFO("Streaming disabled via configuration");
        } else {
            streaming_enabled = true; // Will disable if it fails
            LOG_INFO("Attempting streaming for first message...");
        }
    }

    // If streaming is enabled, try it
    if (streaming_enabled) {
        tio.begin_response();  // Reset TerminalIO state before streaming

        Response resp = add_message_stream(session, type, content,
            [](const std::string& delta, const std::string& accumulated, const Response& partial) -> bool {
                tio.write(delta.c_str(), delta.length());
                return true;
            },
            tool_name, tool_id, prompt_tokens, max_tokens);

        tio.end_response();  // Consume any incomplete tags

        // If streaming succeeded, mark it and return
        if (resp.success) {
            resp.was_streamed = true;
            return resp;
        }

        // Streaming failed - disable it and fall through to non-streaming
        LOG_WARN("Streaming failed, disabling for future messages: " + resp.error);
        streaming_enabled = false;
        // Fall through to non-streaming code below
    }

    // Otherwise use non-streaming
    const int MAX_RETRIES = 3;
    int retry = 0;

    while (retry < MAX_RETRIES) {
        // Build request with entire session + new message + max_tokens
        nlohmann::json request = build_request(session, type, content, tool_name, tool_id, max_tokens);

        LOG_DEBUG("Sending to API with max_tokens=" + std::to_string(max_tokens));

        // Get headers and endpoint from concrete backend
        auto headers = get_api_headers();
        std::string endpoint = get_api_endpoint();

        // Try to send
//		std::cout << "request: " << request << std::endl;
        HttpResponse http_response = http_client->post(endpoint, request.dump(), headers);
//		std::cout << "response: " << http_response.body << std::endl;

        // Parse response using backend-specific parser
        Response resp = parse_http_response(http_response);

        if (resp.success) {
            // Success! Use delta method to get exact tokens for the new message
            // Delta = current_prompt_tokens - last_prompt_tokens
            int new_message_tokens;
            if (resp.prompt_tokens > 0 && session.last_prompt_tokens > 0) {
                new_message_tokens = resp.prompt_tokens - session.last_prompt_tokens;
                if (new_message_tokens < 0) new_message_tokens = 1; // Safety floor
                LOG_DEBUG("Exact message tokens from delta: " + std::to_string(new_message_tokens) +
                         " (current=" + std::to_string(resp.prompt_tokens) +
                         " - last=" + std::to_string(session.last_prompt_tokens) + ")");
            } else {
                // Fallback to estimate if API doesn't provide usage
                new_message_tokens = estimate_message_tokens(content);
                LOG_DEBUG("Estimated message tokens: " + std::to_string(new_message_tokens));
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
                    const float MIN_CPT = 2.0f;  // Very dense tokens (code, numbers)
                    const float MAX_CPT = 5.0f;  // Very sparse tokens (English prose)
                    if (chars_per_token < MIN_CPT) chars_per_token = MIN_CPT;
                    if (chars_per_token > MAX_CPT) chars_per_token = MAX_CPT;

                    LOG_DEBUG("Updated EMA chars_per_token: " + std::to_string(old_cpt) + " -> " +
                             std::to_string(chars_per_token) +
                             " (actual: " + std::to_string(actual_ratio) +
                             ", deviation: " + std::to_string(deviation_ratio) + "x)");
                } else {
                    LOG_DEBUG("Skipping EMA update - deviation ratio out of bounds: " +
                             std::to_string(deviation_ratio) + "x (actual: " +
                             std::to_string(actual_ratio) + ", likely prompt caching)");
                }
            }

            // Add user/tool message to session with exact token count
            Message user_msg(type, content, new_message_tokens);
            user_msg.tool_name = tool_name;
            user_msg.tool_call_id = tool_id;
            session.messages.push_back(user_msg);

            // Track last user message for context preservation
            if (type == Message::USER) {
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

            LOG_DEBUG("Message tokens - new msg: " + std::to_string(new_message_tokens) +
                     ", assistant: " + std::to_string(assistant_tokens) +
                     ", session total: " + std::to_string(session.total_tokens));

            return resp;
        }

        // Check if context length error
        int tokens_to_evict = extract_tokens_to_evict(http_response);

        if (tokens_to_evict > 0) {
            // Calculate which messages to evict using Session's eviction logic
            auto ranges = session.calculate_messages_to_evict(tokens_to_evict);

            if (ranges.empty()) {
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::CONTEXT_FULL;
                err_resp.finish_reason = "error";
                err_resp.error = "Context full, cannot evict enough messages";
                return err_resp;
            }

            // Evict messages using Session's eviction method
            if (!session.evict_messages(ranges)) {
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::ERROR;
                err_resp.finish_reason = "error";
                err_resp.error = "Failed to evict messages";
                return err_resp;
            }

            retry++;
            LOG_INFO("Evicted messages, retrying (attempt " + std::to_string(retry + 1) + "/" +
                    std::to_string(MAX_RETRIES) + ")");

        } else {
            // Not a context error - return the error from response
            return resp;
        }
    }

    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.finish_reason = "error";
    err_resp.error = "Max retries exceeded trying to fit context";
    return err_resp;
}

Response ApiBackend::generate_from_session(const Session& session, int max_tokens, StreamCallback callback) {
    // Build request using backend-specific format
    nlohmann::json request = build_request_from_session(session, max_tokens);

    // Get headers and endpoint from backend
    auto headers = get_api_headers();
    std::string endpoint = get_api_endpoint();

    // Make HTTP call
    HttpResponse http_response = http_client->post(endpoint, request.dump(), headers);

    // Parse response using backend-specific parser
    Response resp = parse_http_response(http_response);

    return resp;
}

int ApiBackend::estimate_message_tokens(const std::string& content) const {
    // Estimate tokens using adaptive chars/token ratio
    return static_cast<int>(content.length() / chars_per_token);
}

int ApiBackend::count_message_tokens(Message::Type type,
                                     const std::string& content,
                                     const std::string& tool_name,
                                     const std::string& tool_id) {
    // Build the full JSON request exactly as add_message() would
    // This captures all JSON overhead, role labels, formatting, etc.
    Session temp_session;
    temp_session.system_message = ""; // Empty for counting just this message
    temp_session.tools = {}; // No tools for counting just this message

    nlohmann::json request = build_request(temp_session, type, content, tool_name, tool_id);

    // Get formatted JSON string length
    std::string json_str = request.dump();
    int json_length = json_str.length();

    // Apply EMA to estimate tokens
    int estimated_tokens = static_cast<int>(json_length / chars_per_token);

    LOG_DEBUG("count_message_tokens: JSON length=" + std::to_string(json_length) +
             ", estimated tokens=" + std::to_string(estimated_tokens) +
             " (chars_per_token=" + std::to_string(chars_per_token) + ")");

    return estimated_tokens;
}

Response ApiBackend::add_message_stream(Session& session,
                                       Message::Type type,
                                       const std::string& content,
                                       StreamCallback callback,
                                       const std::string& tool_name,
                                       const std::string& tool_id,
                                       int prompt_tokens,
                                       int max_tokens) {
    // Base implementation: just call non-streaming version
    // Derived classes override this for true streaming
    Response resp = add_message(session, type, content, tool_name, tool_id, prompt_tokens, max_tokens);

    // Invoke callback once with full response to simulate streaming
    if (resp.success && callback) {
        callback(resp.content, resp.content, resp);
    }

    return resp;
}

void ApiBackend::calibrate_token_counts(Session& session) {
    LOG_INFO("Calibrating token counts with single probe message...");

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
        LOG_DEBUG("Sending calibration probe (system + tools + dot)...");
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
                LOG_ERROR("Authentication failed: " + parsed.error);
                throw BackendError("Authentication failed: " + parsed.error);
            }

            // 2. Connection errors (status_code 0 means connection failure)
            if (response.status_code == 0 ||
                parsed.error.find("Could not connect") != std::string::npos ||
                parsed.error.find("connect to server") != std::string::npos) {
                LOG_ERROR("Connection failed: " + parsed.error);
                throw BackendError("Connection failed: " + parsed.error);
            }

            LOG_WARN("Calibration probe failed: " + parsed.error + ", using estimates");
            throw std::runtime_error("Calibration probe failed");
        }

        // Use prompt_tokens from parsed response (works for all backends)
        int probe_tokens = parsed.prompt_tokens;

        // Set baseline: probe includes system + tools + ".", but we don't store the dot or response
        // So baseline should be system + tools only (minus the dot)
        session.last_prompt_tokens = probe_tokens - 1;
        LOG_INFO("Calibration baseline set: " + std::to_string(session.last_prompt_tokens) + " tokens (system + tools, excluding probe '.')");

        // For display/debugging purposes, calculate approximate breakdown
        // (Not used for actual token counting - we use delta method)
        session.system_message_tokens = probe_tokens - 1; // Rough estimate: everything minus the dot (includes tools)

        LOG_INFO("Calibration complete - baseline: " + std::to_string(session.last_prompt_tokens) + " tokens");

    } catch (const BackendError& e) {
        // Re-throw authentication and other critical errors - don't catch these
        throw;
    } catch (const std::exception& e) {
        LOG_ERROR("Calibration failed: " + std::string(e.what()));
        // Fall back to estimates for non-critical errors
        session.system_message_tokens = estimate_message_tokens(session.system_message);
        // Note: system_message_tokens includes tools estimate
        session.last_prompt_tokens = session.system_message_tokens;

        // Use conservative default ratio for API models (optimized for code-heavy content)
        chars_per_token = 2.5f;
    }
}
