
#include "json_frontend.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/utf8_sanitizer.h"
#include "tools/api_tools.h"
#include "message.h"
#include "config.h"
#include "provider.h"
#include "backends/factory.h"
#include "rag.h"
#include "nlohmann/json.hpp"

#include <iostream>
#include <string>
#include <climits>

extern std::unique_ptr<Config> config;

JsonFrontend::JsonFrontend() : Frontend() {
}

JsonFrontend::~JsonFrontend() {
}

void JsonFrontend::emit_json(const nlohmann::json& j) {
    std::cout << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
    std::cout.flush();
}

float JsonFrontend::compute_turn_cost(int prompt_tokens, int completion_tokens) {
    Provider* p = get_provider(current_provider);
    if (!p) return 0.0f;
    float prompt_cost = (prompt_tokens / 1000000.0f) * p->pricing.prompt_cost;
    float completion_cost = (completion_tokens / 1000000.0f) * p->pricing.completion_cost;
    return prompt_cost + completion_cost;
}

void JsonFrontend::init(const FrontendFlags& flags) {
    init_flags = flags;

    // Set up the event callback for streaming JSON output
    callback = [this](CallbackEvent type, const std::string& content,
                      const std::string& tool_name, const std::string& tool_call_id) -> bool {

        switch (type) {
            case CallbackEvent::CONTENT:
            case CallbackEvent::CODEBLOCK: {
                nlohmann::json j;
                j["type"] = "text";
                j["content"] = content;
                emit_json(j);
                break;
            }
            case CallbackEvent::THINKING: {
                nlohmann::json j;
                j["type"] = "thinking";
                j["content"] = content;
                emit_json(j);
                break;
            }
            case CallbackEvent::TOOL_CALL: {
                // Emit tool_use event
                nlohmann::json tool_use;
                tool_use["type"] = "tool_use";
                tool_use["name"] = tool_name;
                try {
                    tool_use["params"] = nlohmann::json::parse(content);
                } catch (...) {
                    tool_use["params"] = content;
                }
                tool_use["id"] = tool_call_id;
                emit_json(tool_use);

                // Execute tool locally
                ToolResult result = execute_tool(tools, tool_name, content, tool_call_id, session.user_id);

                // Emit tool_result event
                nlohmann::json tool_result;
                tool_result["type"] = "tool_result";
                tool_result["name"] = tool_name;
                tool_result["id"] = tool_call_id;
                tool_result["success"] = result.success;
                std::string summary = result.summary.empty()
                    ? (result.success ? result.content.substr(0, 200) : result.error)
                    : result.summary;
                tool_result["summary"] = summary;
                if (!result.success) {
                    tool_result["error"] = result.error;
                }
                emit_json(tool_result);

                // Add tool response to session (generation deferred to TOOL_CALLS_COMPLETE)
                {
                    auto lock = backend->acquire_lock();
                    add_message_to_session(Message::TOOL_RESPONSE, result.content, tool_name, tool_call_id);
                }
                break;
            }
            case CallbackEvent::TOOL_CALLS_COMPLETE: {
                // All tool calls processed - generate next response
                auto lock = backend->acquire_lock();
                generate_response();
                break;
            }
            case CallbackEvent::ERROR: {
                nlohmann::json j;
                j["type"] = "error";
                j["message"] = content;
                if (!tool_name.empty()) j["error_type"] = tool_name;
                emit_json(j);
                break;
            }
            case CallbackEvent::SYSTEM: {
                nlohmann::json j;
                j["type"] = "system";
                j["content"] = content;
                emit_json(j);
                break;
            }
            case CallbackEvent::STOP: {
                turn_count++;
                nlohmann::json j;
                j["type"] = "end_turn";
                j["turns"] = turn_count;
                j["total_tokens"] = session.total_tokens;

                // Compute cost from provider pricing
                Provider* p = get_provider(current_provider);
                if (p && (p->pricing.prompt_cost > 0 || p->pricing.completion_cost > 0)) {
                    float cost = compute_turn_cost(session.last_prompt_tokens,
                                                   session.last_assistant_message_tokens);
                    j["cost_usd"] = cost;
                }

                emit_json(j);
                break;
            }
            case CallbackEvent::STATS:
            case CallbackEvent::TOOL_RESULT:
            case CallbackEvent::TOOL_DISP:
            case CallbackEvent::RESULT_DISP:
            case CallbackEvent::USER_PROMPT:
                // Suppressed - not part of JSON protocol
                break;
        }
        return true;  // Never cancel from JSON frontend
    };

    // Common tool initialization
    init_tools(init_flags);
}

int JsonFrontend::run(Provider* cmdline_provider) {
    // Determine which provider to connect
    Provider* provider_to_use = nullptr;
    if (cmdline_provider) {
        provider_to_use = cmdline_provider;
    } else if (!providers.empty()) {
        provider_to_use = &providers[0];
    }

    if (!provider_to_use) {
        emit_json({{"type", "error"}, {"message", "No providers configured. Use 'shepherd provider add' to configure."}});
        return 1;
    }

    // Connect to provider
    if (!connect_provider(provider_to_use->name)) {
        emit_json({{"type", "error"}, {"message", "Failed to connect to provider '" + provider_to_use->name + "'"}});
        return 1;
    }

    // If server_tools mode, fetch tools from server or fall back to local
    if (config->server_tools && !init_flags.no_tools) {
        Provider* p = get_provider(current_provider);
        if (p && !p->base_url.empty()) {
            init_remote_tools(p->base_url, p->api_key);
        } else {
            init_tools(init_flags, true);
        }
    }

    // Register other providers as tools
    if (!init_flags.no_tools) {
        register_provider_tools(tools, current_provider);
    }

    // Populate session.tools from our tools instance
    tools.populate_session_tools(session);

    // Copy tool names to backend for output filtering
    for (const auto& tool : session.tools) {
        backend->valid_tool_names.insert(tool.name);
    }

    // Configure session based on backend capabilities
    if (config->max_tokens == -1) {
        session.desired_completion_tokens = INT_MAX;
    } else if (config->max_tokens > 0) {
        session.desired_completion_tokens = config->max_tokens;
    } else {
        session.desired_completion_tokens = calculate_desired_completion_tokens(
            backend->context_size, backend->max_output_tokens);
    }
    session.auto_evict = (config->context_size > 0 && !backend->is_gpu);

    // Handle warmup if configured
    if (config->warmup && !config->warmup_message.empty()) {
        auto lock = backend->acquire_lock();
        add_message_to_session(Message::USER, config->warmup_message);
        generate_response();
    }

    // Handle --prompt single query mode
    if (config->single_query_mode) {
        if (config->initial_prompt.empty()) return 0;
        auto lock = backend->acquire_lock();
        add_message_to_session(Message::USER, config->initial_prompt);
        enrich_with_rag_context(session);
        generate_response();
        queue_memory_extraction();
        return 0;
    }

    // Main loop: read JSON lines from stdin
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        // Parse input JSON
        nlohmann::json input;
        try {
            input = nlohmann::json::parse(line);
        } catch (const std::exception& e) {
            emit_json({{"type", "error"}, {"message", "Invalid JSON: " + std::string(e.what())}});
            continue;
        }

        std::string type = input.value("type", "");
        if (type != "user") {
            emit_json({{"type", "error"}, {"message", "Unknown input type: " + type}});
            continue;
        }

        std::string content = input.value("content", "");
        if (content.empty()) continue;

        // Handle exit commands
        if (content == "exit" || content == "quit") break;

        // Handle slash commands
        if (!content.empty() && content[0] == '/') {
            if (handle_slash_commands(content, tools)) continue;
        }

        // Sanitize user input
        content = utf8_sanitizer::strip_control_characters(content);

        // Truncate if needed (same logic as CLI)
        double scale = calculate_truncation_scale(backend->context_size);
        int available = backend->context_size - session.system_message_tokens;
        int max_user_input_tokens = available * scale;
        int input_tokens = backend->count_message_tokens(Message::USER, content, "", "");

        if (input_tokens > max_user_input_tokens) {
            std::string truncation_notice = "\n\n[INPUT TRUNCATED: Too large for context window]";
            while (input_tokens >= max_user_input_tokens && content.length() > 100) {
                size_t new_len = content.length() * 0.9;
                content = content.substr(0, new_len);
                input_tokens = backend->count_message_tokens(Message::USER, content + truncation_notice, "", "");
            }
            content += truncation_notice;
        }

        // Generate response
        {
            auto lock = backend->acquire_lock();
            add_message_to_session(Message::USER, content);
            enrich_with_rag_context(session);
            generate_response();
        }

        queue_memory_extraction();
    }

    // Cleanup
    if (extraction_thread) {
        queue_memory_extraction();
        extraction_thread->flush();
        extraction_thread->stop();
    }

    return 0;
}
