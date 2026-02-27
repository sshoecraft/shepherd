
#include "shepherd.h"
#include "grok.h"
#include "nlohmann/json.hpp"
#include "../tools/utf8_sanitizer.h"

using json = nlohmann::json;

GrokBackend::GrokBackend(size_t context_size, Session& session, EventCallback callback)
    : OpenAIBackend(context_size, session, callback) {
    // Parent constructor handled: endpoint setup from config->api_base, API key,
    // model detection, context size query, calibration, parse_backend_config().
    // Default endpoint if none configured:
    if (config->api_base.empty()) {
        api_endpoint = "https://api.x.ai/v1/chat/completions";
    }
    dout(1) << "Grok backend initialized" << std::endl;
}

// Build Grok API request from complete session
nlohmann::json GrokBackend::build_request_from_session(const Session& session, int max_tokens) {
    json request;
    request["model"] = model_name;
    if (!session.user_id.empty()) {
        request["user"] = session.user_id;
    }

    // Build messages array from complete session
    json messages = json::array();

    // Add system message
    if (!session.system_message.empty()) {
        messages.push_back({{"role", "system"}, {"content", session.system_message}});
    }

    // Add all messages from session
    for (const auto& msg : session.messages) {
        json jmsg;
        jmsg["role"] = msg.get_role();
        jmsg["content"] = utf8_sanitizer::sanitize_utf8(msg.content);

        // Restore tool_calls for assistant messages that made tool calls
        if (msg.role == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                dout(1) << std::string("WARNING: ") + "Failed to parse stored tool_calls: " + std::string(e.what()) << std::endl;
            }
        }

        if (msg.role == Message::TOOL_RESPONSE && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    request["messages"] = messages;

    // Add tools if present
    dout(1) << "Session has " + std::to_string(session.tools.size()) + " tools" << std::endl;
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json params = tool.parameters;
            if (params.contains("properties") && params["properties"].is_object()) {
                for (auto& [key, prop] : params["properties"].items()) {
                    if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                        prop["items"] = {{"type", "object"}};
                    }
                }
            }

            json jtool;
            jtool["type"] = "function";
            jtool["function"] = {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", params}
            };
            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add max_tokens if specified
    if (max_tokens > 0) {
        request[model_config.max_tokens_param_name] = max_tokens;
    }

    // Sampling parameters — only send what Grok supports, only when explicitly configured
    if (sampling) {
        if (temperature >= 0.0f) {
            request["temperature"] = round2(temperature);
        }
        if (top_p >= 0.0f) {
            request["top_p"] = round2(top_p);
        }
        if (frequency_penalty >= 0.0f) {
            request["frequency_penalty"] = round2(frequency_penalty);
        }
    }

    // Grok reasoning: {"reasoning": {"effort": "low"|"high"}} (not OpenAI's "reasoning_effort")
    // Supported by grok-3-mini, grok-4-fast-reasoning, etc. Models that don't support it will reject.
    if (!reasoning.empty() && reasoning != "off") {
        json reasoning_config;
        reasoning_config["effort"] = reasoning;
        request["reasoning"] = reasoning_config;
        dout(1) << "Added Grok reasoning: " + reasoning_config.dump() << std::endl;
    }

    // Add stop sequences if configured
    dout(1) << "stop_sequences.size()=" + std::to_string(stop_sequences.size()) << std::endl;
    if (!stop_sequences.empty()) {
        json stop_array = json::array();
        for (const auto& seq : stop_sequences) {
            stop_array.push_back(seq);
        }
        request["stop"] = stop_array;
        dout(1) << "Added stop sequences to request: " + stop_array.dump() << std::endl;
    }

    return request;
}

// Build Grok API request from session + new message
nlohmann::json GrokBackend::build_request(const Session& session,
                                           Message::Role role,
                                           const std::string& content,
                                           const std::string& tool_name,
                                           const std::string& tool_id,
                                           int max_tokens) {
    json request;
    request["model"] = model_name;
    if (!session.user_id.empty()) {
        request["user"] = session.user_id;
    }

    // Build messages array
    json messages = json::array();

    // Add system message
    if (!session.system_message.empty()) {
        messages.push_back({{"role", "system"}, {"content", session.system_message}});
    }

    // Add existing messages
    for (const auto& msg : session.messages) {
        json jmsg;
        jmsg["role"] = msg.get_role();
        jmsg["content"] = utf8_sanitizer::sanitize_utf8(msg.content);

        if (msg.role == Message::ASSISTANT && !msg.tool_calls_json.empty()) {
            try {
                jmsg["tool_calls"] = json::parse(msg.tool_calls_json);
            } catch (const std::exception& e) {
                dout(1) << std::string("WARNING: ") + "Failed to parse stored tool_calls: " + std::string(e.what()) << std::endl;
            }
        }

        if (msg.role == Message::TOOL_RESPONSE && !msg.tool_call_id.empty()) {
            jmsg["tool_call_id"] = msg.tool_call_id;
        }
        messages.push_back(jmsg);
    }

    // Add the new message
    json new_msg;
    std::string role_str;
    switch (role) {
        case Message::SYSTEM: role_str = "system"; break;
        case Message::USER: role_str = "user"; break;
        case Message::ASSISTANT: role_str = "assistant"; break;
        case Message::TOOL_RESPONSE: role_str = "tool"; break;
        case Message::FUNCTION: role_str = "function"; break;
    }
    new_msg["role"] = role_str;
    new_msg["content"] = utf8_sanitizer::sanitize_utf8(content);
    if (!tool_name.empty()) new_msg["name"] = tool_name;
    if (!tool_id.empty()) new_msg["tool_call_id"] = tool_id;
    messages.push_back(new_msg);

    request["messages"] = messages;

    dout(1) << "Built request with " + std::to_string(messages.size()) + " messages (session has " +
             std::to_string(session.messages.size()) + " messages" << std::endl;
    dout(1) << "Session has " + std::to_string(session.tools.size()) + " tools" << std::endl;

    // Add tools if present
    if (!session.tools.empty()) {
        json tools = json::array();
        for (const auto& tool : session.tools) {
            json params = tool.parameters;
            if (params.contains("properties") && params["properties"].is_object()) {
                for (auto& [key, prop] : params["properties"].items()) {
                    if (prop.contains("type") && prop["type"] == "array" && !prop.contains("items")) {
                        prop["items"] = {{"type", "object"}};
                    }
                }
            }

            json jtool;
            jtool["type"] = "function";
            jtool["function"] = {
                {"name", tool.name},
                {"description", tool.description},
                {"parameters", params}
            };
            tools.push_back(jtool);
        }
        request["tools"] = tools;
    }

    // Add max_tokens if specified
    if (max_tokens > 0) {
        request[model_config.max_tokens_param_name] = max_tokens;
    }

    // Sampling parameters — only send what Grok supports, only when explicitly configured
    if (sampling) {
        if (temperature >= 0.0f) {
            request["temperature"] = round2(temperature);
        }
        if (top_p >= 0.0f) {
            request["top_p"] = round2(top_p);
        }
        if (frequency_penalty >= 0.0f) {
            request["frequency_penalty"] = round2(frequency_penalty);
        }
    }

    // Grok reasoning: {"reasoning": {"effort": "low"|"high"}} (not OpenAI's "reasoning_effort")
    // Supported by grok-3-mini, grok-4-fast-reasoning, etc. Models that don't support it will reject.
    if (!reasoning.empty() && reasoning != "off") {
        json reasoning_config;
        reasoning_config["effort"] = reasoning;
        request["reasoning"] = reasoning_config;
        dout(1) << "Added Grok reasoning: " + reasoning_config.dump() << std::endl;
    }

    // Add stop sequences if configured
    dout(1) << "stop_sequences.size()=" + std::to_string(stop_sequences.size()) << std::endl;
    if (!stop_sequences.empty()) {
        json stop_array = json::array();
        for (const auto& seq : stop_sequences) {
            stop_array.push_back(seq);
        }
        request["stop"] = stop_array;
        dout(1) << "Added stop sequences to request: " + stop_array.dump() << std::endl;
    }

    return request;
}
