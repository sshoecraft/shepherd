#include "client_output.h"
#include "../tools/utf8_sanitizer.h"

namespace ClientOutputs {

// ============================================================================
// StreamingOutput Implementation
// ============================================================================

StreamingOutput::StreamingOutput(httplib::DataSink* sink, const std::string& client_id)
    : sink(sink), client_id(client_id), connected(true) {}

bool StreamingOutput::write_sse(const std::string& event_type, const nlohmann::json& data) {
    if (!connected || !sink) return false;

    nlohmann::json event;
    event["type"] = event_type;
    event["data"] = data;
    std::string sse_data = "data: " + event.dump() + "\n\n";

    if (!sink->write(sse_data.c_str(), sse_data.size())) {
        connected = false;
        return false;
    }
    return true;
}

void StreamingOutput::on_delta(const std::string& delta) {
    if (delta.empty()) return;
    std::string sanitized = utf8_sanitizer::sanitize_utf8(delta);
    write_sse("delta", {{"delta", sanitized}});
}

void StreamingOutput::on_codeblock(const std::string& content) {
    if (content.empty()) return;
    std::string sanitized = utf8_sanitizer::sanitize_utf8(content);
    write_sse("codeblock", {{"content", sanitized}});
}

void StreamingOutput::on_user_prompt(const std::string& prompt) {
    nlohmann::json data;
    data["role"] = "user";
    data["content"] = utf8_sanitizer::sanitize_utf8(prompt);
    write_sse("message_added", data);
}

void StreamingOutput::on_message_added(const std::string& role,
                                       const std::string& content,
                                       int tokens) {
    nlohmann::json data;
    data["role"] = role;
    data["content"] = utf8_sanitizer::sanitize_utf8(content);
    data["tokens"] = tokens;
    write_sse("message_added", data);
}

void StreamingOutput::on_tool_call(const std::string& name,
                                   const nlohmann::json& params,
                                   const std::string& id) {
    nlohmann::json data;
    data["tool_call"] = name;
    data["parameters"] = params;
    if (!id.empty()) {
        data["tool_call_id"] = id;
    }
    write_sse("tool_call", data);
}

void StreamingOutput::on_tool_result(const std::string& name,
                                     bool success,
                                     const std::string& error) {
    nlohmann::json data;
    data["tool_name"] = name;
    data["success"] = success;
    if (!success && !error.empty()) {
        data["error"] = error;
    }
    write_sse("tool_result", data);
}

void StreamingOutput::on_complete(const std::string& full_response) {
    nlohmann::json data;
    data["response"] = utf8_sanitizer::sanitize_utf8(full_response);
    write_sse("response_complete", data);
}

void StreamingOutput::on_error(const std::string& error) {
    write_sse("error", {{"error", error}});
}

void StreamingOutput::flush() {
    // For streaming output, nothing to flush - all events sent immediately
    if (sink && connected) {
        // Send done marker for POST /request streaming responses
        nlohmann::json done;
        done["done"] = true;
        std::string sse_data = "data: " + done.dump() + "\n\n";
        sink->write(sse_data.c_str(), sse_data.size());
    }
}

bool StreamingOutput::is_connected() const {
    return connected;
}

bool StreamingOutput::send_keepalive() {
    if (!connected || !sink) return false;

    const char* keepalive = ": keepalive\n\n";
    if (!sink->write(keepalive, strlen(keepalive))) {
        connected = false;
        return false;
    }
    return true;
}

// ============================================================================
// BatchedOutput Implementation
// ============================================================================

BatchedOutput::BatchedOutput(httplib::Response* response)
    : response(response) {}

void BatchedOutput::on_delta(const std::string& delta) {
    // Close code block if switching from codeblock to regular content
    if (in_codeblock) {
        accumulated += "```\n";
        in_codeblock = false;
    }
    // Accumulate deltas - will be returned in flush()
    accumulated += delta;
}

void BatchedOutput::on_codeblock(const std::string& content) {
    // Accumulate code blocks with markers for batched output
    if (!in_codeblock) {
        accumulated += "```\n";
        in_codeblock = true;
    }
    accumulated += content;
}

void BatchedOutput::on_user_prompt(const std::string& prompt) {
    // No-op for batched output - user already knows their prompt
}

void BatchedOutput::on_message_added(const std::string& role,
                                     const std::string& content,
                                     int tokens) {
    // No-op for batched output - complete response returned in flush()
}

void BatchedOutput::on_tool_call(const std::string& name,
                                 const nlohmann::json& params,
                                 const std::string& id) {
    // No-op for batched output - tools are handled server-side
}

void BatchedOutput::on_tool_result(const std::string& name,
                                   bool success,
                                   const std::string& error) {
    // No-op for batched output - tools are handled server-side
}

void BatchedOutput::on_complete(const std::string& full_response) {
    // Store the final response for flush()
    final_response = full_response;
}

void BatchedOutput::on_error(const std::string& error) {
    has_error = true;
    error_message = error;
}

void BatchedOutput::flush() {
    if (flushed || !response) return;
    flushed = true;

    nlohmann::json result;
    if (has_error) {
        result["success"] = false;
        result["error"] = error_message;
        response->status = 500;
    } else {
        result["success"] = true;
        // Use final_response if set, otherwise use accumulated
        result["response"] = utf8_sanitizer::sanitize_utf8(
            final_response.empty() ? accumulated : final_response);
    }
    response->set_content(result.dump(), "application/json");
}

bool BatchedOutput::is_connected() const {
    // Batched output is always "connected" until flushed
    return !flushed;
}

} // namespace ClientOutputs
