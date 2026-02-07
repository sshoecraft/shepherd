#pragma once

#include <string>
#include <functional>
#include <atomic>
#include "nlohmann/json.hpp"
#include "httplib.h"

namespace ClientOutputs {

// Abstract base class for client output handling
// Each connected client gets an output instance based on their preferred mode
class ClientOutput {
public:
    virtual ~ClientOutput() = default;

    // Content streaming - called for each delta from model generation
    virtual void on_delta(const std::string& delta) = 0;

    // Code block content - called for content inside code blocks
    virtual void on_codeblock(const std::string& content) = 0;

    // User prompt echo - called when user message is submitted
    virtual void on_user_prompt(const std::string& prompt) = 0;

    // Message events - called when a complete message is added to session
    virtual void on_message_added(const std::string& role,
                                  const std::string& content,
                                  int tokens) = 0;

    // Tool events
    virtual void on_tool_call(const std::string& name,
                              const nlohmann::json& params,
                              const std::string& id) = 0;
    virtual void on_tool_result(const std::string& name,
                                bool success,
                                const std::string& error = "") = 0;

    // Completion events
    virtual void on_complete(const std::string& full_response) = 0;
    virtual void on_error(const std::string& error) = 0;

    // Lifecycle
    virtual void flush() = 0;
    virtual bool is_connected() const = 0;
};

// Writes SSE events immediately to httplib::DataSink
// Used for /updates observers and streaming POST requests
class StreamingOutput : public ClientOutput {
public:
    StreamingOutput(httplib::DataSink* sink, const std::string& client_id);

    void on_delta(const std::string& delta) override;
    void on_codeblock(const std::string& content) override;
    void on_user_prompt(const std::string& prompt) override;
    void on_message_added(const std::string& role,
                          const std::string& content,
                          int tokens) override;
    void on_tool_call(const std::string& name,
                      const nlohmann::json& params,
                      const std::string& id) override;
    void on_tool_result(const std::string& name,
                        bool success,
                        const std::string& error = "") override;
    void on_complete(const std::string& full_response) override;
    void on_error(const std::string& error) override;
    void flush() override;
    bool is_connected() const override;

    // Send keep-alive comment to detect disconnection
    bool send_keepalive();

    const std::string& get_client_id() const { return client_id; }

private:
    httplib::DataSink* sink;
    std::string client_id;
    std::atomic<bool> connected{true};

    // Helper to write SSE event
    bool write_sse(const std::string& event_type, const nlohmann::json& data);
};

// Accumulates deltas, writes complete JSON response on flush
// Used for non-streaming POST requests
class BatchedOutput : public ClientOutput {
public:
    BatchedOutput(httplib::Response* response);

    void on_delta(const std::string& delta) override;
    void on_codeblock(const std::string& content) override;
    void on_user_prompt(const std::string& prompt) override;
    void on_message_added(const std::string& role,
                          const std::string& content,
                          int tokens) override;
    void on_tool_call(const std::string& name,
                      const nlohmann::json& params,
                      const std::string& id) override;
    void on_tool_result(const std::string& name,
                        bool success,
                        const std::string& error = "") override;
    void on_complete(const std::string& full_response) override;
    void on_error(const std::string& error) override;
    void flush() override;
    bool is_connected() const override;

private:
    httplib::Response* response;
    std::string accumulated;
    std::string final_response;
    bool has_error = false;
    std::string error_message;
    bool flushed = false;
    bool in_codeblock = false;
};

} // namespace ClientOutputs
