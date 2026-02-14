#pragma once

#include "backend.h"
#include "http_client.h"
#include <memory>
#include <string>
#include <sstream>
#include <thread>
#include <atomic>

/// @brief Backend that connects to a CLI server
/// Simple client that POSTs prompts to /request and gets responses
class CLIClientBackend : public Backend {
public:
    CLIClientBackend(const std::string& base_url, Session& session, EventCallback callback);
    ~CLIClientBackend() override;


    void generate_from_session(Session& session, int max_tokens = 0) override;

    void clear_session() override;

    int count_message_tokens(Message::Role role,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;

    void parse_backend_config() override;

    // Per-session flag: when true, send "memory": false in requests to server
    bool send_no_memory = false;

private:
    std::string base_url;
    std::string api_key;  // API key for server authentication
    std::unique_ptr<HttpClient> http_client;

    // SSE listener thread for updates
    std::thread sse_thread;
    std::atomic<bool> sse_running{false};
    void sse_listener_thread();

    // Track when we're making a request (to skip our own user message echo)
    std::atomic<bool> request_in_progress{false};

    Response send_request(const std::string& prompt, int max_tokens = 0, EventCallback callback = nullptr);
};
