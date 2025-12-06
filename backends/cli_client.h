#pragma once

#include "backend.h"
#include "http_client.h"
#include <memory>
#include <string>
#include <sstream>

/// @brief Backend that connects to a CLI server
/// Simple client that POSTs prompts to /request and gets responses
class CLIClientBackend : public Backend {
public:
    explicit CLIClientBackend(const std::string& base_url);
    ~CLIClientBackend() override = default;

    void initialize(Session& session) override;

    Response add_message(Session& session,
                        Message::Type type,
                        const std::string& content,
                        const std::string& tool_name = "",
                        const std::string& tool_id = "",
                        int prompt_tokens = 0,
                        int max_tokens = 0) override;

    Response add_message_stream(Session& session,
                               Message::Type type,
                               const std::string& content,
                               StreamCallback callback,
                               const std::string& tool_name = "",
                               const std::string& tool_id = "",
                               int prompt_tokens = 0,
                               int max_tokens = 0) override;

    Response generate_from_session(const Session& session, int max_tokens = 0, StreamCallback callback = nullptr) override;

    int count_message_tokens(Message::Type type,
                            const std::string& content,
                            const std::string& tool_name = "",
                            const std::string& tool_id = "") override;

    void parse_backend_config() override;

private:
    std::string base_url;
    std::unique_ptr<HttpClient> http_client;

    Response send_request(const std::string& prompt, StreamCallback callback = nullptr);
};
