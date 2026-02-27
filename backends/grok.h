
#pragma once

#include "openai.h"

/// @brief Backend for xAI Grok API (OpenAI-compatible protocol, Grok-specific request format)
class GrokBackend : public OpenAIBackend {
public:
    GrokBackend(size_t context_size, Session& session, EventCallback callback);

    // Override request builders with clean Grok-specific implementations
    nlohmann::json build_request_from_session(const Session& session, int max_tokens) override;

    nlohmann::json build_request(const Session& session,
                                  Message::Role role,
                                  const std::string& content,
                                  const std::string& tool_name,
                                  const std::string& tool_id,
                                  int max_tokens = 0) override;
};
