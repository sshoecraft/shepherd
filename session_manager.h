#pragma once

#include "session.h"
#include "tools/tools.h"
#include "auth.h"
#include "config.h"
#include "backend.h"
#include "nlohmann/json.hpp"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>

/// @brief Per-API-key session state for multi-tenant API server
struct ManagedSession {
    std::unique_ptr<Session> session;
    std::unique_ptr<Tools> tools;
    std::chrono::steady_clock::time_point last_access;
    std::string key_name;
    std::mutex session_mutex;
    uint64_t requests_processed = 0;
    uint64_t tool_executions = 0;
};

/// @brief Manages per-API-key sessions for multi-tenant API server
/// Sessions are created on first request and persist until server restart
/// or manual clear via clear_session()
class SessionManager {
public:
    /// @brief Construct session manager
    /// @param backend Shared backend for all sessions (not owned)
    /// @param config Configuration for session initialization (not owned)
    /// @param no_mcp If true, skip MCP tool initialization
    /// @param no_tools If true, skip all tool initialization
    SessionManager(Backend* backend, Config* config, bool no_mcp, bool no_tools, bool mem_tools = false);
    ~SessionManager();

    /// @brief Get or create session for an API key
    /// @param api_key The validated API key (already authenticated)
    /// @param entry The ApiKeyEntry metadata
    /// @return Pointer to managed session (never null for valid keys)
    ManagedSession* get_session(const std::string& api_key, const ApiKeyEntry& entry);

    /// @brief Check if session exists for key
    bool has_session(const std::string& api_key) const;

    /// @brief Clear/reset a specific session
    /// @param api_key The API key whose session to clear
    void clear_session(const std::string& api_key);

    /// @brief Get number of active sessions
    size_t session_count() const;

    /// @brief Get status for all sessions (for /status endpoint)
    nlohmann::json get_status() const;

private:
    /// @brief Create and initialize a new session
    std::unique_ptr<ManagedSession> create_session(const std::string& api_key,
                                                    const ApiKeyEntry& entry);

    Backend* backend;  // Shared backend (not owned)
    Config* config;    // Configuration (not owned)
    bool no_mcp;
    bool no_tools;
    bool mem_tools;
    std::unordered_map<std::string, std::unique_ptr<ManagedSession>> sessions;
    mutable std::mutex manager_mutex;
};
