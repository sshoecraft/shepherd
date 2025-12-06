#pragma once

#include "backends/backend.h"
#include "provider.h"
#include "session.h"
#include <string>
#include <memory>
#include <vector>

/// @brief Base class for all frontend presentation layers (CLI, Server)
/// Manages backend, providers, and session lifecycle
class Frontend {
public:
    Frontend();
    virtual ~Frontend();

    /// @brief Factory method to create appropriate frontend
    /// @param mode Frontend mode: "cli", "api-server", "cli-server"
    /// @param host Server host (for server modes)
    /// @param port Server port (for server modes)
    /// @param cmdline_provider Optional provider from command-line override
    static std::unique_ptr<Frontend> create(const std::string& mode,
                                             const std::string& host,
                                             int port,
                                             Provider* cmdline_provider = nullptr);

    /// @brief Initialize the frontend (register tools, connect to provider, etc)
    /// @param session Session to initialize with tools
    /// @param no_mcp If true, skip MCP initialization
    /// @param no_tools If true, skip all tool initialization
    /// @param provider_name Specific provider to connect to (empty = auto-select)
    virtual void init(Session& session,
                      bool no_mcp = false,
                      bool no_tools = false,
                      const std::string& provider_name = "") {}

    /// @brief Start the frontend main loop
    /// Pure virtual - subclasses implement their specific behavior
    virtual int run(Session& session) = 0;

    /// @brief Get provider by name (returns nullptr if not found)
    Provider* get_provider(const std::string& name);

    /// @brief List all provider names (sorted by priority)
    std::vector<std::string> list_providers() const;

    /// @brief Connect to next available provider
    /// @param session Session for backend initialization
    /// @return true if connected, false if all providers fail
    bool connect_next_provider(Session& session);

    /// @brief Connect to a specific provider by name
    /// @param name Provider name
    /// @param session Session for backend initialization
    /// @return true if connected, false if connection fails
    bool connect_provider(const std::string& name, Session& session);

    // Provider list owned by frontend
    std::vector<Provider> providers;
    std::string current_provider;

    // Backend owned by frontend after connect
    std::unique_ptr<Backend> backend;
};
