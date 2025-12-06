#pragma once

#include "backends/backend.h"
#include "session.h"
#include <string>
#include <memory>

/// @brief Base class for all frontend presentation layers (CLI, Server)
/// Manages backend and session lifecycle
class Frontend {
public:
    Frontend();
    virtual ~Frontend();

    static std::unique_ptr<Frontend> create( const std::string& mode, const std::string& host, int port);

    /// @brief Initialize the frontend (register tools, etc)
    /// Default does nothing. CLI overrides to register tools.
    /// @param no_mcp If true, skip MCP initialization
    /// @param no_tools If true, skip all tool initialization
    virtual void init(bool no_mcp = false, bool no_tools = false) {}

    /// @brief Start the frontend (initialize backend, run main loop)
    /// Pure virtual - subclasses implement their specific behavior
    virtual int run(std::unique_ptr<Backend>& backend, Session& session) = 0;

protected:
    // Subclasses have access to these during run()
};
