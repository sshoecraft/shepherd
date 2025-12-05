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

    /// @brief Start the frontend (initialize backend, run main loop)
    /// Pure virtual - subclasses implement their specific behavior
    virtual int run(std::unique_ptr<Backend>& backend, Session& session) = 0;

protected:
    // Subclasses have access to these during run()
};
