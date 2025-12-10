#pragma once

#include <string>
#include "nlohmann/json.hpp"

/// @brief Control client for communicating with shepherd server via Unix socket
class ControlClient {
public:
    ControlClient(const std::string& socket_path);

    /// @brief Get server status
    /// @return JSON response or error object with "error" field
    nlohmann::json get_status();

    /// @brief Request server shutdown
    /// @return JSON response or error object with "error" field
    nlohmann::json shutdown();

    /// @brief Check if socket exists and is accessible
    bool socket_exists() const;

private:
    /// @brief Send HTTP request over Unix socket
    /// @param method HTTP method (GET, POST)
    /// @param path URL path (e.g., "/status")
    /// @return Response body as string, or empty on error
    std::string http_request(const std::string& method, const std::string& path);

    std::string socket_path;
};

/// @brief Handle ctl subcommand
/// @param args Arguments after "ctl"
/// @return Exit code
int handle_ctl_args(const std::vector<std::string>& args);
