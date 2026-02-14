#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

class MCPServerError : public std::runtime_error {
public:
    explicit MCPServerError(const std::string& message) : std::runtime_error(message) {}
};

// Represents a running MCP server process
class MCPServer {
public:
    struct Config {
        std::string name;                          // Server identifier
        std::string command;                       // Executable path
        std::vector<std::string> args;            // Command line arguments
        std::map<std::string, std::string> env;   // Environment variables
        std::map<std::string, std::string> smcp_credentials;  // SMCP credentials (sent as JSON)
    };

    explicit MCPServer(const Config& config);
    ~MCPServer();

    // Disable copy, allow move
    MCPServer(const MCPServer&) = delete;
    MCPServer& operator=(const MCPServer&) = delete;
    MCPServer(MCPServer&&) noexcept;
    MCPServer& operator=(MCPServer&&) noexcept;

    // Process lifecycle
    void start();
    void stop();
    bool is_running() const;

    // I/O
    void write_line(const std::string& line);
    std::string read_line();
    bool has_output() const;
    std::string read_stderr() const;

    Config server_config;
    pid_t pid;
    int stdin_fd;
    int stdout_fd;
    int stderr_fd;
    std::string stderr_buffer;

    void close_fds();

    // SMCP handshake support
    bool perform_smcp_handshake();
    std::string read_line_timeout(int timeout_ms);
};
