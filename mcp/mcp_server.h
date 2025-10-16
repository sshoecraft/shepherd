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

    // Info
    const std::string& get_name() const { return config_.name; }
    pid_t get_pid() const { return pid_; }
    std::string get_stderr() const;

private:
    Config config_;
    pid_t pid_;
    int stdin_fd_;
    int stdout_fd_;
    int stderr_fd_;
    std::string stderr_buffer_;

    void close_fds();
};
