#include "shepherd.h"
#include "mcp_server.h"
#include "nlohmann/json.hpp"
#include <unistd.h>

#include <sys/wait.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>
#include <cstring>
#include <sstream>
#include <chrono>

MCPServer::MCPServer(const Config& config)
    : server_config(config), pid(-1), stdin_fd(-1), stdout_fd(-1), stderr_fd(-1) {
}

MCPServer::~MCPServer() {
    if (is_running()) {
        stop();
    }
}

MCPServer::MCPServer(MCPServer&& other) noexcept
    : server_config(std::move(other.server_config)),
      pid(other.pid),
      stdin_fd(other.stdin_fd),
      stdout_fd(other.stdout_fd),
      stderr_fd(other.stderr_fd),
      stderr_buffer(std::move(other.stderr_buffer)) {
    other.pid = -1;
    other.stdin_fd = -1;
    other.stdout_fd = -1;
    other.stderr_fd = -1;
}

MCPServer& MCPServer::operator=(MCPServer&& other) noexcept {
    if (this != &other) {
        if (is_running()) {
            stop();
        }
        server_config = std::move(other.server_config);
        pid = other.pid;
        stdin_fd = other.stdin_fd;
        stdout_fd = other.stdout_fd;
        stderr_fd = other.stderr_fd;
        stderr_buffer = std::move(other.stderr_buffer);
        other.pid = -1;
        other.stdin_fd = -1;
        other.stdout_fd = -1;
        other.stderr_fd = -1;
    }
    return *this;
}

void MCPServer::start() {
    if (is_running()) {
        throw MCPServerError("Server already running");
    }

    // Create pipes: [read_end, write_end]
    int stdin_pipe[2], stdout_pipe[2], stderr_pipe[2];

    if (pipe(stdin_pipe) < 0 || pipe(stdout_pipe) < 0 || pipe(stderr_pipe) < 0) {
        throw MCPServerError("Failed to create pipes: " + std::string(strerror(errno)));
    }

    pid = fork();
    if (pid < 0) {
        throw MCPServerError("Failed to fork: " + std::string(strerror(errno)));
    }

    if (pid == 0) {
        // Child process
        // Redirect stdin from pipe
        dup2(stdin_pipe[0], STDIN_FILENO);
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);

        // Redirect stdout to pipe
        dup2(stdout_pipe[1], STDOUT_FILENO);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        // Redirect stderr to pipe
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);

        // Set environment variables
        for (const auto& [key, value] : server_config.env) {
            setenv(key.c_str(), value.c_str(), 1);
        }

        // Build argument list
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(server_config.command.c_str()));
        for (const auto& arg : server_config.args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        // Execute command
        execvp(server_config.command.c_str(), argv.data());

        // If we get here, exec failed
        std::cerr << "Failed to execute " << server_config.command << ": " << strerror(errno) << std::endl;
        _exit(1);
    }

    // Parent process
    // Close child ends of pipes
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Store our ends
    stdin_fd = stdin_pipe[1];
    stdout_fd = stdout_pipe[0];
    stderr_fd = stderr_pipe[0];

    // Perform SMCP handshake if credentials are configured
    // (must happen before making stdout non-blocking)
    if (!server_config.smcp_credentials.empty()) {
        dout(1) << "Performing SMCP handshake for '" + server_config.name + "'" << std::endl;
        if (!perform_smcp_handshake()) {
            stop();
            throw MCPServerError("SMCP handshake failed for " + server_config.name);
        }
    }

    // Make stdout and stderr non-blocking
    fcntl(stdout_fd, F_SETFL, fcntl(stdout_fd, F_GETFL) | O_NONBLOCK);
    fcntl(stderr_fd, F_SETFL, fcntl(stderr_fd, F_GETFL) | O_NONBLOCK);

    dout(1) << "MCP server '" + server_config.name + "' started with PID " + std::to_string(pid) << std::endl;
}

void MCPServer::stop() {
    if (!is_running()) {
        return;
    }

    dout(1) << "Stopping MCP server '" + server_config.name + "' (PID " + std::to_string(pid) + ")" << std::endl;

    // Close stdin to signal EOF
    if (stdin_fd >= 0) {
        close(stdin_fd);
        stdin_fd = -1;
    }

    // Send SIGTERM and wait briefly
    kill(pid, SIGTERM);

    int status;
    // Quick non-blocking check first
    if (waitpid(pid, &status, WNOHANG) == 0) {
        // Still running, wait up to 100ms
        usleep(100000);
        if (waitpid(pid, &status, WNOHANG) == 0) {
            // Force kill
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
        }
    }

    close_fds();
    pid = -1;
}

bool MCPServer::is_running() const {
    if (pid <= 0) {
        return false;
    }

    int status;
    pid_t result = waitpid(pid, &status, WNOHANG);
    return result == 0;
}

void MCPServer::write_line(const std::string& line) {
    if (!is_running()) {
        throw MCPServerError("Server not running");
    }

    std::string data = line + "\n";
    ssize_t written = write(stdin_fd, data.c_str(), data.size());

    if (written < 0) {
        throw MCPServerError("Failed to write to server: " + std::string(strerror(errno)));
    }
    if (written != static_cast<ssize_t>(data.size())) {
        throw MCPServerError("Partial write to server");
    }
}

std::string MCPServer::read_line() {
    if (!is_running()) {
        throw MCPServerError("Server not running");
    }

    std::string line;
    char buffer[4096];

    while (true) {
        ssize_t n = read(stdout_fd, buffer, sizeof(buffer) - 1);

        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // No data available, wait a bit
                usleep(10000); // 10ms
                continue;
            }
            throw MCPServerError("Failed to read from server: " + std::string(strerror(errno)));
        }

        if (n == 0) {
            throw MCPServerError("Server closed stdout");
        }

        buffer[n] = '\0';
        line += buffer;

        // Check for newline
        auto newline_pos = line.find('\n');
        if (newline_pos != std::string::npos) {
            std::string result = line.substr(0, newline_pos);
            return result;
        }
    }
}

bool MCPServer::has_output() const {
    if (!is_running() || stdout_fd < 0) {
        return false;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(stdout_fd, &fds);

    struct timeval timeout = {0, 0}; // Non-blocking
    return select(stdout_fd + 1, &fds, nullptr, nullptr, &timeout) > 0;
}

std::string MCPServer::read_stderr() const {
    if (stderr_fd < 0) {
        return stderr_buffer;
    }

    // Read any available stderr
    char buffer[4096];
    std::string result = stderr_buffer;

    while (true) {
        ssize_t n = read(stderr_fd, buffer, sizeof(buffer) - 1);
        if (n <= 0) break;
        buffer[n] = '\0';
        result += buffer;
    }

    return result;
}

void MCPServer::close_fds() {
    if (stdin_fd >= 0) {
        close(stdin_fd);
        stdin_fd = -1;
    }
    if (stdout_fd >= 0) {
        close(stdout_fd);
        stdout_fd = -1;
    }
    if (stderr_fd >= 0) {
        close(stderr_fd);
        stderr_fd = -1;
    }
}

std::string MCPServer::read_line_timeout(int timeout_ms) {
    std::string line;
    char buffer[4096];
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // Check timeout
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeout_ms) {
            throw MCPServerError("Timeout waiting for response from " + server_config.name);
        }

        // Use select to wait for data with remaining timeout
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(stdout_fd, &fds);

        int remaining_ms = timeout_ms - elapsed;
        struct timeval tv;
        tv.tv_sec = remaining_ms / 1000;
        tv.tv_usec = (remaining_ms % 1000) * 1000;

        int ret = select(stdout_fd + 1, &fds, nullptr, nullptr, &tv);
        if (ret < 0) {
            throw MCPServerError("select() failed: " + std::string(strerror(errno)));
        }
        if (ret == 0) {
            continue;  // Timeout, loop will check total elapsed time
        }

        // Data available
        ssize_t n = read(stdout_fd, buffer, sizeof(buffer) - 1);
        if (n < 0) {
            throw MCPServerError("Failed to read from server: " + std::string(strerror(errno)));
        }
        if (n == 0) {
            throw MCPServerError("Server closed stdout during SMCP handshake");
        }

        buffer[n] = '\0';
        line += buffer;

        // Check for newline
        auto newline_pos = line.find('\n');
        if (newline_pos != std::string::npos) {
            return line.substr(0, newline_pos);
        }
    }
}

bool MCPServer::perform_smcp_handshake() {
    try {
        // Wait for +READY (10s timeout)
        std::string line = read_line_timeout(10000);
        if (line != "+READY") {
            std::cerr << "SMCP: Expected +READY from " << server_config.name
                      << ", got: " << line << std::endl;
            return false;
        }

        dout(1) << "SMCP: Received +READY from " << server_config.name << std::endl;

        // Send credentials as JSON object
        nlohmann::json creds_json(server_config.smcp_credentials);
        write_line(creds_json.dump());

        dout(1) << "SMCP: Sent " << server_config.smcp_credentials.size()
                << " credentials to " << server_config.name << std::endl;

        // Wait for +OK (5s timeout)
        line = read_line_timeout(5000);
        if (line != "+OK") {
            std::cerr << "SMCP: Expected +OK from " << server_config.name
                      << ", got: " << line << std::endl;
            return false;
        }

        dout(1) << "SMCP: Handshake complete for " << server_config.name << std::endl;
        return true;

    } catch (const MCPServerError& e) {
        std::cerr << "SMCP handshake error for " << server_config.name
                  << ": " << e.what() << std::endl;
        return false;
    }
}
