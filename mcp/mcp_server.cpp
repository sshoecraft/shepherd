#include "shepherd.h"
#include "mcp_server.h"
#include "nlohmann/json.hpp"
#include <unistd.h>

#include <sys/wait.h>
#include <sys/select.h>
#include <sys/prctl.h>
#include <fcntl.h>
#include <signal.h>
#include <cstring>
#include <sstream>
#include <chrono>

MCPServer::MCPServer(const Config& config)
    : config_(config), pid_(-1), stdin_fd_(-1), stdout_fd_(-1), stderr_fd_(-1) {
}

MCPServer::~MCPServer() {
    if (is_running()) {
        stop();
    }
}

MCPServer::MCPServer(MCPServer&& other) noexcept
    : config_(std::move(other.config_)),
      pid_(other.pid_),
      stdin_fd_(other.stdin_fd_),
      stdout_fd_(other.stdout_fd_),
      stderr_fd_(other.stderr_fd_),
      stderr_buffer_(std::move(other.stderr_buffer_)) {
    other.pid_ = -1;
    other.stdin_fd_ = -1;
    other.stdout_fd_ = -1;
    other.stderr_fd_ = -1;
}

MCPServer& MCPServer::operator=(MCPServer&& other) noexcept {
    if (this != &other) {
        if (is_running()) {
            stop();
        }
        config_ = std::move(other.config_);
        pid_ = other.pid_;
        stdin_fd_ = other.stdin_fd_;
        stdout_fd_ = other.stdout_fd_;
        stderr_fd_ = other.stderr_fd_;
        stderr_buffer_ = std::move(other.stderr_buffer_);
        other.pid_ = -1;
        other.stdin_fd_ = -1;
        other.stdout_fd_ = -1;
        other.stderr_fd_ = -1;
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

    pid_ = fork();
    if (pid_ < 0) {
        throw MCPServerError("Failed to fork: " + std::string(strerror(errno)));
    }

    if (pid_ == 0) {
        // Child process
        // Die when parent dies (Linux-specific)
        prctl(PR_SET_PDEATHSIG, SIGTERM);

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
        for (const auto& [key, value] : config_.env) {
            setenv(key.c_str(), value.c_str(), 1);
        }

        // Build argument list
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(config_.command.c_str()));
        for (const auto& arg : config_.args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        // Execute command
        execvp(config_.command.c_str(), argv.data());

        // If we get here, exec failed
        std::cerr << "Failed to execute " << config_.command << ": " << strerror(errno) << std::endl;
        _exit(1);
    }

    // Parent process
    // Close child ends of pipes
    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Store our ends
    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];
    stderr_fd_ = stderr_pipe[0];

    // Perform SMCP handshake if credentials are configured
    // (must happen before making stdout non-blocking)
    if (!config_.smcp_credentials.empty()) {
        dout(1) << "Performing SMCP handshake for '" + config_.name + "'" << std::endl;
        if (!perform_smcp_handshake()) {
            stop();
            throw MCPServerError("SMCP handshake failed for " + config_.name);
        }
    }

    // Make stdout and stderr non-blocking
    fcntl(stdout_fd_, F_SETFL, fcntl(stdout_fd_, F_GETFL) | O_NONBLOCK);
    fcntl(stderr_fd_, F_SETFL, fcntl(stderr_fd_, F_GETFL) | O_NONBLOCK);

    dout(1) << "MCP server '" + config_.name + "' started with PID " + std::to_string(pid_) << std::endl;
}

void MCPServer::stop() {
    if (!is_running()) {
        return;
    }

    dout(1) << "Stopping MCP server '" + config_.name + "' (PID " + std::to_string(pid_) + ")" << std::endl;

    // Close stdin to signal EOF
    if (stdin_fd_ >= 0) {
        close(stdin_fd_);
        stdin_fd_ = -1;
    }

    // Send SIGTERM and wait briefly
    kill(pid_, SIGTERM);

    int status;
    // Quick non-blocking check first
    if (waitpid(pid_, &status, WNOHANG) == 0) {
        // Still running, wait up to 100ms
        usleep(100000);
        if (waitpid(pid_, &status, WNOHANG) == 0) {
            // Force kill
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
        }
    }

    close_fds();
    pid_ = -1;
}

bool MCPServer::is_running() const {
    if (pid_ <= 0) {
        return false;
    }

    int status;
    pid_t result = waitpid(pid_, &status, WNOHANG);
    return result == 0;
}

void MCPServer::write_line(const std::string& line) {
    if (!is_running()) {
        throw MCPServerError("Server not running");
    }

    std::string data = line + "\n";
    ssize_t written = write(stdin_fd_, data.c_str(), data.size());

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
        ssize_t n = read(stdout_fd_, buffer, sizeof(buffer) - 1);

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
    if (!is_running() || stdout_fd_ < 0) {
        return false;
    }

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(stdout_fd_, &fds);

    struct timeval timeout = {0, 0}; // Non-blocking
    return select(stdout_fd_ + 1, &fds, nullptr, nullptr, &timeout) > 0;
}

std::string MCPServer::get_stderr() const {
    if (stderr_fd_ < 0) {
        return stderr_buffer_;
    }

    // Read any available stderr
    char buffer[4096];
    std::string result = stderr_buffer_;

    while (true) {
        ssize_t n = read(stderr_fd_, buffer, sizeof(buffer) - 1);
        if (n <= 0) break;
        buffer[n] = '\0';
        result += buffer;
    }

    return result;
}

void MCPServer::close_fds() {
    if (stdin_fd_ >= 0) {
        close(stdin_fd_);
        stdin_fd_ = -1;
    }
    if (stdout_fd_ >= 0) {
        close(stdout_fd_);
        stdout_fd_ = -1;
    }
    if (stderr_fd_ >= 0) {
        close(stderr_fd_);
        stderr_fd_ = -1;
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
            throw MCPServerError("Timeout waiting for response from " + config_.name);
        }

        // Use select to wait for data with remaining timeout
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(stdout_fd_, &fds);

        int remaining_ms = timeout_ms - elapsed;
        struct timeval tv;
        tv.tv_sec = remaining_ms / 1000;
        tv.tv_usec = (remaining_ms % 1000) * 1000;

        int ret = select(stdout_fd_ + 1, &fds, nullptr, nullptr, &tv);
        if (ret < 0) {
            throw MCPServerError("select() failed: " + std::string(strerror(errno)));
        }
        if (ret == 0) {
            continue;  // Timeout, loop will check total elapsed time
        }

        // Data available
        ssize_t n = read(stdout_fd_, buffer, sizeof(buffer) - 1);
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
            std::cerr << "SMCP: Expected +READY from " << config_.name
                      << ", got: " << line << std::endl;
            return false;
        }

        dout(1) << "SMCP: Received +READY from " << config_.name << std::endl;

        // Send credentials as JSON object
        nlohmann::json creds_json(config_.smcp_credentials);
        write_line(creds_json.dump());

        dout(1) << "SMCP: Sent " << config_.smcp_credentials.size()
                << " credentials to " << config_.name << std::endl;

        // Wait for +OK (5s timeout)
        line = read_line_timeout(5000);
        if (line != "+OK") {
            std::cerr << "SMCP: Expected +OK from " << config_.name
                      << ", got: " << line << std::endl;
            return false;
        }

        dout(1) << "SMCP: Handshake complete for " << config_.name << std::endl;
        return true;

    } catch (const MCPServerError& e) {
        std::cerr << "SMCP handshake error for " << config_.name
                  << ": " << e.what() << std::endl;
        return false;
    }
}
