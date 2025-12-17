#include "shepherd.h"
#include "mcp_server.h"
#include <unistd.h>

#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#include <cstring>
#include <sstream>

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

    // Wait briefly for graceful shutdown
    int status;
    pid_t result = waitpid(pid_, &status, WNOHANG);

    if (result == 0) {
        // Still running, give it 1 second
        usleep(1000000);
        result = waitpid(pid_, &status, WNOHANG);
    }

    if (result == 0) {
        // Force kill
        dout(1) << "Forcefully terminating MCP server '" + config_.name + "'" << std::endl;
        kill(pid_, SIGTERM);
        usleep(500000);
        waitpid(pid_, &status, WNOHANG);
        kill(pid_, SIGKILL);
        waitpid(pid_, &status, 0);
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
