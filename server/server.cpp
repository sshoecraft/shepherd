#include "shepherd.h"
#include "server/server.h"
#include "logger.h"
#include "nlohmann/json.hpp"
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

using json = nlohmann::json;

// Server base class implementation
Server::Server(const std::string& host, int port, const std::string& server_type)
    : Frontend(), host(host), port(port), server_type(server_type) {
    // Set default control socket path
    // Prefer /var/tmp (persistent, user-writable) over /tmp
    if (access("/var/tmp", W_OK) == 0) {
        control_socket_path = "/var/tmp/shepherd.sock";
    } else {
        control_socket_path = "/tmp/shepherd.sock";
    }
}

Server::~Server() {
    // Ensure shutdown is called
    shutdown();
    // Join control thread if still joinable
    if (control_thread.joinable()) {
        control_thread.join();
    }
}

void Server::shutdown() {
    if (!running.exchange(false)) {
        return;  // Already shutting down
    }
    LOG_INFO("Server shutdown requested");

    // Signal any in-flight generation to cancel
    g_generation_cancelled = true;

    tcp_server.stop();
    control_server.stop();
}

void Server::register_common_endpoints() {
    // GET /health - Health check
    tcp_server.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "ok"},
            {"server_type", server_type},
            {"backend_connected", backend != nullptr}
        };
        res.set_content(response.dump(), "application/json");
    });

    // GET /status - Server status
    tcp_server.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        json status = {
            {"status", "ok"},
            {"server_type", server_type},
            {"uptime_seconds", uptime},
            {"requests_processed", requests_processed.load()},
            {"pid", getpid()}
        };

        if (backend) {
            status["model"] = backend->model_name;
            status["context_size"] = backend->context_size;
        }

        // Let subclass add additional info
        add_status_info(status);

        res.set_content(status.dump(), "application/json");
    });
}

void Server::register_control_endpoints() {
    // GET /status - Status via control socket
    control_server.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        json status = {
            {"status", "ok"},
            {"server_type", server_type},
            {"uptime_seconds", uptime},
            {"requests_processed", requests_processed.load()},
            {"pid", getpid()},
            {"host", host},
            {"port", port},
            {"control_socket", control_socket_path}
        };

        if (backend) {
            status["model"] = backend->model_name;
            status["context_size"] = backend->context_size;
        }

        // Let subclass add additional info
        add_status_info(status);

        res.set_content(status.dump(), "application/json");
    });

    // POST /shutdown - Graceful shutdown via control socket
    control_server.Post("/shutdown", [this](const httplib::Request&, httplib::Response& res) {
        LOG_INFO("Shutdown command received via control socket");

        json response = {
            {"status", "shutting_down"}
        };
        res.set_content(response.dump(), "application/json");

        // Schedule shutdown after response is sent
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            shutdown();
        }).detach();
    });
}

bool Server::start_control_socket() {
    // Remove stale socket file if it exists
    if (access(control_socket_path.c_str(), F_OK) == 0) {
        LOG_WARN("Removing stale control socket: " + control_socket_path);
        unlink(control_socket_path.c_str());
    }

    // Configure for Unix domain socket
    control_server.set_address_family(AF_UNIX);
    control_server.set_error_logger([](const auto& err, const auto* /*req*/) {
        LOG_ERROR("Control socket error: " + httplib::to_string(err));
    });

    // Register control endpoints
    register_control_endpoints();

    // Start control socket in a separate thread
    control_thread = std::thread([this]() {
        LOG_INFO("Control socket listening on " + control_socket_path);
        // Port must be non-zero to avoid httplib bug in bind_internal that doesn't handle AF_UNIX
        if (!control_server.listen(control_socket_path.c_str(), 1)) {
            LOG_ERROR("Failed to start control socket on " + control_socket_path + " (errno=" + std::to_string(errno) + ": " + strerror(errno) + ")");
            exit(1);
        }
    });

    // Wait for socket file to appear
    for (int i = 0; i < 50; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (access(control_socket_path.c_str(), F_OK) == 0) {
            chmod(control_socket_path.c_str(), 0600);
            return true;
        }
    }

    // Timeout - socket never appeared, exit
    LOG_ERROR("Control socket failed to start (timeout)");
    exit(1);
}

void Server::cleanup_control_socket() {
    // Join control thread (shutdown() already called stop())
    if (control_thread.joinable()) {
        control_thread.join();
    }

    // Remove socket file
    if (!control_socket_path.empty() && access(control_socket_path.c_str(), F_OK) == 0) {
        unlink(control_socket_path.c_str());
    }
}

int Server::run(Session& session) {

    // Record start time
    start_time = std::chrono::steady_clock::now();

    // Register common endpoints
    register_common_endpoints();

    // Let subclass register its endpoints
    register_endpoints(session);

    // Start control socket (required for graceful shutdown)
    if (!start_control_socket()) {
        LOG_ERROR("Failed to start control socket - server cannot start");
        return 1;
    }

    // Let subclass do any startup work
    on_server_start();

    LOG_INFO(server_type + " server ready on " + host + ":" + std::to_string(port));

    // Start TCP server (blocks until stopped)
    bool success = tcp_server.listen(host.c_str(), port);

    // Server stopped - cleanup
    running = false;

    // Let subclass do any cleanup
    on_server_stop();

    // Cleanup control socket
    cleanup_control_socket();

    if (!success) {
        LOG_ERROR("Failed to start " + server_type + " server on " + host + ":" + std::to_string(port));
        return 1;
    }

    LOG_INFO(server_type + " server stopped");
    return 0;
}
