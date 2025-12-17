#include "shepherd.h"
#include "server.h"
#include "nlohmann/json.hpp"
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

using json = nlohmann::json;

// ControlClient implementation

ControlClient::ControlClient(const std::string& path)
    : socket_path(path) {
}

bool ControlClient::socket_exists() const {
    return access(socket_path.c_str(), F_OK) == 0;
}

std::string ControlClient::http_request(const std::string& method, const std::string& path) {
    httplib::Client client(socket_path);
    client.set_address_family(AF_UNIX);
    client.set_connection_timeout(5);
    client.set_read_timeout(5);

    httplib::Result res;
    if (method == "GET") {
        res = client.Get(path);
    } else if (method == "POST") {
        res = client.Post(path, "", "application/json");
    } else {
        return "";
    }

    if (!res || res->status != 200) {
        return "";
    }

    return res->body;
}

json ControlClient::get_status() {
    std::string response = http_request("GET", "/status");
    if (response.empty()) {
        return json{{"error", "Failed to connect to server"}};
    }

    try {
        return json::parse(response);
    } catch (const json::exception& e) {
        return json{{"error", "Invalid response from server"}};
    }
}

json ControlClient::shutdown() {
    std::string response = http_request("POST", "/shutdown");
    if (response.empty()) {
        return json{{"error", "Failed to connect to server"}};
    }

    try {
        return json::parse(response);
    } catch (const json::exception& e) {
        return json{{"error", "Invalid response from server"}};
    }
}

int handle_ctl_args(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Usage: shepherd ctl <command> [options]\n\n";
        std::cout << "Commands:\n";
        std::cout << "  status [--socket PATH]    Get server status\n";
        std::cout << "  shutdown [--socket PATH]  Request server shutdown\n";
        std::cout << "\nDefault socket path: /var/tmp/shepherd.sock (or /tmp/shepherd.sock)\n";
        return 0;
    }

    std::string command = args[0];
    std::string socket_path;

    // Parse --socket option
    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "--socket" && i + 1 < args.size()) {
            socket_path = args[++i];
        }
    }

    // Auto-detect socket if not specified
    if (socket_path.empty()) {
        if (access("/var/tmp/shepherd.sock", F_OK) == 0) {
            socket_path = "/var/tmp/shepherd.sock";
        } else if (access("/tmp/shepherd.sock", F_OK) == 0) {
            socket_path = "/tmp/shepherd.sock";
        } else {
            std::cerr << "Error: No running server found.\n";
            std::cerr << "Expected socket at /var/tmp/shepherd.sock or /tmp/shepherd.sock\n";
            return 1;
        }
    }

    ControlClient client(socket_path);

    if (!client.socket_exists()) {
        std::cerr << "Error: Socket not found: " << socket_path << "\n";
        return 1;
    }

    if (command == "status") {
        json status = client.get_status();

        if (status.contains("error")) {
            std::cerr << "Error: " << status["error"].get<std::string>() << "\n";
            return 1;
        }

        std::cout << "Server Status:\n";
        std::cout << "  Type:              " << status.value("server_type", "unknown") << "\n";
        std::cout << "  Status:            " << status.value("status", "unknown") << "\n";
        std::cout << "  PID:               " << status.value("pid", 0) << "\n";
        std::cout << "  Uptime:            " << status.value("uptime_seconds", 0) << " seconds\n";
        std::cout << "  Requests:          " << status.value("requests_processed", 0) << "\n";

        if (status.contains("model")) {
            std::cout << "  Model:             " << status["model"].get<std::string>() << "\n";
        }
        if (status.contains("context_size")) {
            std::cout << "  Context Size:      " << status["context_size"].get<int>() << "\n";
        }
        if (status.contains("host")) {
            std::cout << "  Host:              " << status["host"].get<std::string>() << "\n";
        }
        if (status.contains("port")) {
            std::cout << "  Port:              " << status["port"].get<int>() << "\n";
        }
        if (status.contains("processing")) {
            std::cout << "  Processing:        " << (status["processing"].get<bool>() ? "yes" : "no") << "\n";
        }
        if (status.contains("queue_depth")) {
            std::cout << "  Queue Depth:       " << status["queue_depth"].get<int>() << "\n";
        }
        if (status.contains("current_request")) {
            std::cout << "  Current Request:   " << status["current_request"].get<std::string>() << "\n";
        }

        std::cout << "  Socket:            " << socket_path << "\n";
        return 0;
    }

    if (command == "shutdown") {
        std::cout << "Requesting server shutdown...\n";
        json response = client.shutdown();

        if (response.contains("error")) {
            std::cerr << "Error: " << response["error"].get<std::string>() << "\n";
            return 1;
        }

        if (response.value("status", "") == "shutting_down") {
            std::cout << "Server shutdown initiated.\n";
            return 0;
        }

        std::cerr << "Unexpected response: " << response.dump() << "\n";
        return 1;
    }

    std::cerr << "Unknown command: " << command << "\n";
    std::cerr << "Use 'shepherd ctl' for usage.\n";
    return 1;
}

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

    // Stub callback for backend construction
    // Servers handle streaming via per-request HTTP response writers rather than
    // this callback. Each request creates its own streaming context.
    callback = [](CallbackEvent event, const std::string& content,
                  const std::string& name, const std::string& id) -> bool {
        return true;
    };
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
    dout(1) << "Server shutdown requested" << std::endl;

    // Signal any in-flight generation to cancel
    g_generation_cancelled = true;

    // Let subclass signal threads to exit before stopping server
    on_shutdown();

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
        dout(1) << "Shutdown command received via control socket" << std::endl;

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
        dout(1) << std::string("WARNING: ") +"Removing stale control socket: " + control_socket_path << std::endl;
        unlink(control_socket_path.c_str());
    }

    // Configure for Unix domain socket
    control_server.set_address_family(AF_UNIX);
    control_server.set_error_logger([](const auto& err, const auto* /*req*/) {
        std::cerr << "Control socket error: " + httplib::to_string(err) << std::endl;
    });

    // Register control endpoints
    register_control_endpoints();

    // Start control socket in a separate thread
    control_thread = std::thread([this]() {
        dout(1) << "Control socket listening on " + control_socket_path << std::endl;
        // Port must be non-zero to avoid httplib bug in bind_internal that doesn't handle AF_UNIX
        if (!control_server.listen(control_socket_path.c_str(), 1)) {
            std::cerr << "Failed to start control socket on " + control_socket_path + " (errno=" + std::to_string(errno) + ": " + strerror(errno) + ")" << std::endl;
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
    std::cerr << "Control socket failed to start (timeout)" << std::endl;
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

int Server::run() {

    // Record start time
    start_time = std::chrono::steady_clock::now();

    // Register common endpoints
    register_common_endpoints();

    // Let subclass register its endpoints (uses frontend's session member)
    register_endpoints();

    // Start control socket (required for graceful shutdown)
    if (!start_control_socket()) {
        std::cerr << "Failed to start control socket - server cannot start" << std::endl;
        return 1;
    }

    // Let subclass do any startup work
    on_server_start();

    std::cout << server_type << " server listening on " << host << ":" << port << std::endl;

    // Start TCP server (blocks until stopped)
    bool success = tcp_server.listen(host.c_str(), port);

    // Server stopped - cleanup
    running = false;

    // Let subclass do any cleanup
    on_server_stop();

    // Cleanup control socket
    cleanup_control_socket();

    if (!success) {
        std::cerr << "Failed to start " + server_type + " server on " + host + ":" + std::to_string(port) << std::endl;
        return 1;
    }

    dout(1) << server_type + " server stopped" << std::endl;
    return 0;
}
