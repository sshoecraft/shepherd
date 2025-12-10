#include "control.h"
#include "../llama.cpp/vendor/cpp-httplib/httplib.h"
#include <unistd.h>
#include <iostream>

using json = nlohmann::json;

ControlClient::ControlClient(const std::string& socket_path)
    : socket_path(socket_path) {
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
        // Try /var/tmp first, then /tmp
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

        // Pretty print status
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
