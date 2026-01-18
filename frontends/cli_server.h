#pragma once

#include "server.h"
#include "backend.h"
#include "client_output.h"
#include "../session.h"
#include "../tools/tools.h"
#include <string>
#include <memory>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <set>
#include <functional>

// Forward declaration
class CLIServer;

/// @brief CLI Server state with input queue
struct CliServerState {
    CLIServer* server = nullptr;  // For calling execute_tool
    Backend* backend = nullptr;
    Session* session = nullptr;
    Tools* tools = nullptr;

    // Input queue with condition variable
    std::deque<std::string> input_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    // Request mutex - ensures only one request is processed at a time
    std::mutex request_mutex;

    // SSE observer clients (connected to /updates)
    std::vector<ClientOutputs::StreamingOutput*> observers;
    std::mutex observers_mutex;

    std::atomic<bool> processing{false};
    std::atomic<bool> running{true};
    std::string current_request;

    // Add prompt to queue
    void add_input(const std::string& prompt);

    // Get queue depth
    size_t queue_size();

    // Wait for and get next input
    std::string get_next_input();

    // Send event to all observers - removes disconnected ones
    void send_to_observers(const std::function<void(ClientOutputs::ClientOutput&)>& action);

    // Register/unregister observer
    void register_observer(ClientOutputs::StreamingOutput* observer);
    void unregister_observer(ClientOutputs::StreamingOutput* observer);
};

/// @brief CLI Server - HTTP server that executes tools locally
class CLIServer : public Server {
public:
    CLIServer(const std::string& host, int port, const std::string& auth_mode = "none");
    ~CLIServer();

    /// @brief Initialize tools and RAG
    void init(bool no_mcp = false, bool no_tools = false) override;

protected:
    /// @brief Register CLI server endpoints
    void register_endpoints() override;

    /// @brief Add CLI-specific status info
    void add_status_info(nlohmann::json& status) override;

    /// @brief Start processor thread
    void on_server_start() override;

    /// @brief Stop processor thread
    void on_server_stop() override;

    /// @brief Signal threads to exit before server stops
    void on_shutdown() override;

private:
    // Server state shared with endpoints
    CliServerState state;

    // Processor thread for async requests
    std::thread processor_thread;
};
