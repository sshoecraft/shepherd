#include "session_manager.h"
#include "shepherd.h"
#include "rag.h"
#include "mcp/mcp.h"
#include "tools/filesystem_tools.h"
#include "tools/command_tools.h"
#include "tools/json_tools.h"
#include "tools/http_tools.h"
#include "tools/memory_tools.h"
#include "tools/mcp_resource_tools.h"
#include "tools/core_tools.h"
#include "tools/scheduler_tools.h"

extern std::unique_ptr<Config> config;

SessionManager::SessionManager(Backend* backend, Config* cfg, bool no_mcp, bool no_tools, bool mem_tools)
    : backend(backend), config(cfg), no_tools(no_tools), mem_tools(mem_tools) {
}

SessionManager::~SessionManager() {
    // Sessions are automatically cleaned up by unique_ptr
}

ManagedSession* SessionManager::get_session(const std::string& api_key, const ApiKeyEntry& entry) {
    std::lock_guard<std::mutex> lock(manager_mutex);

    auto it = sessions.find(api_key);
    if (it != sessions.end()) {
        // Update last access time
        it->second->last_access = std::chrono::steady_clock::now();
        return it->second.get();
    }

    // Create new session
    auto session = create_session(api_key, entry);
    ManagedSession* ptr = session.get();
    sessions[api_key] = std::move(session);
    return ptr;
}

bool SessionManager::has_session(const std::string& api_key) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return sessions.find(api_key) != sessions.end();
}

void SessionManager::clear_session(const std::string& api_key) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    sessions.erase(api_key);
}

size_t SessionManager::session_count() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return sessions.size();
}

nlohmann::json SessionManager::get_status() const {
    std::lock_guard<std::mutex> lock(manager_mutex);

    nlohmann::json status;
    status["active_sessions"] = sessions.size();

    nlohmann::json session_list = nlohmann::json::array();
    for (const auto& [key, managed] : sessions) {
        nlohmann::json s;
        s["key_name"] = managed->key_name;
        s["message_count"] = managed->session->messages.size();
        s["requests_processed"] = managed->requests_processed;
        s["tool_executions"] = managed->tool_executions;

        // Calculate time since last access
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - managed->last_access).count();
        s["idle_seconds"] = elapsed;

        session_list.push_back(s);
    }
    status["sessions"] = session_list;

    return status;
}

std::unique_ptr<ManagedSession> SessionManager::create_session(
    const std::string& api_key,
    const ApiKeyEntry& entry) {

    auto managed = std::make_unique<ManagedSession>();
    managed->key_name = entry.name;
    managed->last_access = std::chrono::steady_clock::now();

    // Create session
    managed->session = std::make_unique<Session>();
    managed->session->backend = backend;
    managed->session->system_message = config->system_message;
    managed->session->desired_completion_tokens =
        calculate_desired_completion_tokens(backend->context_size, backend->max_output_tokens);

    // Create tools instance
    managed->tools = std::make_unique<Tools>();

    if (!no_tools) {
        // Register all native tools
        register_filesystem_tools(*managed->tools);
        register_command_tools(*managed->tools);
        register_json_tools(*managed->tools);
        register_http_tools(*managed->tools);
        register_memory_tools(*managed->tools, mem_tools);
        register_mcp_resource_tools(*managed->tools);
        register_core_tools(*managed->tools);
        register_scheduler_tools(*managed->tools);

        // Note: MCP tools are initialized globally via MCP::instance()
        // For multi-tenant sessions, MCP tools would need special handling
        // since they register to a single Tools instance.
        // For now, we skip MCP for per-session tools.
        // TODO: Support MCP tools for multi-tenant sessions

        // Build the combined tool list
        managed->tools->build_all_tools();

        // Populate session.tools from Tools instance
        managed->tools->populate_session_tools(*managed->session);
    }

    return managed;
}
