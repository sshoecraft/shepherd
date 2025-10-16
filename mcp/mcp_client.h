#pragma once

#include "mcp_server.h"
#include "../nlohmann/json.hpp"
#include <string>
#include <memory>
#include <map>
#include <vector>

using json = nlohmann::json;

class MCPClientError : public std::runtime_error {
public:
    explicit MCPClientError(const std::string& message) : std::runtime_error(message) {}
};

// MCP tool schema
struct MCPTool {
    std::string name;
    std::string description;
    json input_schema;
};

// MCP resource schema
struct MCPResource {
    std::string uri;
    std::string name;
    std::string description;
    std::string mime_type;
};

// MCP prompt schema
struct MCPPrompt {
    std::string name;
    std::string description;
    json arguments;
};

// MCP server capabilities
struct MCPCapabilities {
    bool supports_tools = false;
    bool supports_resources = false;
    bool supports_prompts = false;
};

// JSON-RPC client for MCP protocol
class MCPClient {
public:
    explicit MCPClient(std::unique_ptr<MCPServer> server);
    ~MCPClient();

    // Protocol methods
    void initialize();
    std::vector<MCPTool> list_tools();
    json call_tool(const std::string& name, const json& arguments);

    // Resources
    std::vector<MCPResource> list_resources();
    json read_resource(const std::string& uri);

    // Prompts
    std::vector<MCPPrompt> list_prompts();
    json get_prompt(const std::string& name, const json& arguments = json::object());

    // Server info
    const std::string& get_server_name() const { return server_->get_name(); }
    const MCPCapabilities& get_capabilities() const { return capabilities_; }
    bool is_initialized() const { return initialized_; }

private:
    std::unique_ptr<MCPServer> server_;
    MCPCapabilities capabilities_;
    bool initialized_;
    int next_request_id_;

    // JSON-RPC helpers
    json create_request(const std::string& method, const json& params = json::object());
    json send_request(const json& request);
    void send_notification(const std::string& method, const json& params = json::object());
    json parse_response(const std::string& line);
    void validate_response(const json& response, int expected_id);
};
