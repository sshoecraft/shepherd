#pragma once

#include "mcp_server.h"
#include "nlohmann/json.hpp"
#include <string>
#include <memory>
#include <map>
#include <vector>

class MCPClientError : public std::runtime_error {
public:
    explicit MCPClientError(const std::string& message) : std::runtime_error(message) {}
};

// MCP tool schema
struct MCPTool {
    std::string name;
    std::string description;
    nlohmann::json input_schema;
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
    nlohmann::json arguments;
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
    nlohmann::json call_tool(const std::string& name, const nlohmann::json& arguments);

    // Resources
    std::vector<MCPResource> list_resources();
    nlohmann::json read_resource(const std::string& uri);

    // Prompts
    std::vector<MCPPrompt> list_prompts();
    nlohmann::json get_prompt(const std::string& name, const nlohmann::json& arguments = nlohmann::json::object());

    std::unique_ptr<MCPServer> server;
    MCPCapabilities capabilities;
    bool initialized;
    int next_request_id;

    // JSON-RPC helpers
    nlohmann::json create_request(const std::string& method, const nlohmann::json& params = nlohmann::json::object());
    nlohmann::json send_request(const nlohmann::json& request);
    void send_notification(const std::string& method, const nlohmann::json& params = nlohmann::json::object());
    nlohmann::json parse_response(const std::string& line);
    void validate_response(const nlohmann::json& response, int expected_id);
};
