#include "mcp_client.h"
#include "../logger.h"

MCPClient::MCPClient(std::unique_ptr<MCPServer> server)
    : server_(std::move(server)), initialized_(false), next_request_id_(1) {
}

MCPClient::~MCPClient() {
    if (server_ && server_->is_running()) {
        server_->stop();
    }
}

void MCPClient::initialize() {
    if (initialized_) {
        return;
    }

    LOG_INFO("Initializing MCP client for server: " + server_->get_name());

    // Start server if not running
    if (!server_->is_running()) {
        server_->start();
    }

    // Send initialize request
    nlohmann::json init_params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", nlohmann::json::object()},
        {"clientInfo", {
            {"name", "shepherd"},
            {"version", "1.0.0"}
        }}
    };

    nlohmann::json request = create_request("initialize", init_params);
    nlohmann::json response = send_request(request);

    // Parse server info and capabilities
    if (response.contains("result")) {
        auto result = response["result"];

        if (result.contains("capabilities")) {
            auto caps = result["capabilities"];
            capabilities_.supports_tools = caps.value("tools", nlohmann::json::object()).value("listChanged", false) ||
                                           caps.contains("tools");
            capabilities_.supports_resources = caps.contains("resources");
            capabilities_.supports_prompts = caps.contains("prompts");
        }

        if (result.contains("serverInfo")) {
            LOG_INFO("Connected to MCP server: " + result["serverInfo"].value("name", "unknown"));
        }
    }

    // Send initialized notification
    send_notification("notifications/initialized");

    initialized_ = true;
    LOG_DEBUG("MCP client initialized for: " + server_->get_name());
}

std::vector<MCPTool> MCPClient::list_tools() {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    if (!capabilities_.supports_tools) {
        return {};
    }

    LOG_DEBUG("Listing tools from MCP server: " + server_->get_name());

    nlohmann::json request = create_request("tools/list");
    nlohmann::json response = send_request(request);

    std::vector<MCPTool> tools;

    if (response.contains("result") && response["result"].contains("tools")) {
        for (const auto& tool_json : response["result"]["tools"]) {
            MCPTool tool;
            tool.name = tool_json.value("name", "");
            tool.description = tool_json.value("description", "");
            tool.input_schema = tool_json.value("inputSchema", nlohmann::json::object());

            if (!tool.name.empty()) {
                tools.push_back(tool);
            }
        }
    }

    LOG_DEBUG("Found " + std::to_string(tools.size()) + " tools from: " + server_->get_name());
    return tools;
}

nlohmann::json MCPClient::call_tool(const std::string& name, const nlohmann::json& arguments) {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    LOG_DEBUG("Calling MCP tool '" + name + "' on server: " + server_->get_name());

    nlohmann::json params = {
        {"name", name},
        {"arguments", arguments}
    };

    nlohmann::json request = create_request("tools/call", params);
    nlohmann::json response = send_request(request);

    if (response.contains("result")) {
        return response["result"];
    }

    throw MCPClientError("Tool call failed: no result in response");
}

std::vector<MCPResource> MCPClient::list_resources() {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    if (!capabilities_.supports_resources) {
        return {};
    }

    LOG_DEBUG("Listing resources from MCP server: " + server_->get_name());

    nlohmann::json request = create_request("resources/list");
    nlohmann::json response = send_request(request);

    std::vector<MCPResource> resources;

    if (response.contains("result") && response["result"].contains("resources")) {
        for (const auto& res_json : response["result"]["resources"]) {
            MCPResource resource;
            resource.uri = res_json.value("uri", "");
            resource.name = res_json.value("name", "");
            resource.description = res_json.value("description", "");
            resource.mime_type = res_json.value("mimeType", "text/plain");

            if (!resource.uri.empty()) {
                resources.push_back(resource);
            }
        }
    }

    LOG_DEBUG("Found " + std::to_string(resources.size()) + " resources from: " + server_->get_name());
    return resources;
}

nlohmann::json MCPClient::read_resource(const std::string& uri) {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    LOG_DEBUG("Reading resource '" + uri + "' from server: " + server_->get_name());

    nlohmann::json params = {{"uri", uri}};
    nlohmann::json request = create_request("resources/read", params);
    nlohmann::json response = send_request(request);

    if (response.contains("result")) {
        return response["result"];
    }

    throw MCPClientError("Resource read failed: no result in response");
}

std::vector<MCPPrompt> MCPClient::list_prompts() {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    if (!capabilities_.supports_prompts) {
        return {};
    }

    LOG_DEBUG("Listing prompts from MCP server: " + server_->get_name());

    nlohmann::json request = create_request("prompts/list");
    nlohmann::json response = send_request(request);

    std::vector<MCPPrompt> prompts;

    if (response.contains("result") && response["result"].contains("prompts")) {
        for (const auto& prompt_json : response["result"]["prompts"]) {
            MCPPrompt prompt;
            prompt.name = prompt_json.value("name", "");
            prompt.description = prompt_json.value("description", "");
            prompt.arguments = prompt_json.value("arguments", nlohmann::json::array());

            if (!prompt.name.empty()) {
                prompts.push_back(prompt);
            }
        }
    }

    LOG_DEBUG("Found " + std::to_string(prompts.size()) + " prompts from: " + server_->get_name());
    return prompts;
}

nlohmann::json MCPClient::get_prompt(const std::string& name, const nlohmann::json& arguments) {
    if (!initialized_) {
        throw MCPClientError("Client not initialized");
    }

    LOG_DEBUG("Getting prompt '" + name + "' from server: " + server_->get_name());

    nlohmann::json params = {
        {"name", name},
        {"arguments", arguments}
    };

    nlohmann::json request = create_request("prompts/get", params);
    nlohmann::json response = send_request(request);

    if (response.contains("result")) {
        return response["result"];
    }

    throw MCPClientError("Prompt get failed: no result in response");
}

nlohmann::json MCPClient::create_request(const std::string& method, const nlohmann::json& params) {
    nlohmann::json request = {
        {"jsonrpc", "2.0"},
        {"id", next_request_id_++},
        {"method", method}
    };

    if (!params.is_null() && !params.empty()) {
        request["params"] = params;
    }

    return request;
}

nlohmann::json MCPClient::send_request(const nlohmann::json& request) {
    int request_id = request["id"];
    std::string request_str = request.dump();

    LOG_DEBUG("MCP request: " + request_str);

    server_->write_line(request_str);
    std::string response_line = server_->read_line();

    // Truncate long responses in debug output
    std::string debug_response = response_line;
    const size_t MAX_DEBUG_LEN = 500;
    if (debug_response.length() > MAX_DEBUG_LEN) {
        debug_response = debug_response.substr(0, MAX_DEBUG_LEN) + "... [truncated " + std::to_string(debug_response.length() - MAX_DEBUG_LEN) + " chars]";
    }
    LOG_DEBUG("MCP response: " + debug_response);

    nlohmann::json response = parse_response(response_line);
    validate_response(response, request_id);

    return response;
}

void MCPClient::send_notification(const std::string& method, const nlohmann::json& params) {
    nlohmann::json notification = {
        {"jsonrpc", "2.0"},
        {"method", method}
    };

    if (!params.is_null() && !params.empty()) {
        notification["params"] = params;
    }

    std::string notification_str = notification.dump();
    LOG_DEBUG("MCP notification: " + notification_str);

    server_->write_line(notification_str);
}

nlohmann::json MCPClient::parse_response(const std::string& line) {
    try {
        return nlohmann::json::parse(line);
    } catch (const nlohmann::json::exception& e) {
        throw MCPClientError("Failed to parse JSON response: " + std::string(e.what()) +
                           "\nResponse: " + line);
    }
}

void MCPClient::validate_response(const nlohmann::json& response, int expected_id) {
    if (!response.contains("jsonrpc") || response["jsonrpc"] != "2.0") {
        throw MCPClientError("Invalid JSON-RPC response");
    }

    if (response.contains("id") && response["id"] != expected_id) {
        throw MCPClientError("Response ID mismatch");
    }

    if (response.contains("error")) {
        auto error = response["error"];
        std::string message = error.value("message", "Unknown error");
        int code = error.value("code", -1);
        throw MCPClientError("MCP error " + std::to_string(code) + ": " + message);
    }
}
