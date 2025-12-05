#include "shepherd.h"
#include "server/server.h"
#include "server/api_server.h"
#include "tools/tool_parser.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Server base class implementation
Server::Server(const std::string& host, int port)
    : Frontend(), host(host), port(port) {
}

Server::~Server() {
}

// Helper: Extract tool call from Response (handles both structured and text-based)
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    // If backend already parsed tool calls, use them
    if (!resp.tool_calls.empty()) {
        return resp.tool_calls[0];
    }

    // Otherwise parse from text content
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

int run_server(std::unique_ptr<Backend>& backend,
                const std::string& server_host,
                int server_port) {

    LOG_INFO("Starting API server on " + server_host + ":" + std::to_string(server_port));

    // Use the C++ HTTP server
    return run_api_server(backend.get(), server_host, server_port);
}
