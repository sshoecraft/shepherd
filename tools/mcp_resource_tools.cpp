
#include "mcp_resource_tools.h"
#include "mcp/mcp.h"
#include "logger.h"
#include <sstream>

std::string ListMcpResourcesTool::unsanitized_name() const {
    return "list_mcp_resources";
}

std::string ListMcpResourcesTool::description() const {
    return "List resources from MCP servers. Optionally filter by server name.";
}

std::string ListMcpResourcesTool::parameters() const {
    return "server=\"server_name\" (optional)";
}

std::map<std::string, std::any> ListMcpResourcesTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string server_name = tool_utils::get_string(args, "server", "");

    try {
        auto& manager = MCP::instance();
        std::ostringstream oss;

        if (server_name.empty()) {
            // List resources from all servers
            auto all_resources = manager.list_all_resources();

            if (all_resources.empty()) {
                result["success"] = true;
                result["content"] = std::string("No resources found from any MCP server");
                return result;
            }

            oss << "Resources from MCP servers:\n\n";
            for (const auto& [srv_name, resources] : all_resources) {
                oss << "Server: " << srv_name << " (" << resources.size() << " resources)\n";
                for (const auto& resource : resources) {
                    oss << "  - " << resource.uri << "\n";
                    oss << "    Name: " << resource.name << "\n";
                    if (!resource.description.empty()) {
                        oss << "    Description: " << resource.description << "\n";
                    }
                    oss << "    MIME Type: " << resource.mime_type << "\n";
                }
                oss << "\n";
            }
        } else {
            // List resources from specific server
            auto resources = manager.list_resources(server_name);

            if (resources.empty()) {
                result["success"] = true;
                result["content"] = std::string("No resources found from server: ") + server_name;
                return result;
            }

            oss << "Resources from " << server_name << " (" << resources.size() << " resources):\n\n";
            for (const auto& resource : resources) {
                oss << "  - " << resource.uri << "\n";
                oss << "    Name: " << resource.name << "\n";
                if (!resource.description.empty()) {
                    oss << "    Description: " << resource.description << "\n";
                }
                oss << "    MIME Type: " << resource.mime_type << "\n";
            }
        }

        result["success"] = true;
        result["content"] = oss.str();

    } catch (const std::exception& e) {
        LOG_ERROR("Error listing MCP resources: " + std::string(e.what()));
        result["success"] = false;
        result["error"] = std::string("Failed to list resources: ") + e.what();
    }

    return result;
}

std::string ReadMcpResourcesTool::unsanitized_name() const {
    return "read_mcp_resource";
}

std::string ReadMcpResourcesTool::description() const {
    return "Read a specific resource from an MCP server by URI";
}

std::string ReadMcpResourcesTool::parameters() const {
    return "server=\"server_name\", uri=\"resource_uri\"";
}

std::map<std::string, std::any> ReadMcpResourcesTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string server_name = tool_utils::get_string(args, "server");
    std::string uri = tool_utils::get_string(args, "uri");

    if (server_name.empty()) {
        result["success"] = false;
        result["error"] = std::string("Server name is required");
        return result;
    }

    if (uri.empty()) {
        result["success"] = false;
        result["error"] = std::string("Resource URI is required");
        return result;
    }

    LOG_DEBUG("Reading resource '" + uri + "' from server: " + server_name);

    try {
        auto& manager = MCP::instance();
        auto resource_data = manager.read_resource(server_name, uri);

        // Format the response
        std::ostringstream oss;
        oss << "Resource: " << uri << "\n";
        oss << "Server: " << server_name << "\n\n";
        oss << "Content:\n" << resource_data.dump(2);

        result["success"] = true;
        result["content"] = oss.str();

    } catch (const std::exception& e) {
        LOG_ERROR("Error reading MCP resource: " + std::string(e.what()));
        result["success"] = false;
        result["error"] = std::string("Failed to read resource: ") + e.what();
    }

    return result;
}

void register_mcp_resource_tools() {
    auto& registry = ToolRegistry::instance();
    registry.register_tool(std::make_unique<ListMcpResourcesTool>());
    registry.register_tool(std::make_unique<ReadMcpResourcesTool>());
}
