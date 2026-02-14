#pragma once

#include "../tools/tool.h"
#include "mcp_client.h"
#include <memory>

// Adapter that wraps an MCP tool as a Shepherd Tool
class MCPToolAdapter : public Tool {
public:
    MCPToolAdapter(std::shared_ptr<MCPClient> client, const MCPTool& mcp_tool);

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;

    std::shared_ptr<MCPClient> client;
    MCPTool mcp_tool;

    // Convert MCP JSON schema to Shepherd parameter string
    std::string schema_to_parameters(const nlohmann::json& schema) const;

    // Convert Shepherd args to MCP JSON arguments
    nlohmann::json args_to_json(const std::map<std::string, std::any>& args) const;
};
