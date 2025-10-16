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

private:
    std::shared_ptr<MCPClient> client_;
    MCPTool mcp_tool_;

    // Convert MCP JSON schema to Shepherd parameter string
    std::string schema_to_parameters(const json& schema) const;

    // Convert Shepherd args to MCP JSON arguments
    json args_to_json(const std::map<std::string, std::any>& args) const;
};
