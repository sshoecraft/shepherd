#pragma once

#include "tool.h"

class ListMcpResourcesTool : public Tool {
public:
    ListMcpResourcesTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class ReadMcpResourcesTool : public Tool {
public:
    ReadMcpResourcesTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

void register_mcp_resource_tools();
