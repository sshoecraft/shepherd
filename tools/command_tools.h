#pragma once

#include "tool.h"

class ExecuteCommandTool : public Tool {
public:
    std::string unsanitized_name() const override { return "execute_command"; }
    std::string description() const override { return "Execute a shell command"; }
    std::string parameters() const override { return "command=\"command\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class GetEnvironmentVariableTool : public Tool {
public:
    std::string unsanitized_name() const override { return "get_env"; }
    std::string description() const override { return "Get an environment variable"; }
    std::string parameters() const override { return "name=\"env_var_name\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class ListProcessesTool : public Tool {
public:
    std::string unsanitized_name() const override { return "list_processes"; }
    std::string description() const override { return "List running processes"; }
    std::string parameters() const override { return ""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Function to register all command tools
class Tools;
void register_command_tools(Tools& tools);