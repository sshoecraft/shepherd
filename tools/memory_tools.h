#pragma once

#include "tool.h"
#include <iomanip>

class SearchMemoryTool : public Tool {
public:
    SearchMemoryTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class SetFactTool : public Tool {
public:
    SetFactTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class GetFactTool : public Tool {
public:
    GetFactTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class ClearFactTool : public Tool {
public:
    ClearFactTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class StoreMemoryTool : public Tool {
public:
    StoreMemoryTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class ClearMemoryTool : public Tool {
public:
    ClearMemoryTool() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class Tools;
void register_memory_tools(Tools& tools, bool enable = false);
