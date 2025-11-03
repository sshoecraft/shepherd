#pragma once

#include "../tools/tool.h"
#include "api_tool_config.h"
#include <memory>

/// @brief Adapter that wraps an API backend as a Shepherd Tool
/// This allows one backend to call another backend as a tool
class APIToolAdapter : public Tool {
public:
    explicit APIToolAdapter(const APIToolEntry& entry);

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;

private:
    APIToolEntry config;
};
