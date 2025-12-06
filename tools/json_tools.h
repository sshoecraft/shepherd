#pragma once

#include "tool.h"

class ParseJSONTool : public Tool {
public:
    std::string unsanitized_name() const override { return "parse_json"; }
    std::string description() const override { return "Parse JSON string and return structured data"; }
    std::string parameters() const override { return "json=\"json_string\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class SerializeJSONTool : public Tool {
public:
    std::string unsanitized_name() const override { return "serialize_json"; }
    std::string description() const override { return "Serialize data to JSON string"; }
    std::string parameters() const override { return "data=\"data\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class QueryJSONTool : public Tool {
public:
    std::string unsanitized_name() const override { return "query_json"; }
    std::string description() const override { return "Query JSON data with JSONPath expressions"; }
    std::string parameters() const override { return "json=\"json_data\", path=\"jsonpath_expression\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Function to register all JSON tools
class Tools;
void register_json_tools(Tools& tools);