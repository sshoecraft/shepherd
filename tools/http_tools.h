#pragma once

#include "tool.h"

class HTTPRequestTool : public Tool {
public:
    std::string unsanitized_name() const override { return "http_request"; }
    std::string description() const override { return "Make HTTP requests with GET, POST, PUT, DELETE methods"; }
    std::string parameters() const override { return "url=\"url\", method=\"GET\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class HTTPGetTool : public Tool {
public:
    std::string unsanitized_name() const override { return "http_get"; }
    std::string description() const override { return "Make HTTP GET request"; }
    std::string parameters() const override { return "url=\"url\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class HTTPPostTool : public Tool {
public:
    std::string unsanitized_name() const override { return "http_post"; }
    std::string description() const override { return "Make HTTP POST request with JSON body"; }
    std::string parameters() const override { return "url=\"url\", data=\"json_data\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Function to register all HTTP tools
void register_http_tools();