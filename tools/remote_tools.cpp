#include "remote_tools.h"
#include "tools.h"
#include "../shepherd.h"
#include <iostream>
#include <set>

using json = nlohmann::json;

RemoteToolProxy::RemoteToolProxy(const std::string& name,
                                 const std::string& description,
                                 const nlohmann::json& parameters_schema,
                                 const std::string& server_url,
                                 const std::string& api_key)
    : tool_name(name)
    , tool_description(description)
    , params_schema(parameters_schema)
    , server_url(server_url)
    , api_key(api_key) {
}

void RemoteToolProxy::ensure_client() const {
    if (!http_client) {
        http_client = std::make_unique<HttpClient>();
        http_client->set_timeout(300);  // 5 minute timeout for tool execution
    }
}

std::string RemoteToolProxy::unsanitized_name() const {
    return tool_name;
}

std::string RemoteToolProxy::description() const {
    return tool_description;
}

std::string RemoteToolProxy::parameters() const {
    // Legacy format - extract from schema
    std::string result;
    if (params_schema.contains("properties")) {
        for (auto& [key, value] : params_schema["properties"].items()) {
            if (!result.empty()) result += ", ";
            result += key + ": " + value.value("type", "any");
        }
    }
    return result;
}

std::vector<ParameterDef> RemoteToolProxy::get_parameters_schema() const {
    std::vector<ParameterDef> params;

    if (!params_schema.contains("properties")) {
        return params;
    }

    std::set<std::string> required_set;
    if (params_schema.contains("required") && params_schema["required"].is_array()) {
        for (const auto& r : params_schema["required"]) {
            required_set.insert(r.get<std::string>());
        }
    }

    for (auto& [key, value] : params_schema["properties"].items()) {
        ParameterDef param;
        param.name = key;
        param.type = value.value("type", "string");
        param.description = value.value("description", "");
        param.required = required_set.count(key) > 0;
        params.push_back(param);
    }

    return params;
}

std::map<std::string, std::any> RemoteToolProxy::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    ensure_client();

    // Build request JSON
    json request;
    request["name"] = tool_name;
    request["tool_call_id"] = "";

    // Convert args to JSON
    json arguments = json::object();
    for (const auto& [key, value] : args) {
        try {
            if (value.type() == typeid(std::string)) {
                arguments[key] = std::any_cast<std::string>(value);
            } else if (value.type() == typeid(int)) {
                arguments[key] = std::any_cast<int>(value);
            } else if (value.type() == typeid(double)) {
                arguments[key] = std::any_cast<double>(value);
            } else if (value.type() == typeid(bool)) {
                arguments[key] = std::any_cast<bool>(value);
            } else if (value.type() == typeid(long)) {
                arguments[key] = std::any_cast<long>(value);
            } else if (value.type() == typeid(long long)) {
                arguments[key] = std::any_cast<long long>(value);
            } else {
                // Try to get as string as last resort
                try {
                    arguments[key] = std::any_cast<std::string>(value);
                } catch (...) {
                    arguments[key] = "[unsupported type]";
                }
            }
        } catch (...) {
            arguments[key] = "[conversion error]";
        }
    }
    request["arguments"] = arguments;

    // Build headers
    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    if (!api_key.empty()) {
        headers["Authorization"] = "Bearer " + api_key;
    }

    // Make request to /tools/execute
    std::string url = server_url + "/tools/execute";
    dout(1) << "RemoteToolProxy: executing " + tool_name + " via " + url << std::endl;

    HttpResponse response = http_client->post(url, request.dump(), headers);

    // Handle HTTP errors
    if (response.is_error()) {
        result["success"] = false;
        std::string error_msg = "HTTP error: ";
        if (!response.error_message.empty()) {
            error_msg += response.error_message;
        } else {
            error_msg += "status " + std::to_string(response.status_code);
        }
        if (response.status_code == 401) {
            error_msg += " (authentication failed - check API key)";
        } else if (response.status_code == 404) {
            error_msg += " (tool not found or server does not have --server-tools enabled)";
        }
        result["error"] = error_msg;
        return result;
    }

    // Parse response JSON
    try {
        json resp = json::parse(response.body);

        bool success = resp.value("success", false);
        result["success"] = success;

        if (success) {
            result["content"] = resp.value("content", "");
            if (resp.contains("summary") && !resp["summary"].is_null()) {
                result["summary"] = resp.value("summary", "");
            }
        } else {
            result["error"] = resp.value("error", "Unknown remote error");
        }
    } catch (const json::exception& e) {
        result["success"] = false;
        result["error"] = "Failed to parse response: " + std::string(e.what());
    }

    return result;
}

int register_remote_tools(Tools& tools, const std::string& server_url, const std::string& api_key) {
    HttpClient client;
    client.set_timeout(30);  // 30 second timeout for tool listing

    std::map<std::string, std::string> headers;
    headers["Accept"] = "application/json";
    if (!api_key.empty()) {
        headers["Authorization"] = "Bearer " + api_key;
    }

    std::string url = server_url + "/tools";
    dout(1) << "Fetching remote tools from " + url << std::endl;

    HttpResponse response = client.get(url, headers);

    if (response.is_error()) {
        std::string error_msg = "Failed to fetch remote tools: ";
        if (!response.error_message.empty()) {
            error_msg += response.error_message;
        } else {
            error_msg += "HTTP " + std::to_string(response.status_code);
        }
        if (response.status_code == 401) {
            error_msg += " (authentication failed - check API key)";
        } else if (response.status_code == 404) {
            error_msg += " (server may not have --server-tools enabled)";
        }
        std::cerr << error_msg << std::endl;
        return -1;
    }

    try {
        json resp = json::parse(response.body);

        if (!resp.contains("tools") || !resp["tools"].is_array()) {
            std::cerr << "Invalid response format from /tools endpoint" << std::endl;
            return -1;
        }

        int count = 0;
        for (const auto& tool : resp["tools"]) {
            std::string name = tool.value("name", "");
            std::string desc = tool.value("description", "");
            json params = tool.value("parameters", json::object());

            if (name.empty()) continue;

            auto proxy = std::make_unique<RemoteToolProxy>(name, desc, params, server_url, api_key);
            tools.register_tool(std::move(proxy), "core");
            count++;
        }

        dout(1) << "Registered " + std::to_string(count) + " remote tool proxies" << std::endl;
        return count;

    } catch (const json::exception& e) {
        std::cerr << "Failed to parse remote tools response: " + std::string(e.what()) << std::endl;
        return -1;
    }
}
