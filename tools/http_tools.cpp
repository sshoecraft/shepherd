#include "../shepherd.h"
#include "http_tools.h"
#include <map>

// Simple HTTP client using curl command (for cross-platform compatibility)
class SimpleHTTPClient {
private:
    std::string execute_curl(const std::string& curl_command) {
        std::array<char, 128> buffer;
        std::string result;

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(curl_command.c_str(), "r"), pclose);

        if (!pipe) {
            throw std::runtime_error("Failed to execute curl command");
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }

        return result;
    }

public:
    struct HTTPResponse {
        int status_code;
        std::string body;
        std::map<std::string, std::string> headers;
        bool success;
        std::string error;
    };

    HTTPResponse request(const std::string& method, const std::string& url,
                        const std::map<std::string, std::string>& headers = {},
                        const std::string& body = "") {
        HTTPResponse response;
        response.success = false;
        response.status_code = 0;

        try {
            std::string curl_cmd = "curl -s -w \"\\nHTTP_STATUS:%{http_code}\\n\" ";

            // Add method
            if (method != "GET") {
                curl_cmd += "-X " + method + " ";
            }

            // Add headers
            for (const auto& header : headers) {
                curl_cmd += "-H \"" + header.first + ": " + header.second + "\" ";
            }

            // Add body for POST/PUT
            if (!body.empty() && (method == "POST" || method == "PUT")) {
                curl_cmd += "-d '" + body + "' ";
            }

            // Add URL
            curl_cmd += "\"" + url + "\"";

            // Debug output controlled by global flag
            if (g_debug_mode) {
                std::cout << "HTTPClient: Executing: " << curl_cmd << std::endl;
            }

            std::string output = execute_curl(curl_cmd);

            // Parse output to separate body and status
            size_t status_pos = output.find("HTTP_STATUS:");
            if (status_pos != std::string::npos) {
                response.body = output.substr(0, status_pos);

                // Remove trailing newline from body
                if (!response.body.empty() && response.body.back() == '\n') {
                    response.body.pop_back();
                }

                std::string status_line = output.substr(status_pos + 12); // Skip "HTTP_STATUS:"
                response.status_code = std::stoi(status_line);
                response.success = (response.status_code >= 200 && response.status_code < 300);
            } else {
                response.body = output;
                response.status_code = 200;
                response.success = true;
            }

        } catch (const std::exception& e) {
            response.error = std::string("HTTP request failed: ") + e.what();
            response.success = false;
        }

        return response;
    }
};

std::vector<ParameterDef> HTTPRequestTool::get_parameters_schema() const {
    return {
        {"url", "string", "The URL to make the HTTP request to", true, "", "", {}},
        {"method", "string", "HTTP method (GET, POST, PUT, DELETE)", false, "GET", "", {}},
        {"body", "string", "Optional request body (JSON format)", false, "", "", {}}
    };
}

std::map<std::string, std::any> HTTPRequestTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string url = tool_utils::get_string(args, "url");
    std::string method = tool_utils::get_string(args, "method", "GET");
    std::string body = tool_utils::get_string(args, "body");

    if (url.empty()) {
        result["error"] = std::string("url is required");
        result["success"] = false;
        return result;
    }

    try {
        SimpleHTTPClient client;

        // Parse headers if provided
        std::map<std::string, std::string> headers;
        headers["User-Agent"] = "Shepherd/1.0";

        // If body is provided, set content-type to JSON
        if (!body.empty()) {
            headers["Content-Type"] = "application/json";
        }

        auto response = client.request(method, url, headers, body);

        result["status_code"] = response.status_code;
        result["body"] = response.body;
        result["success"] = response.success;

        if (!response.error.empty()) {
            result["error"] = response.error;
        }

        if (g_debug_mode) {
            std::cout << "HTTPRequest: " << method << " " << url
                      << " -> " << response.status_code << std::endl;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error making HTTP request: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> HTTPGetTool::get_parameters_schema() const {
    return {
        {"url", "string", "The URL to make the HTTP GET request to", true, "", "", {}}
    };
}

std::map<std::string, std::any> HTTPGetTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string url = tool_utils::get_string(args, "url");

    if (url.empty()) {
        result["error"] = std::string("url is required");
        result["success"] = false;
        return result;
    }

    try {
        SimpleHTTPClient client;

        std::map<std::string, std::string> headers;
        headers["User-Agent"] = "Shepherd/1.0";

        auto response = client.request("GET", url, headers);

        result["status_code"] = response.status_code;
        result["body"] = response.body;
        result["success"] = response.success;

        if (!response.error.empty()) {
            result["error"] = response.error;
        }

        if (g_debug_mode) {
            std::cout << "HTTPGet: " << url << " -> " << response.status_code << std::endl;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error making HTTP GET request: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> HTTPPostTool::get_parameters_schema() const {
    return {
        {"url", "string", "The URL to make the HTTP POST request to", true, "", "", {}},
        {"body", "string", "Optional request body (JSON format)", false, "", "", {}}
    };
}

std::map<std::string, std::any> HTTPPostTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string url = tool_utils::get_string(args, "url");
    std::string body = tool_utils::get_string(args, "body");

    if (url.empty()) {
        result["error"] = std::string("url is required");
        result["success"] = false;
        return result;
    }

    try {
        SimpleHTTPClient client;

        std::map<std::string, std::string> headers;
        headers["User-Agent"] = "Shepherd/1.0";
        headers["Content-Type"] = "application/json";

        auto response = client.request("POST", url, headers, body);

        result["status_code"] = response.status_code;
        result["body"] = response.body;
        result["success"] = response.success;

        if (!response.error.empty()) {
            result["error"] = response.error;
        }

        if (g_debug_mode) {
            std::cout << "HTTPPost: " << url << " -> " << response.status_code << std::endl;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error making HTTP POST request: ") + e.what();
        result["success"] = false;
    }

    return result;
}

void register_http_tools() {
    auto& registry = ToolRegistry::instance();

    registry.register_tool(std::make_unique<HTTPRequestTool>());
    registry.register_tool(std::make_unique<HTTPGetTool>());
    registry.register_tool(std::make_unique<HTTPPostTool>());

    LOG_DEBUG("Registered HTTP tools: http_request, http_get, http_post");
}