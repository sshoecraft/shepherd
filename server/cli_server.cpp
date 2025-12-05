#include "cli_server.h"
#include "../shepherd.h"
#include "../tools/tool.h"
#include "../tools/tool_parser.h"
#include "../tools/utf8_sanitizer.h"
#include "../llama.cpp/vendor/cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <mutex>
#include <queue>
#include <atomic>
#include <future>

using json = nlohmann::json;

extern std::unique_ptr<Config> config;

// CLIServer class implementation
CLIServer::CLIServer(const std::string& host, int port)
    : Server(host, port) {
}

CLIServer::~CLIServer() {
}

int CLIServer::run(std::unique_ptr<Backend>& backend, Session& session) {
    return run_cli_server(backend, session, host, port);
}

// Helper: Extract tool call from Response
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    if (!resp.tool_calls.empty()) {
        return resp.tool_calls[0];
    }
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

// Request queue entry
struct CliRequest {
    std::string prompt;
    std::promise<json> response_promise;
};

// CLI Server state
struct CliServerState {
    Backend* backend;
    Session* session;
    std::mutex request_mutex;
    std::atomic<bool> processing{false};
    std::atomic<int> queue_depth{0};
    std::string current_request;
};

// Process a single request with tool execution loop
static json process_request(CliServerState& state, const std::string& prompt) {
    json result;
    result["success"] = true;
    result["prompt"] = prompt;

    std::string accumulated_response;
    std::vector<json> tool_executions;

    // Add user message and generate initial response
    Response resp = state.session->add_message(Message::USER, prompt);

    if (!resp.success) {
        result["success"] = false;
        result["error"] = resp.error;
        return result;
    }

    accumulated_response = resp.content;

    // Tool execution loop
    const int max_tool_iterations = 50;
    int iteration = 0;

    while (iteration < max_tool_iterations) {
        iteration++;

        // Check for tool calls
        auto tool_call_opt = extract_tool_call(resp, state.backend);

        if (!tool_call_opt) {
            // No more tool calls - done
            break;
        }

        auto& tool_call = *tool_call_opt;
        std::string tool_name = tool_call.name;

        // Log tool execution
        json tool_exec;
        tool_exec["name"] = tool_name;
        tool_exec["parameters"] = tool_call.raw_json;

        // Execute the tool
        ToolResult tool_result = execute_tool(tool_name, tool_call.parameters);

        tool_exec["success"] = tool_result.success;
        if (tool_result.success) {
            // Sanitize UTF-8 in result
            std::string sanitized = utf8_sanitizer::sanitize_utf8(tool_result.content);
            tool_exec["result"] = sanitized;

            // Send tool result to model and get next response
            resp = state.session->add_message(Message::TOOL, sanitized, tool_name, tool_call.tool_call_id);
        } else {
            tool_exec["error"] = tool_result.error;

            // Send error to model
            std::string error_msg = "Error: " + tool_result.error;
            resp = state.session->add_message(Message::TOOL, error_msg, tool_name, tool_call.tool_call_id);
        }

        tool_executions.push_back(tool_exec);

        if (!resp.success) {
            result["success"] = false;
            result["error"] = resp.error;
            result["partial_response"] = accumulated_response;
            result["tool_executions"] = tool_executions;
            return result;
        }

        accumulated_response = resp.content;
    }

    result["response"] = accumulated_response;
    result["tool_executions"] = tool_executions;
    result["iterations"] = iteration;
    result["tokens"] = {
        {"prompt", resp.prompt_tokens},
        {"completion", resp.completion_tokens}
    };

    return result;
}

int run_cli_server(std::unique_ptr<Backend>& backend, Session& session,
                   const std::string& host, int port) {

    LOG_INFO("Starting CLI server on " + host + ":" + std::to_string(port));

    httplib::Server server;

    CliServerState state;
    state.backend = backend.get();
    state.session = &session;

    // Health endpoint
    server.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json response;
        response["status"] = "ok";
        response["mode"] = "cli_server";
        response["model"] = backend->model_name;
        res.set_content(response.dump(), "application/json");
    });

    // Status endpoint
    server.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        json response;
        response["status"] = "ok";
        response["processing"] = state.processing.load();
        response["queue_depth"] = state.queue_depth.load();
        response["model"] = backend->model_name;
        response["context_size"] = backend->context_size;
        response["messages"] = state.session->messages.size();

        if (state.processing.load() && !state.current_request.empty()) {
            // Truncate for display
            std::string truncated = state.current_request;
            if (truncated.length() > 100) {
                truncated = truncated.substr(0, 100) + "...";
            }
            response["current_request"] = truncated;
        }

        res.set_content(response.dump(), "application/json");
    });

    // Request endpoint
    server.Post("/request", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.request_mutex);

        state.processing = true;
        state.queue_depth++;

        try {
            // Parse request
            json request_json;
            std::string prompt;

            if (!req.body.empty()) {
                try {
                    request_json = json::parse(req.body);
                    if (request_json.contains("prompt")) {
                        prompt = request_json["prompt"].get<std::string>();
                    }
                } catch (const json::exception&) {
                    // Not JSON, treat body as raw prompt
                    prompt = req.body;
                }
            }

            if (prompt.empty()) {
                json error_response;
                error_response["success"] = false;
                error_response["error"] = "Missing prompt";
                res.status = 400;
                res.set_content(error_response.dump(), "application/json");
                state.processing = false;
                state.queue_depth--;
                return;
            }

            state.current_request = prompt;

            // Process the request
            json response = process_request(state, prompt);

            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            json error_response;
            error_response["success"] = false;
            error_response["error"] = e.what();
            res.status = 500;
            res.set_content(error_response.dump(), "application/json");
        }

        state.current_request.clear();
        state.processing = false;
        state.queue_depth--;
    });

    // Clear endpoint - reset conversation
    server.Post("/clear", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.request_mutex);

        state.session->messages.clear();
        state.session->last_prompt_tokens = 0;
        state.session->total_tokens = 0;
        state.session->last_user_message_index = -1;
        state.session->last_user_message_tokens = 0;
        state.session->last_assistant_message_index = -1;
        state.session->last_assistant_message_tokens = 0;

        json response;
        response["success"] = true;
        response["message"] = "Conversation cleared";
        res.set_content(response.dump(), "application/json");
    });

    LOG_INFO("CLI server endpoints: /health, /status, /request, /clear");
    LOG_INFO("CLI server listening on " + host + ":" + std::to_string(port));

    if (!server.listen(host.c_str(), port)) {
        LOG_ERROR("Failed to start CLI server on " + host + ":" + std::to_string(port));
        return 1;
    }

    return 0;
}
