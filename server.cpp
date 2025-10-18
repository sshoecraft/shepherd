#include "server.h"
#include "session_context.h"
#include "logger.h"
#include "tools/tool_parser.h"
#include "nlohmann/json.hpp"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <csignal>

using json = nlohmann::json;

int run_server_mode(std::unique_ptr<BackendManager>& backend,
                    const std::string& server_host,
                    int server_port) {

    LOG_INFO("Starting server mode on " + server_host + ":" + std::to_string(server_port));

    // Create Unix domain socket pair for bidirectional communication
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1) {
        LOG_ERROR("Failed to create socket pair: " + std::string(strerror(errno)));
        return 1;
    }

    int parent_fd = sv[0];  // Shepherd uses this
    int child_fd = sv[1];   // FastAPI uses this

    // Fork to spawn FastAPI server
    pid_t pid = fork();

    if (pid == -1) {
        LOG_ERROR("Failed to fork: " + std::string(strerror(errno)));
        close(parent_fd);
        close(child_fd);
        return 1;
    }

    if (pid == 0) {
        // Child process - run FastAPI server
        close(parent_fd);

        // Pass socket fd to Python via command line
        std::string fd_str = std::to_string(child_fd);
        std::string port_str = std::to_string(server_port);

        // Find Python interpreter
        const char* python = "python3";

        // Get executable directory to find server/api_server.py
        std::string exe_path = std::filesystem::canonical("/proc/self/exe").parent_path().string();
        std::string script_path = exe_path + "/server/api_server.py";

        // Check if script exists
        if (!std::filesystem::exists(script_path)) {
            // Try relative path (for development)
            script_path = "./server/api_server.py";
            if (!std::filesystem::exists(script_path)) {
                fprintf(stderr, "Error: Could not find server/api_server.py\n");
                exit(1);
            }
        }

        // Execute FastAPI server
        execl("/usr/bin/env", "env", python, script_path.c_str(),
              "--fd", fd_str.c_str(),
              "--port", port_str.c_str(),
              "--host", server_host.c_str(),
              nullptr);

        // If execl returns, it failed
        fprintf(stderr, "Failed to execute FastAPI server: %s\n", strerror(errno));
        exit(1);
    }

    // Parent process - Shepherd continues
    close(child_fd);

    // Store child PID for cleanup
    pid_t fastapi_pid = pid;

    printf("\nShepherd API Server starting...\n");
    printf("Backend: %s\n", backend->get_backend_name().c_str());
    printf("Model: %s\n", backend->get_model_name().c_str());
    printf("Listening on: http://%s:%d\n", server_host.c_str(), server_port);
    printf("\nEndpoints:\n");
    printf("  POST http://%s:%d/v1/chat/completions\n", server_host.c_str(), server_port);
    printf("  GET  http://%s:%d/v1/models\n", server_host.c_str(), server_port);
    printf("  GET  http://%s:%d/health\n", server_host.c_str(), server_port);
    printf("\nPress Ctrl+C to stop.\n\n");
    fflush(stdout);

    // Main server loop - read JSON from pipe, process, write back
    FILE* pipe_in = fdopen(parent_fd, "r");
    FILE* pipe_out = fdopen(dup(parent_fd), "w");

    if (!pipe_in || !pipe_out) {
        LOG_ERROR("Failed to open pipe streams");
        kill(fastapi_pid, SIGTERM);
        waitpid(fastapi_pid, nullptr, 0);
        return 1;
    }

    // Session context for single-session server with prefix caching
    // All requests are processed in sequence using this shared session
    SessionContext session;

    char* line = nullptr;
    size_t len = 0;
    ssize_t read_len;

    while ((read_len = getline(&line, &len, pipe_in)) != -1) {
        try {
            // Parse request
            std::string request_str(line);
            auto request = json::parse(request_str);

            LOG_DEBUG("Received request: " + request["action"].get<std::string>());

            json response;

            if (request["action"] == "generate") {
                // Single-session server with prefix caching
                // Parse incoming request into session context, then only send NEW items to backend

                // Parse tools from request (if provided)
                if (request.contains("tools") && request["tools"].is_array()) {
                    // Convert OpenAI tools format to SessionContext format
                    session.tools.clear();  // Replace tools with new set
                    for (const auto& tool : request["tools"]) {
                        SessionContext::ToolDefinition td;

                        // OpenAI format: tool["function"]["name"]
                        if (tool.contains("function") && tool["function"].is_object()) {
                            td.name = tool["function"].value("name", "");
                            td.description = tool["function"].value("description", "");
                            if (tool["function"].contains("parameters")) {
                                td.parameters_json = tool["function"]["parameters"].dump();
                            }
                        } else {
                            // Fallback: direct format (for testing)
                            td.name = tool.value("name", "");
                            td.description = tool.value("description", "");
                            if (tool.contains("parameters")) {
                                td.parameters_json = tool["parameters"].dump();
                            }
                        }
                        session.tools.push_back(td);
                    }
                    LOG_DEBUG("Parsed " + std::to_string(session.tools.size()) + " tools from request");
                }

                // Parse messages from request into session
                // OpenAI protocol sends FULL conversation history each time, so REPLACE not append
                session.messages.clear();  // Clear before adding new messages
                for (const auto& msg : request["messages"]) {
                    SessionContext::Message m;
                    m.role = msg["role"];
                    m.content = msg.value("content", "");
                    if (msg.contains("name")) m.name = msg["name"];
                    if (msg.contains("tool_call_id")) m.tool_call_id = msg["tool_call_id"];
                    session.messages.push_back(m);
                }

                // NEW UNIFIED INTERFACE: Pass entire SessionContext to backend
                // Backend reads SessionContext and formats request appropriately
                // For API backends: formats to JSON and sends API request
                // For local backends: formats to prompt and runs inference
                LOG_DEBUG("Calling generate_from_session with " + std::to_string(session.messages.size()) +
                          " messages and " + std::to_string(session.tools.size()) + " tools");

                // Generate response from SessionContext
                int max_tokens = request["parameters"].value("max_tokens", 0);
                std::string generated = backend->generate_from_session(session, max_tokens);

                // Check for tool calls
                auto tool_call_opt = ToolParser::parse_tool_call(
                    generated,
                    backend->get_tool_call_markers()
                );

                response["status"] = "success";

                if (tool_call_opt.has_value()) {
                    auto tool_call = tool_call_opt.value();
                    response["content"] = "";
                    response["finish_reason"] = "tool_calls";
                    response["tool_calls"] = json::array();

                    json tc;
                    tc["id"] = tool_call.tool_call_id.empty() ?
                               "call_" + std::to_string(std::time(nullptr)) :
                               tool_call.tool_call_id;
                    tc["type"] = "function";

                    // Build function object with name and arguments
                    json function_obj;
                    function_obj["name"] = tool_call.name;

                    // Convert parameters to JSON object
                    json parameters = json::object();
                    for (const auto& [key, value] : tool_call.parameters) {
                        if (value.type() == typeid(std::string)) {
                            parameters[key] = std::any_cast<std::string>(value);
                        } else if (value.type() == typeid(int)) {
                            parameters[key] = std::any_cast<int>(value);
                        } else if (value.type() == typeid(double)) {
                            parameters[key] = std::any_cast<double>(value);
                        } else if (value.type() == typeid(bool)) {
                            parameters[key] = std::any_cast<bool>(value);
                        }
                    }

                    // OpenAI expects arguments as a JSON string, not an object
                    function_obj["arguments"] = parameters.dump();

                    tc["function"] = function_obj;
                    response["tool_calls"].push_back(tc);
                } else {
                    response["content"] = generated;
                    response["finish_reason"] = "stop";
                    response["tool_calls"] = json::array();
                }

                // Add usage statistics
                response["usage"] = {
                    {"prompt_tokens", backend->get_context_manager().get_total_tokens()},
                    {"completion_tokens", 0},  // TODO: Track separately
                    {"total_tokens", backend->get_context_manager().get_total_tokens()}
                };

            } else if (request["action"] == "list_models") {
                response["status"] = "success";
                response["models"] = json::array();
                response["models"].push_back({
                    {"id", backend->get_model_name()},
                    {"backend", backend->get_backend_name()},
                    {"max_model_len", backend->get_max_context_size()}
                });

            } else if (request["action"] == "get_model_info") {
                response["status"] = "success";
                response["model_info"] = {
                    {"id", "shepherd"},
                    {"object", "model"},
                    {"created", std::time(nullptr)},
                    {"owned_by", "shepherd"},
                    {"context_window", backend->get_max_context_size()},
                    {"backend", backend->get_backend_name()},
                    {"model_name", backend->get_model_name()}
                };

            } else {
                response["status"] = "error";
                response["error"] = "Unknown action: " + request["action"].get<std::string>();
            }

            // Write response
            std::string response_str = response.dump() + "\n";
            fprintf(pipe_out, "%s", response_str.c_str());
            fflush(pipe_out);

        } catch (const std::exception& e) {
            LOG_ERROR("Error processing request: " + std::string(e.what()));

            json error_response;
            error_response["status"] = "error";
            error_response["error"] = e.what();

            std::string error_str = error_response.dump() + "\n";
            fprintf(pipe_out, "%s", error_str.c_str());
            fflush(pipe_out);
        }
    }

    free(line);
    fclose(pipe_in);
    fclose(pipe_out);

    // Clean up child process
    printf("Stopping FastAPI server...\n");
    kill(fastapi_pid, SIGTERM);
    waitpid(fastapi_pid, nullptr, 0);

    return 0;
}
