
#include "shepherd.h"
#include "server/server.h"
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

// Helper: Extract tool call from Response (handles both structured and text-based)
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
    // If backend already parsed tool calls, use them
    if (!resp.tool_calls.empty()) {
        return resp.tool_calls[0];
    }

    // Otherwise parse from text content
    return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

int run_server(std::unique_ptr<Backend>& backend,
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

        // Pass socket fd and parent PID to Python via command line
        std::string fd_str = std::to_string(child_fd);
        std::string port_str = std::to_string(server_port);
        std::string parent_pid_str = std::to_string(getppid());  // Parent C++ process PID for cancellation signaling

        // Find Python interpreter
        const char* python = "python3";

        // Get executable directory to find server/api_server.py
        std::string exe_path = std::filesystem::canonical("/proc/self/exe").parent_path().string();
        std::string script_path = exe_path + "/server/api_server.py";

        // Check if script exists
        if (!std::filesystem::exists(script_path)) {
            // Try installed location (e.g., /usr/local/share/shepherd/server/)
            script_path = exe_path + "/../share/shepherd/server/api_server.py";
            if (!std::filesystem::exists(script_path)) {
                // Try relative path (for development)
                script_path = "./server/api_server.py";
                if (!std::filesystem::exists(script_path)) {
                    fprintf(stderr, "Error: Could not find server/api_server.py\n");
                    exit(1);
                }
            }
        }

        // Execute FastAPI server
        execl("/usr/bin/env", "env", python, script_path.c_str(),
              "--fd", fd_str.c_str(),
              "--port", port_str.c_str(),
              "--host", server_host.c_str(),
              "--parent-pid", parent_pid_str.c_str(),
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
    printf("Backend: %s\n", backend->backend_name.c_str());
    printf("Model: %s\n", backend->model_name.c_str());
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

    // Session for single-session server with prefix caching
    // All requests are processed in sequence using this shared session
    Session session;

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
                // Always clear tools first - client controls what tools are available
                session.tools.clear();

                if (request.contains("tools") && request["tools"].is_array()) {
                    // Convert OpenAI tools format to Session::Tool format
                    for (const auto& tool : request["tools"]) {
                        Session::Tool st;

                        // OpenAI format: tool["function"]["name"]
                        if (tool.contains("function") && tool["function"].is_object()) {
                            st.name = tool["function"].value("name", "");
                            st.description = tool["function"].value("description", "");
                            if (tool["function"].contains("parameters")) {
                                st.parameters = tool["function"]["parameters"];  // Store as JSON object
                            } else {
                                st.parameters = nlohmann::json::object();
                            }
                        } else {
                            // Fallback: direct format (for testing)
                            st.name = tool.value("name", "");
                            st.description = tool.value("description", "");
                            if (tool.contains("parameters")) {
                                st.parameters = tool["parameters"];  // Store as JSON object
                            } else {
                                st.parameters = nlohmann::json::object();
                            }
                        }
                        session.tools.push_back(st);
                    }
                    LOG_DEBUG("Parsed " + std::to_string(session.tools.size()) + " tools from request");
                }

                // Parse messages from request into session
                // OpenAI protocol sends FULL conversation history each time, so REPLACE not append
                session.messages.clear();  // Clear before adding new messages
                for (const auto& msg : request["messages"]) {
                    std::string role = msg["role"];
                    std::string content = msg.value("content", "");

                    // Convert role to Message::Type
                    Message::Type type;
                    if (role == "system") {
                        // System messages handled separately via session.system_message
                        session.system_message = content;
                        continue;
                    } else if (role == "user") {
                        type = Message::USER;
                    } else if (role == "assistant") {
                        type = Message::ASSISTANT;
                    } else if (role == "tool") {
                        type = Message::TOOL;
                    } else {
                        type = Message::USER; // fallback
                    }

                    // Create Message with estimated tokens
                    Message m(type, content, content.length() / 4); // rough estimate
                    if (msg.contains("tool_call_id")) {
                        m.tool_call_id = msg["tool_call_id"];
                    }
                    if (msg.contains("name")) {
                        m.tool_name = msg["name"];
                    }
                    session.messages.push_back(m);

                    // Track last user/assistant messages for context preservation
                    if (type == Message::USER) {
                        session.last_user_message_index = session.messages.size() - 1;
                        session.last_user_message_tokens = m.tokens;
                    } else if (type == Message::ASSISTANT) {
                        session.last_assistant_message_index = session.messages.size() - 1;
                        session.last_assistant_message_tokens = m.tokens;
                    }
                }

                // Generate response from Session
                LOG_DEBUG("Calling generate_from_session with " + std::to_string(session.messages.size()) +
                          " messages and " + std::to_string(session.tools.size()) + " tools");

                int max_tokens = request["parameters"].value("max_tokens", 0);
                Response resp = backend->generate_from_session(session, max_tokens);

                // Check for errors
                if (!resp.success) {
                    response["status"] = "error";
                    response["error"] = resp.error;
                    response["finish_reason"] = "error";
                } else {
                    // Check for tool calls using helper
                    auto tool_call_opt = extract_tool_call(resp, backend.get());

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
                        // No tool call - regular response
                        response["content"] = resp.content;
                        response["finish_reason"] = resp.finish_reason.empty() ? "stop" : resp.finish_reason;
                        response["tool_calls"] = json::array();
                    }

                    // Add usage statistics from Response
                    int prompt_tokens = resp.prompt_tokens;
                    int completion_tokens = resp.completion_tokens;

                    // Fall back to backend tracking if Response doesn't have counts
                    if (prompt_tokens == 0 && completion_tokens == 0) {
                        prompt_tokens = backend->last_prompt_tokens;
                        completion_tokens = backend->last_completion_tokens;
                    }

                    // Final fallback for local backends
                    if (prompt_tokens == 0 && completion_tokens == 0) {
                        prompt_tokens = backend->context_token_count;
                        completion_tokens = 0;
                    }

                    response["usage"] = {
                        {"prompt_tokens", prompt_tokens},
                        {"completion_tokens", completion_tokens},
                        {"total_tokens", prompt_tokens + completion_tokens}
                    };
                }

            } else if (request["action"] == "list_models") {
                response["status"] = "success";
                response["models"] = json::array();

                // Extract just the filename from the model path
                std::string model_path = backend->model_name;
                std::string model_id = std::filesystem::path(model_path).filename().string();

                response["models"].push_back({
                    {"id", model_id},
                    {"backend", backend->backend_name},
                    {"max_model_len", backend->context_size}
                });

            } else if (request["action"] == "get_model_info") {
                response["status"] = "success";
                response["model_info"] = json::object();
                response["model_info"]["id"] = "shepherd";
                response["model_info"]["object"] = "model";
                response["model_info"]["created"] = std::time(nullptr);
                response["model_info"]["owned_by"] = "shepherd";
                response["model_info"]["context_window"] = backend->context_size;
                response["model_info"]["backend"] = backend->backend_name;
                response["model_info"]["model_name"] = backend->model_name;

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
