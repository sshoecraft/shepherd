#include "cli_server.h"
#include "../shepherd.h"
#include "../tools/tool.h"
#include "../tools/tool_parser.h"
#include "../tools/utf8_sanitizer.h"
#include "../mcp/mcp.h"
#include "../rag.h"
#include "../config.h"
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

void CLIServer::init(bool no_mcp, bool no_tools) {
    // Initialize RAG system using global config
    std::string db_path = config->memory_database;
    if (db_path.empty()) {
        try {
            db_path = Config::get_default_memory_db_path();
        } catch (const ConfigError& e) {
            LOG_ERROR("Failed to determine memory database path: " + std::string(e.what()));
            throw;
        }
    } else if (db_path[0] == '~') {
        // Expand ~ if present
        db_path = Config::get_home_directory() + db_path.substr(1);
    }

    if (!RAGManager::initialize(db_path, config->max_db_size)) {
        throw std::runtime_error("Failed to initialize RAG system");
    }
    LOG_INFO("RAG initialized with database: " + db_path);

    if (no_tools) {
        LOG_INFO("Tools disabled for CLI server");
        return;
    }

    LOG_INFO("Initializing tools for CLI server...");

    // Register all native tools
    register_filesystem_tools(tools);
    register_command_tools(tools);
    register_json_tools(tools);
    register_http_tools(tools);
    register_memory_tools(tools);
    register_mcp_resource_tools(tools);
    register_core_tools(tools);

    // Initialize MCP servers
    if (!no_mcp) {
        auto& mcp = MCP::instance();
        mcp.initialize(tools);
        LOG_INFO("MCP initialized with " + std::to_string(mcp.get_tool_count()) + " tools");
    } else {
        LOG_INFO("MCP system disabled");
    }

    // Build the combined tool list
    tools.build_all_tools();

    LOG_INFO("CLI server tools initialized: " + std::to_string(tools.all_tools.size()) + " total");
}

int CLIServer::run(std::unique_ptr<Backend>& backend, Session& session) {
    return run_cli_server(backend, session, host, port, tools);
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
    Tools* tools;
    std::mutex request_mutex;
    std::atomic<bool> processing{false};
    std::atomic<int> queue_depth{0};
    std::string current_request;
};

// Process a single request with tool execution loop
static json process_request(CliServerState& state, const std::string& prompt) {
    json result;
    result["success"] = true;

    std::string accumulated_response;

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

        // Execute the tool
        ToolResult tool_result = state.tools->execute(tool_name, tool_call.parameters);

        if (tool_result.success) {
            // Sanitize UTF-8 in result
            std::string sanitized = utf8_sanitizer::sanitize_utf8(tool_result.content);

            // Send tool result to model and get next response
            resp = state.session->add_message(Message::TOOL, sanitized, tool_name, tool_call.tool_call_id);
        } else {
            // Send error to model
            std::string error_msg = "Error: " + tool_result.error;
            resp = state.session->add_message(Message::TOOL, error_msg, tool_name, tool_call.tool_call_id);
        }

        if (!resp.success) {
            result["success"] = false;
            result["error"] = resp.error;
            return result;
        }

        accumulated_response = resp.content;
    }

    result["response"] = accumulated_response;
    return result;
}

int run_cli_server(std::unique_ptr<Backend>& backend, Session& session,
                   const std::string& host, int port, Tools& tools) {

    LOG_INFO("Starting CLI server on " + host + ":" + std::to_string(port));

    httplib::Server server;

    CliServerState state;
    state.backend = backend.get();
    state.session = &session;
    state.tools = &tools;

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
            bool stream = false;

            if (!req.body.empty()) {
                try {
                    request_json = json::parse(req.body);
                    if (request_json.contains("prompt")) {
                        prompt = request_json["prompt"].get<std::string>();
                    }
                    stream = request_json.value("stream", false);
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

            if (stream) {
                // Streaming response using SSE
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("X-Accel-Buffering", "no");

                res.set_content_provider(
                    "text/event-stream",
                    [&state, prompt](size_t offset, httplib::DataSink& sink) mutable {
                        std::string accumulated_response;

                        // Tool call filtering state
                        std::vector<std::string> tool_markers = state.backend->get_tool_call_markers();
                        std::string pending_buffer;
                        bool in_tool_call = false;

                        // Helper to check if buffer could be start of any marker
                        auto could_be_marker_start = [&tool_markers](const std::string& buf) -> bool {
                            for (const auto& marker : tool_markers) {
                                if (marker.substr(0, buf.size()) == buf) {
                                    return true;
                                }
                            }
                            // Also check for raw JSON tool call pattern
                            if (buf.size() > 0 && buf[0] == '{') {
                                std::string json_pattern = "{\"name\"";
                                if (json_pattern.substr(0, buf.size()) == buf) {
                                    return true;
                                }
                            }
                            return false;
                        };

                        // Helper to check if buffer matches any marker
                        auto matches_marker = [&tool_markers](const std::string& buf) -> bool {
                            for (const auto& marker : tool_markers) {
                                if (buf.find(marker) != std::string::npos) {
                                    return true;
                                }
                            }
                            // Check for raw JSON tool call
                            if (buf.find("{\"name\"") != std::string::npos) {
                                return true;
                            }
                            return false;
                        };

                        // Streaming callback with tool call filtering
                        auto stream_callback = [&](const std::string& delta,
                                                   const std::string& accumulated,
                                                   const Response& partial) -> bool {
                            if (delta.empty()) return true;

                            // Add delta to pending buffer
                            pending_buffer += delta;

                            // If we're in a tool call, just buffer it
                            if (in_tool_call) {
                                return true;
                            }

                            // Check if this could be the start of a tool call
                            if (matches_marker(pending_buffer)) {
                                // We've hit a tool call marker - stop streaming content
                                in_tool_call = true;
                                return true;
                            }

                            // Check if we might be starting a tool call
                            if (could_be_marker_start(pending_buffer)) {
                                // Keep buffering - might be a tool call
                                return true;
                            }

                            // Safe to output the pending buffer
                            if (!pending_buffer.empty()) {
                                json chunk;
                                chunk["delta"] = pending_buffer;
                                std::string data = "data: " + chunk.dump() + "\n\n";
                                pending_buffer.clear();
                                if (!sink.write(data.c_str(), data.size())) {
                                    return false;
                                }
                            }
                            return true;
                        };

                        // Add user message with streaming
                        Response resp = state.session->add_message_stream(
                            Message::USER, prompt, stream_callback);

                        if (!resp.success) {
                            json error_chunk;
                            error_chunk["error"] = resp.error;
                            error_chunk["done"] = true;
                            std::string data = "data: " + error_chunk.dump() + "\n\n";
                            sink.write(data.c_str(), data.size());
                            sink.done();
                            return true;
                        }

                        accumulated_response = resp.content;

                        // Tool execution loop
                        const int max_tool_iterations = 50;
                        int iteration = 0;

                        while (iteration < max_tool_iterations) {
                            iteration++;

                            auto tool_call_opt = extract_tool_call(resp, state.backend);
                            if (!tool_call_opt) {
                                break;
                            }

                            auto& tool_call = *tool_call_opt;
                            std::string tool_name = tool_call.name;

                            // Send tool call notification
                            json tool_chunk;
                            tool_chunk["tool_call"] = tool_name;
                            std::string tool_data = "data: " + tool_chunk.dump() + "\n\n";
                            sink.write(tool_data.c_str(), tool_data.size());

                            // Execute the tool
                            ToolResult tool_result = state.tools->execute(tool_name, tool_call.parameters);

                            if (tool_result.success) {
                                std::string sanitized = utf8_sanitizer::sanitize_utf8(tool_result.content);
                                resp = state.session->add_message_stream(
                                    Message::TOOL, sanitized, stream_callback, tool_name, tool_call.tool_call_id);
                            } else {
                                std::string error_msg = "Error: " + tool_result.error;
                                resp = state.session->add_message_stream(
                                    Message::TOOL, error_msg, stream_callback, tool_name, tool_call.tool_call_id);
                            }

                            if (!resp.success) {
                                json error_chunk;
                                error_chunk["error"] = resp.error;
                                error_chunk["done"] = true;
                                std::string data = "data: " + error_chunk.dump() + "\n\n";
                                sink.write(data.c_str(), data.size());
                                sink.done();
                                return true;
                            }

                            accumulated_response = resp.content;
                        }

                        // Send final done message
                        json done_chunk;
                        done_chunk["done"] = true;
                        done_chunk["response"] = accumulated_response;
                        std::string done_data = "data: " + done_chunk.dump() + "\n\n";
                        sink.write(done_data.c_str(), done_data.size());
                        sink.done();
                        return false;  // Signal no more content
                    }
                );
            } else {
                // Non-streaming response
                json response = process_request(state, prompt);
                res.set_content(response.dump(), "application/json");
            }

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
