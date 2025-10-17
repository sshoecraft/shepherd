#include "logger.h"
#include "config.h"
#include "backend_manager.h"
#include "rag.h"
#include "tools/tools.h"
#include "tools/tool_parser.h"
#include "mcp/mcp_manager.h"
#include "mcp/mcp_config.h"
#include "web_search.h"
#include "server.h"
#include "session_context.h"
#include "nlohmann/json.hpp"

#ifdef ENABLE_API_BACKENDS
#include "backends/openai.h"
#include "backends/ollama.h"
#include "backends/api_backend.h"
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <getopt.h>
#include <csignal>
#include <memory>
#include <unistd.h>
#include <filesystem>
#include <termios.h>
#include <sys/select.h>
#include <fcntl.h>
#include <atomic>
#include <thread>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cerrno>

// Global storage for original command-line arguments (for MPI re-exec)
static int g_argc = 0;
static char** g_argv = nullptr;

void set_global_args(int argc, char** argv) {
    g_argc = argc;
    g_argv = argv;
}

void get_global_args(int& argc, char**& argv) {
    argc = g_argc;
    argv = g_argv;
}

#ifdef USE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

// Global debug flag
bool g_debug_mode = false;

// Get input line with readline support if available
// Global flag to track EOF from readline
bool g_eof_received = false;

// Global cancellation flag
std::atomic<bool> g_generation_cancelled{false};

// Terminal state management
struct termios g_original_term;
bool g_term_raw_mode = false;

// Set terminal to raw mode for immediate key detection
void set_terminal_raw() {
    if (g_term_raw_mode) return;

    struct termios raw;
    tcgetattr(STDIN_FILENO, &g_original_term);
    raw = g_original_term;

    // Disable canonical mode and echo
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;   // Non-blocking read
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    g_term_raw_mode = true;
}

// Restore terminal to normal mode
void restore_terminal() {
    if (!g_term_raw_mode) return;
    tcsetattr(STDIN_FILENO, TCSANOW, &g_original_term);
    g_term_raw_mode = false;
}

// Check if Escape key was pressed (non-blocking)
bool check_escape_pressed() {
    if (!g_term_raw_mode) return false;

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    struct timeval tv = {0, 0};  // No wait

    if (select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv) > 0) {
        char c;
        if (read(STDIN_FILENO, &c, 1) == 1) {
            if (c == 27) {  // ESC key
                return true;
            }
        }
    }

    return false;
}

// Helper to clean bracketed paste sequences from a string
static std::string strip_bracketed_paste_sequences(const std::string& input) {
    std::string result = input;

    // Strip common bracketed paste markers that leak through EditLine
    // Variants seen: [200~, 200~, 00~ (start) and [201~, 201~, 01~ (end)

    // Remove paste start sequences (various formats)
    std::vector<std::string> start_markers = {"[200~", "200~", "00~"};
    for (const auto& marker : start_markers) {
        size_t pos = 0;
        while ((pos = result.find(marker, pos)) != std::string::npos) {
            result.erase(pos, marker.length());
        }
    }

    // Remove paste end sequences (various formats)
    std::vector<std::string> end_markers = {"[201~", "201~", "01~"};
    for (const auto& marker : end_markers) {
        size_t pos = 0;
        while ((pos = result.find(marker, pos)) != std::string::npos) {
            result.erase(pos, marker.length());
        }
    }

    return result;
}

std::string get_input_line(const char* prompt, bool is_interactive) {
#ifdef USE_READLINE
    if (is_interactive) {
        char* line = readline(prompt);
        if (line == nullptr) {
            g_eof_received = true;
            return "";  // EOF
        }
        g_eof_received = false;
        std::string result(line);

        // Debug: Show what we actually received
        if (g_debug_mode) {
            LOG_DEBUG("Raw input received: '" + result + "'");
            // Show hex bytes for debugging escape sequences
            std::string hex_dump;
            for (unsigned char c : result.substr(0, std::min(size_t(50), result.length()))) {
                char buf[10];
                snprintf(buf, sizeof(buf), "%02x ", c);
                hex_dump += buf;
            }
            LOG_DEBUG("First 50 bytes (hex): " + hex_dump);
        }

        // Detect bracketed paste mode (EditLine on macOS doesn't strip these automatically)
        // If we see the paste start sequence, accumulate until paste end
        // Check for various formats: 200~, 00~, [200~, etc.
        if (result.find("200~") != std::string::npos || result.find("00~") != std::string::npos) {
            LOG_DEBUG("Detected bracketed paste start");
            std::string pasted_content = strip_bracketed_paste_sequences(result);

            // Keep reading lines until we see the paste end marker (various formats)
            while (result.find("201~") == std::string::npos && result.find("01~") == std::string::npos) {
                free(line);
                line = readline("");  // No prompt for continuation
                if (line == nullptr) {
                    g_eof_received = true;
                    break;
                }
                result = std::string(line);

                // Check for end marker (various formats)
                if (result.find("201~") != std::string::npos || result.find("01~") != std::string::npos) {
                    // Add final content before end marker
                    std::string final_line = strip_bracketed_paste_sequences(result);
                    if (!final_line.empty()) {
                        pasted_content += "\n" + final_line;
                    }
                    break;
                } else {
                    // Accumulate this line
                    pasted_content += "\n" + result;
                }
            }

            free(line);
            if (!pasted_content.empty()) {
                add_history(pasted_content.c_str());
            }
            LOG_DEBUG("Bracketed paste complete, length: " + std::to_string(pasted_content.length()));
            return pasted_content;
        }

        // Check for triple-quote multi-line mode (alternative for explicit multi-line)
        if (result == "\"\"\"" || result.find("\"\"\"") == 0) {
            // Multi-line mode - keep reading until closing """
            std::string multi_line;
            bool first_line = true;

            // If line has content after """, include it
            if (result.length() > 3) {
                multi_line = result.substr(3);
                first_line = false;
            }

            while (true) {
                free(line);  // Free previous line
                line = readline("");  // No prompt for continuation lines
                if (line == nullptr) {
                    g_eof_received = true;
                    break;
                }

                std::string continued(line);

                // Check for closing """
                if (continued == "\"\"\"" || continued.find("\"\"\"") != std::string::npos) {
                    // If """ is at end of line with content before it, include that content
                    size_t pos = continued.find("\"\"\"");
                    if (pos > 0) {
                        if (!first_line) multi_line += "\n";
                        multi_line += continued.substr(0, pos);
                    }
                    break;
                }

                // Add line to multi-line buffer
                if (!first_line) multi_line += "\n";
                multi_line += continued;
                first_line = false;
            }

            if (!multi_line.empty()) {
                add_history(multi_line.c_str());
            }
            free(line);
            return multi_line;
        }

        if (!result.empty()) {
            add_history(line);
        }
        free(line);
        return result;
    } else {
        // Non-interactive mode - use getline
        std::string line;
        if (!std::getline(std::cin, line)) {
            g_eof_received = true;
            return "";  // EOF
        }
        g_eof_received = false;
        return line;
    }
#else
    // No readline - use basic getline
    if (is_interactive) {
        std::cout << prompt << std::flush;
    }
    std::string line;
    if (!std::getline(std::cin, line)) {
        g_eof_received = true;
        return "";  // EOF
    }
    g_eof_received = false;
    return line;
#endif
}

static void print_usage(int, char** argv) {
    printf("\n=== Shepherd - Advanced LLM Management System ===\n");
    printf("\nUsage:\n");
    printf("    %s [OPTIONS]\n", argv[0]);
    printf("    %s edit-system              Edit system prompt in $EDITOR\n", argv[0]);
    printf("    %s list-tools               List all available tools\n", argv[0]);
    printf("    %s mcp <add|remove|list> [args...]\n", argv[0]);
    printf("\nOptions:\n");
    printf("    -c, --config FILE  Specify config file (default: ~/.shepherd/config.json)\n");
    printf("    -d, --debug        Enable debug mode\n");
    printf("    -l, --log-file     Log to file instead of console\n");
    printf("    -m, --model        Model path (overrides config)\n");
    printf("    --backend          Backend (llamacpp, openai, anthropic, gemini, grok, ollama)\n");
    printf("    --api-key          API key for cloud backends\n");
    printf("    --api-base         API base URL (for OpenAI-compatible APIs)\n");
    printf("    --context-size     Set context window size (0 = use model's full context, default: from config)\n");
    printf("    --max-tokens       Set max generation tokens (default: auto)\n");
    printf("    --nomcp            Disable MCP system (no MCP servers loaded)\n");
    printf("    --template         Custom chat template file (Jinja format, llamacpp only)\n");
    printf("    --temperature      Sampling temperature 0.0-2.0 (default: 0.7, local backends)\n");
    printf("    --top-p            Nucleus sampling probability 0.0-1.0 (default: 0.95, local backends)\n");
    printf("    --top-k            Top-K sampling (default: 40, local backends)\n");
    printf("    --min-keep         Minimum tokens to keep (default: 1, local backends)\n");
    printf("    --penalty-repeat   Repetition penalty >= 1.0 (default: 1.1, local backends)\n");
    printf("    --penalty-freq     Frequency penalty >= 0.0 (default: 0.1, local backends)\n");
    printf("    --penalty-present  Presence penalty >= 0.0 (default: 0.0, local backends)\n");
    printf("    --penalty-last-n   Penalty window size (default: 64, 0=disabled, local backends)\n");
    printf("    --server           Start HTTP API server mode (OpenAI-compatible)\n");
    printf("    --port PORT        Server port (default: 8080, requires --server)\n");
    printf("    --host HOST        Server host to bind to (default: 0.0.0.0, requires --server)\n");
    printf("    -h, --help         Show this help message\n");
    printf("\nMCP Management:\n");
    printf("    mcp list                              List all configured MCP servers\n");
    printf("    mcp add <name> <cmd> [args] [-e ...]  Add a new MCP server\n");
    printf("                                          -e KEY=VALUE  Set environment variable\n");
    printf("    mcp remove <name>                     Remove an MCP server\n");
    printf("\nConfiguration:\n");
    printf("    Edit ~/.shepherd/config.json to configure:\n");
    printf("    - backend: llamacpp, openai, anthropic");
#ifdef PLATFORM_LINUX
#ifdef ENABLE_TENSORRT
    printf(", tensorrt");
#endif
#endif
    printf("\n");
    printf("    - model: model name or path\n");
    printf("    - model_path: directory for models (optional, defaults to ~/.shepherd/models)\n");
    printf("    - key: API key for cloud backends (optional)\n");
    printf("    - context_size: context window size (optional, 0 = auto)\n");
    printf("\nFeatures:\n");
    printf("    - Direct document reading and processing\n");
    printf("    - Conversation memory with search capabilities\n");
    printf("    - Multiple inference backends (local + cloud)\n");
    printf("    - Tool execution support\n");
    printf("    - Model Context Protocol (MCP) server integration\n");
    printf("\n");
}

static void signal_handler(int signal) {
    restore_terminal();
    printf("\n\nReceived signal %d, shutting down gracefully...\n", signal);
    exit(0);
}

#if 0
static void show_config(const Config& config) {
    printf("\n=== Current Configuration ===\n");
    printf("Backend: %s\n", config.get_backend().c_str());
    printf("Model: %s\n", config.get_model().c_str());

    if (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt") {
        printf("Model path: %s\n", config.get_model_path().c_str());
    } else if (config.get_backend() == "openai" || config.get_backend() == "anthropic") {
        printf("API key: %s\n", config.get_key().empty() ? "(not set)" : "***");
    }

    printf("Context size: %zu\n", config.get_context_size());

    printf("\nAvailable backends: ");
    auto available = Config::get_available_backends();
    for (size_t i = 0; i < available.size(); ++i) {
        if (i > 0) printf(", ");
        printf("%s", available[i].c_str());
    }
    printf("\n\n");
}
#endif

static int handle_mcp_command(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: shepherd mcp <add|remove|list> [args...]" << std::endl;
        return 1;
    }

    std::string subcommand = argv[2];
    std::string config_path = std::string(getenv("HOME")) + "/.shepherd/config.json";

    if (subcommand == "list") {
        // Suppress logs during health check
        Logger::instance().set_log_level(LogLevel::FATAL);
        MCPConfig::list_servers(config_path, true);  // Check health
        return 0;
    }

    if (subcommand == "add") {
        if (argc < 5) {
            std::cerr << "Usage: shepherd mcp add <name> <command> [args...] [-e KEY=VALUE ...]" << std::endl;
            return 1;
        }

        MCPServerEntry server;
        server.name = argv[3];
        server.command = argv[4];

        // Parse arguments and environment variables
        for (int i = 5; i < argc; i++) {
            std::string arg = argv[i];

            if (arg == "-e" || arg == "--env") {
                // Next arg should be KEY=VALUE
                if (i + 1 < argc) {
                    i++;
                    std::string env_pair = argv[i];
                    size_t eq_pos = env_pair.find('=');
                    if (eq_pos != std::string::npos) {
                        std::string key = env_pair.substr(0, eq_pos);
                        std::string value = env_pair.substr(eq_pos + 1);
                        server.env[key] = value;
                    } else {
                        std::cerr << "Warning: Invalid env format (use KEY=VALUE): " << env_pair << std::endl;
                    }
                }
            } else {
                server.args.push_back(arg);
            }
        }

        if (MCPConfig::add_server(config_path, server)) {
            std::cout << "Added MCP server '" << server.name << "'" << std::endl;
            return 0;
        }
        return 1;
    }

    if (subcommand == "remove") {
        if (argc < 4) {
            std::cerr << "Usage: shepherd mcp remove <name>" << std::endl;
            return 1;
        }

        std::string name = argv[3];
        if (MCPConfig::remove_server(config_path, name)) {
            std::cout << "Removed MCP server '" << name << "'" << std::endl;
            return 0;
        }
        return 1;
    }

    std::cerr << "Unknown mcp subcommand: " << subcommand << std::endl;
    std::cerr << "Available: add, remove, list" << std::endl;
    return 1;
}

static int handle_edit_system_command(int argc, char** argv) {
    // Load current config
    Config config;
    try {
        config.load();
    } catch (const ConfigError& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return 1;
    }

    // Create temp file with current system prompt
    std::string temp_path = "/tmp/shepherd_system_prompt_XXXXXX";
    char temp_template[256];
    strncpy(temp_template, temp_path.c_str(), sizeof(temp_template) - 1);
    temp_template[sizeof(temp_template) - 1] = '\0';

    int temp_fd = mkstemp(temp_template);
    if (temp_fd == -1) {
        std::cerr << "Error: Failed to create temporary file" << std::endl;
        return 1;
    }
    temp_path = temp_template;

    // Write current system prompt to temp file
    std::string current_prompt = config.get_system_prompt();
    if (current_prompt.empty()) {
        // Use default prompt as starting point
        current_prompt = "You have access to tools, but they are OPTIONAL. Only use tools when you need external information that you don't have:\n\n";
        current_prompt += "When you see a NOTICE about conversations moved to long-term memory, you can use the search_memory tool to retrieve that information if the user asks about it.\n\n";
        current_prompt += "Directive: Whenever you invoke a tool, output exactly one line containing only the JSON object for the tool call.\nThe line must start with { and end with }.\nDo not include any text, explanations, extra lines, or formatting before or after the JSON.\nThe JSON must appear on its own line with nothing else.\n\n";
    }

    if (write(temp_fd, current_prompt.c_str(), current_prompt.length()) == -1) {
        std::cerr << "Error: Failed to write to temporary file" << std::endl;
        close(temp_fd);
        unlink(temp_path.c_str());
        return 1;
    }
    close(temp_fd);

    // Get editor from environment, fallback to vi
    const char* editor = getenv("EDITOR");
    if (!editor || editor[0] == '\0') {
        editor = getenv("VISUAL");
    }
    if (!editor || editor[0] == '\0') {
        editor = "vi";
    }

    // Launch editor
    std::string editor_cmd = std::string(editor) + " " + temp_path;
    int result = system(editor_cmd.c_str());
    if (result != 0) {
        std::cerr << "Error: Editor exited with non-zero status" << std::endl;
        unlink(temp_path.c_str());
        return 1;
    }

    // Read edited content
    std::ifstream temp_file(temp_path);
    if (!temp_file.is_open()) {
        std::cerr << "Error: Failed to read edited file" << std::endl;
        unlink(temp_path.c_str());
        return 1;
    }

    std::string new_prompt;
    std::string line;
    while (std::getline(temp_file, line)) {
        if (!new_prompt.empty()) {
            new_prompt += "\n";
        }
        new_prompt += line;
    }
    temp_file.close();

    // Clean up temp file
    unlink(temp_path.c_str());

    // Update config.json
    std::string config_path = Config::get_home_directory() + "/.shepherd/config.json";

    try {
        // Read existing config
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << "Error: Failed to open config file: " << config_path << std::endl;
            return 1;
        }

        nlohmann::json config_json;
        config_file >> config_json;
        config_file.close();

        // Update system prompt
        config_json["system"] = new_prompt;

        // Write back
        std::ofstream out_file(config_path);
        if (!out_file.is_open()) {
            std::cerr << "Error: Failed to write config file: " << config_path << std::endl;
            return 1;
        }

        out_file << config_json.dump(2) << std::endl;
        out_file.close();

        std::cout << "System prompt updated successfully" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error updating config: " << e.what() << std::endl;
        return 1;
    }
}

/// @brief Generate JSON schema for a tool
/// @param tool_name Tool name
/// @param description Tool description
/// @param params Parameter definitions
/// @return JSON schema string
std::string generate_tool_json_schema(const std::string& tool_name,
                                      const std::string& description,
                                      const std::vector<ParameterDef>& params) {
    std::string schema = "{\n";
    schema += "  \"type\": \"function\",\n";
    schema += "  \"function\": {\n";
    schema += "    \"name\": \"" + tool_name + "\",\n";
    schema += "    \"description\": \"" + description + "\",\n";
    schema += "    \"parameters\": {\n";
    schema += "      \"type\": \"object\",\n";
    schema += "      \"properties\": {\n";

    // Add properties
    for (size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        schema += "        \"" + param.name + "\": {\n";
        schema += "          \"type\": \"" + param.type + "\"";
        if (!param.description.empty()) {
            schema += ",\n          \"description\": \"" + param.description + "\"";
        }
        schema += "\n        }";
        if (i < params.size() - 1) {
            schema += ",";
        }
        schema += "\n";
    }

    schema += "      }";

    // Add required fields
    std::vector<std::string> required_params;
    for (const auto& param : params) {
        if (param.required) {
            required_params.push_back(param.name);
        }
    }
    if (!required_params.empty()) {
        schema += ",\n      \"required\": [";
        for (size_t i = 0; i < required_params.size(); ++i) {
            schema += "\"" + required_params[i] + "\"";
            if (i < required_params.size() - 1) {
                schema += ", ";
            }
        }
        schema += "]";
    }

    schema += "\n    }\n";
    schema += "  }\n";
    schema += "}";

    return schema;
}

/// @brief Generate JSON schemas section for Llama 3.x first user message
/// @param tool_descriptions Map of tool names to descriptions
/// @param registry Tool registry for accessing tool objects
/// @return Formatted JSON schemas string to prepend to first user message
std::string generate_llama3_tool_schemas(const std::map<std::string, std::string>& tool_descriptions,
                                         ToolRegistry& registry) {
    std::string result;

    // Modified BFCL format for multi-turn agent (original BFCL is single-turn only)
    // Make it VERY clear that tools are optional - only use when actually needed
    result += "The following functions are available IF needed to answer the user's request:\n\n";
    result += "IMPORTANT: Only call a function if you actually need external information or capabilities. ";
    result += "For greetings, casual conversation, or questions you can answer directly - respond normally without calling any function.\n\n";
    result += "When you DO need to call a function, respond with ONLY a JSON object in this format: ";
    result += "{\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\n";

    // Sort tools so memory tools appear first
    std::vector<std::string> memory_tools = {"search_memory", "get_fact", "set_fact", "clear_fact"};
    std::vector<std::string> other_tools;

    // Separate memory tools from other tools
    for (const auto& pair : tool_descriptions) {
        if (std::find(memory_tools.begin(), memory_tools.end(), pair.first) == memory_tools.end()) {
            other_tools.push_back(pair.first);
        }
    }

    // Generate JSON schemas for each tool
    // Memory tools first
    for (const auto& tool_name : memory_tools) {
        if (tool_descriptions.find(tool_name) != tool_descriptions.end()) {
            Tool* tool = registry.get_tool(tool_name);
            if (tool) {
                auto params_schema = tool->get_parameters_schema();
                if (!params_schema.empty()) {
                    result += generate_tool_json_schema(tool_name, tool_descriptions.at(tool_name), params_schema);
                    result += "\n\n";
                }
            }
        }
    }

    // Other tools
    for (const auto& tool_name : other_tools) {
        Tool* tool = registry.get_tool(tool_name);
        if (tool) {
            auto params_schema = tool->get_parameters_schema();
            if (!params_schema.empty()) {
                result += generate_tool_json_schema(tool_name, tool_descriptions.at(tool_name), params_schema);
                result += "\n\n";
            }
        }
    }

    return result;
}

/// @brief Format tools list based on model family
/// @param config Model configuration
/// @param tool_descriptions Map of tool names to descriptions
/// @param registry Tool registry for accessing tool objects
/// @return Formatted tools string for system prompt
std::string format_tools_for_model(const ModelConfig& config,
                                   const std::map<std::string, std::string>& tool_descriptions,
                                   ToolRegistry& registry) {
    std::string result;

    // Sort tools so memory tools appear first
    std::vector<std::string> memory_tools = {"search_memory", "get_fact", "set_fact", "clear_fact"};
    std::vector<std::string> other_tools;

    // Separate memory tools from other tools
    for (const auto& pair : tool_descriptions) {
        if (std::find(memory_tools.begin(), memory_tools.end(), pair.first) == memory_tools.end()) {
            other_tools.push_back(pair.first);
        }
    }

    // Llama 3.x uses JSON-based zero-shot tool calling (per PDF page 10-11)
    // Per the spec, JSON schemas should go in the FIRST USER MESSAGE, not system message
    // System message should be minimal - just context about handling tool responses
    // Note: The chat template will add "Environment: ipython" automatically when builtin_tools is set
    if (config.family == ModelFamily::LLAMA_3_X) {
        // For Llama 3.x, return empty string - system message already set above
        // JSON schemas are added to first user message, not system message
        return "";
    }

    // Generic/fallback format (for other models)
    result += "Here are the available tools:  \n\n";

    // Add memory tools first
    for (const auto& tool_name : memory_tools) {
        if (tool_descriptions.find(tool_name) != tool_descriptions.end()) {
            Tool* tool = registry.get_tool(tool_name);
            if (tool) {
                result += "- " + tool_name + ": " + tool_descriptions.at(tool_name) + " (parameters: " + tool->parameters() + ")\n";
            }
        }
    }

    // Add other tools
    for (const auto& tool_name : other_tools) {
        Tool* tool = registry.get_tool(tool_name);
        if (tool) {
            result += "- " + tool_name + ": " + tool_descriptions.at(tool_name) + " (parameters: " + tool->parameters() + ")\n";
        }
    }

    return result;
}

int main(int argc, char** argv) {
    // Store original arguments for potential MPI re-exec
    set_global_args(argc, argv);

    // Detect MPI rank early (for multi-GPU TensorRT models)
    int mpi_rank = 0;
    const char* mpi_rank_env = getenv("OMPI_COMM_WORLD_RANK");
    if (mpi_rank_env) {
        mpi_rank = std::atoi(mpi_rank_env);
    }
    bool is_mpi_leader = (mpi_rank == 0);

    // Handle edit-system subcommand
    if (argc >= 2 && std::string(argv[1]) == "edit-system") {
        return handle_edit_system_command(argc, argv);
    }

    // Handle list-tools subcommand
    if (argc >= 2 && std::string(argv[1]) == "list-tools") {
        // Initialize tools system
        register_filesystem_tools();
        register_command_tools();
        register_json_tools();
        register_http_tools();
        register_memory_tools();
        register_mcp_resource_tools();
        register_core_tools();

        auto& registry = ToolRegistry::instance();
        auto tool_descriptions = registry.list_tools_with_descriptions();

        printf("\n=== Available Tools (%zu) ===\n\n", tool_descriptions.size());
        for (const auto& pair : tool_descriptions) {
            Tool* tool = registry.get_tool(pair.first);
            if (tool) {
                printf("• %s\n", pair.first.c_str());
                printf("  %s\n", pair.second.c_str());

                // Show parameters if available
                auto params = tool->get_parameters_schema();
                if (!params.empty()) {
                    printf("  Parameters:\n");
                    for (const auto& param : params) {
                        printf("    - %s (%s)%s: %s\n",
                               param.name.c_str(),
                               param.type.c_str(),
                               param.required ? ", required" : "",
                               param.description.c_str());
                    }
                } else {
                    printf("  Parameters: %s\n", tool->parameters().c_str());
                }
                printf("\n");
            }
        }
        return 0;
    }

    // Handle MCP subcommand
    if (argc >= 2 && std::string(argv[1]) == "mcp") {
        return handle_mcp_command(argc, argv);
    }

    bool debug_override = false;
    bool no_mcp = false;
    bool server_mode = false;
    std::string log_file;
    std::string config_file_path;
    std::string model_path_override;
    std::string backend_override;
    std::string api_key_override;
    std::string api_base_override;
    std::string template_override;
    int server_port = 8080;
    std::string server_host = "0.0.0.0";
    int context_size_override = -1;  // -1 means not specified, 0 means use model's full context
    int max_tokens_override = 0;
    float temperature_override = -1.0f;  // -1 means use config/default
    float top_p_override = -1.0f;
    int top_k_override = -1;
    int min_keep_override = -1;
    float penalty_repeat_override = -1.0f;
    float penalty_freq_override = -1.0f;
    float penalty_present_override = -1.0f;
    int penalty_last_n_override = -1;

    static struct option long_options[] = {
        {"config", required_argument, 0, 'c'},
        {"debug", no_argument, 0, 'd'},
        {"log-file", required_argument, 0, 'l'},
        {"model", required_argument, 0, 'm'},
        {"backend", required_argument, 0, 1002},
        {"api-key", required_argument, 0, 1003},
        {"api-base", required_argument, 0, 1004},
        {"context-size", required_argument, 0, 1000},
        {"max-tokens", required_argument, 0, 1001},
        {"nomcp", no_argument, 0, 1005},
        {"template", required_argument, 0, 1006},
        {"temperature", required_argument, 0, 1007},
        {"top-p", required_argument, 0, 1008},
        {"top-k", required_argument, 0, 1009},
        {"min-keep", required_argument, 0, 1010},
        {"penalty-repeat", required_argument, 0, 1011},
        {"penalty-freq", required_argument, 0, 1012},
        {"penalty-present", required_argument, 0, 1013},
        {"penalty-last-n", required_argument, 0, 1014},
        {"server", no_argument, 0, 1015},
        {"port", required_argument, 0, 1016},
        {"host", required_argument, 0, 1017},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "c:dl:m:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'c':
                config_file_path = optarg;
                break;
            case 'd':
                debug_override = true;
                break;
            case 'l':
                log_file = optarg;
                break;
            case 'm':
                model_path_override = optarg;
                break;
            case 1002: // --backend
                backend_override = optarg;
                break;
            case 1003: // --api-key
                api_key_override = optarg;
                break;
            case 1004: // --api-base
                api_base_override = optarg;
                break;
            case 1000: // --context-size
                context_size_override = std::atoi(optarg);
                if (context_size_override < 0) {
                    printf("Error: context-size cannot be negative (use 0 for model's full context)\n");
                    return 1;
                }
                break;
            case 1001: // --max-tokens
                max_tokens_override = std::atoi(optarg);
                if (max_tokens_override <= 0) {
                    printf("Error: max-tokens must be positive\n");
                    return 1;
                }
                break;
            case 1005: // --nomcp
                no_mcp = true;
                break;
            case 1006: // --template
                template_override = optarg;
                break;
            case 1007: // --temperature
                temperature_override = std::atof(optarg);
                if (temperature_override < 0.0f || temperature_override > 2.0f) {
                    printf("Error: temperature must be between 0.0 and 2.0\n");
                    return 1;
                }
                break;
            case 1008: // --top-p
                top_p_override = std::atof(optarg);
                if (top_p_override <= 0.0f || top_p_override > 1.0f) {
                    printf("Error: top-p must be between 0.0 and 1.0\n");
                    return 1;
                }
                break;
            case 1009: // --top-k
                top_k_override = std::atoi(optarg);
                if (top_k_override < 1) {
                    printf("Error: top-k must be at least 1\n");
                    return 1;
                }
                break;
            case 1010: // --min-keep
                min_keep_override = std::atoi(optarg);
                if (min_keep_override < 1) {
                    printf("Error: min-keep must be at least 1\n");
                    return 1;
                }
                break;
            case 1011: // --penalty-repeat
                penalty_repeat_override = std::atof(optarg);
                if (penalty_repeat_override < 1.0f) {
                    printf("Error: penalty-repeat must be >= 1.0\n");
                    return 1;
                }
                break;
            case 1012: // --penalty-freq
                penalty_freq_override = std::atof(optarg);
                if (penalty_freq_override < 0.0f) {
                    printf("Error: penalty-freq must be >= 0.0\n");
                    return 1;
                }
                break;
            case 1013: // --penalty-present
                penalty_present_override = std::atof(optarg);
                if (penalty_present_override < 0.0f) {
                    printf("Error: penalty-present must be >= 0.0\n");
                    return 1;
                }
                break;
            case 1014: // --penalty-last-n
                penalty_last_n_override = std::atoi(optarg);
                if (penalty_last_n_override < 0) {
                    printf("Error: penalty-last-n must be >= 0 (0 = disabled)\n");
                    return 1;
                }
                break;
            case 1015: // --server
                server_mode = true;
                break;
            case 1016: // --port
                server_port = std::atoi(optarg);
                if (server_port <= 0 || server_port > 65535) {
                    printf("Error: port must be between 1 and 65535\n");
                    return 1;
                }
                break;
            case 1017: // --host
                server_host = optarg;
                break;
            case 'h':
                print_usage(argc, argv);
                return 0;
            default:
                print_usage(argc, argv);
                return 1;
        }
    }

    // Check if input is from a terminal or piped (do this early)
    bool is_interactive = isatty(STDIN_FILENO);

    // Initialize logger early
    Logger& logger = Logger::instance();
    g_debug_mode = debug_override;

    if (g_debug_mode) {
        logger.set_log_level(LogLevel::DEBUG);
        std::cout << "Debug mode enabled" << std::endl;
    } else {
        // Suppress INFO logs unless in debug mode
        logger.set_log_level(LogLevel::WARN);
    }

    // Load configuration
    Config config;

    // Set custom config path if specified
    if (!config_file_path.empty()) {
        config.set_config_path(config_file_path);
    }

    try {
        config.load();
    } catch (const ConfigError& e) {
        fprintf(stderr, "Configuration error: %s\n", e.what());
        return 1;
    }

    // Apply command-line overrides
    if (!backend_override.empty()) {
        config.set_backend(backend_override);
    }
    if (!api_key_override.empty()) {
        config.set_key(api_key_override);
    }
    if (!api_base_override.empty()) {
        config.set_api_base(api_base_override);
    }
    if (context_size_override >= 0) {  // -1 means not set, 0+ are valid values
        config.set_context_size(context_size_override);
    }

    // Validate configuration (skip model path check if overridden)
    if (model_path_override.empty()) {
        try {
            config.validate();
        } catch (const ConfigError& e) {
            fprintf(stderr, "Configuration error: %s\n", e.what());
            fprintf(stderr, "Edit ~/.shepherd/config.json or use -h for help\n");
            return 1;
        }
    } else {
        // With model override, just validate backend availability
        auto available = Config::get_available_backends();
        bool backend_found = false;
        for (const auto& b : available) {
            if (b == config.get_backend()) {
                backend_found = true;
                break;
            }
        }
        if (!backend_found) {
            std::string available_str;
            for (size_t i = 0; i < available.size(); ++i) {
                if (i > 0) available_str += ", ";
                available_str += available[i];
            }
            fprintf(stderr, "Invalid backend '%s'. Available: %s\n",
                    config.get_backend().c_str(), available_str.c_str());
            return 1;
        }
        // Validate model file exists (only for local backends)
        if (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt") {
            if (!std::filesystem::exists(model_path_override)) {
                fprintf(stderr, "Model file not found: %s\n", model_path_override.c_str());
                return 1;
            }
        }
    }

    if (!log_file.empty()) {
        logger.set_log_file(log_file);
        logger.set_console_output(false);
        LOG_INFO("Logging to file: " + log_file);
    }

    LOG_INFO("Shepherd starting up...");
    LOG_INFO("Backend: " + config.get_backend());

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    try {
        // Create backend manager
        std::string model_path;

        if (!model_path_override.empty()) {
            // Use command line override as full path
            model_path = model_path_override;
            LOG_INFO("Model path overridden from command line: " + model_path);
        } else if (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt") {
            // For local backends, construct full path from directory + model name
            std::filesystem::path full_path;
            if (!config.get_model().empty() && (config.get_model()[0] == '/' || config.get_model()[0] == '~')) {
                // Model is already a full path
                full_path = config.get_model();
            } else {
                // Combine model_path directory with model name
                full_path = std::filesystem::path(config.get_model_path()) / config.get_model();
            }
            model_path = full_path.string();
        }

        size_t context_size = context_size_override >= 0 ?  // -1 means not set
                              static_cast<size_t>(context_size_override) :
                              config.get_context_size();

        if (context_size_override >= 0) {
            if (context_size_override == 0) {
                LOG_INFO("Context size overridden from command line: 0 (use model's full context)");
            } else {
                LOG_INFO("Context size overridden from command line: " + std::to_string(context_size));
            }
            config.set_context_size(context_size);
        }

        if (max_tokens_override > 0) {
            LOG_INFO("Max tokens overridden from command line: " + std::to_string(max_tokens_override));
        }

        // Show status to user (only for local backends that actually load models)
        bool is_local_backend = (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt");
        if (is_interactive && is_local_backend) {
            printf("Initializing Engine...\n");
            fflush(stdout);
        }

        std::unique_ptr<BackendManager> backend = BackendFactory::create_backend(
            config.get_backend(),
            model_path,
            context_size,
            "" // api_key will be set during initialization
        );

        if (!backend) {
            LOG_ERROR("Failed to create backend: " + config.get_backend());
            return 1;
        }

        // Initialize RAG system with default path
        std::string db_path;
        try {
            db_path = Config::get_home_directory() + "/.shepherd/memory.db";
        } catch (const ConfigError& e) {
            LOG_ERROR("Failed to determine home directory: " + std::string(e.what()));
            return 1;
        }
        if (!RAGManager::initialize(db_path, config.get_max_db_size())) {
            LOG_ERROR("Failed to initialize RAG system");
            return 1;
        }

        // Show status to user before the slow model load (only for local backends)
        if (is_interactive && is_local_backend) {
            printf("Loading Model...\n");
            fflush(stdout);
        }

        // Initialize backend with appropriate credentials
        bool init_success = false;
        if (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt") {
            // Set GPU layers via environment variable (will be read by llamacpp backend during init)
            if (config.get_backend() == "llamacpp" && config.get_gpu_layers() >= 0) {
                setenv("GGML_N_GPU_LAYERS", std::to_string(config.get_gpu_layers()).c_str(), 0); // 0 = don't overwrite if already set
                LOG_INFO("Setting GPU layers from config: " + std::to_string(config.get_gpu_layers()));
            }
            init_success = backend->initialize(model_path, "", template_override);
        } else if (config.get_backend() == "openai" || config.get_backend() == "anthropic") {
            // Set API base if specified (for OpenAI-compatible APIs)
            if (!config.get_api_base().empty() && config.get_backend() == "openai") {
                // Cast to OpenAIBackend to call set_api_base
                #ifdef ENABLE_API_BACKENDS
                auto* openai_backend = dynamic_cast<OpenAIBackend*>(backend.get());
                if (openai_backend) {
                    openai_backend->set_api_base(config.get_api_base());
                }
                #endif
            }
            // Use model override if provided, otherwise use config
            std::string api_model_name = model_path_override.empty() ? config.get_model() : model_path_override;
            init_success = backend->initialize(api_model_name, config.get_key());
        } else if (config.get_backend() == "ollama") {
            // Set API base if specified
            if (!config.get_api_base().empty()) {
                #ifdef ENABLE_API_BACKENDS
                auto* ollama_backend = dynamic_cast<OllamaBackend*>(backend.get());
                if (ollama_backend) {
                    ollama_backend->set_api_base(config.get_api_base());
                }
                #endif
            }
            // Use model override if provided, otherwise use config
            std::string api_model_name = model_path_override.empty() ? config.get_model() : model_path_override;
            // Ollama doesn't require a real API key
            std::string api_key = config.get_key().empty() ? "dummy" : config.get_key();
            init_success = backend->initialize(api_model_name, api_key);
        } else {
            init_success = backend->initialize("", "");
        }

        if (!init_success) {
            LOG_ERROR("Failed to initialize backend: " + config.get_backend());
            return 1;
        }

        // Apply sampling parameters for local backends (llamacpp, tensorrt)
        if (config.get_backend() == "llamacpp" || config.get_backend() == "tensorrt") {
            // Use overrides if specified, otherwise use config values
            float temperature = temperature_override >= 0 ? temperature_override : config.get_temperature();
            float top_p = top_p_override >= 0 ? top_p_override : config.get_top_p();
            int top_k = top_k_override >= 0 ? top_k_override : config.get_top_k();
            int min_keep = min_keep_override >= 0 ? min_keep_override : config.get_min_keep();
            float penalty_repeat = penalty_repeat_override >= 0 ? penalty_repeat_override : config.get_penalty_repeat();
            float penalty_freq = penalty_freq_override >= 0 ? penalty_freq_override : config.get_penalty_freq();
            float penalty_present = penalty_present_override >= 0 ? penalty_present_override : config.get_penalty_present();
            int penalty_last_n = penalty_last_n_override >= 0 ? penalty_last_n_override : config.get_penalty_last_n();

            backend->set_sampling_params(temperature, top_p, top_k, min_keep);
            LOG_INFO("Sampling parameters: temperature=" + std::to_string(temperature) +
                     ", top_p=" + std::to_string(top_p) +
                     ", top_k=" + std::to_string(top_k) +
                     ", min_keep=" + std::to_string(min_keep));

            // Set penalty parameters (works for any backend that supports it via virtual method)
            backend->set_penalty_params(penalty_repeat, penalty_freq, penalty_present, penalty_last_n);
            LOG_INFO("Repetition penalties: repeat=" + std::to_string(penalty_repeat) +
                     ", freq=" + std::to_string(penalty_freq) +
                     ", present=" + std::to_string(penalty_present) +
                     ", last_n=" + std::to_string(penalty_last_n));
        }

        LOG_INFO("Backend initialized: " + backend->get_backend_name() + " with model: " + backend->get_model_name());

        // Initialize tools system (skip in server mode - tools handled by client)
        if (!server_mode) {
            LOG_INFO("Initializing tools system...");
            try {
                // Register all native tools including memory search
                register_filesystem_tools();
                register_command_tools();
                register_json_tools();
                register_http_tools();
                register_memory_tools();
                register_mcp_resource_tools();
                register_core_tools();  // IMPORTANT: Register core tools (Bash, Glob, Grep, Edit, WebSearch, etc.)

                auto& registry = ToolRegistry::instance();
                auto tools = registry.list_tools();
                LOG_INFO("Native tools initialized with " + std::to_string(tools.size()) + " tools");
                if (g_debug_mode) {
                    for (const auto& tool_name : tools) {
                        LOG_DEBUG("Registered tool: " + tool_name);
                    }
                }

                // Initialize MCP servers (will register additional tools)
                if (!no_mcp) {
                    auto& mcp_manager = MCPManager::instance();
                    mcp_manager.initialize(config);

                    // Show total tool count after MCP
                    tools = registry.list_tools();
                    LOG_INFO("Total tools available: " + std::to_string(tools.size()) +
                             " (native + " + std::to_string(mcp_manager.get_tool_count()) + " MCP)");
                } else {
                    LOG_INFO("MCP system disabled via --nomcp flag");
                    tools = registry.list_tools();
                    LOG_INFO("Total tools available: " + std::to_string(tools.size()) + " (native only)");
                }

                // Initialize web search if configured
                std::string web_provider = config.get_web_search_provider();
                if (!web_provider.empty()) {
                    LOG_INFO("Initializing web search: " + web_provider);

                    auto& search_manager = WebSearchManager::instance();
                    std::string api_key = config.get_web_search_api_key();
                    std::string instance_url = config.get_web_search_instance_url();

                    search_manager.initialize(web_provider, api_key, instance_url);
                    if (search_manager.is_available()) {
                        LOG_INFO("Web search initialized successfully");
                    } else {
                        LOG_WARN("Web search initialization failed");
                    }
                } else {
                    LOG_DEBUG("Web search not configured");
                }

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to initialize tools system: " + std::string(e.what()));
                return 1;
            }
        } else {
            LOG_INFO("Server mode: Tool registration skipped (client-side tools)");
        }

        // Check what tools the model already knows from chat template
        if (g_debug_mode) {
            LOG_DEBUG("Checking model's built-in tool support...");
        }

        // Build system prompt: always start with default instructions, then append custom if provided
        std::string system_prompt = R"(You are a highly effective AI assistant with persistent memory. Follow this STRICT sequence for every interaction:

**Step 1: Check Memory (MANDATORY - NO EXCEPTIONS)**
- For EVERY query, your FIRST action MUST be a memory tool call
- Specific facts (e.g., "what is my name?"): get_fact(key=...) with plausible key
- Everything else: search_memory(query=...)
- Use the user's exact question as search query for best matching
- Do NOT proceed until memory has been checked

**Step 2: Use Other Tools (Only if Memory Returns Nothing)**
- Local files: read, grep, glob
- General knowledge: WebSearch(query=...)
- NEVER use websearch for local file content

**Step 3: Store Your Answer (MANDATORY - NO EXCEPTIONS)**
- CRITICAL: After deriving ANY answer from non-memory sources, you MUST store it
- Use the user's original question and your final answer:
  store_memory(question="<user's exact question>", answer="<your complete answer>")
- This applies to: file analysis, calculations, research, code findings - EVERYTHING
- EXCEPTION: Do NOT store if the answer came from get_fact or search_memory (already stored)

**Step 4: Update Outdated Information**
- When new info contradicts old: clear_memory(question=...) then store_memory(...)
- Only when explicitly told or clearly superseded

**Handling Truncated Tool Results:**

When you see [TRUNCATED]:

1. Assess First
   - Can you answer with visible data? If YES, answer and store in memory
   - If NO, proceed to recovery

2. Smart Recovery
   For code/text files:
   - Need specific section: read(file_path=..., offset=N)
   - Searching for keyword: grep(pattern="literal_string") with SIMPLE patterns only

   For grep failures:
   - Remove special chars: ( ) [ ] . * + ? { } | ^ $
   - Use literal strings only
   - Example: NOT "Config::parse_size\(" but USE "Config parse_size"

3. Stop Conditions
   - After 2-3 attempts with no progress: STOP
   - Answer with available data
   - Still store the partial answer in memory

4. Tool Boundaries
   - Local files: read, grep, glob ONLY
   - Past conversations: search_memory, get_fact ONLY
   - General knowledge: WebSearch ONLY
   - NO MIXING domains - never websearch for file content

**Enforcement Rules:**

ALWAYS check memory FIRST - even if query seems like obvious file operation
ALWAYS store answer LAST - unless it came from memory
NEVER skip memory check - this wastes computation and breaks continuity
NEVER forget to store - every answer you derive must be saved for next time

**Example Correct Flow:**
User: "What is the private variable in config.cpp?"
1. search_memory(query="private variable config.cpp") → empty
2. read(file_path="config.cpp") → find: private int m_max_size
3. store_memory(question="What is the private variable in config.cpp?", answer="The private variable is m_max_size, an int defined at line 47")
4. Respond to user

**Example Violation:**
User: "What is the private variable in config.cpp?"
1. read(file_path="config.cpp") ← WRONG! Didn't check memory first
2. Respond to user ← WRONG! Didn't store the answer)";

        // Append custom user prompt if configured
        std::string custom_prompt = config.get_system_prompt();
        if (!custom_prompt.empty()) {
            system_prompt += "\n\n" + custom_prompt;
            LOG_INFO("System prompt: default + custom additions");
        } else {
            LOG_INFO("System prompt: default only");
        }

        // Get registry and tool descriptions (needed for both interactive and server modes)
        auto& registry = ToolRegistry::instance();
        auto tool_descriptions = registry.list_tools_with_descriptions();

        // In server mode, DON'T cache the system message at startup
        // The client will provide it with the correct tools in each request
        if (!server_mode) {
            backend->add_system_message(system_prompt);
            LOG_INFO("Added system prompt with " + std::to_string(tool_descriptions.size()) + " available tools");
        } else {
            LOG_INFO("Server mode: System message will be provided by client with each request (empty KV cache)");
        }

        // Create session context for non-server mode
        // In non-server mode, this is a single continuous session
        // SessionContext organizes system prompt, tools, and messages in a unified structure
        SessionContext session;
        session.system_prompt = system_prompt;

        // Populate tools from registry
        for (const auto& [tool_name, tool_desc] : tool_descriptions) {
            SessionContext::ToolDefinition td;
            td.name = tool_name;
            td.description = tool_desc;

            // Get parameters schema if available
            Tool* tool = registry.get_tool(tool_name);
            if (tool) {
                auto params_schema = tool->get_parameters_schema();
                if (!params_schema.empty()) {
                    // Convert parameter schema to JSON string
                    // For now, just store the tool name - full schema conversion can be added later if needed
                    td.parameters_json = "{}";  // Placeholder
                }
            }

            session.tools.push_back(td);
        }

        LOG_DEBUG("SessionContext created with " + std::to_string(session.tools.size()) + " tools");

        LOG_INFO("Shepherd initialization complete");

        // ============================================================================
        // Server Mode - HTTP API via FastAPI
        // ============================================================================
        if (server_mode) {
            return run_server_mode(backend, server_host, server_port);
        }

        // For MPI multi-GPU setups, only rank 0 handles user interaction
        // Non-zero ranks keep their backend alive for MPI communication
        LOG_DEBUG("MPI rank check: mpi_rank=" + std::to_string(mpi_rank) + ", is_mpi_leader=" + std::string(is_mpi_leader ? "true" : "false"));
        if (!is_mpi_leader) {
            LOG_INFO("MPI rank " + std::to_string(mpi_rank) + " initialization complete, waiting for work from rank 0...");
            // Keep process alive - backend will be controlled via MPI by rank 0
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(3600));
            }
        }

        // From here on, only rank 0 continues...
        LOG_DEBUG("Rank 0 continuing to user input loop...");

        std::string user_input;

        // Force interactive mode for rank 0 (mpirun can make isatty return false)
        if (is_mpi_leader) {
            is_interactive = true;
        }

        // Warmup flag for llamacpp and ollama backends
        // The first iteration will send "ok" to warm up the model
        bool warmup_done = false;
        std::string backend_name = backend->get_backend_name();
        bool needs_warmup = (backend_name == "llamacpp" || backend_name == "ollama");

        while (true) {
            // Ensure terminal is in normal mode before reading input
            // (it may have been left in raw mode from previous iteration's ESC detection)
            restore_terminal();

            // Check if we need to do warmup first
            if (needs_warmup && !warmup_done) {
                LOG_DEBUG("Warming up " + backend_name + " model with warmup message...");
                user_input = "I want you to respond with exactly 'Ready.' and absolutely nothing else one time only at the start.";
                warmup_done = true;
            } else {
                // Normal user input
                // Under mpirun, stdout might not work correctly, so print prompt to stderr
                if (is_mpi_leader && is_interactive) {
                    std::cerr << "\n\033[32m> \033[0m" << std::flush;
                }

                user_input = get_input_line("", is_interactive);
            }

            if (user_input.empty()) {
                // Empty string indicates EOF (Ctrl+D) or empty input
                if (g_eof_received || std::cin.eof()) {
                    if (!is_interactive) {
                        LOG_DEBUG("End of piped input");
                    } else {
                        LOG_INFO("User pressed Ctrl+D - exiting");
                    }
                    break;
                } else if (is_interactive) {
                    // Interactive mode: empty line continues (just prompt again)
                    continue;
                } else {
                    // In non-interactive mode, skip empty lines
                    continue;
                }
            }

            // Check for exit commands
            if (user_input == "exit" || user_input == "quit") {
                LOG_INFO("User requested exit with command: " + user_input);
                break;
            }

            LOG_DEBUG("User input: " + user_input);

            // Show user message in transcript (only in non-interactive mode)
            if (!is_interactive) {
                printf("> %s\n", user_input.c_str());
                fflush(stdout);
            }

            try {
                // Add user message to backend context
                LOG_DEBUG("Adding user message to backend");

                backend->add_user_message(user_input);

                // Reset cancellation flag
                g_generation_cancelled = false;

                // Enable raw terminal mode for escape detection during generation
                // Only if stdin is actually a TTY (not piped input)
                if (is_interactive && isatty(STDIN_FILENO)) {
                    set_terminal_raw();
                }

                // Tool execution loop - orchestrated by main
                // User can cancel with Ctrl+C or ESC if needed
                std::string response;
                int tool_loop_iteration = 0;

                while (true) {
                    // Check for cancellation between iterations
                    if (g_generation_cancelled || (is_interactive && check_escape_pressed())) {
                        g_generation_cancelled = true;
                        LOG_DEBUG("Generation cancelled by user");
                        if (is_interactive) {
                            restore_terminal();
                            printf("\n\033[31m[Cancelled]\033[0m\n");
                        }
                        break;
                    }

                    tool_loop_iteration++;
                    LOG_DEBUG("Tool loop iteration: " + std::to_string(tool_loop_iteration));

                    // Generate response from backend
                    LOG_DEBUG("Calling backend->generate()");
                    response = backend->generate(max_tokens_override);
                    LOG_DEBUG("backend->generate() returned, length: " + std::to_string(response.length()));

                    // Check if user cancelled during generation
                    if (is_interactive && check_escape_pressed()) {
                        g_generation_cancelled = true;
                        restore_terminal();
                        printf("\n\033[31m[Cancelled]\033[0m\n");
                        break;
                    }

                    // Parse for tool calls
                    auto tool_call_opt = ToolParser::parse_tool_call(response, backend->get_tool_call_markers());

                    if (tool_call_opt.has_value()) {
                        // Tool call detected
                        auto tool_call = tool_call_opt.value();
                        std::string tool_name = tool_call.name;
                        std::string tool_call_id = tool_call.tool_call_id;
                        std::string json_str = tool_call.raw_json;

                        LOG_DEBUG("Tool call detected: " + tool_name + (tool_call_id.empty() ? "" : " (id: " + tool_call_id + ")"));

                        // Show tool call in transcript with actual parameter values
                        std::string params_str;
                        bool first_param = true;
                        for (const auto& param : tool_call.parameters) {
                            if (!first_param) params_str += ", ";
                            first_param = false;

                            // Convert std::any value to string for display
                            std::string value_str;
                            try {
                                if (param.second.type() == typeid(std::string)) {
                                    value_str = std::any_cast<std::string>(param.second);
                                } else if (param.second.type() == typeid(int)) {
                                    value_str = std::to_string(std::any_cast<int>(param.second));
                                } else if (param.second.type() == typeid(double)) {
                                    value_str = std::to_string(std::any_cast<double>(param.second));
                                } else if (param.second.type() == typeid(bool)) {
                                    value_str = std::any_cast<bool>(param.second) ? "true" : "false";
                                } else {
                                    value_str = "<unknown>";
                                }
                            } catch (...) {
                                value_str = "<error>";
                            }

                            // Truncate long values
                            if (value_str.length() > 50) {
                                value_str = value_str.substr(0, 47) + "...";
                            }

                            params_str += param.first + "=" + value_str;
                        }
                        if (is_interactive) {
                            printf("\033[33m< %s(%s)\033[0m\n", tool_name.c_str(), params_str.c_str());
                        } else {
                            printf("< %s(%s)\n", tool_name.c_str(), params_str.c_str());
                        }
                        fflush(stdout);

                        // Get tool from registry
                        auto& registry = ToolRegistry::instance();
                        Tool* tool = registry.get_tool(tool_name);
                        if (!tool) {
                            LOG_ERROR("Tool not found: " + tool_name);

                            // Feed error back to model so it can try a different approach
                            std::string error_msg = "Error: Tool '" + tool_name + "' not found. ";
                            error_msg += "Available tools you can use: WebSearch, Bash, Read, Write, Edit, Glob, Grep, get_fact, set_fact, search_memory, clear_fact";

                            if (is_interactive) {
                                printf("\033[36m> %s\033[0m\n", error_msg.substr(0, 100).c_str());
                            } else {
                                printf("> %s\n", error_msg.substr(0, 100).c_str());
                            }
                            fflush(stdout);

                            // Add the tool call to context
                            backend->add_assistant_message(json_str);

                            // Add error as tool result
                            backend->add_tool_result(tool_name, error_msg, tool_call_id);

                            // Continue loop to let model try again
                            continue;
                        }

                        // Execute tool
                        LOG_DEBUG("Executing tool: " + tool_name);
                        auto result = tool->execute(tool_call.parameters);

                        // Extract tool result content
                        std::string tool_result;
                        auto success_it = result.find("success");
                        if (success_it != result.end() && std::any_cast<bool>(success_it->second)) {
                            auto content_it = result.find("content");
                            if (content_it != result.end()) {
                                tool_result = std::any_cast<std::string>(content_it->second);
                            } else {
                                tool_result = "Tool executed successfully";
                            }
                        } else {
                            auto error_it = result.find("error");
                            if (error_it != result.end()) {
                                tool_result = "Error: " + std::any_cast<std::string>(error_it->second);
                            } else {
                                tool_result = "Error: Tool execution failed";
                            }
                        }

                        // Truncate large tool results to prevent context overflow
                        // Use actual tokenizer to count tokens accurately
                        int actual_tool_result_tokens = backend->get_context_manager().count_tokens(tool_result);

                        // Get current total tokens from context manager (includes ALL messages: system, user, assistant, tools)
                        int current_total_tokens = backend->get_context_manager().get_total_tokens();

                        // Reserve space for model's response - scale with context size
                        // Small contexts need proportionally more space for generation
                        size_t min_response_tokens = 2000; // Minimum 2K tokens for response
                        size_t context_based_reserve = context_size / 4; // Reserve 25% of context for response
                        size_t reserved_for_response = std::max(min_response_tokens, context_based_reserve);

                        LOG_DEBUG("Context calculation: size=" + std::to_string(context_size) +
                                 ", current_total=" + std::to_string(current_total_tokens) +
                                 ", tool_result=" + std::to_string(actual_tool_result_tokens) +
                                 ", reserved=" + std::to_string(reserved_for_response));

                        int max_tool_result_tokens = static_cast<int>(context_size) - current_total_tokens - static_cast<int>(reserved_for_response);
                        if (max_tool_result_tokens < 500) {
                            max_tool_result_tokens = 500; // Minimum fallback if context is tiny
                        }

                        LOG_DEBUG("Max tool result: " + std::to_string(max_tool_result_tokens) + " tokens allowed, actual: " +
                                 std::to_string(actual_tool_result_tokens) + " tokens");

                        if (actual_tool_result_tokens > max_tool_result_tokens) {
                            // Count lines for helpful guidance
                            size_t original_line_count = std::count(tool_result.begin(), tool_result.end(), '\n');

                            // Iteratively truncate by characters until it fits within token budget
                            // Start with a ratio estimate, then refine
                            double ratio = static_cast<double>(max_tool_result_tokens) / static_cast<double>(actual_tool_result_tokens);
                            size_t target_chars = static_cast<size_t>(tool_result.length() * ratio * 0.95); // 95% for safety margin

                            std::string truncation_notice = "\n\n[TRUNCATED: Output too large for context window]";
                            truncation_notice += "\nOriginal length: " + std::to_string(original_line_count) + " lines";
                            truncation_notice += "\nIf you need more: use Read(offset=X, limit=Y), Grep(pattern=...), or Glob with specific patterns";

                            // Reserve space for the truncation notice itself
                            int notice_tokens = backend->get_context_manager().count_tokens(truncation_notice);
                            int available_for_content = max_tool_result_tokens - notice_tokens;

                            if (available_for_content < 100) {
                                // If we can't even fit 100 tokens, just send an error message
                                tool_result = "[ERROR: Tool result too large for context window. Please use pagination or filters.]";
                            } else {
                                // Truncate to estimated size
                                std::string truncated = tool_result.substr(0, target_chars);

                                // Check if it fits now
                                int truncated_tokens = backend->get_context_manager().count_tokens(truncated);

                                // If still too large, do binary search to find the right size
                                int max_iterations = 5;
                                int iteration = 0;
                                while (truncated_tokens > available_for_content && iteration < max_iterations && target_chars > 100) {
                                    // Adjust ratio based on actual result
                                    ratio = static_cast<double>(available_for_content) / static_cast<double>(truncated_tokens);
                                    target_chars = static_cast<size_t>(target_chars * ratio * 0.9); // Smaller safety margin
                                    truncated = tool_result.substr(0, target_chars);
                                    truncated_tokens = backend->get_context_manager().count_tokens(truncated);
                                    iteration++;
                                }

                                tool_result = truncated + truncation_notice;

                                size_t final_line_count = std::count(truncated.begin(), truncated.end(), '\n');
                                if (g_debug_mode) {
                                    LOG_WARN("Truncated large tool result from " + tool_name + " (" +
                                            std::to_string(original_line_count) + " lines / " +
                                            std::to_string(actual_tool_result_tokens) + " tokens -> " +
                                            std::to_string(final_line_count) + " lines / " +
                                            std::to_string(truncated_tokens + notice_tokens) + " tokens)");
                                }
                            }
                        }

                        std::string result_preview = tool_result.substr(0, std::min(size_t(80), tool_result.length()));
                        if (tool_result.length() > 80) result_preview += "...";
                        LOG_DEBUG("Tool result: " + result_preview);

                        // Show tool result in transcript (truncated)
                        std::string truncated_result = tool_result;
                        size_t first_newline = truncated_result.find('\n');
                        if (first_newline != std::string::npos) {
                            size_t second_newline = truncated_result.find('\n', first_newline + 1);
                            if (second_newline != std::string::npos) {
                                truncated_result = truncated_result.substr(0, second_newline) + "\n...";
                            }
                        } else if (truncated_result.length() > 100) {
                            truncated_result = truncated_result.substr(0, 100) + "...";
                        }
                        if (is_interactive) {
                            printf("\033[36m> %s\033[0m\n", truncated_result.c_str());
                        } else {
                            printf("> %s\n", truncated_result.c_str());
                        }
                        fflush(stdout);

                        // Add assistant message with tool call JSON
                        backend->add_assistant_message(json_str);

                        // Add tool result to context (with tool_call_id if available)
                        backend->add_tool_result(tool_name, tool_result, tool_call_id);

                        // Continue loop to get model's response to tool result
                        continue;
                    } else {
                        // No tool call - this is the final response
                        LOG_DEBUG("No tool call detected, final response");

                        // Clean up malformed/incomplete tool call markers from display
                        std::string display_response = response;

                        // Remove incomplete <tool_call> without closing tag
                        size_t incomplete_marker = display_response.rfind("<tool_call>");
                        if (incomplete_marker != std::string::npos) {
                            size_t closing_tag = display_response.find("</tool_call>", incomplete_marker);
                            if (closing_tag == std::string::npos) {
                                display_response = display_response.substr(0, incomplete_marker);
                                size_t end = display_response.find_last_not_of(" \t\n\r");
                                if (end != std::string::npos) {
                                    display_response = display_response.substr(0, end + 1);
                                }
                            }
                        }

                        // Remove orphaned </tool_call> without opening tag
                        // IMPORTANT: Only remove the orphaned tag itself, not valid content before it
                        size_t closing_marker = display_response.rfind("</tool_call>");
                        if (closing_marker != std::string::npos) {
                            size_t opening_tag = display_response.rfind("<tool_call>", closing_marker);
                            if (opening_tag == std::string::npos || opening_tag > closing_marker) {
                                // Remove the orphaned closing tag and any preceding incomplete function tag
                                // But DON'T remove valid completed <function>...</function> blocks before it
                                size_t function_start = display_response.rfind("<function=", closing_marker);
                                size_t function_end = display_response.rfind("</function>", closing_marker);

                                // If there's an incomplete function tag (no closing </function>), remove from there
                                // Otherwise just remove the orphaned </tool_call>
                                size_t remove_from;
                                if (function_start != std::string::npos &&
                                    (function_end == std::string::npos || function_end < function_start)) {
                                    // Incomplete function tag - remove from function_start
                                    remove_from = function_start;
                                } else {
                                    // Just remove the orphaned closing tag
                                    remove_from = closing_marker;
                                }

                                display_response = display_response.substr(0, remove_from);
                                size_t end = display_response.find_last_not_of(" \t\n\r");
                                if (end != std::string::npos) {
                                    display_response = display_response.substr(0, end + 1);
                                }
                            }
                        }

                        // Filter out chat template tokens and <think> tags unless debug mode is enabled
                        if (!g_debug_mode && !display_response.empty()) {
                            // Remove Llama 3.x chat template tokens
                            // Pattern: <|start_header_id|>assistant<|end_header_id|>\n\n
                            size_t header_start = display_response.find("<|start_header_id|>");
                            if (header_start != std::string::npos) {
                                size_t header_end = display_response.find("<|end_header_id|>", header_start);
                                if (header_end != std::string::npos) {
                                    // Remove from start_header_id to end_header_id + token length
                                    display_response.erase(header_start, (header_end + 17) - header_start);
                                }
                            }

                            // Remove <think>...</think> blocks
                            size_t think_start = 0;
                            while ((think_start = display_response.find("<think>", think_start)) != std::string::npos) {
                                size_t think_end = display_response.find("</think>", think_start);
                                if (think_end != std::string::npos) {
                                    // Remove the entire <think>...</think> block
                                    display_response.erase(think_start, (think_end + 8) - think_start);
                                } else {
                                    // No closing tag found, break to avoid infinite loop
                                    break;
                                }
                            }

                            // Trim leading/trailing whitespace after removing tags
                            size_t first_non_space = display_response.find_first_not_of(" \n\r\t");
                            if (first_non_space != std::string::npos) {
                                display_response = display_response.substr(first_non_space);
                            }
                            size_t last_non_space = display_response.find_last_not_of(" \n\r\t");
                            if (last_non_space != std::string::npos) {
                                display_response = display_response.substr(0, last_non_space + 1);
                            }
                        }

                        // Show assistant response in transcript
                        if (!display_response.empty()) {
                            if (is_interactive) {
                                printf("\033[33m< %s\033[0m\n", display_response.c_str());
                            } else {
                                printf("< %s\n", display_response.c_str());
                            }
                            fflush(stdout);
                        }

                        // Add the FULL response (including <think> tags) to backend context
                        backend->add_assistant_message(response);
                        break;
                    }
                }

                LOG_DEBUG("Response from backend (first 100 chars): '" + response.substr(0, std::min(size_t(100), response.length())) + "'");
                LOG_DEBUG("Response from backend length: " + std::to_string(response.length()));

                // Note: Response printing is now handled inline in the tool loop above
                // with transcript-style formatting (< for assistant, > for user/tool results)

                // Restore terminal to normal mode after generation
                if (is_interactive) {
                    restore_terminal();
                }

            } catch (const std::exception& e) {
                // Restore terminal on error
                if (is_interactive) {
                    restore_terminal();
                    printf("\033[31mError: %s\033[0m\n", e.what());
                } else {
                    fprintf(stderr, "Error: %s\n", e.what());
                }
                LOG_ERROR("Error processing turn: " + std::string(e.what()));
            }
        }

    } catch (const std::exception& e) {
        restore_terminal();
        printf("\033[31mFatal error: %s\033[0m\n", e.what());
        LOG_FATAL("Fatal error: " + std::string(e.what()));
        return 1;
    }

    restore_terminal();
    LOG_INFO("Shutting down Shepherd...");
    LOG_INFO("Shutdown complete");

    return 0;
}
