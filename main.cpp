
#include "shepherd.h"
#include "tools/tools.h"
#include "backends/llamacpp.h"  // Include before mcp.h to define json as ordered_json
#include "mcp/mcp.h"
#include "api_tools/api_tools.h"
#include "rag.h"
#include "server/server.h"
#include "cli.h"
#include "backends/backend.h"
#include "backends/factory.h"
#include "backends/models.h"
#include "version.h"

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

// Global debug level (0=off, 1-9=increasing verbosity)
// Used by dprintf() macro in debug.h for fine-grained debug control
int g_debug_level = 0;

// Global config instance
std::unique_ptr<Config> config;

// Global server mode flag
bool g_server_mode = false;

// Global generation cancellation flag (for llamacpp backend)
bool g_generation_cancelled = false;


static void print_usage(int, char** argv) {
	printf("\n=== Shepherd - Advanced LLM Management System ===\n");
	printf("\nUsage:\n");
	printf("	%s [OPTIONS]\n", argv[0]);
	printf("	%s edit-system				Edit system prompt in $EDITOR\n", argv[0]);
	printf("	%s list-tools				List all available tools\n", argv[0]);
	printf("	%s mcp <add|remove|list> [args...]\n", argv[0]);
	printf("	%s api <add|remove|list> [args...]\n", argv[0]);
	printf("\nOptions:\n");
	printf("	-c, --config FILE  Specify config file (default: ~/.shepherd/config.json)\n");
	printf("	-d, --debug[=N]    Enable debug mode with optional level (1-9, default: 1)\n");
	printf("	-l, --log-file	   Log to file instead of console\n");
	printf("	-m, --model		   Model name or file (overrides config)\n");
	printf("	--model_path	   Model directory path (overrides config, e.g., ~/models)\n");
	printf("	--backend		   Backend (llamacpp, openai, anthropic, gemini, grok, ollama)\n");
	printf("	--api-key		   API key for cloud backends\n");
	printf("	--api-base		   API base URL (for OpenAI-compatible APIs)\n");
	printf("	--context-size	   Set context window size (0 = use model's full context, default: from config)\n");
	printf("	--gpu-layers N	   Number of model layers to offload to GPU (-1=auto/all, 0=CPU only, >0=specific count)\n");
	printf("	--models-file FILE Path to models database JSON file (default: ~/.shepherd/models.json)\n");
	printf("	--max-tokens	   Set max generation tokens (default: auto)\n");
	printf("	--memory-db		   Path to RAG memory database (default: ~/.shepherd/memory.db)\n");
	printf("	--nomcp			   Disable MCP system (no MCP servers loaded)\n");
	printf("	--template		   Custom chat template file (Jinja format, llamacpp only)\n");
	printf("	--server		   Start HTTP API server mode (OpenAI-compatible)\n");
	printf("	--port PORT		   Server port (default: 8000, requires --server)\n");
	printf("	--host HOST		   Server host to bind to (default: 0.0.0.0, requires --server)\n");
	printf("	--truncate LIMIT   Truncate tool results to LIMIT tokens (0 = auto 85%% of available space)\n");
	printf("	-v, --version	   Show version information\n");
	printf("	-h, --help		   Show this help message\n");
	printf("\nMCP Management:\n");
	printf("	mcp list							  List all configured MCP servers\n");
	printf("	mcp add <name> <cmd> [args] [-e ...]  Add a new MCP server\n");
	printf("										  -e KEY=VALUE	Set environment variable\n");
	printf("	mcp remove <name>					  Remove an MCP server\n");
	printf("\nAPI Tool Management:\n");
	printf("	api list							  List all configured API tools\n");
	printf("	api add <name> <backend> --model <model> [options]  Add a new API tool\n");
	printf("										  Options: --api-key, --api-base, --context-size, --max-tokens\n");
	printf("	api remove <name>					  Remove an API tool\n");
	printf("\nConfiguration:\n");
	printf("	Edit ~/.shepherd/config.json to configure:\n");
	printf("	- backend: llamacpp, openai, anthropic");
#ifdef PLATFORM_LINUX
#ifdef ENABLE_TENSORRT
	printf(", tensorrt");
#endif
#endif
	printf("\n");
	printf("	- model: model name or path\n");
	printf("	- model_path: directory for models (optional, defaults to ~/.shepherd/models)\n");
	printf("	- key: API key for cloud backends (optional)\n");
	printf("	- context_size: context window size (optional, 0 = auto)\n");
	printf("\nFeatures:\n");
	printf("	- Direct document reading and processing\n");
	printf("	- Conversation memory with search capabilities\n");
	printf("	- Multiple inference backends (local + cloud)\n");
	printf("	- Tool execution support\n");
	printf("	- Model Context Protocol (MCP) server integration\n");
	printf("\n");
}

static void signal_handler(int signal) {
	// Terminal cleanup is handled by CLI destructor
	printf("\n\nReceived signal %d, shutting down gracefully...\n", signal);
	exit(0);
}

static void cancel_handler(int signal) {
	(void)signal; // Suppress unused parameter warning
	// Set cancellation flag when SIGUSR1 received (from FastAPI on client disconnect)
	// g_generation_cancelled = true;  // This would need to be in CLI now
	LOG_INFO("Received SIGUSR1 - cancelling generation");
}

#if 0
static void show_config(const Config& config) {
	printf("\n=== Current Configuration ===\n");
	printf("Backend: %s\n", config.backend.c_str());
	printf("Model: %s\n", config.model.c_str());

	if (config.backend == "llamacpp" || config.backend == "tensorrt") {
		printf("Model path: %s\n", config.model_path.c_str());
	} else if (config.backend == "openai" || config.backend == "anthropic") {
		printf("API key: %s\n", config.key.empty() ? "(not set)" : "***");
	}

	printf("Context size: %zu\n", config.context_size);

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

static int handle_api_command(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: shepherd api <add|remove|list> [args...]" << std::endl;
		return 1;
	}

	std::string subcommand = argv[2];
	std::string config_path = std::string(getenv("HOME")) + "/.shepherd/config.json";

	if (subcommand == "list") {
		APIToolConfig::list_tools(config_path, false);
		return 0;
	}

	if (subcommand == "add") {
		if (argc < 5) {
			std::cerr << "Usage: shepherd api add <name> <backend> --model <model> [options]" << std::endl;
			std::cerr << "\nOptions:" << std::endl;
			std::cerr << "  --model <model>         Model name (required)" << std::endl;
			std::cerr << "  --api-key <key>         API key" << std::endl;
			std::cerr << "  --api-base <url>        Custom API base URL" << std::endl;
			std::cerr << "  --context-size <size>   Context window size" << std::endl;
			std::cerr << "  --max-tokens <tokens>   Max generation tokens" << std::endl;
			std::cerr << "\nExamples:" << std::endl;
			std::cerr << "  shepherd api add ask_claude anthropic --model claude-sonnet-4 --api-key sk-ant-..." << std::endl;
			std::cerr << "  shepherd api add ask_gpt openai --model gpt-4 --api-key sk-... --max-tokens 8000" << std::endl;
			std::cerr << "  shepherd api add local_llama ollama --model llama3 --api-base http://localhost:11434" << std::endl;
			return 1;
		}

		APIToolEntry tool;
		tool.name = argv[3];
		tool.backend = argv[4];
		tool.context_size = 0;
		tool.max_tokens = 0;

		// Parse options
		for (int i = 5; i < argc; i++) {
			std::string arg = argv[i];

			if (arg == "--model") {
				if (i + 1 < argc) {
					tool.model = argv[++i];
				} else {
					std::cerr << "Error: --model requires an argument" << std::endl;
					return 1;
				}
			} else if (arg == "--api-key") {
				if (i + 1 < argc) {
					tool.api_key = argv[++i];
				} else {
					std::cerr << "Error: --api-key requires an argument" << std::endl;
					return 1;
				}
			} else if (arg == "--api-base") {
				if (i + 1 < argc) {
					tool.api_base = argv[++i];
				} else {
					std::cerr << "Error: --api-base requires an argument" << std::endl;
					return 1;
				}
			} else if (arg == "--context-size") {
				if (i + 1 < argc) {
					tool.context_size = std::stoull(argv[++i]);
				} else {
					std::cerr << "Error: --context-size requires an argument" << std::endl;
					return 1;
				}
			} else if (arg == "--max-tokens") {
				if (i + 1 < argc) {
					tool.max_tokens = std::stoi(argv[++i]);
				} else {
					std::cerr << "Error: --max-tokens requires an argument" << std::endl;
					return 1;
				}
			} else {
				std::cerr << "Error: Unknown option: " << arg << std::endl;
				std::cerr << "Use 'shepherd api add' without arguments to see usage and examples" << std::endl;
				return 1;
			}
		}

		// Validate required fields
		if (tool.model.empty()) {
			std::cerr << "Error: --model is required" << std::endl;
			std::cerr << "Use 'shepherd api add' without arguments to see usage and examples" << std::endl;
			return 1;
		}

		if (APIToolConfig::add_tool(config_path, tool)) {
			std::cout << "Added API tool '" << tool.name << "'" << std::endl;
			return 0;
		}
		return 1;
	}

	if (subcommand == "remove") {
		if (argc < 4) {
			std::cerr << "Usage: shepherd api remove <name>" << std::endl;
			return 1;
		}

		std::string name = argv[3];
		if (APIToolConfig::remove_tool(config_path, name)) {
			std::cout << "Removed API tool '" << name << "'" << std::endl;
			return 0;
		}
		return 1;
	}

	std::cerr << "Unknown api subcommand: " << subcommand << std::endl;
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
	std::string current_prompt = config.system_prompt;
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

#if 0
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
	schema += "		 \"type\": \"object\",\n";
	schema += "		 \"properties\": {\n";

	// Add properties
	for (size_t i = 0; i < params.size(); ++i) {
		const auto& param = params[i];
		schema += "		   \"" + param.name + "\": {\n";
		schema += "			 \"type\": \"" + param.type + "\"";
		if (!param.description.empty()) {
			schema += ",\n			\"description\": \"" + param.description + "\"";
		}
		schema += "\n		 }";
		if (i < params.size() - 1) {
			schema += ",";
		}
		schema += "\n";
	}

	schema += "		 }";

	// Add required fields
	std::vector<std::string> required_params;
	for (const auto& param : params) {
		if (param.required) {
			required_params.push_back(param.name);
		}
	}
	if (!required_params.empty()) {
		schema += ",\n		\"required\": [";
		for (size_t i = 0; i < required_params.size(); ++i) {
			schema += "\"" + required_params[i] + "\"";
			if (i < required_params.size() - 1) {
				schema += ", ";
			}
		}
		schema += "]";
	}

	schema += "\n	 }\n";
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
#endif

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

		// Initialize API tools
		auto& api_tools = APITools::instance();
		api_tools.initialize();

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
						printf("	- %s (%s)%s: %s\n",
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

	// Handle API subcommand
	if (argc >= 2 && std::string(argv[1]) == "api") {
		return handle_api_command(argc, argv);
	}

	bool debug_override = false;
	bool no_mcp = false;
	int truncate_limit = 0;
	std::string log_file;
	std::string config_file_path;
	std::string model_override;
	std::string model_path_override;
	std::string backend_override;
	std::string api_key_override;
	std::string api_base_override;
	std::string template_override;
	std::string memory_db_override;
	std::string models_file_override;
	int server_port = 8000;
	std::string server_host = "0.0.0.0";
	int context_size_override = -1;  // -1 means not specified, 0 means use model's full context
	int gpu_layers_override = -999;  // -999 means not specified, -1=auto/all, 0=CPU only, >0=specific

	static struct option long_options[] = {
		{"config", required_argument, 0, 'c'},
		{"debug", optional_argument, 0, 'd'},
		{"log-file", required_argument, 0, 'l'},
		{"model", required_argument, 0, 'm'},
		{"model_path", required_argument, 0, 1018},
		{"backend", required_argument, 0, 1002},
		{"api-key", required_argument, 0, 1003},
		{"api-base", required_argument, 0, 1004},
		{"context-size", required_argument, 0, 1000},
		{"gpu-layers", required_argument, 0, 1025},
		{"memory-db", required_argument, 0, 1023},
		{"models-file", required_argument, 0, 1024},
		{"nomcp", no_argument, 0, 1005},
		{"template", required_argument, 0, 1006},
		{"server", no_argument, 0, 1015},
		{"port", required_argument, 0, 1016},
		{"host", required_argument, 0, 1017},
		{"truncate", required_argument, 0, 1019},
		{"version", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "c:dl:m:vh", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'c':
				config_file_path = optarg;
				break;
			case 'd':
				debug_override = true;
				// Parse optional debug level (default to 1 if not specified)
				if (optarg) {
					g_debug_level = atoi(optarg);
				} else {
					g_debug_level = 1;
				}
				break;
			case 'l':
				log_file = optarg;
				break;
			case 'm':
				model_override = optarg;
				break;
			case 1018: // --model_path
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
			case 1025: // --gpu-layers
				gpu_layers_override = std::atoi(optarg);
				break;
			case 1005: // --nomcp
				no_mcp = true;
				break;
			case 1006: // --template
				template_override = optarg;
				break;
			case 1015: // --server
				g_server_mode = true;
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
			case 1019: // --truncate
				truncate_limit = std::atoi(optarg);
				break;
			case 1023: // --memory-db
				memory_db_override = optarg;
				break;
			case 1024: // --models-file
				models_file_override = optarg;
				break;
			case 'v':
				printf("Shepherd version %s\n", SHEPHERD_VERSION);
				return 0;
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
	// g_debug_level already set during argument parsing, don't overwrite it

	if (g_debug_level) {
		logger.set_log_level(LogLevel::DEBUG);
		std::cout << "Debug mode enabled (level " << g_debug_level << ")" << std::endl;
	} else {
		// Suppress INFO logs unless in debug mode
		logger.set_log_level(LogLevel::WARN);
	}

	// Load configuration
	config = std::make_unique<Config>();

	// Set custom config path if specified
	if (!config_file_path.empty()) {
		config->set_config_path(config_file_path);
	}

	try {
		config->load();
	} catch (const ConfigError& e) {
		fprintf(stderr, "Configuration error: %s\n", e.what());
		return 1;
	}

	// Apply command-line overrides
	if (!backend_override.empty()) {
		config->set_backend(backend_override);
	}
	if (!api_key_override.empty()) {
		config->key = api_key_override;
	}
	if (!api_base_override.empty()) {
		config->api_base = api_base_override;
	}
	if (context_size_override >= 0) {  // -1 means not set, 0+ are valid values
		config->context_size = context_size_override;
	}
	if (truncate_limit > 0) {
		config->truncate_limit = truncate_limit;
	}
	if (gpu_layers_override != -999) {  // -999 means not specified
		// Update llamacpp backend config with gpu_layers
		json backend_config;
		if (config->backend_configs.find("llamacpp") != config->backend_configs.end()) {
			backend_config = json::parse(config->backend_configs["llamacpp"]);
		}
		backend_config["gpu_layers"] = gpu_layers_override;
		config->backend_configs["llamacpp"] = backend_config.dump();
	}

	// Validate configuration (skip model path check if overridden)
	if (model_override.empty() && model_path_override.empty()) {
		// No overrides - validate everything from config
		try {
			config->validate();
		} catch (const ConfigError& e) {
			fprintf(stderr, "Configuration error: %s\n", e.what());
			fprintf(stderr, "Edit ~/.shepherd/config->json or use -h for help\n");
			return 1;
		}
	} else {
		// With overrides, just validate backend availability
		auto available = Config::get_available_backends();
		bool backend_found = false;
		for (const auto& b : available) {
			if (b == config->backend) {
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
					config->backend.c_str(), available_str.c_str());
			return 1;
		}
		// Validate model file exists (only for local backends with overrides)
		if (config->backend == "llamacpp" || config->backend == "tensorrt") {
			// Construct the full path to validate
			std::string model_file_to_check;
			std::string model_name = model_override.empty() ? config->model : model_override;
			std::string model_dir = model_path_override.empty() ? config->model_path : model_path_override;

			// Check if model_name is already a full path
			if (!model_name.empty() && (model_name[0] == '/' || model_name[0] == '~')) {
				model_file_to_check = model_name;
			} else {
				model_file_to_check = std::filesystem::path(model_dir) / model_name;
			}

			// Expand ~ if present
			if (!model_file_to_check.empty() && model_file_to_check[0] == '~') {
				model_file_to_check = Config::get_home_directory() + model_file_to_check.substr(1);
			}

			if (!std::filesystem::exists(model_file_to_check)) {
				fprintf(stderr, "Model file not found: %s\n", model_file_to_check.c_str());
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
	LOG_INFO("Backend: " + config->backend);

	// Set up signal handlers for graceful shutdown
	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);
	signal(SIGUSR1, cancel_handler);  // For cancellation from FastAPI on client disconnect

	size_t context_size = context_size_override >= 0 ? static_cast<size_t>(context_size_override) : config->context_size;

	// Initialize models database if custom file specified (command-line takes precedence)
	std::string models_file = models_file_override.empty() ? config->models_file : models_file_override;
	if (!models_file.empty()) {
		Models::init(models_file);
	}

	// Create and initialize backend (needed for both CLI and server mode)
	try {
		// Create backend manager
		std::string model_path;

		// Apply command-line overrides to config (for local backends)
		if (config->backend == "llamacpp" || config->backend == "tensorrt") {
			if (!model_override.empty()) {
				config->model = model_override;
				LOG_INFO("Model override from command line: " + model_override);
			}
			if (!model_path_override.empty()) {
				config->model_path = model_path_override;
				LOG_INFO("Model path override from command line: " + model_path_override);
			}
		}

		// Show status to user (only for local backends that actually load models)
		bool is_local_backend = (config->backend == "llamacpp" || config->backend == "tensorrt");
		if (is_interactive && is_local_backend) {
			printf("Initializing Engine...\n");
			fflush(stdout);
		}

		// Create the backend
		auto backend = BackendFactory::create_backend(config->backend, context_size);

		if (!backend) {
			LOG_ERROR("Failed to create backend: " + config->backend);
			return 1;
		}


		// Initialize RAG system with specified or default path (skip in server mode - clients have their own RAG)
		if (!g_server_mode) {
			std::string db_path;
			if (!memory_db_override.empty()) {
				// Command-line override takes precedence
				db_path = memory_db_override;
				// Expand ~ if present
				if (db_path[0] == '~') {
					db_path = Config::get_home_directory() + db_path.substr(1);
				}
				LOG_INFO("Using memory database from command line: " + db_path);
			} else if (!config->memory_database.empty()) {
				// Config file value
				db_path = config->memory_database;
				// Expand ~ if present
				if (db_path[0] == '~') {
					db_path = Config::get_home_directory() + db_path.substr(1);
				}
				LOG_INFO("Using memory database from config: " + db_path);
			} else {
				// Default
				try {
					db_path = Config::get_home_directory() + "/.shepherd/memory.db";
				} catch (const ConfigError& e) {
					LOG_ERROR("Failed to determine home directory: " + std::string(e.what()));
					return 1;
				}
			}
			if (!RAGManager::initialize(db_path, config->max_db_size)) {
				LOG_ERROR("Failed to initialize RAG system");
				return 1;
			}
		}

		// Show status to user before the slow model load (only for local backends)
		if (is_interactive && is_local_backend) {
			printf("Loading Model...\n");
			fflush(stdout);
		}

		// Create the session that will be used throughout (not in server mode)
		Session session;
		session.system_message = config->system_message;

		// Initialize tools system (skip in server mode - tools handled by client)
		if (!g_server_mode) {
			LOG_INFO("Initializing tools system...");
			try {
				// Register all native tools including memory search
				register_filesystem_tools();
				register_command_tools();
				register_json_tools();
				register_http_tools();
				register_memory_tools();
				register_mcp_resource_tools();
				register_core_tools();	// IMPORTANT: Register core tools (Bash, Glob, Grep, Edit, WebSearch, etc.)

				auto& registry = ToolRegistry::instance();
				auto tools = registry.list_tools();
				LOG_INFO("Native tools initialized with " + std::to_string(tools.size()) + " tools");
				if (g_debug_level) {
					for (const auto& tool_name : tools) {
						LOG_DEBUG("Registered tool: " + tool_name);
					}
				}

				// Initialize MCP servers (will register additional tools)
				size_t mcp_count = 0;
				if (!no_mcp) {
					auto& mcp = MCP::instance();
					mcp.initialize();
					mcp_count = mcp.get_tool_count();
				} else {
					LOG_INFO("MCP system disabled via --nomcp flag");
				}

				// Initialize API tools (will register additional tools)
				auto& api_tools = APITools::instance();
				api_tools.initialize();
				size_t api_tool_count = api_tools.get_tool_count();

				// Show total tool count after MCP and API tools
				tools = registry.list_tools();
				if (mcp_count > 0 || api_tool_count > 0) {
					LOG_INFO("Total tools available: " + std::to_string(tools.size()) +
							 " (native + " + std::to_string(mcp_count) + " MCP + " +
							 std::to_string(api_tool_count) + " API)");
				} else {
					LOG_INFO("Total tools available: " + std::to_string(tools.size()) + " (native only)");
				}

			} catch (const std::exception& e) {
				LOG_ERROR("Failed to initialize tools system: " + std::string(e.what()));
				return 1;
			}
		} else {
			LOG_INFO("Server mode: Tool registration skipped (client-side tools)");
		}

		// Get registry and tool descriptions AFTER all tools are registered
		// (needed for both interactive and server modes)
		auto& registry = ToolRegistry::instance();
		auto tool_descriptions = registry.list_tools_with_descriptions();

		LOG_INFO("Shepherd initialization complete");

		// ============================================================================
		// Server Mode - HTTP API via FastAPI
		// ============================================================================
		if (g_server_mode) {
			// Initialize backend before starting server
			LOG_DEBUG("Initializing backend for server mode...");
			backend->initialize(session);
			return run_server(backend, server_host, server_port);
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

		// Force interactive mode for rank 0 (mpirun can make isatty return false)
		if (is_mpi_leader) {
			is_interactive = true;
		}

		// Populate tools from registry
		// This converts from ToolRegistry format to Session::Tool format
		for (const auto& [tool_name, tool_desc] : tool_descriptions) {
			Session::Tool st;
			st.name = tool_name;
			st.description = tool_desc;

			// Get parameters schema from the actual tool
			Tool* tool = registry.get_tool(tool_name);
			if (tool) {
				auto params_schema = tool->get_parameters_schema();
				if (!params_schema.empty()) {
					// Build proper JSON schema from ParameterDef vector
					nlohmann::json params_json;
					params_json["type"] = "object";
					params_json["properties"] = nlohmann::json::object();

					std::vector<std::string> required_fields;
					for (const auto& param : params_schema) {
						nlohmann::json prop;
						prop["type"] = param.type;
						if (!param.description.empty()) {
							prop["description"] = param.description;
						}
						if (!param.default_value.empty()) {
							prop["default"] = param.default_value;
						}
						params_json["properties"][param.name] = prop;

						if (param.required) {
							required_fields.push_back(param.name);
						}
					}

					if (!required_fields.empty()) {
						params_json["required"] = required_fields;
					}

					st.parameters = params_json;
				} else {
					// Legacy tools without structured schema - create minimal schema
					st.parameters = nlohmann::json::object();
					st.parameters["type"] = "object";
					st.parameters["properties"] = nlohmann::json::object();
				}
			}

			session.tools.push_back(st);
		}

		LOG_DEBUG("Session created with " + std::to_string(session.tools.size()) + " tools");

		// Initialize backend (calibrate tokens, validate setup, etc.)
		LOG_DEBUG("Initializing backend...");
		backend->initialize(session);
		session.backend = backend.get();

		// Calculate desired completion tokens once (used throughout session lifetime)
		// Must be calculated AFTER backend initialization when context_size is finalized
		session.desired_completion_tokens = calculate_desired_completion_tokens(
			backend->context_size,
			backend->max_output_tokens
		);

		// Enable auto-eviction for API backends when context size is specified
		// Local backends (llamacpp) handle eviction through reactive callbacks
		// Must be set AFTER initialization when backend->context_size is finalized
		session.auto_evict = (backend->context_size > 0 && !backend->is_local);
		if (session.auto_evict) {
			LOG_INFO("Auto-eviction enabled (context_size=" + std::to_string(backend->context_size) +
			         ", desired_completion=" + std::to_string(session.desired_completion_tokens) + ")");
		}

		// ============================================================================

	// ============================================================================
	// Server Mode - HTTP API via FastAPI
	// ============================================================================
	if (g_server_mode) {
		return run_server(backend, server_host, server_port);
	}

	// ============================================================================
	// CLI Mode - Interactive or Piped Input
	// ============================================================================
	return run_cli(backend, session);

} catch (const std::exception& e) {
	fprintf(stderr, "Fatal error: %s\n", e.what());
	LOG_FATAL("Fatal error: " + std::string(e.what()));
	return 1;
}
}

