
#include "shepherd.h"
#include "tools/tools.h"
#include "backends/llamacpp.h"  // Include before mcp.h to define json as ordered_json
#include "mcp/mcp.h"
#include "mcp/mcp_config.h"
#include "provider.h"
#include "api_tools/api_tools.h"
#include "rag.h"
#include "cli.h"
#include "server/api_server.h"
#include "server/cli_server.h"
#include "terminal_io.h"
#include "backends/backend.h"
#include "backends/factory.h"
#include "backends/models.h"
#include "version.h"
#include "scheduler.h"

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

// MPI support for multi-GPU TensorRT
#ifdef ENABLE_TENSORRT
#include <mpi.h>
#endif

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
bool g_show_thinking = false;

// Global config instance
std::unique_ptr<Config> config;

// Global server mode flag
bool g_server_mode = false;

// Global generation cancellation flag (for llamacpp backend)
bool g_generation_cancelled = false;

// Scheduler disable flag (--nosched)
bool g_disable_scheduler = false;


static void print_usage(int, char** argv) {
	printf("\n=== Shepherd - Advanced LLM Management System ===\n");
	printf("\nUsage:\n");
	printf("	%s [OPTIONS]\n", argv[0]);
	printf("	%s edit-system				Edit system prompt in $EDITOR\n", argv[0]);
	printf("	%s list-tools				List all available tools\n", argv[0]);
	printf("	%s provider <add|list|show|remove|use> [args...]\n", argv[0]);
	printf("	%s config <show|set> [args...]\n", argv[0]);
	printf("	%s mcp <add|remove|list> [args...]\n", argv[0]);
	printf("	%s api <add|remove|list> [args...]\n", argv[0]);
	printf("	%s sched <list|add|remove|enable|disable|show|next> [args...]\n", argv[0]);
	printf("\nOptions:\n");
	printf("	-c, --config FILE  Specify config file (default: ~/.shepherd/config.json)\n");
	printf("	-d, --debug[=N]    Enable debug mode with optional level (1-9, default: 1)\n");
	printf("	-l, --log-file	   Log to file instead of console\n");
	printf("	-m, --model		   Model name or file (overrides config)\n");
	printf("	-p, --provider	   Provider name to use (from provider list)\n");
	printf("	--model_path	   Model directory path (overrides config, e.g., ~/models)\n");
	printf("	--backend		   Backend (llamacpp, tensorrt, openai, anthropic, gemini, grok, ollama)\n");
	printf("	--api-key		   API key for cloud backends\n");
	printf("	--api-base		   API base URL (for OpenAI-compatible APIs)\n");
	printf("	--context-size	   Set context window size (0 = use model's full context, default: from config)\n");
	printf("	--gpu-layers N	   Number of model layers to offload to GPU (-1=auto/all, 0=CPU only, >0=specific count)\n");
	printf("	--tp N			   Tensor parallelism size (llamacpp only, default: 1)\n");
	printf("	--pp N			   Pipeline parallelism size (llamacpp only, default: 1)\n");
	printf("	--ubatch N		   Micro-batch size for prompt processing (llamacpp only, default: auto)\n");
	printf("	--models-file FILE Path to models database JSON file (default: ~/.config/shepherd/models.json)\n");
	printf("	--max-tokens	   Set max generation tokens (default: auto)\n");
	printf("	--memory-db		   Path to RAG memory database (default: ~/.local/share/shepherd/memory.db)\n");
	printf("	--nomcp			   Disable MCP system (no MCP servers loaded)\n");
	printf("	--nosched		   Disable scheduler (no scheduled tasks run)\n");
	printf("	--notools		   Disable all tools (no tool registration or use)\n");
	printf("	--system-prompt	   Override system prompt (useful with --notools)\n");
	printf("	--template		   Custom chat template file (Jinja format, llamacpp only)\n");
	printf("	--apiserver		   Start HTTP API server mode (OpenAI-compatible)\n");
	printf("	--server		   Alias for --apiserver\n");
	printf("	--cliserver		   Start CLI server mode (local tool execution)\n");
	printf("	--port PORT		   Server port (default: 8000, requires --apiserver or --cliserver)\n");
	printf("	--host HOST		   Server host to bind to (default: 0.0.0.0, requires --apiserver or --cliserver)\n");
	printf("	--truncate LIMIT   Truncate tool results to LIMIT tokens (0 = auto 85%% of available space)\n");
	printf("	--warmup		   Send warmup message before first user prompt (initializes model)\n");
	printf("	--colors		   Force enable colored output (overrides environment)\n");
	printf("	--no-colors		   Force disable colored output (overrides environment)\n");
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
	printf("	Edit ~/.config/shepherd/config.json to configure:\n");
	printf("	- backend: llamacpp, openai, anthropic");
#ifdef PLATFORM_LINUX
#ifdef ENABLE_TENSORRT
	printf(", tensorrt");
#endif
#endif
	printf("\n");
	printf("	- model: model name or path\n");
	printf("	- model_path: directory for models (optional, defaults to ~/.local/share/shepherd/models)\n");
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

static int handle_config_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "config")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation
	return handle_config_args(args);
}

static int handle_provider_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "provider")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation
	return handle_provider_args(args);
}

static int handle_mcp_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "mcp")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation
	return handle_mcp_args(args);
}

static int handle_api_command(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "Usage: shepherd api <add|remove|list> [args...]" << std::endl;
		return 1;
	}

	std::string subcommand = argv[2];
	std::string config_path = Config::get_default_config_path();

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

static int handle_sched_command(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "sched")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation
	return handle_sched_args(args);
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
	std::string config_path = Config::get_default_config_path();

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

int main(int argc, char** argv) {
	// Store original arguments for potential MPI re-exec
	set_global_args(argc, argv);

	// Detect interactivity BEFORE any MPI re-exec (isatty fails after mpirun)
	// Only check if not already set (preserve across MPI re-exec)
	if (!getenv("SHEPHERD_INTERACTIVE")) {
		int is_interactive = isatty(STDIN_FILENO);
		setenv("SHEPHERD_INTERACTIVE", is_interactive ? "1" : "0", 0);
		LOG_DEBUG("Set SHEPHERD_INTERACTIVE=" + std::string(is_interactive ? "1" : "0") +
		          " (isatty=" + std::to_string(is_interactive) + ")");
	} else {
		LOG_DEBUG("SHEPHERD_INTERACTIVE already set to: " + std::string(getenv("SHEPHERD_INTERACTIVE")));
	}

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

	// Handle provider subcommand
	if (argc >= 2 && std::string(argv[1]) == "provider") {
		return handle_provider_subcommand(argc, argv);
	}

	// Handle config subcommand
	if (argc >= 2 && std::string(argv[1]) == "config") {
		return handle_config_subcommand(argc, argv);
	}

	// Handle MCP subcommand
	if (argc >= 2 && std::string(argv[1]) == "mcp") {
		return handle_mcp_subcommand(argc, argv);
	}

	// Handle API subcommand
	if (argc >= 2 && std::string(argv[1]) == "api") {
		return handle_api_command(argc, argv);
	}

	// Handle sched subcommand
	if (argc >= 2 && std::string(argv[1]) == "sched") {
		return handle_sched_command(argc, argv);
	}

	bool debug_override = false;
	bool no_mcp = false;
	bool no_tools = false;
	bool warmup_enable = false;
	bool calibration_enable = false;
	bool thinking_override = false;
	int truncate_limit = 0;
	int color_override = -1;  // -1 = auto, 0 = off, 1 = on
	std::string log_file;
	std::string config_file_path;
	std::string provider_override;
	std::string model_override;
	std::string model_path_override;
	std::string backend_override;
	std::string api_key_override;
	std::string api_base_override;
	std::string template_override;
	std::string memory_db_override;
	std::string models_file_override;
	std::string system_prompt_override;
	bool system_prompt_was_set = false;
	int server_port = 8000;
	std::string server_host = "0.0.0.0";
	int context_size_override = -1;  // -1 means not specified, 0 means use model's full context
	int gpu_layers_override = -999;  // -999 means not specified, -1=auto/all, 0=CPU only, >0=specific
	int tp_override = -1;  // -1 means not specified
	int pp_override = -1;  // -1 means not specified
	int ubatch_override = -1;  // -1 means not specified
	std::string frontend_mode = "cli";

	static struct option long_options[] = {
		{"config", required_argument, 0, 'c'},
		{"debug", optional_argument, 0, 'd'},
		{"log-file", required_argument, 0, 'l'},
		{"model", required_argument, 0, 'm'},
		{"provider", required_argument, 0, 'p'},
		{"model_path", required_argument, 0, 1018},
		{"backend", required_argument, 0, 1002},
		{"api-key", required_argument, 0, 1003},
		{"api-base", required_argument, 0, 1004},
		{"context-size", required_argument, 0, 1000},
		{"gpu-layers", required_argument, 0, 1025},
		{"memory-db", required_argument, 0, 1023},
		{"models-file", required_argument, 0, 1024},
		{"nomcp", no_argument, 0, 1005},
		{"nosched", no_argument, 0, 1037},
		{"notools", no_argument, 0, 1027},
		{"system-prompt", required_argument, 0, 1028},
		{"template", required_argument, 0, 1006},
		{"apiserver", no_argument, 0, 1015},
		{"server", no_argument, 0, 1015},  // Alias for --apiserver
		{"cliserver", no_argument, 0, 1036},
		{"port", required_argument, 0, 1016},
		{"host", required_argument, 0, 1017},
		{"truncate", required_argument, 0, 1019},
		{"warmup", no_argument, 0, 1026},
		{"tp", required_argument, 0, 1029},
		{"pp", required_argument, 0, 1030},
		{"ubatch", required_argument, 0, 1035},
		{"thinking", no_argument, 0, 1031},
		{"colors", no_argument, 0, 1032},
		{"no-colors", no_argument, 0, 1033},
		{"calibration", no_argument, 0, 1034},
		{"version", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
	while ((opt = getopt_long(argc, argv, "c:dl:m:p:vh", long_options, &option_index)) != -1) {
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
			case 'p':
				provider_override = optarg;
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
			case 1037: // --nosched
				g_disable_scheduler = true;
				break;
			case 1027: // --notools
				no_tools = true;
				break;
			case 1028: // --system-prompt
				system_prompt_override = optarg;
				system_prompt_was_set = true;
				break;
			case 1006: // --template
				template_override = optarg;
				break;
			case 1015: // --apiserver
				frontend_mode = "api-server";
				break;
			case 1036: // --cliserver
				frontend_mode = "cli-server";
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
			case 1026: // --warmup
				warmup_enable = true;
				break;
			case 1029: // --tp
				tp_override = std::atoi(optarg);
				if (tp_override < 1) {
					printf("Error: tp must be at least 1\n");
					return 1;
				}
				break;
			case 1030: // --pp
				pp_override = std::atoi(optarg);
				if (pp_override < 1) {
					printf("Error: pp must be at least 1\n");
					return 1;
				}
				break;
			case 1035: // --ubatch
				ubatch_override = std::atoi(optarg);
				if (ubatch_override < 1) {
					printf("Error: ubatch must be at least 1\n");
					return 1;
				}
				break;
			case 1031: // --thinking
				g_show_thinking = true;
				thinking_override = true;
				break;
			case 1032: // --colors
				color_override = 1;
				break;
			case 1033: // --no-colors
				color_override = 0;
				break;
			case 1034: // --calibration
				calibration_enable = true;
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

	// Initialize terminal I/O early (before logger and other systems)
	if (!tio.init(color_override)) {
		return 1;  // Failed to initialize terminal
	}

	// Initialize logger early
	Logger& logger = Logger::instance();
	// g_debug_level already set during argument parsing, don't overwrite it

	if (g_debug_level) {
		logger.set_log_level(LogLevel::DEBUG);
		std::cout << "Debug mode enabled (level " << g_debug_level << ")" << std::endl;

		// Log interactivity detection for debugging
		const char* interactive_env = getenv("SHEPHERD_INTERACTIVE");
		std::string debug_msg = "SHEPHERD_INTERACTIVE=" + std::string(interactive_env ? interactive_env : "not set") +
		                        " (tio.interactive_mode=" + std::to_string(tio.interactive_mode) + ")";
		std::cerr << "[DEBUG-MAIN] " << debug_msg << std::endl;
		LOG_DEBUG(debug_msg);
	} else if (!g_server_mode) {
		// In client mode, suppress INFO logs unless debug is enabled
		logger.set_log_level(LogLevel::WARN);
	}
	// In server mode, default log level is INFO (set in Logger constructor)

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
		try {
			config->set_backend(backend_override);
		} catch (const ConfigError& e) {
			fprintf(stderr, "Error: %s\n", e.what());
			return 1;
		}
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
	if (warmup_enable) {
		config->warmup = true;
	}
	if (calibration_enable) {
		config->calibration = true;
	}
	// Set g_show_thinking from config, then apply command-line override
	g_show_thinking = config->thinking;
	if (thinking_override) {
		config->thinking = true;
		g_show_thinking = true;
	}
	if (gpu_layers_override != -999) {  // -999 means not specified
		config->json["gpu_layers"] = gpu_layers_override;
	}
	if (tp_override != -1) {  // -1 means not specified
		config->json["tp"] = tp_override;
	}
	if (pp_override != -1) {  // -1 means not specified
		config->json["pp"] = pp_override;
	}
	if (ubatch_override != -1) {  // -1 means not specified
		config->json["ubatch"] = ubatch_override;
	}

	// Validate configuration (skip model path check if overridden)
	if (model_override.empty() && model_path_override.empty()) {
		// No overrides - validate everything from config
		try {
			config->validate();
		} catch (const ConfigError& e) {
			fprintf(stderr, "Configuration error: %s\n", e.what());
			fprintf(stderr, "Edit ~/.config/shepherd/config.json or use -h for help\n");
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

		// Create the backend from provider system or legacy config
		std::unique_ptr<Backend> backend;
		bool use_provider_mode = backend_override.empty() && model_override.empty() && api_key_override.empty();
		Provider provider_mgr;  // Need this for provider mode

		// Check if using command-line overrides (legacy mode)
		if (!use_provider_mode) {
			// Legacy mode: create from command-line args (will initialize after session is ready)
			LOG_INFO("Using command-line overrides (legacy mode)");
			backend = BackendFactory::create_backend(config->backend, context_size);
			if (!backend) {
				LOG_ERROR("Failed to create backend");
				return 1;
			}
		}
		// Provider mode: backend will be created after session/tools are ready


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
					db_path = Config::get_default_memory_db_path();
				} catch (const ConfigError& e) {
					LOG_ERROR("Failed to determine memory database path: " + std::string(e.what()));
					return 1;
				}
			}
			if (!RAGManager::initialize(db_path, config->max_db_size)) {
				LOG_ERROR("Failed to initialize RAG system");
				return 1;
			}
		}

		// Create the session that will be used throughout (not in server mode)
		Session session;
		session.system_message = system_prompt_was_set ? system_prompt_override : config->system_message;

		// Initialize tools system (skip in server mode - tools handled by client)
		if (!g_server_mode) {
			if (!no_tools) {
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

				// Populate session.tools from ToolRegistry for API backends
				for (const auto& tool_name : tools) {
					Tool* tool_ptr = registry.get_tool(tool_name);
					if (tool_ptr) {
						Session::Tool session_tool;
						session_tool.name = tool_ptr->name();
						session_tool.description = tool_ptr->description();

						// Build JSON schema from parameter definitions
						auto params = tool_ptr->get_parameters_schema();
						nlohmann::json schema;
						schema["type"] = "object";
						schema["properties"] = nlohmann::json::object();
						nlohmann::json required = nlohmann::json::array();

						for (const auto& param : params) {
							nlohmann::json param_schema;
							param_schema["type"] = param.type;
							if (!param.description.empty()) {
								param_schema["description"] = param.description;
							}
							schema["properties"][param.name] = param_schema;

							if (param.required) {
								required.push_back(param.name);
							}
						}

						if (!required.empty()) {
							schema["required"] = required;
						}

						session_tool.parameters = schema;
						session.tools.push_back(session_tool);
					}
				}
				LOG_DEBUG("Session initialized with " + std::to_string(session.tools.size()) + " tools");

				} catch (const std::exception& e) {
					LOG_ERROR("Failed to initialize tools system: " + std::string(e.what()));
					return 1;
				}
			} else {
				LOG_INFO("Tools disabled via --notools flag");
			}
		} else {
			LOG_INFO("Server mode: Tool registration skipped (client-side tools)");
		}

		// Get registry and tool descriptions AFTER all tools are registered
		// (needed for both interactive and server modes)
		auto& registry = ToolRegistry::instance();
		std::map<std::string, std::string> tool_descriptions;

		// Only get tool descriptions if tools are enabled
		if (!no_tools) {
			tool_descriptions = registry.list_tools_with_descriptions();
		}

		// Now that session/tools are ready, connect to provider (if using provider mode)
		if (use_provider_mode) {
			if (!provider_override.empty()) {
				// Specific provider requested - only try that one
				backend = provider_mgr.connect_provider(provider_override, session, context_size);
			} else {
				// Try providers in priority order (respects auto_provider setting)
				backend = provider_mgr.connect_next_provider(session, context_size);
			}

			if (!backend) {
				fprintf(stderr, "Failed to connect to provider. Use 'shepherd provider add' to configure providers.\n");
				return 1;
			}
		} else {
			// Legacy mode: initialize the backend we created earlier
			backend->initialize(session);
		}

		LOG_INFO("Shepherd initialization complete");

		// ============================================================================
		// Server Mode - HTTP API via FastAPI
		// ============================================================================
		if (g_server_mode) {
			// Backend already initialized above via connect_provider or legacy init

			// MPI rank check: Only rank 0 should run the server
			// (backend->initialize() may have re-exec'd with mpirun for multi-GPU models)
			LOG_DEBUG("MPI rank check: mpi_rank=" + std::to_string(mpi_rank) + ", is_mpi_leader=" + std::string(is_mpi_leader ? "true" : "false"));
			if (!is_mpi_leader) {
				LOG_INFO("MPI rank " + std::to_string(mpi_rank) + " initialization complete, waiting for work from rank 0...");
				// Keep process alive - backend will be controlled via MPI by rank 0
				while (true) {
					std::this_thread::sleep_for(std::chrono::seconds(3600));
				}
			}

			// Only rank 0 continues to run server
			return run_server(backend, server_host, server_port);
		}

		// Populate tools from registry (all ranks need this for Session creation)
		// This converts from ToolRegistry format to Session::Tool format
		// Skip if tools were already added during initialization above
		if (session.tools.empty()) {
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
		} // end if (session.tools.empty())

		LOG_DEBUG("Session created with " + std::to_string(session.tools.size()) + " tools");

		// Backend already initialized above via connect_provider or legacy init
		session.backend = backend.get();

		// Calculate desired completion tokens once (used throughout session lifetime)
		// Must be calculated AFTER backend initialization when context_size is finalized
		session.desired_completion_tokens = calculate_desired_completion_tokens(
			backend->context_size,
			backend->max_output_tokens
		);

		// Enable auto-eviction ONLY for API backends in CLI mode
		// In server mode, never auto-evict - return 400 error and let client handle cleanup
		// Local backends (llamacpp) handle eviction through reactive callbacks
		// Must be set AFTER initialization when backend->context_size is finalized
		session.auto_evict = (!g_server_mode && backend->context_size > 0 && !backend->is_local);
		if (session.auto_evict) {
			LOG_INFO("Auto-eviction enabled (context_size=" + std::to_string(backend->context_size) +
			         ", desired_completion=" + std::to_string(session.desired_completion_tokens) + ")");
		}

		// ============================================================================
		// MPI Multi-GPU: Non-leader ranks wait for shutdown signal
		// ============================================================================
		LOG_DEBUG("MPI rank check: mpi_rank=" + std::to_string(mpi_rank) + ", is_mpi_leader=" + std::string(is_mpi_leader ? "true" : "false"));
		if (!is_mpi_leader) {
			LOG_INFO("MPI rank " + std::to_string(mpi_rank) + " initialization complete, waiting for shutdown signal...");
			// Workers wait at barrier - rank 0 will hit this barrier when done
			// The executor's internal threads handle MPI work distribution
#ifdef ENABLE_TENSORRT
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			LOG_INFO("MPI rank " + std::to_string(mpi_rank) + " received shutdown signal, exiting");
			return 0;
		}

		// From here on, only rank 0 continues...
		LOG_DEBUG("Rank 0 continuing to user input loop...");


	// ============================================================================
	// Create and Run Frontend
	// ============================================================================
	std::unique_ptr<Frontend> frontend;
	if (frontend_mode == "cli") {
		frontend = std::make_unique<CLI>();
	} else if (frontend_mode == "api-server") {
		frontend = std::make_unique<APIServer>(server_host, server_port);
	} else if (frontend_mode == "cli-server") {
		frontend = std::make_unique<CLIServer>(server_host, server_port);
	}

	int result = frontend->run(backend, session);

	// Clean shutdown for MPI
	// Signal workers to exit via barrier, then all ranks exit together
	const char* mpi_size_env = getenv("OMPI_COMM_WORLD_SIZE");
	bool is_mpi = mpi_size_env && std::atoi(mpi_size_env) > 1;

	if (is_mpi) {
		LOG_DEBUG("Rank 0 done, signaling workers via barrier");
#ifdef ENABLE_TENSORRT
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}

	// TensorRT-LLM registers MPI_Finalize as atexit handler (mpiUtils.cpp:186)
	return result;

} catch (const std::exception& e) {
	LOG_FATAL("Fatal error: " + std::string(e.what()));

	// For fatal errors, use MPI_Abort to terminate all ranks immediately
	const char* mpi_size_env = getenv("OMPI_COMM_WORLD_SIZE");
	if (mpi_size_env && std::atoi(mpi_size_env) > 1) {
#ifdef ENABLE_TENSORRT
		MPI_Abort(MPI_COMM_WORLD, 1);
#endif
	}

	return 1;
}
}

