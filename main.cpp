
#include "shepherd.h"
#include "tools/tools.h"
#include "tools/api_tools.h"
#include "tools/scheduler_tools.h"
#ifdef ENABLE_LLAMACPP
#include "backends/llamacpp.h"  // Include before mcp.h to define json as ordered_json
#endif
#include "mcp/mcp.h"
#include "mcp/mcp_config.h"
#include "provider.h"
#include "frontend.h"
#include "cli.h"
#include "tui.h"
#include "cli_server.h"
#include "server.h"
#include "backend.h"
#include "backends/factory.h"
#include "backends/models.h"
#include "version.h"
#include "scheduler.h"
#include "auth.h"
#include "azure_msi.h"

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
#include <chrono>
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

#ifdef _DEBUG
// Global debug level (0=off, 1-9=increasing verbosity)
// Used by dout() macro for fine-grained debug control
int g_debug_level = 0;

// Null stream for discarding debug output when level not met
static std::ofstream null_stream("/dev/null");

// Get timestamp string for debug output
static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "[%02d:%02d:%02d.%03d] ",
                  tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
    return std::string(buf);
}

// Debug output stream with timestamp
std::ostream& dout(int level) {
    if (g_debug_level >= level) {
        std::cerr << get_timestamp();
        return std::cerr;
    }
    return null_stream;
}
#endif

// g_show_thinking removed - use config->thinking instead

// Global config instance
std::unique_ptr<Config> config;

// Global server mode flag
bool g_server_mode = false;

// Global generation cancellation flag (for llamacpp backend) - atomic for thread safety
std::atomic<bool> g_generation_cancelled{false};

// Scheduler disable flag (--nosched)
bool g_disable_scheduler = false;


static void print_usage(int, char** argv) {
	printf("\n=== Shepherd - Advanced LLM Management System ===\n");
	printf("\nUsage:\n");
	printf("	%s [OPTIONS]\n", argv[0]);
	printf("	%s edit-system				Edit system prompt in $EDITOR\n", argv[0]);
	printf("	%s tools [list|enable|disable] [args...]	Manage tools\n", argv[0]);
	printf("	%s provider <add|list|show|remove|use> [args...]\n", argv[0]);
	printf("	%s config <show|set> [args...]\n", argv[0]);
	printf("	%s mcp <add|remove|list> [args...]\n", argv[0]);
	printf("	%s smcp <add|remove|list> [args...]\n", argv[0]);
	printf("	%s sched <list|add|remove|enable|disable|show|next> [args...]\n", argv[0]);
	printf("	%s ctl <status|shutdown> [--socket PATH]\n", argv[0]);
	printf("	%s apikey <create|list|remove> [options]\n", argv[0]);
	printf("\nOptions:\n");
	printf("	-c, --config FILE  Specify config file (default: ~/.shepherd/config.json)\n");
	printf("	                   Use '--config msi --kv <vault>' to load from Azure Key Vault\n");
	printf("	--kv VAULT         Azure Key Vault name (requires --config msi)\n");
#ifdef _DEBUG
	printf("	-d, --debug[=N]    Enable debug mode with optional level (1-9, default: 1)\n");
#endif
	printf("	-m, --model		   Model name or file (overrides config)\n");
	printf("	-p, --provider	   Provider name to use (from provider list)\n");
	printf("	--model_path	   Model directory path (overrides config, e.g., ~/models)\n");
	printf("	--backend		   Backend (llamacpp, tensorrt, openai, anthropic, gemini, ollama, cli)\n");
	printf("	--api-key		   API key for cloud backends\n");
	printf("	--api-base		   API base URL (for OpenAI-compatible APIs)\n");
	printf("	--context-size	   Set context window size (0 = use model's full context, default: from config)\n");
	printf("	--gpu-layers N	   Number of model layers to offload to GPU (-1=auto/all, 0=CPU only, >0=specific count)\n");
	printf("	--tp N			   Tensor parallelism size (llamacpp only, default: 1)\n");
	printf("	--pp N			   Pipeline parallelism size (llamacpp only, default: 1)\n");
	printf("	--parallel N       Number of parallel sequences/slots (llamacpp only, placeholder)\n");
	printf("	--n_batch N		   Logical batch size for prompt processing (llamacpp only, default: auto)\n");
	printf("	--ubatch N		   Physical micro-batch size (llamacpp only, default: 512)\n");
	printf("	--cache-type TYPE  KV cache data type: f16, f32, q8_0, q4_0 (llamacpp only, default: f16)\n");
	printf("	--flash-attn       Force Flash Attention on (faster TTFT, slightly slower decode)\n");
	printf("	--model-draft PATH Draft model for speculative decoding (llamacpp only)\n");
	printf("	--draft-max N      Max tokens to draft per iteration (default: 16)\n");
	printf("	--temperature F    Sampling temperature (default: 0.7)\n");
	printf("	--top-p F          Top-p (nucleus) sampling (default: 0.95)\n");
	printf("	--top-k N          Top-k sampling (default: 40)\n");
	printf("	--freq F           Frequency penalty (default: 0.1)\n");
	printf("	--models-file FILE Path to models database JSON file (default: ~/.config/shepherd/models.json)\n");
	printf("	--max-tokens	   Set max generation tokens (default: auto)\n");
	printf("	--memory-db		   Path to RAG memory database (default: ~/.local/share/shepherd/memory.db)\n");
	printf("	--nomcp			   Disable MCP system (no MCP servers loaded)\n");
	printf("	--nosched		   Disable scheduler (no scheduled tasks run)\n");
	printf("	--nostream		   Disable streaming (wait for complete response)\n");
	printf("	--raw			   Raw output mode (no channel parsing, like vLLM)\n");
	printf("	--notools		   Disable all tools (no tool registration or use)\n");
	printf("	--system-prompt	   Override system prompt (useful with --notools)\n");
	printf("	-e, --prompt TEXT  Initial user prompt (non-interactive single query)\n");
	printf("	--template		   Custom chat template file (Jinja format, llamacpp only)\n");
	printf("	--apiserver		   Start HTTP API server mode (OpenAI-compatible)\n");
	printf("	--server		   Alias for --apiserver\n");
	printf("	--cliserver		   Start CLI server mode (local tool execution)\n");
	printf("	--port PORT		   Server port (default: 8000, requires --apiserver or --cliserver)\n");
	printf("	--host HOST		   Server host to bind to (default: 0.0.0.0, requires --apiserver or --cliserver)\n");
	printf("	--server-tools	   Expose /v1/tools endpoints (requires --apiserver)\n");
	printf("	--auth-mode MODE   Authentication mode: none (default), json\n");
	printf("	--truncate LIMIT   Truncate tool results to LIMIT tokens (0 = auto 85%% of available space)\n");
	printf("	--warmup		   Send warmup message before first user prompt (initializes model)\n");
	printf("	--colors		   Force enable colored output (overrides environment)\n");
	printf("	--no-colors		   Force disable colored output (overrides environment)\n");
	printf("	--tui			   Force enable TUI mode (boxed input, status line)\n");
	printf("	--no-tui		   Force disable TUI mode (classic scrolling terminal)\n");
	printf("	--stats			   Show performance stats (prefill/decode speed, KV cache)\n");
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


static int handle_config_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "config")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation with stdout output
	auto out = [](const std::string& msg) { std::cout << msg; };
	return handle_config_args(args, out);
}

static int handle_provider_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "provider")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Initialize config if not already done (for CLI subcommands)
	if (!config) {
		config = std::make_unique<Config>();
		config->load();
	}

	// Load providers for CLI commands
	std::vector<Provider> providers = Provider::load_providers();

	// Call common implementation with stdout output
	auto out = [](const std::string& msg) { std::cout << msg; };
	return handle_provider_args(args, out, nullptr, nullptr, &providers, nullptr, nullptr);
}

static int handle_mcp_subcommand(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "mcp")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation with stdout output
	auto out = [](const std::string& msg) { std::cout << msg; };
	return handle_mcp_args(args, out);
}

static int handle_smcp_subcommand(int argc, char** argv) {
	// SMCP management - works directly with config file, no server connection
	std::string config_path = Config::get_default_config_path();

	// Load config
	std::ifstream infile(config_path);
	nlohmann::json config_json;
	if (infile.is_open()) {
		try {
			config_json = nlohmann::json::parse(infile);
		} catch (const std::exception& e) {
			std::cerr << "Failed to parse config: " << e.what() << std::endl;
			return 1;
		}
		infile.close();
	}

	// Ensure smcp_servers array exists
	if (!config_json.contains("smcp_servers") || !config_json["smcp_servers"].is_array()) {
		config_json["smcp_servers"] = nlohmann::json::array();
	}

	// No args - show SMCP servers
	if (argc < 3) {
		auto& servers = config_json["smcp_servers"];
		if (servers.empty()) {
			std::cout << "No SMCP servers configured" << std::endl;
		} else {
			std::cout << "SMCP servers:" << std::endl;
			for (const auto& s : servers) {
				std::cout << "  " << s.value("name", "unnamed") << ": "
				          << s.value("command", "") << std::endl;
			}
		}
		return 0;
	}

	// Determine if first arg is an action (action-first) or a name (name-first)
	std::string first_arg = argv[2];
	bool action_first = (first_arg == "list" || first_arg == "add" ||
	                     first_arg == "help" || first_arg == "--help" || first_arg == "-h");

	std::string name;
	std::string subcmd;

	if (action_first) {
		subcmd = first_arg;
		name = (argc >= 4) ? argv[3] : "";
	} else {
		// Name-first: smcp NAME [action]
		name = first_arg;
		subcmd = (argc >= 4) ? argv[3] : "help";  // No action = show help
	}

	// Help
	if (subcmd == "help" || subcmd == "--help" || subcmd == "-h") {
		if (!name.empty()) {
			std::cout << "Usage: shepherd smcp " << name << " <action>" << std::endl;
			std::cout << "\nActions:" << std::endl;
			std::cout << "  show     - Show server details" << std::endl;
			std::cout << "  remove   - Remove server" << std::endl;
		} else {
			std::cout << "Usage: shepherd smcp <name> <action>" << std::endl;
			std::cout << "\nActions (after name):" << std::endl;
			std::cout << "  show     - Show server details" << std::endl;
			std::cout << "  remove   - Remove server" << std::endl;
			std::cout << "\nOther commands:" << std::endl;
			std::cout << "  list     - List all servers" << std::endl;
			std::cout << "  add <name> <command> [--cred KEY=VALUE ...]" << std::endl;
		}
		return 0;
	}

	// List
	if (subcmd == "list") {
		auto& servers = config_json["smcp_servers"];
		if (servers.empty()) {
			std::cout << "No SMCP servers configured" << std::endl;
		} else {
			std::cout << "SMCP servers:" << std::endl;
			for (const auto& s : servers) {
				std::cout << "  " << s.value("name", "unnamed") << ": "
				          << s.value("command", "") << std::endl;
			}
		}
		return 0;
	}

	// Add (action-first: smcp add NAME COMMAND ...)
	if (subcmd == "add") {
		if (argc < 5) {
			std::cout << "Usage: shepherd smcp add <name> <command> [--cred KEY=VALUE ...]" << std::endl;
			return 1;
		}

		nlohmann::json server;
		server["name"] = argv[3];
		server["command"] = argv[4];
		server["credentials"] = nlohmann::json::object();

		// Parse --cred arguments
		for (int i = 5; i < argc; i++) {
			if (std::string(argv[i]) == "--cred" && i + 1 < argc) {
				i++;
				std::string pair = argv[i];
				size_t eq = pair.find('=');
				if (eq != std::string::npos) {
					server["credentials"][pair.substr(0, eq)] = pair.substr(eq + 1);
				}
			}
		}

		config_json["smcp_servers"].push_back(server);

		// Save config
		std::ofstream outfile(config_path);
		outfile << config_json.dump(4) << std::endl;
		std::cout << "Added SMCP server '" << argv[3] << "'" << std::endl;
		return 0;
	}

	// Show (name-first)
	if (subcmd == "show") {
		if (name.empty()) {
			std::cout << "Usage: shepherd smcp <name> show" << std::endl;
			return 1;
		}

		auto& servers = config_json["smcp_servers"];
		for (const auto& s : servers) {
			if (s.value("name", "") == name) {
				std::cout << "=== SMCP Server: " << name << " ===" << std::endl;
				std::cout << "command = " << s.value("command", "") << std::endl;
				if (s.contains("credentials") && !s["credentials"].empty()) {
					std::cout << "credentials:" << std::endl;
					for (auto& [k, v] : s["credentials"].items()) {
						std::cout << "  " << k << " = " << v.get<std::string>() << std::endl;
					}
				}
				return 0;
			}
		}
		std::cerr << "SMCP server '" << name << "' not found" << std::endl;
		return 1;
	}

	// Remove (name-first)
	if (subcmd == "remove") {
		if (name.empty()) {
			std::cout << "Usage: shepherd smcp <name> remove" << std::endl;
			return 1;
		}

		auto& servers = config_json["smcp_servers"];
		bool found = false;

		for (auto it = servers.begin(); it != servers.end(); ++it) {
			if ((*it).value("name", "") == name) {
				servers.erase(it);
				found = true;
				break;
			}
		}

		if (!found) {
			std::cerr << "SMCP server '" << name << "' not found" << std::endl;
			return 1;
		}

		// Save config
		std::ofstream outfile(config_path);
		outfile << config_json.dump(4) << std::endl;
		std::cout << "Removed SMCP server '" << name << "'" << std::endl;
		return 0;
	}

	std::cerr << "Unknown smcp command: " << subcmd << std::endl;
	std::cerr << "Use 'shepherd smcp help' to see available commands" << std::endl;
	return 1;
}

static int handle_sched_command(int argc, char** argv) {
	// Convert argc/argv to vector of args (skip "shepherd" and "sched")
	std::vector<std::string> args;
	for (int i = 2; i < argc; i++) {
		args.push_back(argv[i]);
	}

	// Call common implementation with stdout output
	auto out = [](const std::string& msg) { std::cout << msg; };
	return handle_sched_args(args, out);
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
	bool is_interactive = true;
	if (!getenv("SHEPHERD_INTERACTIVE")) {
		is_interactive = isatty(STDIN_FILENO);
		setenv("SHEPHERD_INTERACTIVE", is_interactive ? "1" : "0", 0);
		dout(1) << "Set SHEPHERD_INTERACTIVE=" << (is_interactive ? "1" : "0")
		        << " (isatty=" << is_interactive << ")" << std::endl;
	} else {
		is_interactive = (std::string(getenv("SHEPHERD_INTERACTIVE")) == "1");
		dout(1) << "SHEPHERD_INTERACTIVE already set to: " + std::string(getenv("SHEPHERD_INTERACTIVE")) << std::endl;
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

	// Handle tools subcommand
	if (argc >= 2 && std::string(argv[1]) == "tools") {
		// Initialize tools system
		Tools tools;
		register_filesystem_tools(tools);
		register_command_tools(tools);
		register_json_tools(tools);
		register_http_tools(tools);
		register_memory_tools(tools);
		register_mcp_resource_tools(tools);
		register_core_tools(tools);
		register_scheduler_tools(tools);

		tools.build_all_tools();

		// Build args vector from argv[2] onwards
		std::vector<std::string> args;
		for (int i = 2; i < argc; i++) {
			args.push_back(argv[i]);
		}

		auto out = [](const std::string& msg) { std::cout << msg; };
		return tools.handle_tools_args(args, out);
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

	// Handle SMCP subcommand (config-only, no server connection)
	if (argc >= 2 && std::string(argv[1]) == "smcp") {
		return handle_smcp_subcommand(argc, argv);
	}

	// Handle sched subcommand
	if (argc >= 2 && std::string(argv[1]) == "sched") {
		return handle_sched_command(argc, argv);
	}

	// Handle ctl subcommand (server control)
	if (argc >= 2 && std::string(argv[1]) == "ctl") {
		std::vector<std::string> args;
		for (int i = 2; i < argc; i++) {
			args.push_back(argv[i]);
		}
		return handle_ctl_args(args);
	}

	// Handle apikey subcommand (API key management)
	if (argc >= 2 && std::string(argv[1]) == "apikey") {
		std::vector<std::string> args;
		for (int i = 2; i < argc; i++) {
			args.push_back(argv[i]);
		}
		auto out = [](const std::string& msg) { std::cout << msg; };
		return handle_apikey_args(args, out);
	}

	// Flags and settings that don't map directly to config
	bool no_mcp = false;
	bool no_stream = false;
	bool no_tools = false;
	bool server_tools = false;
	int color_override = -1;  // -1 = auto, 0 = off, 1 = on
	int tui_override = -1;    // -1 = auto, 0 = off, 1 = on
	std::string config_file_path;
	std::string keyvault_name;  // Azure Key Vault name for --config msi --kv <vault>
	int server_port = 8000;
	std::string server_host = "0.0.0.0";
	std::string auth_mode = "none";
	std::string frontend_mode = "cli";

	// Temporary storage for command-line overrides (applied to config after load)
	struct {
		bool debug = false;
		bool warmup = false;
		bool calibration = false;
		bool thinking = false;
		bool stats = false;
		bool raw_output = false;
		bool system_prompt_set = false;
		int truncate_limit = 0;
		int context_size = -1;      // -1 = not specified
		int gpu_layers = -999;      // -999 = not specified
		int tp = -1;                // -1 = not specified
		int pp = -1;                // -1 = not specified
		int n_batch = -1;           // -1 = not specified
		int ubatch = -1;            // -1 = not specified
		std::string provider;
		std::string model;
		std::string model_path;
		std::string backend;
		std::string api_key;
		std::string api_base;
		std::string template_name;
		std::string memory_db;
		std::string models_file;
		std::string system_prompt;
		std::string initial_prompt;
		bool single_query_mode = false;
		std::string cache_type;
		bool flash_attn = false;
		std::string model_draft;
		int draft_max = -1;
		int n_parallel = -1;        // -1 = not specified
		int max_tokens = -2;        // -2 = not specified, -1 = max, 0 = auto, >0 = explicit
		float temperature = -1.0f;
		float top_p = -1.0f;
		int top_k = -1;
		float freq_penalty = -1.0f;
	} override;

	static struct option long_options[] = {
		{"config", required_argument, 0, 'c'},
#ifdef _DEBUG
		{"debug", optional_argument, 0, 'd'},
#endif
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
		{"nostream", no_argument, 0, 1041},
		{"raw", no_argument, 0, 1043},
		{"notools", no_argument, 0, 1027},
		{"system-prompt", required_argument, 0, 1028},
		{"prompt", required_argument, 0, 'e'},
		{"template", required_argument, 0, 1006},
		{"apiserver", no_argument, 0, 1015},
		{"server", no_argument, 0, 1015},  // Alias for --apiserver
		{"cliserver", no_argument, 0, 1036},
		{"port", required_argument, 0, 1016},
		{"host", required_argument, 0, 1017},
		{"auth-mode", required_argument, 0, 1045},
		{"server-tools", no_argument, 0, 1047},
		{"truncate", required_argument, 0, 1019},
		{"warmup", no_argument, 0, 1026},
		{"tp", required_argument, 0, 1029},
		{"pp", required_argument, 0, 1030},
		{"n_batch", required_argument, 0, 1044},
		{"ubatch", required_argument, 0, 1035},
		{"cache-type", required_argument, 0, 1040},
		{"flash-attn", no_argument, 0, 1048},
		{"model-draft", required_argument, 0, 1049},
		{"draft-max", required_argument, 0, 1050},
		{"temperature", required_argument, 0, 1051},
		{"top-p", required_argument, 0, 1052},
		{"top-k", required_argument, 0, 1053},
		{"freq", required_argument, 0, 1054},
		{"parallel", required_argument, 0, 1055},
		{"max-tokens", required_argument, 0, 1056},
		{"thinking", no_argument, 0, 1031},
		{"stats", no_argument, 0, 1042},
		{"colors", no_argument, 0, 1032},
		{"no-colors", no_argument, 0, 1033},
		{"tui", no_argument, 0, 1038},
		{"no-tui", no_argument, 0, 1039},
		{"calibration", no_argument, 0, 1034},
		{"kv", required_argument, 0, 1046},
		{"version", no_argument, 0, 'v'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	int option_index = 0;
#ifdef _DEBUG
	while ((opt = getopt_long(argc, argv, "c:d::e:m:p:vh", long_options, &option_index)) != -1) {
#else
	while ((opt = getopt_long(argc, argv, "c:e:m:p:vh", long_options, &option_index)) != -1) {
#endif
		switch (opt) {
			case 'c':
				config_file_path = optarg;
				break;
#ifdef _DEBUG
			case 'd':
				override.debug = true;
				// Parse optional debug level (default to 1 if not specified)
				if (optarg) {
					g_debug_level = atoi(optarg);
				} else {
					g_debug_level = 1;
				}
				break;
#endif
			case 'm':
				override.model = optarg;
				break;
			case 'p':
				override.provider = optarg;
				break;
			case 'e': // --prompt
				override.initial_prompt = optarg ? optarg : "";
				override.single_query_mode = true;
				break;
			case 1018: // --model_path
				override.model_path = optarg;
				break;
			case 1002: // --backend
				override.backend = optarg;
				break;
			case 1003: // --api-key
				override.api_key = optarg;
				break;
			case 1004: // --api-base
				override.api_base = optarg;
				break;
			case 1000: // --context-size
				override.context_size = std::atoi(optarg);
				if (override.context_size < 0) {
					printf("Error: context-size cannot be negative (use 0 for model's full context)\n");
					return 1;
				}
				break;
			case 1025: // --gpu-layers
				override.gpu_layers = std::atoi(optarg);
				break;
			case 1005: // --nomcp
				no_mcp = true;
				break;
			case 1037: // --nosched
				g_disable_scheduler = true;
				break;
			case 1041: // --nostream
				no_stream = true;
				break;
			case 1043: // --raw
				override.raw_output = true;
				break;
			case 1027: // --notools
				no_tools = true;
				break;
			case 1028: // --system-prompt
				override.system_prompt = optarg;
				override.system_prompt_set = true;
				break;
			case 1006: // --template
				override.template_name = optarg;
				break;
			case 1015: // --apiserver
				frontend_mode = "api-server";
				g_server_mode = true;
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
			case 1045: // --auth-mode
				auth_mode = optarg;
				if (auth_mode != "none" && auth_mode != "json") {
					printf("Error: auth-mode must be one of: none, json\n");
					return 1;
				}
				break;
			case 1047: // --server-tools
				server_tools = true;
				break;
			case 1019: // --truncate
				override.truncate_limit = std::atoi(optarg);
				break;
			case 1023: // --memory-db
				override.memory_db = optarg;
				break;
			case 1024: // --models-file
				override.models_file = optarg;
				break;
			case 1026: // --warmup
				override.warmup = true;
				break;
			case 1029: // --tp
				override.tp = std::atoi(optarg);
				if (override.tp < 1) {
					printf("Error: tp must be at least 1\n");
					return 1;
				}
				break;
			case 1030: // --pp
				override.pp = std::atoi(optarg);
				if (override.pp < 1) {
					printf("Error: pp must be at least 1\n");
					return 1;
				}
				break;
			case 1044: // --n_batch
				override.n_batch = std::atoi(optarg);
				if (override.n_batch < 1) {
					printf("Error: n_batch must be at least 1\n");
					return 1;
				}
				break;
			case 1035: // --ubatch
				override.ubatch = std::atoi(optarg);
				if (override.ubatch < 1) {
					printf("Error: ubatch must be at least 1\n");
					return 1;
				}
				break;
			case 1040: // --cache-type
				override.cache_type = optarg;
				if (override.cache_type != "f16" && override.cache_type != "f32" &&
				    override.cache_type != "q8_0" && override.cache_type != "q4_0") {
					printf("Error: cache-type must be one of: f16, f32, q8_0, q4_0\n");
					return 1;
				}
				break;
			case 1048: // --flash-attn
				override.flash_attn = true;
				break;
			case 1049: // --model-draft
				override.model_draft = optarg;
				break;
			case 1050: // --draft-max
				override.draft_max = std::atoi(optarg);
				break;
			case 1051: // --temperature
				override.temperature = std::atof(optarg);
				break;
			case 1052: // --top-p
				override.top_p = std::atof(optarg);
				break;
			case 1053: // --top-k
				override.top_k = std::atoi(optarg);
				break;
			case 1054: // --freq
				override.freq_penalty = std::atof(optarg);
				break;
			case 1055: // --parallel
				override.n_parallel = std::atoi(optarg);
				if (override.n_parallel < 1) {
					printf("Error: parallel must be at least 1\n");
					return 1;
				}
				break;
			case 1056: // --max-tokens
				override.max_tokens = std::atoi(optarg);
				// -1 = max possible, 0 = auto, >0 = explicit value
				if (override.max_tokens < -1) {
					printf("Error: max-tokens must be -1 (max), 0 (auto), or a positive number\n");
					return 1;
				}
				break;
			case 1031: // --thinking
				override.thinking = true;
				break;
			case 1042: // --stats
				override.stats = true;
				break;
			case 1032: // --colors
				color_override = 1;
				break;
			case 1033: // --no-colors
				color_override = 0;
				break;
			case 1038: // --tui
				tui_override = 1;
				break;
			case 1039: // --no-tui
				tui_override = 0;
				break;
			case 1034: // --calibration
				override.calibration = true;
				break;
			case 1046: // --kv (Azure Key Vault name for MSI config)
				keyvault_name = optarg;
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

	// Load configuration early (before terminal init to get TUI setting)
	config = std::make_unique<Config>();

	// Check for Azure Managed Identity config source
	if (config_file_path == "msi") {
		// Load config from Azure Key Vault using Managed Identity
		if (keyvault_name.empty()) {
			fprintf(stderr, "Error: --config msi requires --kv <vault-name>\n");
			return 1;
		}

		auto secret = azure::get_keyvault_secret(keyvault_name, "shepherd-config");
		if (!secret) {
			fprintf(stderr, "Error: Failed to fetch config from Key Vault '%s'\n",
			        keyvault_name.c_str());
			return 1;
		}

		try {
			config->load_from_json_string(*secret);
			config->source_mode = Config::SourceMode::KEY_VAULT;
			config->keyvault_name = keyvault_name;
			dout(1) << "Loaded config from Key Vault (read-only mode)" << std::endl;
		} catch (const ConfigError& e) {
			fprintf(stderr, "Error parsing Key Vault config: %s\n", e.what());
			return 1;
		}
	} else {
		// Standard file-based config loading
		if (!config_file_path.empty()) {
			config->set_config_path(config_file_path);
		}

		try {
			config->load();
		} catch (const ConfigError& e) {
			fprintf(stderr, "Configuration error: %s\n", e.what());
			return 1;
		}

		// Store vault name if provided (for --auth-mode msi without --config msi)
		if (!keyvault_name.empty()) {
			config->keyvault_name = keyvault_name;
		}
	}

	// Determine if we need TUI mode
	// Server modes should never use TUI or interactive input - force TUI off
	bool use_tui = false;
	if (g_server_mode || frontend_mode == "cli-server") {
		tui_override = 0;  // Force TUI off for server modes
	} else if (tui_override == -1) {
		// No command-line override - use config setting
		if (config->tui) {
			tui_override = 1;
		}
	}
	use_tui = (tui_override == 1);

	// Set frontend mode based on TUI flag
	if (use_tui) {
		frontend_mode = "tui";
		dout(1) << "TUI mode enabled, frontend_mode=tui" << std::endl;
	}

#ifdef _DEBUG
	// Debug mode initialization
	if (g_debug_level) {
		std::cout << "Debug mode enabled (level " << g_debug_level << ")" << std::endl;

		// Log interactivity detection for debugging
		const char* interactive_env = getenv("SHEPHERD_INTERACTIVE");
		std::string debug_msg = "SHEPHERD_INTERACTIVE=" + std::string(interactive_env ? interactive_env : "not set");
		dout(1) << debug_msg << std::endl;
	}
#endif

	// Apply command-line overrides to config
	if (!override.backend.empty()) {
		try {
			config->set_backend(override.backend);
		} catch (const ConfigError& e) {
			fprintf(stderr, "Error: %s\n", e.what());
			return 1;
		}
	}
	if (!override.api_key.empty()) {
		config->key = override.api_key;
	}
	if (!override.api_base.empty()) {
		config->api_base = override.api_base;
	}
	if (!override.model.empty()) {
		config->model = override.model;
	}
	if (!override.model_path.empty()) {
		config->model_path = override.model_path;
	}
	if (!override.memory_db.empty()) {
		config->memory_database = override.memory_db;
	}
	if (!override.models_file.empty()) {
		config->models_file = override.models_file;
	}
	if (!override.system_prompt.empty()) {
		config->system_message = override.system_prompt;
	}
	if (override.single_query_mode) {
		config->single_query_mode = true;
		config->initial_prompt = override.initial_prompt;
	}
	if (override.context_size >= 0) {
		config->context_size = override.context_size;
	}
	if (override.truncate_limit > 0) {
		config->truncate_limit = override.truncate_limit;
	}
	if (override.max_tokens != -2) {  // -2 in override means not set; -1=max, 0=auto, >0=explicit
		config->max_tokens = override.max_tokens;
	}
	if (override.warmup) {
		config->warmup = true;
	}
	if (override.calibration) {
		config->calibration = true;
	}
	if (override.thinking) {
		config->thinking = true;
	}

	if (override.stats) {
		config->stats = true;
	}
	if (override.raw_output) {
		config->raw_output = true;
	}
	if (no_stream) {
		config->streaming = false;
		dout(1) << "Streaming disabled via --nostream flag" << std::endl;
	}
	if (override.gpu_layers != -999) {
		config->json["gpu_layers"] = override.gpu_layers;
	}
	if (override.tp != -1) {
		config->json["tp"] = override.tp;
	}
	if (override.pp != -1) {
		config->json["pp"] = override.pp;
	}
	if (override.n_batch != -1) {
		config->json["n_batch"] = override.n_batch;
	}
	if (override.ubatch != -1) {
		config->json["ubatch"] = override.ubatch;
	}
	if (!override.cache_type.empty()) {
		config->json["cache_type"] = override.cache_type;
	}
	if (override.flash_attn) {
		config->json["flash_attn"] = true;
	}
	if (!override.model_draft.empty()) {
		config->json["model_draft"] = override.model_draft;
	}
	if (override.draft_max > 0) {
		config->json["draft_max"] = override.draft_max;
	}
	if (override.temperature >= 0.0f) {
		config->json["temperature"] = override.temperature;
		config->temperature_override = override.temperature;
	}
	if (override.top_p >= 0.0f) {
		config->json["top_p"] = override.top_p;
		config->top_p_override = override.top_p;
	}
	if (override.top_k >= 0) {
		config->json["top_k"] = override.top_k;
		config->top_k_override = override.top_k;
	}
	if (override.freq_penalty >= 0.0f) {
		config->json["penalty_freq"] = override.freq_penalty;
		config->frequency_penalty_override = override.freq_penalty;
	}
	if (override.n_parallel != -1) {
		config->json["n_parallel"] = override.n_parallel;
	}

	// Apply server settings from command line (these override config file values)
	if (auth_mode != "none") {
		config->auth_mode = auth_mode;
	}
	if (server_tools) {
		config->server_tools = true;
	}

	// Validate configuration (skip model path check if overridden)
	if (override.model.empty() && override.model_path.empty()) {
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
		// Validate model file exists (only for local backends)
		if (config->backend == "llamacpp" || config->backend == "tensorrt") {
			// Construct the full path to validate (config already has overrides applied)
			std::string model_file_to_check;

			// Check if model is already a full path
			if (!config->model.empty() && (config->model[0] == '/' || config->model[0] == '~')) {
				model_file_to_check = config->model;
			} else {
				model_file_to_check = std::filesystem::path(config->model_path) / config->model;
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

	dout(1) << "Shepherd starting up..." << std::endl;
	dout(1) << "Backend: " + config->backend << std::endl;


	// Initialize models database if specified
	if (!config->models_file.empty()) {
		Models::init(config->models_file);
	}

	// Create and initialize frontend/backend
	try {
		// If command-line overrides were provided, create ephemeral provider
		bool has_cmdline_override = !override.backend.empty() || !override.model.empty() || !override.api_key.empty();
		Provider* cmdline_provider_ptr = nullptr;
		Provider cmdline_provider;
		if (has_cmdline_override) {
			cmdline_provider = Provider::from_config();
			// Build descriptive name from backend type and model
			std::string cmdline_name = "cmdline";
			if (!cmdline_provider.type.empty()) {
				cmdline_name += ":" + cmdline_provider.type;
			}
			if (!cmdline_provider.model.empty()) {
				// Extract just the filename from path, without extension
				std::string model_name = cmdline_provider.model;
				size_t last_slash = model_name.find_last_of("/\\");
				if (last_slash != std::string::npos) {
					model_name = model_name.substr(last_slash + 1);
				}
				size_t last_dot = model_name.rfind('.');
				if (last_dot != std::string::npos) {
					model_name = model_name.substr(0, last_dot);
				}
				cmdline_name += "/" + model_name;
			}
			cmdline_provider.name = cmdline_name;
			cmdline_provider.priority = 0;  // Highest priority (reserved for ephemeral)
			cmdline_provider_ptr = &cmdline_provider;
		}

		// Create and initialize frontend (loads providers, registers tools)
		// Session is owned by frontend
		auto frontend = Frontend::create(frontend_mode, server_host, server_port,
		                                 cmdline_provider_ptr, no_mcp, no_tools);

		// Determine which provider to pass to run()
		// Provider connection now happens inside run() for proper UI sequencing
		Provider* provider_for_run = nullptr;
		if (!override.provider.empty()) {
			// Specific provider requested via --provider
			provider_for_run = frontend->get_provider(override.provider);
			if (!provider_for_run) {
				fprintf(stderr, "Provider '%s' not found. Use 'shepherd provider list' to see available providers.\n", override.provider.c_str());
				return 1;
			}
		} else if (has_cmdline_override) {
			// Command-line overrides → use ephemeral cmdline provider
			provider_for_run = frontend->get_provider(cmdline_provider.name);
		}
		// If provider_for_run is nullptr, run() will use first available provider

		dout(1) << "Shepherd initialization complete" << std::endl;

		// ============================================================================
		// Server Mode - HTTP API via FastAPI
		// ============================================================================
		if (g_server_mode) {
			// MPI rank check: Only rank 0 should run the server
			// (backend->initialize() may have re-exec'd with mpirun for multi-GPU models)
			dout(1) << "MPI rank check: mpi_rank=" + std::to_string(mpi_rank) + ", is_mpi_leader=" + std::string(is_mpi_leader ? "true" : "false") << std::endl;
			if (!is_mpi_leader) {
				dout(1) << "MPI rank " + std::to_string(mpi_rank) + " initialization complete, waiting for work from rank 0..." << std::endl;
				// Keep process alive - backend will be controlled via MPI by rank 0
				while (true) {
					std::this_thread::sleep_for(std::chrono::seconds(3600));
				}
			}

			// Only rank 0 continues to run server
			// Provider connection happens inside run()
			return frontend->run(provider_for_run);
		}

		// ============================================================================
		// MPI Multi-GPU: Non-leader ranks wait for shutdown signal
		// ============================================================================
		dout(1) << "MPI rank check: mpi_rank=" + std::to_string(mpi_rank) + ", is_mpi_leader=" + std::string(is_mpi_leader ? "true" : "false") << std::endl;
		if (!is_mpi_leader) {
			dout(1) << "MPI rank " + std::to_string(mpi_rank) + " initialization complete, waiting for shutdown signal..." << std::endl;
			// Workers wait at barrier - rank 0 will hit this barrier when done
			// The executor's internal threads handle MPI work distribution
#ifdef ENABLE_TENSORRT
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			dout(1) << "MPI rank " + std::to_string(mpi_rank) + " received shutdown signal, exiting" << std::endl;
			return 0;
		}

		// From here on, only rank 0 continues...
		dout(1) << "Rank 0 continuing to user input loop..." << std::endl;


	// ============================================================================
	// Run Frontend - provider connection happens inside run()
	// ============================================================================
	int result = frontend->run(provider_for_run);

	// Clean shutdown for MPI
	// Signal workers to exit via barrier, then all ranks exit together
	const char* mpi_size_env = getenv("OMPI_COMM_WORLD_SIZE");
	bool is_mpi = mpi_size_env && std::atoi(mpi_size_env) > 1;

	if (is_mpi) {
		dout(1) << "Rank 0 done, signaling workers via barrier" << std::endl;
#ifdef ENABLE_TENSORRT
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}

	// TensorRT-LLM registers MPI_Finalize as atexit handler (mpiUtils.cpp:186)
	return result;

} catch (const std::exception& e) {
	std::string error_msg = std::string("Fatal error: ") + e.what();
	fprintf(stderr, "Error: %s\n", error_msg.c_str());

	// For fatal errors, use MPI_Abort to terminate all ranks immediately
	const char* mpi_size_env = getenv("OMPI_COMM_WORLD_SIZE");
	if (mpi_size_env && std::atoi(mpi_size_env) > 1) {
#ifdef ENABLE_TENSORRT
		MPI_Abort(MPI_COMM_WORLD, 1);
#endif
	}

	return 1;
} catch (...) {
	fprintf(stderr, "Fatal error: Unknown exception\n");
	return 1;
}
}

