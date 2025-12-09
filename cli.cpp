
#include "debug.h"
#include "cli.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "tools/utf8_sanitizer.h"
#include "tools/filesystem_tools.h"
#include "tools/command_tools.h"
#include "tools/json_tools.h"
#include "tools/http_tools.h"
#include "tools/memory_tools.h"
#include "tools/mcp_resource_tools.h"
#include "tools/core_tools.h"
#include "message.h"
#include "config.h"
#include "provider.h"
#include "scheduler.h"
#include "mcp/mcp.h"
#include "backends/api.h"  // For ApiBackend::set_chars_per_token()
#include "backends/factory.h"
#include "rag.h"
#include "input_reader.h"
#include "tui_screen.h"
#include "output_queue.h"
#include "generation_thread.h"

#include <iostream>
#include <sstream>
#include <tuple>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <cstring>
#include <ctime>
#include <atomic>

#include "terminal_io.h"

// External globals from main.cpp
extern int g_debug_level;
extern bool g_show_thinking;
extern std::unique_ptr<Config> config;

// Forward declaration
static int run_cli_impl(CLI& cli, std::unique_ptr<Backend>& backend, Session& session);

// Helper: Extract tool call from Response (handles both structured and text-based)
static std::optional<ToolParser::ToolCall> extract_tool_call(const Response& resp, Backend* backend) {
	// If backend already parsed tool calls, use them
	if (!resp.tool_calls.empty()) {
		return resp.tool_calls[0];
	}

	// Otherwise parse from text content
	return ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers());
}

// CLI Implementation
CLI::CLI() : Frontend(), eof_received(false), generation_cancelled(false) {
}

CLI::~CLI() {
}

int CLI::run(Session& session) {
	// Populate session.tools from our tools instance
	tools.populate_session_tools(session);

	// Run the main CLI loop using this instance
	return run_cli_impl(*this, backend, session);
}

void CLI::show_tool_call(const std::string& name, const std::string& params) {
	std::string msg = "* " + name + "(" + params + ")\n";
	tio.write(msg.c_str(), msg.length(), Color::YELLOW);
}

void CLI::show_tool_result(const std::string& result) {
	// Truncate for display
	std::string truncated = result;
	size_t first_newline = truncated.find('\n');
	if (first_newline != std::string::npos) {
		size_t second_newline = truncated.find('\n', first_newline + 1);
		if (second_newline != std::string::npos) {
			truncated = truncated.substr(0, second_newline) + "\n...";
		}
	} else if (truncated.length() > 100) {
		truncated = truncated.substr(0, 100) + "...";
	}

	// Split by lines - first line gets >, rest are indented
	std::istringstream stream(truncated);
	std::string line;
	bool first_line = true;

	while (std::getline(stream, line)) {
		std::string msg;
		if (first_line) {
			msg = "> " + line + "\n";
			first_line = false;
		} else {
			msg = "  " + line + "\n";
		}
		tio.write(msg.c_str(), msg.length(), Color::CYAN);
	}
}

void CLI::show_error(const std::string& error) {
	std::string msg = "Error: " + error + "\n";
	tio.write(msg.c_str(), msg.length(), Color::RED);
}

void CLI::show_cancelled() {
	if (tio.interactive_mode) {
		std::string msg = "\n[Cancelled]\n";
		tio.write(msg.c_str(), msg.length(), Color::RED);
	}
}

void CLI::send_message(const std::string& message) {
	tio.write(message.c_str(), message.length());
}

std::string CLI::receive_message(const std::string& prompt) {
	return tio.read(prompt.c_str());
}

void CLI::init(Session& session, bool no_mcp, bool no_tools, const std::string& provider_name) {
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
		LOG_INFO("Tools disabled via --notools flag");
		return;
	}

	LOG_INFO("Initializing tools system...");

	// Register all native tools
	register_filesystem_tools(tools);
	register_command_tools(tools);
	register_json_tools(tools);
	register_http_tools(tools);
	register_memory_tools(tools);
	register_mcp_resource_tools(tools);
	register_core_tools(tools);

	// Initialize MCP servers (registers additional tools)
	if (!no_mcp) {
		auto& mcp = MCP::instance();
		mcp.initialize(tools);
		LOG_INFO("MCP initialized with " + std::to_string(mcp.get_tool_count()) + " tools");
	} else {
		LOG_INFO("MCP system disabled");
	}

	// Build the combined tool list
	tools.build_all_tools();

	LOG_INFO("Tools initialized: " + std::to_string(tools.all_tools.size()) + " total");

	// Populate session.tools from Tools instance
	if (!no_tools) {
		tools.populate_session_tools(session);
		LOG_DEBUG("Session initialized with " + std::to_string(session.tools.size()) + " tools");
	}
}

// Handle all slash commands
// Returns true if command was handled, false otherwise
bool CLI::handle_slash_commands(const std::string& input, Session& session) {
	// Tokenize the input
	std::istringstream iss(input);
	std::string cmd;
	iss >> cmd;

	// Parse remaining arguments - handle quoted strings
	std::vector<std::string> args;
	std::string rest;
	std::getline(iss, rest);

	std::string token;
	bool in_quotes = false;
	std::string quoted_arg;

	for (size_t i = 0; i < rest.size(); ++i) {
		char c = rest[i];
		if (c == '"') {
			if (in_quotes) {
				args.push_back(quoted_arg);
				quoted_arg.clear();
				in_quotes = false;
			} else {
				in_quotes = true;
			}
		} else if (std::isspace(c) && !in_quotes) {
			if (!token.empty()) {
				args.push_back(token);
				token.clear();
			}
		} else {
			if (in_quotes) {
				quoted_arg += c;
			} else {
				token += c;
			}
		}
	}
	if (!token.empty()) args.push_back(token);
	if (!quoted_arg.empty()) args.push_back(quoted_arg);

	// /provider commands
	if (cmd == "/provider") {
		// Call common implementation with backend, session, and providers from frontend
		int result = handle_provider_args(args, &backend, &session, &providers, &current_provider);
		return true;  // Command was handled (even if it returned error)
	}

	// /model commands
	if (cmd == "/model") {
		handle_model_args(args, &backend, &providers, &current_provider);
		return true;
	}

	// /config command
	if (cmd == "/config") {
		handle_config_args(args);
		return true;
	}

	// /sched commands
	if (cmd == "/sched") {
		handle_sched_args(args);
		return true;
	}

	// /tools command
	if (cmd == "/tools") {
		tools.handle_tools_args(args);
		return true;
	}

	// Command not recognized
	return false;
}

// Helper to format status line
static void update_status_line(Backend* backend, Session& session, const std::string& provider_name) {
	if (!tio.tui_mode) return;

	// Left side: current working directory
	char cwd[1024];
	std::string left;
	if (getcwd(cwd, sizeof(cwd))) {
		// Shorten home directory to ~
		std::string cwd_str = cwd;
		const char* home = getenv("HOME");
		if (home && cwd_str.find(home) == 0) {
			cwd_str = "~" + cwd_str.substr(strlen(home));
		}
		left = cwd_str;
	}

	// Right side: provider | model | tokens/limit
	std::string right;
	if (!provider_name.empty()) {
		right = provider_name;
	}
	if (backend && !backend->model_name.empty()) {
		if (!right.empty()) right += " | ";
		right += backend->model_name;
	}
	// Add token count
	right += " | " + std::to_string(session.total_tokens) + "/" + std::to_string(backend ? backend->context_size : 0);

	tio.set_status(left, right);
}

// Main CLI loop implementation - handles all user interaction
static int run_cli_impl(CLI& cli, std::unique_ptr<Backend>& backend, Session& session) {
	LOG_DEBUG("Starting CLI mode (interactive: " + std::string(tio.interactive_mode ? "yes" : "no") + ")");

	// Configure TerminalIO output filtering for this session
	// Get tool call markers from backend (model-specific from chat template)
	tio.markers.tool_call_start = backend->get_tool_call_markers();
	tio.markers.tool_call_end = backend->get_tool_call_end_markers();

	// Add fallback markers if backend didn't provide any
	// These cover common tool call formats used by various models
	if (tio.markers.tool_call_start.empty()) {
		LOG_DEBUG("Adding fallback tool call markers for terminal filtering");
		tio.markers.tool_call_start = {
			"<tool_call>", "<function_call>", "<tools>",
			"<execute_command>", "<.execute_command>",
			"<read>", "<.read>", "<write>", "<.write>",
			"<bash>", "<.bash>", "<edit>", "<.edit>",
			"<glob>", "<.glob>", "<grep>", "<.grep>"
		};
		tio.markers.tool_call_end = {
			"</tool_call>", "</function_call>", "</tools>",
			"</execute_command>", "</.execute_command>",
			"</read>", "</.read>", "</write>", "</.write>",
			"</bash>", "</.bash>", "</edit>", "</.edit>",
			"</glob>", "</.glob>", "</grep>", "</.grep>"
		};
	} else {
		LOG_DEBUG("Using backend-provided tool call markers: " + std::to_string(tio.markers.tool_call_start.size()));
	}

	// Get thinking markers from backend (model-specific)
	tio.markers.thinking_start = backend->get_thinking_start_markers();
	tio.markers.thinking_end = backend->get_thinking_end_markers();
	tio.show_thinking = g_show_thinking;  // From --thinking flag

	// Queue warmup message if needed
	if (config->warmup) {
		LOG_DEBUG("Queueing warmup message...");
		tio.add_input(config->warmup_message);
	}

	// Initialize and start scheduler (unless disabled)
	Scheduler scheduler;
	if (!g_disable_scheduler) {
		scheduler.load();
		scheduler.start();
		LOG_DEBUG("Scheduler initialized with " + std::to_string(scheduler.list().size()) + " schedules");
	} else {
		LOG_INFO("Scheduler disabled via --nosched flag");
	}

	// Initialize input reader thread (only in non-TUI mode)
	// In TUI mode, FTXUI handles input via callback set in TerminalIO::init()
	InputReader input_reader;
	std::atomic<bool> eof_signaled{false};

	if (!tio.tui_mode) {
		// Callback to queue input and handle EOF
		auto input_callback = [&eof_signaled](const std::string& input) {
			if (input.empty()) {
				// EOF signal
				eof_signaled = true;
				tio.add_input("", false);  // Add empty to wake up main loop
				tio.notify_input();
			} else {
				// User typed this - replxx already displayed it, no echo needed
				tio.add_input(input, false);
			}
		};

		if (!input_reader.init(tio.interactive_mode, tio.colors_enabled, input_callback)) {
			LOG_ERROR("Failed to initialize input reader");
			return 1;
		}
		input_reader.start();
	}

	// Initial status line update
	update_status_line(backend.get(), session, cli.current_provider);

	// Initialize generation thread for async generation
	GenerationThread gen_thread;
	gen_thread.init(&session);
	gen_thread.start();
	g_generation_thread = &gen_thread;

	std::string user_input;
	std::string pending_user_input;  // Input being processed
	bool awaiting_generation = false;  // True when generation thread is working

	// State for tool loop
	Response resp;
	int tool_loop_iteration = 0;
	const int max_consecutive_identical_calls = 10;
	std::vector<std::string> recent_tool_calls;
	bool in_tool_loop = false;
	bool local_with_tools = false;

	// Main interaction loop - event driven
	while (true) {
		// Poll scheduler for pending alarms
		if (!g_disable_scheduler) {
			scheduler.poll();
		}

		// Drain output queue - display any tokens that were pushed by backends
		while (auto token = g_output_queue.pop()) {
			tio.write(token->c_str(), token->length());
		}

		// In TUI mode, run event loop iteration
		if (tio.tui_mode && g_tui_screen) {
			if (!g_tui_screen->run_once()) {
				LOG_INFO("User quit TUI - exiting");
				break;
			}
		}

		// Check if generation completed
		if (awaiting_generation && gen_thread.is_complete()) {
			awaiting_generation = false;
			tio.is_generating = false;
			if (g_tui_screen) {
				g_tui_screen->clear_queued_input_display();
			}
			resp = gen_thread.last_response;
			gen_thread.reset();

			// Drain any remaining output
			while (auto token = g_output_queue.pop()) {
				tio.write(token->c_str(), token->length());
			}

			// Check for error
			if (!resp.success) {
				g_output_queue.push("\n\033[31mError: " + resp.error + "\033[0m\n");
				if (!tio.tui_mode) {
					input_reader.resume_prompting();
				}
				in_tool_loop = false;
				continue;
			}

			// Check if generation was cancelled
			if (resp.finish_reason == "cancelled") {
				cli.show_cancelled();
				if (!tio.tui_mode) {
					input_reader.resume_prompting();
				}
				in_tool_loop = false;
				continue;
			}

			LOG_DEBUG("Got response, length: " + std::to_string(resp.content.length()));

			// For non-streamed responses, output content
			bool is_tool_call = !resp.tool_calls.empty() ||
				ToolParser::parse_tool_call(resp.content, backend->get_tool_call_markers()).has_value();

			if (!resp.was_streamed && (!backend->is_local || local_with_tools) && !is_tool_call) {
				tio.begin_response();
				tio.write(resp.content.c_str(), resp.content.length());
				tio.end_response();
			}

			// Enter tool loop processing
			in_tool_loop = true;
		}

		// Tool loop processing (runs in main thread)
		if (in_tool_loop && !awaiting_generation) {
			tool_loop_iteration++;
			LOG_DEBUG("Tool loop iteration: " + std::to_string(tool_loop_iteration));

			// Pending input just queues - don't cancel generation
			// User can press Escape to cancel if needed

			// Check for tool calls
			std::optional<ToolParser::ToolCall> tool_call_opt;
			if (!tio.buffered_tool_call.empty()) {
				tool_call_opt = ToolParser::parse_tool_call(
					tio.buffered_tool_call,
					backend->get_tool_call_markers()
				);
			} else {
				tool_call_opt = extract_tool_call(resp, backend.get());
			}

			if (tool_call_opt.has_value()) {
				auto tool_call = tool_call_opt.value();
				std::string tool_name = tool_call.name;
				std::string tool_call_id = tool_call.tool_call_id;

				LOG_DEBUG("Tool call detected: " + tool_name);

				// Build call signature for loop detection
				std::string call_signature = tool_name + "(";
				bool first_sig_param = true;
				for (const auto& param : tool_call.parameters) {
					if (!first_sig_param) call_signature += ", ";
					first_sig_param = false;
					call_signature += param.first + "=";
					try {
						if (param.second.type() == typeid(std::string)) {
							call_signature += "\"" + std::any_cast<std::string>(param.second) + "\"";
						} else if (param.second.type() == typeid(int)) {
							call_signature += std::to_string(std::any_cast<int>(param.second));
						} else if (param.second.type() == typeid(double)) {
							call_signature += std::to_string(std::any_cast<double>(param.second));
						} else if (param.second.type() == typeid(bool)) {
							call_signature += std::any_cast<bool>(param.second) ? "true" : "false";
						} else {
							call_signature += "?";
						}
					} catch (...) {
						call_signature += "?";
					}
				}
				call_signature += ")";

				// Check for infinite loop
				int consecutive_count = 1;
				for (int i = (int)recent_tool_calls.size() - 1;
					 i >= 0 && i >= (int)recent_tool_calls.size() - 10; i--) {
					if (recent_tool_calls[i] == call_signature) {
						consecutive_count++;
						if (consecutive_count >= max_consecutive_identical_calls) {
							cli.show_error("Detected infinite loop: " + call_signature +
										 " called " + std::to_string(consecutive_count) +
										 " times consecutively. Stopping.");
							in_tool_loop = false;
							tool_loop_iteration = 0;
							recent_tool_calls.clear();
							continue;
						}
					} else {
						break;
					}
				}
				recent_tool_calls.push_back(call_signature);

				// Build params string for display
				std::string params_str;
				bool first_param = true;
				for (const auto& param : tool_call.parameters) {
					if (!first_param) params_str += ", ";
					first_param = false;
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
					if (value_str.length() > 50) {
						value_str = value_str.substr(0, 47) + "...";
					}
					params_str += param.first + "=" + value_str;
				}

				cli.show_tool_call(tool_name, params_str);

				// Execute tool (in main thread)
				LOG_DEBUG("Executing tool: " + tool_name);
				ToolResult tool_result = cli.tools.execute(tool_name, tool_call.parameters);

				std::string result_content;
				if (tool_result.success) {
					result_content = tool_result.content;
				} else {
					if (!tool_result.error.empty()) {
						result_content = "Error: " + tool_result.error;
					} else {
						result_content = "Error: Tool execution failed";
					}
				}

				result_content = utf8_sanitizer::sanitize_utf8(result_content);
				cli.show_tool_result(result_content);

				// Truncate tool result if needed
				int reserved = session.system_message_tokens;
				if (session.last_user_message_index >= 0) {
					reserved += session.last_user_message_tokens;
				}
				if (session.last_assistant_message_index >= 0) {
					reserved += session.last_assistant_message_tokens;
				}

				double scale = calculate_truncation_scale(backend->context_size);
				int max_tool_result_tokens = (backend->context_size - reserved) * scale;

				if (config->truncate_limit > 0 && config->truncate_limit < max_tool_result_tokens) {
					max_tool_result_tokens = config->truncate_limit;
				}

				int tool_result_tokens = backend->count_message_tokens(Message::TOOL, result_content, tool_name, tool_call_id);

				if (tool_result_tokens >= max_tool_result_tokens) {
					size_t original_line_count = std::count(result_content.begin(), result_content.end(), '\n');
					std::string truncation_notice = "\n\n[TRUNCATED: Output too large for context window]";
					truncation_notice += "\nOriginal length: " + std::to_string(original_line_count) + " lines";
					truncation_notice += "\nIf you need more: use Read(offset=X, limit=Y), Grep(pattern=...), or Glob with specific patterns";

					while (tool_result_tokens >= max_tool_result_tokens && result_content.length() > 100) {
						double ratio = (static_cast<double>(max_tool_result_tokens) / static_cast<double>(tool_result_tokens)) * 0.95;
						int new_len = static_cast<int>(result_content.length() * ratio) - truncation_notice.length();
						if (new_len < 0) new_len = 0;
						result_content = result_content.substr(0, new_len) + truncation_notice;
						tool_result_tokens = backend->count_message_tokens(Message::TOOL, result_content, tool_name, tool_call_id);
					}
				}

				// Submit tool result for generation (async)
				tio.reset();
				GenerationRequest req;
				req.type = Message::TOOL;
				req.content = result_content;
				req.tool_name = tool_name;
				req.tool_id = tool_call_id;
				req.prompt_tokens = tool_result_tokens;
				req.max_tokens = 0;
				gen_thread.submit(req);
				awaiting_generation = true;
				tio.is_generating = true;

				// Clear buffered tool call
				tio.buffered_tool_call.clear();
				continue;
			} else {
				// No tool call - generation complete
				LOG_DEBUG("tokens: " + std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));
				update_status_line(backend.get(), session, cli.current_provider);

				if (!backend->is_local) {
					g_output_queue.push("\n");
				}

				// Resume input prompting now that generation is done (non-TUI mode)
				if (!tio.tui_mode) {
					input_reader.resume_prompting();
				}

				in_tool_loop = false;
				tool_loop_iteration = 0;
				recent_tool_calls.clear();
				continue;
			}
		}

		// If generation is in progress, keep loop responsive but don't block on input
		if (awaiting_generation) {
			// Check for EOF during generation (piped input finished)
			if (eof_signaled && !tio.tui_mode) {
				// Wait for generation to complete before exiting
				while (!gen_thread.is_complete()) {
					while (auto token = g_output_queue.pop()) {
						tio.write(token->c_str(), token->length());
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
				// Process final result before exiting
				continue;
			}

			// In TUI mode, input is handled by the TUI event loop
			// No special handling needed here - input stays in queue until generation completes

			// Don't sleep - just loop back to drain output and run TUI
			// The TUI's wgetch timeout provides the pacing
			continue;
		}

		// Check for EOF before waiting (important for piped input)
		if (eof_signaled && !tio.has_pending_input() && !awaiting_generation && !in_tool_loop) {
			LOG_DEBUG("EOF signaled, exiting main loop");
			break;
		}

		// Wait for input (with timeout for scheduler polling)
		int wait_timeout = tio.tui_mode ? 10 : 1000;
		if (!tio.wait_for_input(wait_timeout)) {
			continue;
		}

		// Get next input
		bool needs_echo;
		std::tie(user_input, needs_echo) = tio.read_with_echo_flag(">");

		if (user_input.empty()) {
			if (eof_signaled || (tio.tui_mode && g_tui_screen && g_tui_screen->has_quit())) {
				if (!tio.interactive_mode) {
					LOG_DEBUG("End of piped input");
				} else {
					LOG_INFO("User pressed Ctrl+D - exiting");
				}
				break;
			}
			continue;
		}

		// Echo input to output window (only if not already displayed by replxx)
		if (needs_echo) {
			tio.echo_user_input(user_input);
		}

		if (user_input == "exit" || user_input == "quit") {
			LOG_DEBUG("User requested exit");
			break;
		}

		if (!user_input.empty() && user_input[0] == '/') {
			if (cli.handle_slash_commands(user_input, session)) {
				continue;
			}
		}

		LOG_DEBUG("User input: " + user_input);

		// Reset state
		cli.generation_cancelled = false;
		tool_loop_iteration = 0;
		recent_tool_calls.clear();
		local_with_tools = backend->is_local && !cli.tools.all_tools.empty();

		// Sanitize and truncate user input
		user_input = utf8_sanitizer::strip_control_characters(user_input);

		double scale = calculate_truncation_scale(backend->context_size);
		int available = backend->context_size - session.system_message_tokens;
		int max_user_input_tokens = available * scale;
		int input_tokens = backend->count_message_tokens(Message::USER, user_input, "", "");

		if (input_tokens > max_user_input_tokens) {
			LOG_DEBUG("User input too large, truncating");
			std::string truncation_notice = "\n\n[INPUT TRUNCATED: Too large for context window]";
			while (input_tokens >= max_user_input_tokens && user_input.length() > 100) {
				size_t new_len = user_input.length() * 0.9;
				user_input = user_input.substr(0, new_len);
				input_tokens = backend->count_message_tokens(Message::USER, user_input + truncation_notice, "", "");
			}
			user_input += truncation_notice;
		}

		// Submit user message for generation (async)
		LOG_DEBUG("Submitting user message to generation thread");

		// Mark any queued input as now processing (gray -> cyan)
		if (tio.tui_mode && g_tui_screen) {
			g_tui_screen->mark_input_processing();
		}

		// Note: InputReader auto-pauses after submitting input in non-TUI mode
		// Main loop resumes prompting when generation completes

		GenerationRequest req;
		req.type = Message::USER;
		req.content = user_input;
		req.tool_name = "";
		req.tool_id = "";
		req.prompt_tokens = 0;
		req.max_tokens = 0;
		gen_thread.submit(req);
		awaiting_generation = true;
		tio.is_generating = true;
	}

	// Cleanup - threads are stopped in destructors
	g_generation_thread = nullptr;

	LOG_DEBUG("CLI loop ended");
	return 0;
}

