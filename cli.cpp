
#include "debug.h"
#include "cli.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "tools/utf8_sanitizer.h"
#include "message.h"
#include "config.h"
#include "provider.h"
#include "backends/api.h"  // For ApiBackend::set_chars_per_token()
#include "backends/factory.h"

#include <iostream>
#include <sstream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <cstring>
#include <ctime>

#include "terminal_io.h"

// External globals from main.cpp
extern int g_debug_level;
extern bool g_show_thinking;
extern std::unique_ptr<Config> config;

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
CLI::CLI() : eof_received(false), generation_cancelled(false) {
}

CLI::~CLI() {
}

void CLI::show_tool_call(const std::string& name, const std::string& params) {
	if (tio.interactive_mode) {
		printf("\033[33m* %s(%s)\033[0m\n", name.c_str(), params.c_str());
	} else {
		printf("* %s(%s)\n", name.c_str(), params.c_str());
	}
	fflush(stdout);
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
		if (first_line) {
			// First line gets the > prefix
			if (tio.interactive_mode) {
				printf("\033[36m> %s\033[0m\n", line.c_str());
			} else {
				printf("> %s\n", line.c_str());
			}
			first_line = false;
		} else {
			// Subsequent lines are indented
			if (tio.interactive_mode) {
				printf("\033[36m  %s\033[0m\n", line.c_str());
			} else {
				printf("  %s\n", line.c_str());
			}
		}
	}
	fflush(stdout);
}

void CLI::show_error(const std::string& error) {
	if (tio.interactive_mode) {
		printf("\033[31mError: %s\033[0m\n", error.c_str());
	} else {
		fprintf(stderr, "Error: %s\n", error.c_str());
	}
}

void CLI::show_cancelled() {
	if (tio.interactive_mode) {
		printf("\n\033[31m[Cancelled]\033[0m\n");
	}
}

// Handle provider slash commands
// Returns true if command was handled, false otherwise
bool handle_provider_command(const std::string& input, std::unique_ptr<Backend>& backend, Session& session) {
	static Provider provider_manager;
	// Reload providers to pick up any external changes
	provider_manager.load_providers();

	// Tokenize the input
	std::istringstream iss(input);
	std::string cmd;
	iss >> cmd;

	// Parse remaining arguments
	std::vector<std::string> args;
	std::string arg;
	while (iss >> arg) {
		args.push_back(arg);
	}

	// /provider commands
	if (cmd == "/provider") {
		if (args.empty()) {
			// Show current provider
			std::string current = provider_manager.get_current_provider();
			if (current.empty()) {
				std::cout << "No provider configured\n";
			} else {
				auto prov = provider_manager.get_provider(current);
				if (prov) {
					std::cout << "Current provider: " << prov->name << "\n";
					std::cout << "  Type: " << prov->type << "\n";
					std::cout << "  Model: " << prov->model << "\n";
					if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
						if (!api->base_url.empty()) {
							std::cout << "  Base URL: " << api->base_url << "\n";
						}
					}
				}
			}
			return true;
		}

		std::string subcmd = args[0];

		if (subcmd == "list") {
			auto providers = provider_manager.list_providers();
			if (providers.empty()) {
				std::cout << "No providers configured\n";
			} else {
				std::cout << "Available providers:\n";
				for (const auto& name : providers) {
					if (name == provider_manager.get_current_provider()) {
						std::cout << "  * " << name << " (current)\n";
					} else {
						std::cout << "    " << name << "\n";
					}
				}
			}
			return true;
		}

		if (subcmd == "add") {
			if (args.size() < 2) {
				std::cout << "Usage: /provider add <type> --name <name> [options...]\n";
				std::cout << "Types: llamacpp, tensorrt, openai, anthropic, gemini, grok, ollama\n";
				return true;
			}

			// Command-line mode: /provider add <type> --name <name> ...
			std::string type = args[1];
			std::vector<std::string> cmd_args(args.begin() + 2, args.end());
			auto new_config = provider_manager.parse_provider_args(type, cmd_args);

			if (new_config->name.empty()) {
				std::cout << "Error: Provider name is required (use --name <name>)\n";
			} else {
				provider_manager.save_provider(*new_config);
				std::cout << "Provider '" << new_config->name << "' added successfully\n";
			}
			return true;
		}

		if (subcmd == "edit" && args.size() >= 2) {
			std::string name = args[1];
			auto prov = provider_manager.get_provider(name);
			if (!prov) {
				std::cout << "Provider '" << name << "' not found\n";
			} else {
				if (provider_manager.interactive_edit(*prov)) {
					provider_manager.save_provider(*prov);
					std::cout << "Provider '" << prov->name << "' updated successfully\n";
				} else {
					std::cout << "Edit cancelled\n";
				}
			}
			return true;
		}

		if (subcmd == "set" && args.size() >= 3) {
			std::string name = args[1];
			std::string field = args[2];
			std::string value = (args.size() >= 4) ? args[3] : "";

			auto prov = provider_manager.get_provider(name);
			if (!prov) {
				std::cout << "Provider '" << name << "' not found\n";
			} else {
				// Update specific field
				if (field == "model") {
					prov->model = value;
				} else if (field == "tokens_per_month") {
					prov->rate_limits.tokens_per_month = std::stoi(value);
				} else if (field == "max_cost") {
					prov->rate_limits.max_cost_per_month = std::stof(value);
				} else if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
					if (field == "api_key" || field == "key") {
						api->api_key = value;
					} else if (field == "base_url" || field == "url") {
						api->base_url = value;
					} else {
						std::cout << "Unknown field: " << field << "\n";
						return true;
					}
				} else {
					std::cout << "Unknown field: " << field << "\n";
					return true;
				}

				provider_manager.save_provider(*prov);
				std::cout << "Provider '" << name << "' updated\n";
			}
			return true;
		}

		if (subcmd == "remove" && args.size() >= 2) {
			std::string name = args[1];
			provider_manager.remove_provider(name);
			std::cout << "Provider '" << name << "' removed\n";
			return true;
		}

		if (subcmd == "use" && args.size() >= 2) {
			std::string name = args[1];
			auto prov = provider_manager.get_provider(name);
			if (!prov) {
				std::cout << "Provider '" << name << "' not found\n";
			} else {
				// Create new backend from provider
				auto new_backend = BackendFactory::create_from_provider(prov, backend->context_size);
				if (!new_backend) {
					std::cout << "Failed to create backend for provider '" << name << "'\n";
					return true;
				}

				// Initialize the new backend
				new_backend->initialize(session);

				// Switch session to new backend
				session.switch_backend(new_backend.release());
				provider_manager.set_current_provider(name);

				std::cout << "Switched to provider '" << name << "' (" << prov->type << " / " << prov->model << ")\n";
			}
			return true;
		}

		if (subcmd == "next") {
			auto next = provider_manager.get_next_provider();
			if (!next) {
				std::cout << "No available providers (all rate limited or none configured)\n";
			} else {
				auto prov = provider_manager.get_provider(*next);
				if (!prov) {
					std::cout << "Failed to load provider '" << *next << "'\n";
					return true;
				}

				// Create new backend from provider
				auto new_backend = BackendFactory::create_from_provider(prov, backend->context_size);
				if (!new_backend) {
					std::cout << "Failed to create backend for provider '" << *next << "'\n";
					return true;
				}

				// Initialize the new backend
				new_backend->initialize(session);

				// Switch session to new backend
				session.switch_backend(new_backend.release());
				provider_manager.set_current_provider(*next);

				std::cout << "Switched to provider '" << *next << "' (" << prov->type << " / " << prov->model << ")\n";
			}
			return true;
		}

		if (subcmd == "show") {
			std::string name = (args.size() >= 2) ? args[1] : provider_manager.get_current_provider();
			if (name.empty()) {
				std::cout << "No provider specified\n";
			} else {
				auto prov = provider_manager.get_provider(name);
				if (!prov) {
					std::cout << "Provider '" << name << "' not found\n";
				} else {
					std::cout << "Provider: " << prov->name << "\n";
					std::cout << "  Type: " << prov->type << "\n";
					if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
						std::cout << "  API Key: " << (api->api_key.empty() ? "not set" : "****") << "\n";
						if (!api->base_url.empty()) {
							std::cout << "  Base URL: " << api->base_url << "\n";
						}
					}
					std::cout << "  Model: " << prov->model << "\n";

					if (prov->rate_limits.tokens_per_month > 0) {
						std::cout << "  Monthly token limit: " << prov->rate_limits.tokens_per_month << "\n";
					}
					if (prov->rate_limits.max_cost_per_month > 0) {
						std::cout << "  Monthly cost limit: $" << prov->rate_limits.max_cost_per_month << "\n";
					}

					if (prov->pricing.dynamic) {
						std::cout << "  Pricing: dynamic (from API)\n";
					} else if (prov->pricing.prompt_cost > 0 || prov->pricing.completion_cost > 0) {
						std::cout << "  Pricing: $" << prov->pricing.prompt_cost << " / $"
						          << prov->pricing.completion_cost << " per million tokens\n";
					}
				}
			}
			return true;
		}

		// Unknown subcommand
		std::cout << "Unknown provider command: " << subcmd << "\n";
		std::cout << "Available: list, add, edit, set, remove, use, next, show\n";
		return true;
	}

	// /model commands
	if (cmd == "/model") {
		if (args.empty()) {
			// Show current model
			std::string current_provider = provider_manager.get_current_provider();
			if (current_provider.empty()) {
				std::cout << "No provider configured\n";
			} else {
				auto prov = provider_manager.get_provider(current_provider);
				if (prov) {
					std::cout << "Current model: " << prov->model << "\n";
				}
			}
			return true;
		}

		std::string subcmd = args[0];

		if (subcmd == "list") {
			std::cout << "Available models depend on your provider.\n";
			std::cout << "Refer to your provider's documentation for model list.\n";
			std::cout << "Common models:\n";
			std::cout << "  OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo\n";
			std::cout << "  Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku\n";
			std::cout << "  Google: gemini-pro, gemini-ultra\n";
			std::cout << "  OpenRouter: anthropic/claude-3.5-sonnet, google/gemini-pro\n";
			return true;
		}

		if (subcmd == "set" && args.size() >= 2) {
			std::string model = args[1];
			std::string current_provider = provider_manager.get_current_provider();
			if (current_provider.empty()) {
				std::cout << "No provider configured\n";
			} else {
				auto prov = provider_manager.get_provider(current_provider);
				if (prov) {
					prov->model = model;
					provider_manager.save_provider(*prov);

					// Update backend's model
					backend->model_name = model;

					std::cout << "Model updated to: " << model << "\n";
					std::cout << "Note: Change takes effect on next message\n";
				}
			}
			return true;
		}

		// Unknown subcommand
		std::cout << "Unknown model command: " << subcmd << "\n";
		std::cout << "Available: list, set\n";
		return true;
	}

	// /config command
	if (cmd == "/config") {
		if (args.empty() || args[0] == "show") {
			// Show all configuration
			std::cout << "=== Configuration ===\n";
			std::cout << "Current provider: " << provider_manager.get_current_provider() << "\n";
			std::cout << "Streaming: " << (config->streaming ? "enabled" : "disabled") << "\n";
			std::cout << "Warmup: " << (config->warmup ? "enabled" : "disabled") << "\n";
			std::cout << "Calibration: " << (config->calibration ? "enabled" : "disabled") << "\n";
			std::cout << "Truncate limit: " << config->truncate_limit << "\n";
			// Add more config fields as needed
			return true;
		}

		if (args[0] == "set" && args.size() >= 3) {
			std::string key = args[1];
			std::string value = args[2];

			// Update config fields
			if (key == "streaming") {
				config->streaming = (value == "true" || value == "1" || value == "on");
				std::cout << "Streaming " << (config->streaming ? "enabled" : "disabled") << "\n";
			} else if (key == "warmup") {
				config->warmup = (value == "true" || value == "1" || value == "on");
				std::cout << "Warmup " << (config->warmup ? "enabled" : "disabled") << "\n";
			} else if (key == "calibration") {
				config->calibration = (value == "true" || value == "1" || value == "on");
				std::cout << "Calibration " << (config->calibration ? "enabled" : "disabled") << "\n";
			} else if (key == "truncate_limit") {
				config->truncate_limit = std::stoi(value);
				std::cout << "Truncate limit set to: " << config->truncate_limit << "\n";
			} else {
				std::cout << "Unknown config key: " << key << "\n";
			}

			// Save config
			config->save();
			return true;
		}

		std::cout << "Usage: /config [show | set <key> <value>]\n";
		return true;
	}

	if (cmd == "/tools") {
		auto& registry = ToolRegistry::instance();

		if (args.empty() || args[0] == "list") {
			auto tool_descriptions = registry.list_tools_with_descriptions();

			std::cout << "\n=== Available Tools (" << tool_descriptions.size() << ") ===\n\n";

			size_t enabled_count = 0;
			size_t disabled_count = 0;

			for (const auto& pair : tool_descriptions) {
				bool is_enabled = registry.enabled(pair.first);
				if (is_enabled) {
					enabled_count++;
				} else {
					disabled_count++;
				}

				std::string status = is_enabled ? "[enabled]" : "[DISABLED]";
				std::cout << "  " << status << " " << pair.first << "\n";
				std::cout << "    " << pair.second << "\n\n";
			}

			std::cout << "Total: " << tool_descriptions.size() << " tools";
			std::cout << " (" << enabled_count << " enabled, " << disabled_count << " disabled)\n";
			return true;
		}

		if (args[0] == "enable" && args.size() >= 2) {
			std::string tool_name = args[1];
			Tool* tool = registry.get_tool(tool_name);

			if (!tool) {
				std::cout << "Tool not found: " << tool_name << "\n";
				return true;
			}

			registry.enable_tool(tool_name);
			std::cout << "Tool '" << tool_name << "' enabled\n";
			return true;
		}

		if (args[0] == "disable" && args.size() >= 2) {
			std::string tool_name = args[1];
			Tool* tool = registry.get_tool(tool_name);

			if (!tool) {
				std::cout << "Tool not found: " << tool_name << "\n";
				return true;
			}

			registry.disable_tool(tool_name);
			std::cout << "Tool '" << tool_name << "' disabled\n";
			return true;
		}

		std::cout << "Usage: /tools [list | enable <tool_name> | disable <tool_name>]\n";
		return true;
	}

	// Command not recognized
	return false;
}

// Main CLI loop - handles all user interaction
int run_cli(std::unique_ptr<Backend>& backend, Session& session) {
	CLI cli;

	LOG_DEBUG("Starting CLI mode (interactive: " + std::string(tio.interactive_mode ? "yes" : "no") + ")");

	// Configure TerminalIO output filtering for this session
	// Get tool call markers from backend (model-specific from chat template)
	tio.markers.tool_call_start = backend->get_tool_call_markers();
	tio.markers.tool_call_end = backend->get_tool_call_end_markers();

	// Get thinking markers from backend (model-specific)
	tio.markers.thinking_start = backend->get_thinking_start_markers();
	tio.markers.thinking_end = backend->get_thinking_end_markers();
	tio.show_thinking = g_show_thinking;  // From --thinking flag

	// Queue warmup message if needed
	if (config->warmup) {
		LOG_DEBUG("Queueing warmup message...");
		tio.add_input(config->warmup_message);
	}

	std::string user_input;

	// Main interaction loop
	while (true) {
		// Get next input (from queue or user)
		user_input = tio.read(">");

		if (user_input.empty()) {
			// Empty string indicates EOF (Ctrl+D) or empty input
			if (cli.eof_received || std::cin.eof()) {
				if (!tio.interactive_mode) {
					LOG_DEBUG("End of piped input");
				} else {
					LOG_INFO("User pressed Ctrl+D - exiting");
				}
				break;
			} else if (tio.interactive_mode) {
				// Interactive mode: empty line continues (just prompt again)
				continue;
			} else {
				// In non-interactive mode, skip empty lines
				continue;
			}
		}

		// Check for exit commands
		if (user_input == "exit" || user_input == "quit") {
			LOG_DEBUG("User requested exit");
			break;
		}

		// Check for slash commands
		if (!user_input.empty() && user_input[0] == '/') {
			// Handle provider commands
			if (handle_provider_command(user_input, backend, session)) {
				continue;  // Command handled, get next input
			}
			// If not a recognized command, treat as regular input
		}

		// Show user message in transcript (for piped mode)
		LOG_DEBUG("User input: " + user_input);

		try {
			// Reset cancellation flag
			cli.generation_cancelled = false;

			// Enable raw terminal mode for escape detection during generation
			// Only if stdin is actually a TTY (not piped input)
			if (tio.interactive_mode && isatty(STDIN_FILENO)) {
			}

			// Sanitize user input: strip control characters and validate size
			user_input = utf8_sanitizer::strip_control_characters(user_input);

			// Check if input is too large for context
			// Use scaling function to determine max user input size based on context window
			double scale = calculate_truncation_scale(backend->context_size);
			int available = backend->context_size - session.system_message_tokens;
			int max_user_input_tokens = available * scale;
			int input_tokens = backend->count_message_tokens(Message::USER, user_input, "", "");

			if (input_tokens > max_user_input_tokens) {
				LOG_DEBUG("User input too large (" + std::to_string(input_tokens) + " tokens), truncating to fit context");

				// Truncate user input to fit in available space
				std::string truncation_notice = "\n\n[INPUT TRUNCATED: Too large for context window]";
				while (input_tokens >= max_user_input_tokens && user_input.length() > 100) {
					// Remove 10% at a time
					size_t new_len = user_input.length() * 0.9;
					user_input = user_input.substr(0, new_len);
					input_tokens = backend->count_message_tokens(Message::USER, user_input + truncation_notice, "", "");
				}
				user_input += truncation_notice;
			}

			// Send user message and get initial response
			LOG_DEBUG("Sending user message to backend");
			Response resp = session.add_message(Message::USER, user_input);

			// Check for error
			if (!resp.success) {
				std::cerr << "\n\033[31mError: " << resp.error << "\033[0m\n" << std::flush;
				continue; // Back to prompt
			}

			// Check if generation was cancelled
			if (resp.finish_reason == "cancelled") {
				cli.show_cancelled();
				continue; // Back to prompt
			}

			LOG_DEBUG("Got response, length: " + std::to_string(resp.content.length()));

#if 0
			// For API backends, always process through ModelOutput (they don't stream)
			// For local backends with tools, also process through ModelOutput (streaming was suppressed)
			// For local backends without tools, ModelOutput was already called during streaming
			auto& registry = ToolRegistry::instance();
			auto available_tools = registry.list_tools();
			bool local_with_tools = backend->is_local && !available_tools.empty();

			if (!backend->is_local || local_with_tools) {
				tio.write(resp.content.c_str(), resp.content.length());
			}
#else
			// API backends now stream - only output if it wasn't already streamed
			// Local backends with tools suppress streaming and output here instead
			auto& registry = ToolRegistry::instance();
			auto available_tools = registry.list_tools();
			bool local_with_tools = backend->is_local && !available_tools.empty();

			if (!resp.was_streamed && (!backend->is_local || local_with_tools)) {
				tio.begin_response();
				tio.write(resp.content.c_str(), resp.content.length());
				tio.end_response();
			}
#endif

			// Tool execution loop - orchestrated by CLI
			int tool_loop_iteration = 0;
			const int max_consecutive_identical_calls = 10; // Detect stuck loops (same call repeated)
			std::vector<std::string> recent_tool_calls; // Track call signatures to detect loops

			while (true) {
				tool_loop_iteration++;
				LOG_DEBUG("Tool loop iteration: " + std::to_string(tool_loop_iteration));

				// Check for tool calls from TerminalIO buffer first
				std::optional<ToolParser::ToolCall> tool_call_opt;
				if (!tio.buffered_tool_call.empty()) {
					// Tool call was intercepted by TerminalIO filtering
					tool_call_opt = ToolParser::parse_tool_call(
						tio.buffered_tool_call,
						backend->get_tool_call_markers()
					);
				} else {
					// Fallback to extracting from Response (for backwards compatibility)
					tool_call_opt = extract_tool_call(resp, backend.get());
				}

				if (tool_call_opt.has_value()) {
					auto tool_call = tool_call_opt.value();
					// Tool call detected
					std::string tool_name = tool_call.name;
					std::string tool_call_id = tool_call.tool_call_id;
					std::string json_str = tool_call.raw_json;

					LOG_DEBUG("Tool call detected: " + tool_name + (tool_call_id.empty() ? "" : " (id: " + tool_call_id + ")"));

					// Create a signature for this tool call to detect loops
					std::string call_signature = tool_name + "(";
					bool first_sig_param = true;
					for (const auto& param : tool_call.parameters) {
						if (!first_sig_param) call_signature += ", ";
						first_sig_param = false;
						call_signature += param.first + "=";

						// Serialize parameter value for comparison
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

					// Check for consecutive identical calls (infinite loop detection)
					int consecutive_count = 1;
					for (int i = (int)recent_tool_calls.size() - 1;
						 i >= 0 && i >= (int)recent_tool_calls.size() - 10; // Check last 10 calls
						 i--) {
						if (recent_tool_calls[i] == call_signature) {
							consecutive_count++;
							if (consecutive_count >= max_consecutive_identical_calls) {
								cli.show_error("Detected infinite loop: " + call_signature +
											 " called " + std::to_string(consecutive_count) +
											 " times consecutively. Stopping.");
								LOG_DEBUG("Loop detected - same call signature " +
										std::to_string(consecutive_count) + " times: " + call_signature);
								break; // Exit while loop
							}
						} else {
							break; // Not consecutive anymore
						}
					}

					// If we detected a loop, break out
					if (consecutive_count >= max_consecutive_identical_calls) {
						break;
					}

					// Add this call to history
					recent_tool_calls.push_back(call_signature);
					LOG_DEBUG("Tool call signature: " + call_signature);

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

					// ModelOutput already processed and output the response content
					// (thinking tags hidden, tool call intercepted)
					// Just show the tool call indicator for user feedback
					cli.show_tool_call(tool_name, params_str);

					// Execute tool using utility function
					LOG_DEBUG("Executing tool: " + tool_name);
					ToolResult tool_result = execute_tool(tool_name, tool_call.parameters);

					// Determine what to show and send
					std::string result_content;
					if (tool_result.success) {
						result_content = tool_result.content;
					} else {
						// Tool failed - show error
						if (!tool_result.error.empty()) {
							result_content = "Error: " + tool_result.error;
						} else {
							result_content = "Error: Tool execution failed";
						}
					}

					// Show tool result (truncated for display)
					cli.show_tool_result(result_content);

					// Truncate tool result if needed to fit in context
					// Reserve space for critical context:
					// - System message (always needed)
					// - Last user message (the question that triggered this tool)
					// - Last assistant message (the tool call itself)
					// Without these, the tool result would be meaningless to the model
					int reserved = session.system_message_tokens;
					if (session.last_user_message_index >= 0) {
						reserved += session.last_user_message_tokens;
					}
					if (session.last_assistant_message_index >= 0) {
						reserved += session.last_assistant_message_tokens;
					}

					// Calculate max tool result space using scaling function
					// This ensures we leave enough room for the model's response
					double scale = calculate_truncation_scale(backend->context_size);
					int max_tool_result_tokens = (backend->context_size - reserved) * scale;

					// If user set a truncate limit (> 0), use that instead
					if (config->truncate_limit > 0 && config->truncate_limit < max_tool_result_tokens) {
						max_tool_result_tokens = config->truncate_limit;
					}

					LOG_DEBUG("Max tool result tokens: " + std::to_string(max_tool_result_tokens) +
							 " (current=" + std::to_string(session.total_tokens) +
							 ", context=" + std::to_string(backend->context_size) +
							 ", reserved=" + std::to_string(reserved) +
							 " [system=" + std::to_string(session.system_message_tokens) +
							 ", last_user=" + std::to_string(session.last_user_message_tokens) +
							 ", last_asst=" + std::to_string(session.last_assistant_message_tokens) + "]" +
							 ", user_limit=" + std::to_string(config->truncate_limit) + ")");

					// Count tokens in the tool result
					int tool_result_tokens = backend->count_message_tokens(Message::TOOL, result_content, tool_name, tool_call_id);
					LOG_DEBUG("Tool result tokens: " + std::to_string(tool_result_tokens));

					// Truncate if needed
					dprintf(1,"tool_result_tokens: %d, max_tool_result_tokens: %d\n", tool_result_tokens, max_tool_result_tokens);
					if (tool_result_tokens >= max_tool_result_tokens) {

							// Build truncation notice
							size_t original_line_count = std::count(result_content.begin(), result_content.end(), '\n');
							std::string truncation_notice = "\n\n[TRUNCATED: Output too large for context window]";
							truncation_notice += "\nOriginal length: " + std::to_string(original_line_count) + " lines";
							truncation_notice += "\nIf you need more: use Read(offset=X, limit=Y), Grep(pattern=...), or Glob with specific patterns";
							dprintf(1,"truncation_notice.length: %d\n", (int)truncation_notice.length());

							int orig_tool_result_tokens = tool_result_tokens;
							while (tool_result_tokens >= max_tool_result_tokens) {

									// Calculate how much of the original to keep
									dprintf(1,"tool_result_tokens: %d, max_tool_result_tokens: %d\n", tool_result_tokens, max_tool_result_tokens);
									double ratio = (static_cast<double>(max_tool_result_tokens) / static_cast<double>(tool_result_tokens)) * 0.95;
									dprintf(1,"ratio: %f, result_content.length: %d\n", ratio, (int)result_content.length());
									int num_chars = static_cast<int>(result_content.length() * ratio);
									dprintf(1,"num_chars: %d\n", num_chars);
									int new_len = num_chars - static_cast<int>(truncation_notice.length());
									dprintf(1,"new_len: %d\n", new_len);

									if (new_len < 0) {
										new_len = 0;
										dprintf(1,"NEW new_len: %d\n", new_len);
									}
									if (new_len > static_cast<int>(result_content.length())) {
										new_len = result_content.length();
										dprintf(1,"NEW new_len: %d\n", new_len);
									}
									// If the result is so small we can't even fit the truncation notice, abort
									if (new_len < static_cast<int>(truncation_notice.length())) {
										throw std::runtime_error("Tool result truncation failed: context window too small to fit truncation notice");
									}

									// Truncate and add notice
									std::string new_result = result_content.substr(0, new_len);
									dprintf(1,"new_result.length: %d\n", (int)new_result.length());
									result_content = new_result + truncation_notice;
									dprintf(1,"NEW result_content.length: %d\n", (int)result_content.length());

									// Verify final token count
									tool_result_tokens = backend->count_message_tokens(Message::TOOL, result_content, tool_name, tool_call_id);
									dprintf(1,"NEW tool_result_tokens: %d\n", tool_result_tokens);
							}

							LOG_DEBUG("Truncated large tool result from " + tool_name + " (" +
									std::to_string(original_line_count) + " lines / " +
									std::to_string(orig_tool_result_tokens) + " tokens -> " +
									std::to_string(std::count(result_content.begin(), result_content.end(), '\n')) + " lines / " +
									std::to_string(tool_result_tokens) + " tokens)");
					}

					// Send tool result to backend and get next response
					// Retry loop in case tool result is too large (MAX_TOKENS_TOO_HIGH)
					// Note: May need many retries if initial char/token estimate is poor
					const int MAX_TOOL_RETRIES = 10;
					int tool_retry = 0;
					std::string truncated_result = result_content;

					for (tool_retry = 0; tool_retry < MAX_TOOL_RETRIES; tool_retry++) {
						// Pass tool_result_tokens=0, let Session calculate max_tokens automatically
						LOG_DEBUG("Sending tool result to backend (tool_tokens=" + std::to_string(tool_result_tokens) +
								 ", attempt " + std::to_string(tool_retry + 1) + "/" + std::to_string(MAX_TOOL_RETRIES) + ")");

						// Reset TerminalIO filtering state before generating response to tool
						tio.reset();

						resp = session.add_message(Message::TOOL, truncated_result, tool_name, tool_call_id, tool_result_tokens, 0);

						// Check for MAX_TOKENS_TOO_HIGH error
						if (!resp.success && resp.code == Response::MAX_TOKENS_TOO_HIGH && resp.actual_prompt_tokens > 0) {
							LOG_WARN("Tool result too large: actual=" + std::to_string(resp.actual_prompt_tokens) +
									" tokens, overflow=" + std::to_string(resp.overflow_tokens) + " tokens");

							// Calculate new truncation target using actual token count from backend
							// We know: actual_prompt_tokens = what the backend measured for this request
							// We need to reduce by overflow_tokens (+ 20% safety margin for chat template overhead)
							int reduction_tokens = static_cast<int>(resp.overflow_tokens * 1.2);
							int target_tokens = resp.actual_prompt_tokens - reduction_tokens;
							if (target_tokens < 100) target_tokens = 100;  // Minimum viable result

							// Calculate actual chars/token ratio from backend's measurement
							float actual_ratio = static_cast<float>(truncated_result.length()) / resp.actual_prompt_tokens;

							// Update backend's chars_per_token estimator with the actual ratio
							// This gives us an exact measurement rather than EMA blending
							// Future messages will use EMA from this corrected baseline
							ApiBackend* api_backend = dynamic_cast<ApiBackend*>(backend.get());
							if (api_backend) {
								api_backend->set_chars_per_token(actual_ratio);
								LOG_INFO("Updated chars_per_token to " + std::to_string(actual_ratio) +
										" based on backend measurement");
							}

							int new_char_limit = static_cast<int>(target_tokens * actual_ratio);

							LOG_DEBUG("Recalculating truncation: target_tokens=" + std::to_string(target_tokens) +
									 ", actual_ratio=" + std::to_string(actual_ratio) +
									 ", new_char_limit=" + std::to_string(new_char_limit));

							// Truncate more aggressively
							if (new_char_limit < static_cast<int>(truncated_result.length())) {
								size_t original_line_count = std::count(result_content.begin(), result_content.end(), '\n');
								std::string truncation_notice = "\n\n[TRUNCATED: Output too large for context window]";
								truncation_notice += "\nOriginal length: " + std::to_string(original_line_count) + " lines";
								truncation_notice += "\nIf you need more: use Read(offset=X, limit=Y), Grep(pattern=...), or Glob with specific patterns";

								int notice_len = truncation_notice.length();
								int content_len = new_char_limit - notice_len;
								if (content_len < 0) content_len = 0;

								truncated_result = result_content.substr(0, content_len) + truncation_notice;

								// Recalculate tokens for the newly truncated result
								tool_result_tokens = backend->count_message_tokens(Message::TOOL, truncated_result, tool_name, tool_call_id);

								LOG_DEBUG("Re-truncated tool result: " + std::to_string(result_content.length()) +
										" chars -> " + std::to_string(truncated_result.length()) + " chars, " +
										std::to_string(tool_result_tokens) + " estimated tokens");

								// Retry with smaller result
								continue;
							}
						}

						// Either success or a different error - break out of retry loop
						break;
					}

					// Check for error after retries
					if (!resp.success) {
						std::cerr << "\n\033[31mError: " << resp.error << "\033[0m\n" << std::flush;
						break;
					}

					LOG_DEBUG("Got response after tool, length: " + std::to_string(resp.content.length()));

					// For API backends or local backends with tools, process through TerminalIO filtering
					// Only output if it wasn't already streamed
					if (!resp.was_streamed && (!backend->is_local || local_with_tools)) {
						tio.begin_response();
						tio.write(resp.content.c_str(), resp.content.length());
						tio.end_response();
					}

					// Output token count for monitoring (used by test scripts)
					LOG_DEBUG("tokens: "+std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

					// Continue loop to check if new response has another tool call
					continue;
				} else {
					// No tool call - this is the final response
					// ModelOutput already processed and output the response content
					// (thinking tags hidden, output already shown to user)

					// Output token count for monitoring (used by test scripts)
					LOG_DEBUG("tokens: "+std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

					// Break out of tool loop
					break;
				}
			}

			// Add trailing newline for non-local backends (API models don't print decode stats)
			if (!backend->is_local) {
				tio.write("\n", 1);
			}

		} catch (const std::exception& e) {
			cli.show_error(e.what());
			LOG_ERROR("Error during generation: " + std::string(e.what()));
		}
	}

	LOG_DEBUG("CLI loop ended");
	return 0;
}

