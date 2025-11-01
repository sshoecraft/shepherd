
#include "debug.h"
#include "cli.h"
#include "shepherd.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "tools/utf8_sanitizer.h"
#include "message.h"
#include "config.h"
#include "backends/api.h"  // For ApiBackend::set_chars_per_token()

#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/select.h>
#include <cstring>
#include <ctime>

#ifdef USE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

// External globals from main.cpp
extern int g_debug_level;
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
CLI::CLI() : interactive_mode(false), eof_received(false),
			 generation_cancelled(false), term_raw_mode(false) {
}

CLI::~CLI() {
	restore_terminal();
}

void CLI::initialize() {
	// Check if input is from a terminal or piped
	interactive_mode = isatty(STDIN_FILENO);

	if (interactive_mode) {
		LOG_DEBUG("CLI initialized in interactive mode");
	} else {
		LOG_DEBUG("CLI initialized in piped mode");
	}
}

void CLI::set_terminal_raw() {
	if (term_raw_mode) return;

	struct termios raw;
	tcgetattr(STDIN_FILENO, &original_term);
	raw = original_term;

	// Disable canonical mode and echo
	raw.c_lflag &= ~(ICANON | ECHO);
	raw.c_cc[VMIN] = 0;   // Non-blocking read
	raw.c_cc[VTIME] = 0;

	tcsetattr(STDIN_FILENO, TCSANOW, &raw);
	term_raw_mode = true;
}

void CLI::restore_terminal() {
	if (!term_raw_mode) return;
	tcsetattr(STDIN_FILENO, TCSANOW, &original_term);
	term_raw_mode = false;
}

bool CLI::check_escape_pressed() {
	if (!term_raw_mode) return false;

	fd_set readfds;
	FD_ZERO(&readfds);
	FD_SET(STDIN_FILENO, &readfds);

	struct timeval tv = {0, 0};  // No wait

	if (select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &tv) > 0) {
		char c;
		if (read(STDIN_FILENO, &c, 1) == 1) {
			if (c == 27) {	// ESC key
				return true;
			}
		}
	}

	return false;
}

std::string CLI::strip_bracketed_paste_sequences(const std::string& input) {
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

std::string CLI::get_input_line() {
#ifdef USE_READLINE
	if (interactive_mode) {
		char* line = readline("");	// No prompt, we handle it separately
		if (line == nullptr) {
			eof_received = true;
			return "";	// EOF
		}
		eof_received = false;
		std::string result(line);

		// Debug: Show what we actually received
		if (g_debug_level) {
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
					eof_received = true;
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
					eof_received = true;
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
			eof_received = true;
			return "";	// EOF
		}
		eof_received = false;
		return line;
	}
#else
	// No readline - use basic getline
	// Prompt is already shown by caller, just read input
	std::string line;
	if (!std::getline(std::cin, line)) {
		eof_received = true;
		return "";	// EOF
	}
	eof_received = false;
	return line;
#endif
}

void CLI::show_prompt() {
	if (interactive_mode) {
		std::cerr << "\n\033[32m> \033[0m" << std::flush;
	}
}

void CLI::show_user_message(const std::string& msg) {
	// In piped mode, echo the user input as transcript
	if (!interactive_mode) {
		printf("> %s\n", msg.c_str());
		fflush(stdout);
	}
}

void CLI::show_assistant_message(const std::string& msg) {
	if (interactive_mode) {
		printf("\033[33m< %s\033[0m\n", msg.c_str());
	} else {
		printf("< %s\n", msg.c_str());
	}
	fflush(stdout);
}

void CLI::show_tool_call(const std::string& name, const std::string& params) {
	if (interactive_mode) {
		printf("\033[33m< %s(%s)\033[0m\n", name.c_str(), params.c_str());
	} else {
		printf("< %s(%s)\n", name.c_str(), params.c_str());
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

	if (interactive_mode) {
		printf("\033[36m> %s\033[0m\n", truncated.c_str());
	} else {
		printf("> %s\n", truncated.c_str());
	}
	fflush(stdout);
}

void CLI::show_error(const std::string& error) {
	if (interactive_mode) {
		printf("\033[31mError: %s\033[0m\n", error.c_str());
	} else {
		fprintf(stderr, "Error: %s\n", error.c_str());
	}
}

void CLI::show_cancelled() {
	if (interactive_mode) {
		printf("\n\033[31m[Cancelled]\033[0m\n");
	}
}

// Main CLI loop - handles all user interaction
int run_cli(std::unique_ptr<Backend>& backend, Session& session) {
	CLI cli;
	cli.initialize();

	LOG_DEBUG("Starting CLI mode (interactive: " + std::string(cli.interactive_mode ? "yes" : "no") + ")");

	bool warmup_done = false;
	std::string user_input;

	// Main interaction loop
	while (true) {
		// Ensure terminal is in normal mode before reading input
		cli.restore_terminal();

		// Check if we need to do warmup first
		if (backend->warmup && !warmup_done) {
			LOG_DEBUG("Warming up " + backend->name + " model with warmup message...");
			user_input = "I want you to respond with exactly 'Ready.' and absolutely nothing else one time only at the start.";
			warmup_done = true;
		} else {
			// Normal user input
			cli.show_prompt();
			user_input = cli.get_input_line();
		}

		if (user_input.empty()) {
			// Empty string indicates EOF (Ctrl+D) or empty input
			if (cli.eof_received || std::cin.eof()) {
				if (!cli.interactive_mode) {
					LOG_DEBUG("End of piped input");
				} else {
					LOG_INFO("User pressed Ctrl+D - exiting");
				}
				break;
			} else if (cli.interactive_mode) {
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

		// Show user message in transcript (for piped mode)
		cli.show_user_message(user_input);

		LOG_DEBUG("User input: " + user_input);

		try {
			// Reset cancellation flag
			cli.generation_cancelled = false;

			// Enable raw terminal mode for escape detection during generation
			// Only if stdin is actually a TTY (not piped input)
			if (cli.interactive_mode && isatty(STDIN_FILENO)) {
				cli.set_terminal_raw();
			}

			// Sanitize user input: strip control characters and validate size
			user_input = utf8_sanitizer::strip_control_characters(user_input);

			// Check if input is too large for context
			int max_user_input_tokens = backend->context_size - session.system_message_tokens - 1024; // Reserve 1024 for response
			int input_tokens = backend->count_message_tokens(Message::USER, user_input, "", "");

			if (input_tokens > max_user_input_tokens) {
				LOG_WARN("User input too large (" + std::to_string(input_tokens) + " tokens), truncating to fit context");

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
				cli.restore_terminal();
				std::cerr << "\n\033[31mError: " << resp.error << "\033[0m\n" << std::flush;
				continue; // Back to prompt
			}

			LOG_DEBUG("Got response, length: " + std::to_string(resp.content.length()));

			// Tool execution loop - orchestrated by CLI
			int tool_loop_iteration = 0;
			const int max_tool_iterations = 100; // Absolute safety limit for total iterations
			const int max_consecutive_identical_calls = 10; // Detect stuck loops (same call repeated)
			std::vector<std::string> recent_tool_calls; // Track call signatures to detect loops

			while (tool_loop_iteration < max_tool_iterations) {
				tool_loop_iteration++;
				LOG_DEBUG("Tool loop iteration: " + std::to_string(tool_loop_iteration));

				// Check if user cancelled during generation
				if (cli.interactive_mode && cli.check_escape_pressed()) {
					cli.generation_cancelled = true;
					cli.restore_terminal();
					cli.show_cancelled();
					break;
				}

				// Check for tool calls (handles both structured and text-based)
				auto tool_call_opt = extract_tool_call(resp, backend.get());

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
								cli.restore_terminal();
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

					// Calculate max tool result space (session.cpp will reserve completion space)
					int max_tool_result_tokens = backend->context_size - reserved;

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

							LOG_WARN("Truncated large tool result from " + tool_name + " (" +
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

								LOG_WARN("Re-truncated tool result: " + std::to_string(result_content.length()) +
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
						cli.restore_terminal();
						std::cerr << "\n\033[31mError: " << resp.error << "\033[0m\n" << std::flush;
						break;
					}

					LOG_DEBUG("Got response after tool, length: " + std::to_string(resp.content.length()));

					// Output token count for monitoring (used by test scripts)
					LOG_DEBUG("tokens: "+std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

					// Continue loop to check if new response has another tool call
					continue;
				} else {
					// No tool call - this is the final response
					// Show assistant response
					// (Backend already added it to session via add_message)
					if (!resp.content.empty()) {
						// Filter out incomplete <tool_call> markers (ones without closing tags)
						std::string display_content = resp.content;
						if (display_content.find("<tool_call>") != std::string::npos &&
						    display_content.find("</tool_call>") == std::string::npos) {
							// Has opening tag but no closing tag - strip it
							size_t pos = display_content.find("<tool_call>");
							display_content = display_content.substr(0, pos);
							// Trim trailing whitespace
							while (!display_content.empty() && std::isspace(display_content.back())) {
								display_content.pop_back();
							}
						}
						if (!display_content.empty()) {
							cli.show_assistant_message(display_content);
						}
					}

					// Output token count for monitoring (used by test scripts)
					LOG_DEBUG("tokens: "+std::to_string(session.total_tokens) + "/" + std::to_string(backend->context_size));

					// Break out of tool loop
					break;
				}
			}

			// Restore terminal to normal mode after generation
			if (cli.interactive_mode) {
				cli.restore_terminal();
			}

			if (tool_loop_iteration >= max_tool_iterations) {
				cli.show_error("Maximum tool iterations reached - stopping to prevent infinite loop");
			}

		} catch (const std::exception& e) {
			// Restore terminal on error
			if (cli.interactive_mode) {
				cli.restore_terminal();
			}
			cli.show_error(e.what());
			LOG_ERROR("Error during generation: " + std::string(e.what()));
		}
	}

	LOG_DEBUG("CLI loop ended");
	return 0;
}

