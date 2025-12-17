#include "shepherd.h"
#include "command_tools.h"
#include "tools.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <array>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <cerrno>

std::vector<ParameterDef> ExecuteCommandTool::get_parameters_schema() const {
    return {
        {"command", "string", "The shell command to execute", true, "", "", {}},
        {"working_dir", "string", "Optional working directory for command execution", false, "", "", {}},
        {"timeout", "number", "Optional timeout in seconds (not currently enforced)", false, "30", "", {}}
    };
}

std::map<std::string, std::any> ExecuteCommandTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string command = tool_utils::get_string(args, "command");
    std::string working_dir = tool_utils::get_string(args, "working_dir");
    int timeout = tool_utils::get_int(args, "timeout", 30);

    if (command.empty()) {
        result["error"] = std::string("command is required");
        result["success"] = false;
        return result;
    }

    try {
        // Build full command
        std::string full_command = command;

        // Change directory if specified
        if (!working_dir.empty()) {
            full_command = "cd \"" + working_dir + "\" && " + command;
        }

        dout(1) << "ExecuteCommand: Running: " + full_command << std::endl;

        // Create pipes for stdout and stderr
        int stdout_pipe[2];
        int stderr_pipe[2];

        if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            result["error"] = std::string("failed to create pipes");
            result["success"] = false;
            return result;
        }

        pid_t pid = fork();

        if (pid == -1) {
            close(stdout_pipe[0]); close(stdout_pipe[1]);
            close(stderr_pipe[0]); close(stderr_pipe[1]);
            result["error"] = std::string("failed to fork process");
            result["success"] = false;
            return result;
        }

        if (pid == 0) {
            // Child process
            close(stdout_pipe[0]);  // Close read end
            close(stderr_pipe[0]);  // Close read end

            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stderr_pipe[1], STDERR_FILENO);

            close(stdout_pipe[1]);
            close(stderr_pipe[1]);

            execl("/bin/sh", "sh", "-c", full_command.c_str(), nullptr);
            _exit(127);  // exec failed
        }

        // Parent process
        close(stdout_pipe[1]);  // Close write end
        close(stderr_pipe[1]);  // Close write end

        // Set non-blocking and read both pipes
        fcntl(stdout_pipe[0], F_SETFL, O_NONBLOCK);
        fcntl(stderr_pipe[0], F_SETFL, O_NONBLOCK);

        std::string stdout_output;
        std::string stderr_output;
        char buffer[4096];

        // Read from both pipes until child exits
        int status;
        bool stdout_open = true, stderr_open = true;

        while (stdout_open || stderr_open) {
            if (stdout_open) {
                ssize_t n = read(stdout_pipe[0], buffer, sizeof(buffer));
                if (n > 0) {
                    stdout_output.append(buffer, n);
                } else if (n == 0 || (n == -1 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                    stdout_open = false;
                }
            }
            if (stderr_open) {
                ssize_t n = read(stderr_pipe[0], buffer, sizeof(buffer));
                if (n > 0) {
                    stderr_output.append(buffer, n);
                } else if (n == 0 || (n == -1 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                    stderr_open = false;
                }
            }
            if (stdout_open || stderr_open) {
                usleep(1000);  // Small sleep to avoid busy-waiting
            }
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);

        waitpid(pid, &status, 0);
        int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

        result["stdout"] = stdout_output;
        result["stderr"] = stderr_output;
        result["exit_code"] = exit_code;
        result["success"] = (exit_code == 0);

        // Format the output as a readable string
        if (exit_code == 0) {
            if (stdout_output.empty()) {
                result["content"] = std::string("Command completed successfully (no output)");
                result["summary"] = std::string("Command completed successfully");
            } else {
                result["content"] = stdout_output;
                int line_count = std::count(stdout_output.begin(), stdout_output.end(), '\n');
                if (line_count == 0 && !stdout_output.empty()) line_count = 1;
                result["summary"] = std::string("Output: ") + std::to_string(line_count) + " line" + (line_count != 1 ? "s" : "");
            }
        } else {
            // For errors, prefer stderr for summary, fallback to stdout
            std::string error_summary = stderr_output.empty() ? stdout_output : stderr_output;
            size_t newline = error_summary.find('\n');
            if (newline != std::string::npos) {
                error_summary = error_summary.substr(0, newline);
            }
            if (error_summary.empty()) {
                error_summary = "Command failed (exit " + std::to_string(exit_code) + ")";
            }
            if (error_summary.length() > 80) {
                error_summary = error_summary.substr(0, 77) + "...";
            }

            // Combine stdout and stderr for full content
            std::string combined_output;
            if (!stdout_output.empty()) combined_output += stdout_output;
            if (!stderr_output.empty()) {
                if (!combined_output.empty()) combined_output += "\n";
                combined_output += stderr_output;
            }

            if (combined_output.empty()) {
                result["content"] = std::string("Command failed (exit code ") + std::to_string(exit_code) + ")";
            } else {
                result["content"] = combined_output;
            }
            result["summary"] = error_summary;
            result["error"] = error_summary;
        }

        dout(1) << "ExecuteCommand: Completed with exit code " + std::to_string(exit_code) << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error executing command: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> GetEnvironmentVariableTool::get_parameters_schema() const {
    return {
        {"name", "string", "The name of the environment variable to retrieve", true, "", "", {}},
        {"default", "string", "Optional default value if variable is not set", false, "", "", {}}
    };
}

std::map<std::string, std::any> GetEnvironmentVariableTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string name = tool_utils::get_string(args, "name");
    std::string default_value = tool_utils::get_string(args, "default");

    if (name.empty()) {
        result["error"] = std::string("name is required");
        result["success"] = false;
        return result;
    }

    try {
        const char* env_value = std::getenv(name.c_str());
        std::string value;

        if (env_value != nullptr) {
            value = std::string(env_value);
        } else {
            value = default_value;
        }

        result["value"] = value;
        result["content"] = name + "=" + value;
        result["summary"] = name + "=" + (value.length() > 50 ? value.substr(0, 47) + "..." : value);
        result["success"] = true;

        dout(1) << "GetEnvironmentVariable: " + name + " = " + value << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error getting environment variable: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> ListProcessesTool::get_parameters_schema() const {
    // This tool takes no parameters
    return {};
}

std::map<std::string, std::any> ListProcessesTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    try {
        // Use ps command to get process list
        std::string command = "ps aux";

        std::array<char, 128> buffer;
        std::string ps_output;

        std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(command.c_str(), "r"), pclose);

        if (!pipe) {
            result["error"] = std::string("failed to execute ps command");
            result["success"] = false;
            return result;
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            ps_output += buffer.data();
        }

        int exit_code = pclose(pipe.release());

        if (exit_code != 0) {
            result["error"] = std::string("ps command failed");
            result["success"] = false;
            return result;
        }

        // Parse ps output
        std::vector<std::map<std::string, std::any>> processes;
        std::istringstream stream(ps_output);
        std::string line;

        // Skip header line
        std::getline(stream, line);

        while (std::getline(stream, line)) {
            if (line.empty()) continue;

            std::istringstream line_stream(line);
            std::vector<std::string> fields;
            std::string field;

            // Parse fields (space-separated)
            while (line_stream >> field) {
                fields.push_back(field);
            }

            if (fields.size() >= 11) {
                std::map<std::string, std::any> process;
                process["user"] = fields[0];
                process["pid"] = fields[1];
                process["cpu"] = fields[2];
                process["mem"] = fields[3];

                // Command is everything from field 10 onwards
                std::string command_str;
                for (size_t i = 10; i < fields.size(); ++i) {
                    if (i > 10) command_str += " ";
                    command_str += fields[i];
                }
                process["command"] = command_str;

                processes.push_back(process);
            }
        }

        result["processes"] = processes;
        result["summary"] = std::string("Listed ") + std::to_string(processes.size()) + " process" + (processes.size() != 1 ? "es" : "");
        result["success"] = true;

        dout(1) << "ListProcesses: Found " + std::to_string(processes.size()) + " processes" << std::endl;

    } catch (const std::exception& e) {
        result["error"] = std::string("error listing processes: ") + e.what();
        result["success"] = false;
    }

    return result;
}

void register_command_tools(Tools& tools) {
    tools.register_tool(std::make_unique<ExecuteCommandTool>());
    tools.register_tool(std::make_unique<GetEnvironmentVariableTool>());
    tools.register_tool(std::make_unique<ListProcessesTool>());

    dout(1) << "Registered command tools: execute_command, get_env, list_processes" << std::endl;
}