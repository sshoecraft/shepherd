#include "command_tools.h"
#include "../logger.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <array>
#include <sstream>
#include <string>
#include <vector>

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

        LOG_DEBUG("ExecuteCommand: Running: " + full_command);

        // Execute command and capture output
        std::array<char, 128> buffer;
        std::string stdout_output;
        std::string stderr_output;

        // Use popen to capture stdout
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(full_command.c_str(), "r"), pclose);

        if (!pipe) {
            result["error"] = std::string("failed to execute command");
            result["success"] = false;
            return result;
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            stdout_output += buffer.data();
        }

        int exit_code = pclose(pipe.release());

        result["stdout"] = stdout_output;
        result["stderr"] = stderr_output;  // Note: stderr capture would need more complex implementation
        result["exit_code"] = exit_code;
        result["success"] = (exit_code == 0);

        // Format the output as a readable string
        if (exit_code == 0) {
            result["content"] = "Command output:\n\n" + stdout_output;
        } else {
            result["content"] = "Command failed (exit code " + std::to_string(exit_code) + "):\n\n" + stdout_output;
        }

        LOG_DEBUG("ExecuteCommand: Completed with exit code " + std::to_string(exit_code));

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

        if (env_value != nullptr) {
            result["value"] = std::string(env_value);
        } else {
            result["value"] = default_value;
        }

        result["success"] = true;

        LOG_DEBUG("GetEnvironmentVariable: " + name + " = " + tool_utils::get_string(result, "value"));

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
        std::string output;

        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);

        if (!pipe) {
            result["error"] = std::string("failed to execute ps command");
            result["success"] = false;
            return result;
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            output += buffer.data();
        }

        int exit_code = pclose(pipe.release());

        if (exit_code != 0) {
            result["error"] = std::string("ps command failed");
            result["success"] = false;
            return result;
        }

        // Parse ps output
        std::vector<std::map<std::string, std::any>> processes;
        std::istringstream stream(output);
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
        result["success"] = true;

        LOG_DEBUG("ListProcesses: Found " + std::to_string(processes.size()) + " processes");

    } catch (const std::exception& e) {
        result["error"] = std::string("error listing processes: ") + e.what();
        result["success"] = false;
    }

    return result;
}

void register_command_tools() {
    auto& registry = ToolRegistry::instance();

    registry.register_tool(std::make_unique<ExecuteCommandTool>());
    registry.register_tool(std::make_unique<GetEnvironmentVariableTool>());
    registry.register_tool(std::make_unique<ListProcessesTool>());

    LOG_DEBUG("Registered command tools: execute_command, get_env, list_processes");
}