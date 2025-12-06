#include "core_tools.h"
#include "tools.h"
#include "http_client.h"
#include "tools/web_search.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sys/wait.h>
#include <signal.h>
#include <curl/curl.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Helper function for CURL write callback
static size_t curl_string_write_callback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append((char*)contents, total_size);
    return total_size;
}

// Helper function to execute command and capture output
static std::string exec_command(const std::string& cmd, int timeout_ms = 120000) {
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "Error: Failed to execute command";
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }

    int status = pclose(pipe);
    if (status != 0) {
        result += "\nCommand exited with code: " + std::to_string(WEXITSTATUS(status));
    }

    return result;
}

// Helper function to match glob patterns
static bool glob_match(const std::string& pattern, const std::string& text) {
    std::string regex_pattern = pattern;

    // Convert glob to regex
    size_t pos = 0;
    while ((pos = regex_pattern.find("**", pos)) != std::string::npos) {
        regex_pattern.replace(pos, 2, ".*");
        pos += 2;
    }
    pos = 0;
    while ((pos = regex_pattern.find("*", pos)) != std::string::npos) {
        if (pos == 0 || regex_pattern[pos-1] != '.') {
            regex_pattern.replace(pos, 1, "[^/]*");
            pos += 5;
        } else {
            pos++;
        }
    }
    pos = 0;
    while ((pos = regex_pattern.find("?", pos)) != std::string::npos) {
        regex_pattern.replace(pos, 1, ".");
        pos += 1;
    }

    std::regex re(regex_pattern);
    return std::regex_match(text, re);
}

// TaskTool implementation
std::vector<ParameterDef> TaskTool::get_parameters_schema() const {
    return {
        {"prompt", "string", "Detailed task description for the agent to perform autonomously", true, "", "", {}},
        {"description", "string", "Short (3-5 word) summary of the task", false, "Task", "", {}},
        {"subagent_type", "string", "Type of specialized agent (general-purpose, statusline-setup, output-style-setup)", false, "general-purpose", "", {}}
    };
}

std::map<std::string, std::any> TaskTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string prompt = tool_utils::get_string(args, "prompt");
    std::string description = tool_utils::get_string(args, "description", "Task");
    std::string subagent_type = tool_utils::get_string(args, "subagent_type", "general-purpose");

    if (prompt.empty()) {
        result["error"] = std::string("prompt is required");
        result["success"] = false;
        return result;
    }

    result["content"] = std::string("Task agent system not implemented in Shepherd. Prompt: ") + prompt;
    result["success"] = true;
    return result;
}

// BashTool implementation
std::vector<ParameterDef> BashTool::get_parameters_schema() const {
    return {
        {"command", "string", "The bash command to execute", true, "", "", {}},
        {"description", "string", "Clear description of what the command does (5-10 words)", false, "Execute command", "", {}},
        {"timeout", "number", "Optional timeout in milliseconds", false, "120000", "", {}},
        {"run_in_background", "boolean", "Set to true to run command in background", false, "false", "", {}}
    };
}

std::map<std::string, std::any> BashTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string command = tool_utils::get_string(args, "command");
    std::string description = tool_utils::get_string(args, "description", "Execute command");
    int timeout = tool_utils::get_int(args, "timeout", 120000);
    bool run_in_background = tool_utils::get_bool(args, "run_in_background", false);

    if (command.empty()) {
        result["error"] = std::string("command is required");
        result["success"] = false;
        return result;
    }

    if (run_in_background) {
        auto& manager = BackgroundShellManager::instance();
        std::string shell_id = manager.start_shell(command);
        result["content"] = std::string("Started background shell: ") + shell_id;
        result["shell_id"] = shell_id;
        result["success"] = true;
        return result;
    }

    std::string output = exec_command(command, timeout);
    result["content"] = output;
    result["success"] = true;
    return result;
}

// GlobTool implementation
std::vector<ParameterDef> GlobTool::get_parameters_schema() const {
    return {
        {"pattern", "string", "Glob pattern to match files (e.g., **/*.js, src/**/*.cpp)", true, "", "", {}},
        {"path", "string", "Directory to search in (defaults to current directory)", false, ".", "", {}}
    };
}

std::map<std::string, std::any> GlobTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string pattern = tool_utils::get_string(args, "pattern");
    std::string path = tool_utils::get_string(args, "path", ".");

    if (pattern.empty()) {
        result["error"] = std::string("pattern is required");
        result["success"] = false;
        return result;
    }

    std::vector<std::string> matches;

    try {
        fs::path search_path = fs::absolute(path);

        for (const auto& entry : fs::recursive_directory_iterator(search_path)) {
            if (entry.is_regular_file()) {
                std::string filepath = entry.path().string();
                std::string relative = fs::relative(entry.path(), search_path).string();

                if (glob_match(pattern, relative)) {
                    matches.push_back(filepath);
                }
            }
        }

        std::ostringstream oss;
        for (const auto& match : matches) {
            oss << match << "\n";
        }

        result["content"] = oss.str();
        result["count"] = static_cast<int>(matches.size());
        result["success"] = true;

    } catch (const std::exception& e) {
        result["error"] = std::string("Error: ") + e.what();
        result["success"] = false;
    }

    return result;
}

// GrepTool implementation
std::vector<ParameterDef> GrepTool::get_parameters_schema() const {
    return {
        {"pattern", "string", "Regular expression pattern to search for", true, "", "", {}},
        {"path", "string", "File or directory to search in (defaults to current directory)", false, ".", "", {}},
        {"output_mode", "string", "Output format: content, files_with_matches, or count", false, "files_with_matches", "", {}},
        {"case_insensitive", "boolean", "Perform case-insensitive search", false, "false", "", {}},
        {"line_numbers", "boolean", "Show line numbers in output (content mode only)", false, "false", "", {}},
        {"context_after", "number", "Number of lines to show after match", false, "0", "", {}},
        {"context_before", "number", "Number of lines to show before match", false, "0", "", {}},
        {"glob", "string", "Filter files by glob pattern (e.g., *.js)", false, "", "", {}},
        {"multiline", "boolean", "Enable multiline pattern matching", false, "false", "", {}},
        {"head_limit", "number", "Limit output to first N results", false, "0", "", {}}
    };
}

// Helper: Check if ripgrep (rg) is available
static bool is_ripgrep_available() {
    static int cached = -1;  // -1 = not checked, 0 = no, 1 = yes
    if (cached != -1) {
        return cached == 1;
    }

    FILE* fp = popen("which rg 2>/dev/null", "r");
    if (!fp) {
        cached = 0;
        return false;
    }

    char buf[128];
    bool found = (fgets(buf, sizeof(buf), fp) != nullptr);
    pclose(fp);

    cached = found ? 1 : 0;
    return found;
}

std::map<std::string, std::any> GrepTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string pattern = tool_utils::get_string(args, "pattern");
    std::string path = tool_utils::get_string(args, "path", ".");
    std::string output_mode = tool_utils::get_string(args, "output_mode", "files_with_matches");
    bool case_insensitive = tool_utils::get_bool(args, "case_insensitive", false);
    bool line_numbers = tool_utils::get_bool(args, "line_numbers", false);
    int context_after = tool_utils::get_int(args, "context_after", 0);
    int context_before = tool_utils::get_int(args, "context_before", 0);
    std::string glob = tool_utils::get_string(args, "glob", "");

    if (pattern.empty()) {
        result["error"] = std::string("pattern is required");
        result["success"] = false;
        return result;
    }

    try {
        // Use ripgrep if available (much faster and respects .gitignore)
        if (is_ripgrep_available()) {
            std::ostringstream cmd;
            cmd << "rg";

            // Output mode
            if (output_mode == "files_with_matches") {
                cmd << " --files-with-matches";
            } else if (output_mode == "count") {
                cmd << " --count";
            }
            // content mode is default

            // Case sensitivity
            if (case_insensitive) {
                cmd << " --ignore-case";
            }

            // Line numbers
            if (line_numbers && output_mode == "content") {
                cmd << " --line-number";
            }

            // Context
            if (context_after > 0 && output_mode == "content") {
                cmd << " -A " << context_after;
            }
            if (context_before > 0 && output_mode == "content") {
                cmd << " -B " << context_before;
            }

            // Glob filter
            if (!glob.empty()) {
                cmd << " --glob '" << glob << "'";
            }

            // Pattern and path
            cmd << " '" << pattern << "' '" << path << "' 2>&1";

            FILE* fp = popen(cmd.str().c_str(), "r");
            if (!fp) {
                result["error"] = std::string("Failed to execute ripgrep");
                result["success"] = false;
                return result;
            }

            std::ostringstream output;
            char buf[4096];
            int match_count = 0;

            while (fgets(buf, sizeof(buf), fp) != nullptr) {
                output << buf;
                if (output_mode == "files_with_matches" || output_mode == "content") {
                    match_count++;
                }
            }

            int status = pclose(fp);

            // rg returns 0 if matches found, 1 if no matches, 2 if error
            if (WEXITSTATUS(status) == 2) {
                result["error"] = std::string("Ripgrep error: ") + output.str();
                result["success"] = false;
                return result;
            }

            result["content"] = output.str();
            result["count"] = match_count;
            result["success"] = true;
            return result;
        }

        // Fallback: Use C++ regex but ONLY search current directory (non-recursive)
        std::regex::flag_type flags = std::regex::ECMAScript;
        if (case_insensitive) {
            flags |= std::regex::icase;
        }
        std::regex re(pattern, flags);

        std::ostringstream output;
        int match_count = 0;

        fs::path search_path = fs::absolute(path);

        if (fs::is_regular_file(search_path)) {
            // Search in single file
            std::ifstream file(search_path);
            std::string line;
            int line_num = 0;

            while (std::getline(file, line)) {
                line_num++;
                if (std::regex_search(line, re)) {
                    match_count++;
                    if (output_mode == "content") {
                        if (line_numbers) {
                            output << search_path.string() << ":" << line_num << ":" << line << "\n";
                        } else {
                            output << line << "\n";
                        }
                    } else if (output_mode == "files_with_matches") {
                        output << search_path.string() << "\n";
                        break;
                    }
                }
            }
        } else if (fs::is_directory(search_path)) {
            // Search in directory (NON-RECURSIVE to avoid hanging on large directories)
            // If ripgrep is not available, only search immediate directory
            for (const auto& entry : fs::directory_iterator(search_path)) {
                if (!entry.is_regular_file()) continue;

                std::string filepath = entry.path().string();

                // Apply glob filter if specified
                if (!glob.empty()) {
                    std::string filename = entry.path().filename().string();
                    if (!glob_match(glob, filename)) {
                        continue;
                    }
                }

                std::ifstream file(filepath);
                std::string line;
                int line_num = 0;
                bool file_matched = false;

                while (std::getline(file, line)) {
                    line_num++;
                    if (std::regex_search(line, re)) {
                        match_count++;
                        file_matched = true;

                        if (output_mode == "content") {
                            if (line_numbers) {
                                output << filepath << ":" << line_num << ":" << line << "\n";
                            } else {
                                output << line << "\n";
                            }
                        } else if (output_mode == "files_with_matches") {
                            output << filepath << "\n";
                            break;
                        }
                    }
                }
            }
        }

        if (output_mode == "count") {
            output << "Total matches: " << match_count << "\n";
        }

        result["content"] = output.str();
        result["count"] = match_count;
        result["success"] = true;

    } catch (const std::regex_error& e) {
        result["error"] = std::string("Regex error: ") + e.what();
        result["success"] = false;
    } catch (const std::exception& e) {
        result["error"] = std::string("Error: ") + e.what();
        result["success"] = false;
    }

    return result;
}

// EditTool implementation
std::vector<ParameterDef> EditTool::get_parameters_schema() const {
    return {
        {"file_path", "string", "path to the file to be edited", true, "", "", {}},
        {"old_string", "string", "The exact text to replace", true, "", "", {}},
        {"new_string", "string", "The replacement text", true, "", "", {}},
        {"replace_all", "boolean", "Replace all occurrences (false = replace first only)", false, "false", "", {}}
    };
}

std::map<std::string, std::any> EditTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string file_path = tool_utils::get_string(args, "file_path");
    std::string old_string = tool_utils::get_string(args, "old_string");
    std::string new_string = tool_utils::get_string(args, "new_string");
    bool replace_all = tool_utils::get_bool(args, "replace_all", false);

    if (file_path.empty() || old_string.empty()) {
        result["error"] = std::string("file_path and old_string are required");
        result["success"] = false;
        return result;
    }

    try {
        fs::path abs_path = fs::absolute(file_path);

        if (!fs::exists(abs_path)) {
            result["error"] = std::string("File not found: ") + file_path;
            result["success"] = false;
            return result;
        }

        std::ifstream file(abs_path);
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        std::string content = buffer.str();
        int replacement_count = 0;

        if (replace_all) {
            size_t pos = 0;
            while ((pos = content.find(old_string, pos)) != std::string::npos) {
                content.replace(pos, old_string.length(), new_string);
                pos += new_string.length();
                replacement_count++;
            }
        } else {
            size_t pos = content.find(old_string);
            if (pos != std::string::npos) {
                content.replace(pos, old_string.length(), new_string);
                replacement_count = 1;
            } else {
                result["error"] = std::string("String not found in file");
                result["success"] = false;
                return result;
            }
        }

        std::ofstream out_file(abs_path);
        out_file << content;
        out_file.close();

        result["content"] = std::string("Replaced ") + std::to_string(replacement_count) + " occurrence(s)";
        result["success"] = true;

    } catch (const std::exception& e) {
        result["error"] = std::string("Error: ") + e.what();
        result["success"] = false;
    }

    return result;
}

// WebFetchTool implementation
std::vector<ParameterDef> WebFetchTool::get_parameters_schema() const {
    return {
        {"url", "string", "The URL to fetch content from", true, "", "", {}},
        {"prompt", "string", "What information to extract from the page", false, "Extract content", "", {}}
    };
}

std::map<std::string, std::any> WebFetchTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string url = tool_utils::get_string(args, "url");
    std::string prompt = tool_utils::get_string(args, "prompt", "Extract content");

    if (url.empty()) {
        result["error"] = std::string("url is required");
        result["success"] = false;
        return result;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        result["error"] = std::string("Failed to initialize CURL");
        result["success"] = false;
        return result;
    }

    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_string_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        result["error"] = std::string("CURL error: ") + curl_easy_strerror(res);
        result["success"] = false;
        return result;
    }

    result["content"] = response_data;
    result["success"] = true;
    return result;
}

// WebSearchTool implementation
std::vector<ParameterDef> WebSearchTool::get_parameters_schema() const {
    return {
        {"query", "string", "The search query", true, "", "", {}},
        {"allowed_domains", "array", "Only include results from these domains", false, "", "string", {}},
        {"blocked_domains", "array", "Exclude results from these domains", false, "", "string", {}}
    };
}

std::map<std::string, std::any> WebSearchTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string query = tool_utils::get_string(args, "query");

    if (query.empty()) {
        result["error"] = std::string("query is required");
        result["success"] = false;
        return result;
    }

    auto& search_manager = WebSearch::instance();

    if (!search_manager.is_available()) {
        result["error"] = std::string("Web search not configured. Add 'web_search_provider' to config.json");
        result["success"] = false;
        return result;
    }

    try {
        auto search_results = search_manager.search(query);

        std::ostringstream content;
        content << "Search results for: " << query << "\n\n";

        if (search_results.empty()) {
            content << "No results found.\n";
        } else {
            int count = 1;
            for (const auto& item : search_results) {
                content << count << ". " << item.title << "\n";
                content << "   URL: " << item.url << "\n";
                if (!item.description.empty()) {
                    content << "   " << item.description << "\n";
                }
                content << "\n";
                count++;
            }
        }

        result["content"] = content.str();
        result["success"] = true;

    } catch (const std::exception& e) {
        result["error"] = std::string("Search failed: ") + e.what();
        result["success"] = false;
    }

    return result;
}

// TodoWriteTool implementation
std::vector<ParameterDef> TodoWriteTool::get_parameters_schema() const {
    ParameterDef todo_item;
    todo_item.name = "todos";
    todo_item.type = "array";
    todo_item.description = "Array of todo items to manage";
    todo_item.required = true;
    todo_item.default_value = "";
    todo_item.array_item_type = "object";

    // Define the structure of each todo item
    todo_item.object_properties = {
        {"content", "string", "Task description in imperative form (e.g., 'Run tests')", true, "", "", {}},
        {"status", "string", "Task status: pending, in_progress, or completed", true, "pending", "", {}},
        {"activeForm", "string", "Present continuous form shown during execution (e.g., 'Running tests')", true, "", "", {}}
    };

    return {todo_item};
}

std::map<std::string, std::any> TodoWriteTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;
    result["content"] = std::string("Todo management stored in memory");
    result["success"] = true;
    return result;
}

// BashOutputTool implementation
std::vector<ParameterDef> BashOutputTool::get_parameters_schema() const {
    return {
        {"bash_id", "string", "The shell identifier returned by background Bash execution", true, "", "", {}},
        {"filter", "string", "Optional regex pattern to filter output lines", false, "", "", {}}
    };
}

std::map<std::string, std::any> BashOutputTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string bash_id = tool_utils::get_string(args, "bash_id");
    std::string filter = tool_utils::get_string(args, "filter", "");

    if (bash_id.empty()) {
        result["error"] = std::string("bash_id is required");
        result["success"] = false;
        return result;
    }

    auto& manager = BackgroundShellManager::instance();
    std::string output = manager.get_output(bash_id, filter);

    if (output.empty()) {
        result["error"] = std::string("Shell not found or no output");
        result["success"] = false;
        return result;
    }

    result["content"] = output;
    result["success"] = true;
    return result;
}

// KillShellTool implementation
std::vector<ParameterDef> KillShellTool::get_parameters_schema() const {
    return {
        {"shell_id", "string", "The shell identifier to terminate", true, "", "", {}}
    };
}

std::map<std::string, std::any> KillShellTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string shell_id = tool_utils::get_string(args, "shell_id");

    if (shell_id.empty()) {
        result["error"] = std::string("shell_id is required");
        result["success"] = false;
        return result;
    }

    auto& manager = BackgroundShellManager::instance();
    bool killed = manager.kill_shell(shell_id);

    if (!killed) {
        result["error"] = std::string("Shell not found: ") + shell_id;
        result["success"] = false;
        return result;
    }

    result["content"] = std::string("Killed shell: ") + shell_id;
    result["success"] = true;
    return result;
}

// BackgroundShellManager implementation
BackgroundShellManager& BackgroundShellManager::instance() {
    static BackgroundShellManager manager;
    return manager;
}

std::string BackgroundShellManager::start_shell(const std::string& command) {
    std::lock_guard<std::mutex> lock(shells_mutex_);

    std::string id = "shell_" + std::to_string(next_id_++);
    auto shell = std::make_unique<BackgroundShell>();
    shell->id = id;
    shell->command = command;
    shell->running = true;
    shell->exit_code = -1;

    shell->pipe = popen(command.c_str(), "r");
    if (!shell->pipe) {
        return "Error: Failed to start shell";
    }

    shell->reader_thread = std::thread([shell_ptr = shell.get()]() {
        char buffer[256];
        while (shell_ptr->running && fgets(buffer, sizeof(buffer), shell_ptr->pipe) != nullptr) {
            std::lock_guard<std::mutex> lock(shell_ptr->output_mutex);
            shell_ptr->output += buffer;
        }
        shell_ptr->exit_code = pclose(shell_ptr->pipe);
        shell_ptr->running = false;
    });

    shells_[id] = std::move(shell);
    return id;
}

std::string BackgroundShellManager::get_output(const std::string& shell_id, const std::string& filter) {
    std::lock_guard<std::mutex> lock(shells_mutex_);

    auto it = shells_.find(shell_id);
    if (it == shells_.end()) {
        return "";
    }

    std::lock_guard<std::mutex> output_lock(it->second->output_mutex);
    std::string output = it->second->output;

    if (!filter.empty()) {
        std::regex re(filter);
        std::string filtered_output;
        std::istringstream iss(output);
        std::string line;

        while (std::getline(iss, line)) {
            if (std::regex_search(line, re)) {
                filtered_output += line + "\n";
            }
        }
        return filtered_output;
    }

    return output;
}

bool BackgroundShellManager::kill_shell(const std::string& shell_id) {
    std::lock_guard<std::mutex> lock(shells_mutex_);

    auto it = shells_.find(shell_id);
    if (it == shells_.end()) {
        return false;
    }

    it->second->running = false;
    if (it->second->reader_thread.joinable()) {
        it->second->reader_thread.join();
    }

    shells_.erase(it);
    return true;
}

std::vector<std::string> BackgroundShellManager::list_shells() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(shells_mutex_));

    std::vector<std::string> shell_ids;
    for (const auto& pair : shells_) {
        shell_ids.push_back(pair.first);
    }
    return shell_ids;
}

// GetTimeTool implementation
std::vector<ParameterDef> GetTimeTool::get_parameters_schema() const {
    return {};  // No parameters
}

std::map<std::string, std::any> GetTimeTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    auto now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%H:%M:%S");

    result["content"] = oss.str();
    result["success"] = true;
    return result;
}

// GetDateTool implementation
std::vector<ParameterDef> GetDateTool::get_parameters_schema() const {
    return {};  // No parameters
}

std::map<std::string, std::any> GetDateTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    auto now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");

    result["content"] = oss.str();
    result["success"] = true;
    return result;
}

// Registration function
void register_core_tools(Tools& tools) {
    // TaskTool disabled - not implemented for stateful backends
    // tools.register_tool(std::make_unique<TaskTool>());
    tools.register_tool(std::make_unique<BashTool>());
    tools.register_tool(std::make_unique<GlobTool>());
    tools.register_tool(std::make_unique<GrepTool>());
    tools.register_tool(std::make_unique<EditTool>());
    tools.register_tool(std::make_unique<WebFetchTool>());
    tools.register_tool(std::make_unique<WebSearchTool>());
    tools.register_tool(std::make_unique<TodoWriteTool>());
    tools.register_tool(std::make_unique<BashOutputTool>());
    tools.register_tool(std::make_unique<KillShellTool>());
    tools.register_tool(std::make_unique<GetTimeTool>());
    tools.register_tool(std::make_unique<GetDateTool>());
}
