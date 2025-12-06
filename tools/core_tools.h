#pragma once

#include "tool.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

// Task - Launch a specialized agent to handle complex tasks
class TaskTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Task"; }
    std::string description() const override { return "Launch a specialized agent to handle complex, multi-step tasks autonomously"; }
    std::string parameters() const override { return "prompt=\"detailed task description\", description=\"short task summary\", subagent_type=\"general-purpose|statusline-setup|output-style-setup\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Bash - Execute bash commands in a persistent shell with optional timeout
class BashTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Bash"; }
    std::string description() const override { return "Execute bash commands in a persistent shell session with optional timeout"; }
    std::string parameters() const override { return "command=\"bash command\", description=\"what command does\", timeout=120000 (ms), run_in_background=false"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Glob - Fast file pattern matching
class GlobTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Glob"; }
    std::string description() const override { return "Fast file pattern matching tool supporting glob patterns like \"**/*.js\""; }
    std::string parameters() const override { return "pattern=\"glob pattern\", path=\"directory to search\" (optional)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Grep - Powerful search tool with regex support
class GrepTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Grep"; }
    std::string description() const override { return "Powerful search tool with regex support for finding content in files"; }
    std::string parameters() const override { return "pattern=\"regex pattern\", path=\"file or directory\", output_mode=\"content|files_with_matches|count\", case_insensitive=false, line_numbers=false, context_after=0, context_before=0, glob=\"*.js\", type=\"js|py|rust\", multiline=false, head_limit=N"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Edit - Perform exact string replacements in files
class EditTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Edit"; }
    std::string description() const override { return "Perform exact string replacements in files"; }
    std::string parameters() const override { return "file_path=\"absolute path\", old_string=\"text to replace\", new_string=\"replacement text\", replace_all=false"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// WebFetch - Fetch and process webpage content using AI analysis
class WebFetchTool : public Tool {
public:
    std::string unsanitized_name() const override { return "WebFetch"; }
    std::string description() const override { return "Fetch and process webpage content"; }
    std::string parameters() const override { return "url=\"webpage URL\", prompt=\"what to extract from page\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// WebSearch - Search the web and retrieve results
class WebSearchTool : public Tool {
public:
    std::string unsanitized_name() const override { return "WebSearch"; }
    std::string description() const override { return "Search the web and retrieve results"; }
    std::string parameters() const override { return "query=\"search query\", allowed_domains=[\"domain.com\"], blocked_domains=[\"domain.com\"]"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// TodoWrite - Create and manage structured task lists
class TodoWriteTool : public Tool {
public:
    std::string unsanitized_name() const override { return "TodoWrite"; }
    std::string description() const override { return "Create and manage structured task lists"; }
    std::string parameters() const override { return "todos=[{content=\"task description\", status=\"pending|in_progress|completed\", activeForm=\"doing task...\"}]"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// BashOutput - Retrieve output from running background bash shells
class BashOutputTool : public Tool {
public:
    std::string unsanitized_name() const override { return "BashOutput"; }
    std::string description() const override { return "Retrieve output from running background bash shells"; }
    std::string parameters() const override { return "bash_id=\"shell identifier\", filter=\"regex pattern\" (optional)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// KillShell - Terminate a running background bash shell
class KillShellTool : public Tool {
public:
    std::string unsanitized_name() const override { return "KillShell"; }
    std::string description() const override { return "Terminate a running background bash shell"; }
    std::string parameters() const override { return "shell_id=\"shell identifier\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// GetTime - Get the current local time
class GetTimeTool : public Tool {
public:
    std::string unsanitized_name() const override { return "get_time"; }
    std::string description() const override { return "Get the current local time"; }
    std::string parameters() const override { return "(no parameters)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// GetDate - Get the current local date
class GetDateTool : public Tool {
public:
    std::string unsanitized_name() const override { return "get_date"; }
    std::string description() const override { return "Get the current local date"; }
    std::string parameters() const override { return "(no parameters)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Background shell manager for Bash tool
struct BackgroundShell {
    std::string id;
    std::string command;
    FILE* pipe;
    std::thread reader_thread;
    std::string output;
    std::mutex output_mutex;
    std::atomic<bool> running;
    int exit_code;
};

class BackgroundShellManager {
public:
    static BackgroundShellManager& instance();

    std::string start_shell(const std::string& command);
    std::string get_output(const std::string& shell_id, const std::string& filter = "");
    bool kill_shell(const std::string& shell_id);
    std::vector<std::string> list_shells() const;

private:
    BackgroundShellManager() = default;
    std::map<std::string, std::unique_ptr<BackgroundShell>> shells_;
    std::mutex shells_mutex_;
    int next_id_ = 1;
};

// Forward declaration
class Tools;

// Function to register all core tools
void register_core_tools(Tools& tools);
