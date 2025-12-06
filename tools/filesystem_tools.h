#pragma once

#include "tool.h"

class ReadFileTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Read"; }
    std::string description() const override { return "Read files from the filesystem (supports text, images, PDFs, Jupyter notebooks). Accepts relative or absolute paths."; }
    std::string parameters() const override { return "file_path=\"path\", offset=\"line number to start\" (optional), limit=\"number of lines\" (optional)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class WriteFileTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Write"; }
    std::string description() const override { return "Write new files or overwrite existing files. Accepts relative or absolute paths."; }
    std::string parameters() const override { return "file_path=\"path\", content=\"file content\""; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class ListDirectoryTool : public Tool {
public:
    std::string unsanitized_name() const override { return "list_directory"; }
    std::string description() const override { return "List files and directories (like Unix 'ls' command). Use current directory if no path specified."; }
    std::string parameters() const override { return "path=\"directory_path\" (optional, defaults to current directory)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Alias for list_directory (some models use "List" from their training)
class ListTool : public Tool {
public:
    std::string unsanitized_name() const override { return "List"; }
    std::string description() const override { return "List files and directories (alias for list_directory)"; }
    std::string parameters() const override { return "path=\"directory_path\" (optional, defaults to current directory)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

// Function to register all filesystem tools
class Tools;
void register_filesystem_tools(Tools& tools);