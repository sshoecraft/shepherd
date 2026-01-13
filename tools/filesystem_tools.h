#pragma once

#include "tool.h"

class ReadFileTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Read"; }
    std::string description() const override { return "Read files from the filesystem. Text files are returned as-is. Binary files (images, PDFs, etc.) are automatically detected and returned as base64. Use encoding parameter to control: 'auto' (default, detect binary), 'text' (force text), 'base64' (force base64)."; }
    std::string parameters() const override { return "file_path=\"path\", encoding=\"auto|text|base64\" (optional), offset=\"line number\" (text only), limit=\"num lines\" (text only)"; }
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;
};

class WriteFileTool : public Tool {
public:
    std::string unsanitized_name() const override { return "Write"; }
    std::string description() const override { return "Write files to the filesystem. For binary files, pass base64-encoded content with encoding='base64'. Default is text mode."; }
    std::string parameters() const override { return "file_path=\"path\", content=\"file content\", encoding=\"text|base64\" (optional, default: text)"; }
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