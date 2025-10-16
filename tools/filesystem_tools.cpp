#include "filesystem_tools.h"
#include "../logger.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

std::vector<ParameterDef> ReadFileTool::get_parameters_schema() const {
    return {
        {"file_path", "string", "The absolute path to the file to read", true, ""},
        {"offset", "number", "Optional line number to start reading from (1-indexed)", false, "1"},
        {"limit", "number", "Optional number of lines to read (-1 for unlimited)", false, "-1"}
    };
}

std::map<std::string, std::any> ReadFileTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    // Accept both "file_path" (Claude Code) and "path" (legacy) parameter names
    std::string path = tool_utils::get_string(args, "file_path");
    if (path.empty()) {
        path = tool_utils::get_string(args, "path");
    }

    if (path.empty()) {
        result["error"] = std::string("file_path is required");
        result["success"] = false;
        return result;
    }

    try {
        // Resolve absolute path
        fs::path abs_path = fs::absolute(path);

        // Check if file exists
        if (!fs::exists(abs_path)) {
            result["error"] = std::string("file not found: ") + path;
            result["success"] = false;
            return result;
        }

        if (fs::is_directory(abs_path)) {
            result["error"] = std::string("path is a directory, not a file");
            result["success"] = false;
            return result;
        }

        // Get optional line range parameters
        // Accept both Claude Code style (offset, limit) and legacy style (start_line, end_line)
        int start_line = 1;  // Default: start from line 1
        int limit = -1;      // Default: read to end of file (-1 means unlimited)

        // Try "offset" first (Claude Code), then "start_line" (legacy)
        auto offset_it = args.find("offset");
        if (offset_it != args.end()) {
            try {
                start_line = std::any_cast<int>(offset_it->second);
            } catch (...) {
                try {
                    start_line = std::stoi(std::any_cast<std::string>(offset_it->second));
                } catch (...) {}
            }
        } else {
            auto start_it = args.find("start_line");
            if (start_it != args.end()) {
                try {
                    start_line = std::any_cast<int>(start_it->second);
                } catch (...) {
                    try {
                        start_line = std::stoi(std::any_cast<std::string>(start_it->second));
                    } catch (...) {}
                }
            }
        }

        // Try "limit" first (Claude Code), then "end_line" (legacy)
        auto limit_it = args.find("limit");
        if (limit_it != args.end()) {
            try {
                limit = std::any_cast<int>(limit_it->second);
            } catch (...) {
                try {
                    limit = std::stoi(std::any_cast<std::string>(limit_it->second));
                } catch (...) {}
            }
        } else {
            auto end_it = args.find("end_line");
            if (end_it != args.end()) {
                try {
                    int end_line = std::any_cast<int>(end_it->second);
                    // Convert end_line to limit
                    if (end_line > 0) {
                        limit = end_line - start_line + 1;
                    } else {
                        limit = -1; // Read to end
                    }
                } catch (...) {
                    try {
                        int end_line = std::stoi(std::any_cast<std::string>(end_it->second));
                        if (end_line > 0) {
                            limit = end_line - start_line + 1;
                        } else {
                            limit = -1;
                        }
                    } catch (...) {}
                }
            }
        }

        // Read file content
        std::ifstream file(abs_path);
        if (!file.is_open()) {
            result["error"] = std::string("error opening file: ") + path;
            result["success"] = false;
            return result;
        }

        std::ostringstream content_stream;
        std::string line;
        int current_line = 1;
        int lines_read = 0;

        // Default limits: 2000 lines max, 2000 chars per line max
        const int DEFAULT_MAX_LINES = 2000;
        const int MAX_LINE_LENGTH = 2000;
        int effective_limit = (limit > 0) ? limit : DEFAULT_MAX_LINES;

        while (std::getline(file, line)) {
            // Check if we're at or past the start line
            if (current_line >= start_line) {
                // Truncate long lines
                if (line.length() > MAX_LINE_LENGTH) {
                    content_stream << line.substr(0, MAX_LINE_LENGTH) << " [... line truncated]\n";
                } else {
                    content_stream << line << "\n";
                }
                lines_read++;

                // Stop if we've read the requested number of lines
                if (lines_read >= effective_limit) {
                    if (limit < 0) {
                        // Hit default limit, inform user
                        content_stream << "\n[... output truncated after " << DEFAULT_MAX_LINES << " lines. Use offset and limit parameters to read more]\n";
                    }
                    break;
                }
            }

            current_line++;
        }

        result["content"] = content_stream.str();
        result["path"] = abs_path.string();
        result["success"] = true;

        if (start_line > 1 || limit > 0) {
            LOG_DEBUG("Read: Read " + std::to_string(lines_read) + " lines starting from line " +
                      std::to_string(start_line) + " from " + abs_path.string());
        } else {
            LOG_DEBUG("Read: Successfully read " + abs_path.string());
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error reading file: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> WriteFileTool::get_parameters_schema() const {
    return {
        {"file_path", "string", "The absolute path to the file to write", true, ""},
        {"content", "string", "The content to write to the file", true, ""}
    };
}

std::map<std::string, std::any> WriteFileTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    // Accept both "file_path" (Claude Code) and "path" (legacy) parameter names
    std::string path = tool_utils::get_string(args, "file_path");
    if (path.empty()) {
        path = tool_utils::get_string(args, "path");
    }

    std::string content = tool_utils::get_string(args, "content");

    if (path.empty()) {
        result["error"] = std::string("file_path is required");
        result["success"] = false;
        return result;
    }

    try {
        // Resolve absolute path
        fs::path abs_path = fs::absolute(path);

        // Create parent directories if they don't exist
        fs::path parent_dir = abs_path.parent_path();
        if (!parent_dir.empty()) {
            fs::create_directories(parent_dir);
        }

        // Write file content
        std::ofstream file(abs_path);
        if (!file.is_open()) {
            result["error"] = std::string("error creating file: ") + path;
            result["success"] = false;
            return result;
        }

        file << content;
        file.close();

        result["status"] = std::string("success");
        result["path"] = abs_path.string();
        result["success"] = true;

        LOG_DEBUG("Write: Successfully wrote " + std::to_string(content.length()) + " bytes to " + abs_path.string());

    } catch (const std::exception& e) {
        result["error"] = std::string("error writing file: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> ListDirectoryTool::get_parameters_schema() const {
    return {
        {"path", "string", "The directory path to list (defaults to current directory)", false, "."}
    };
}

std::map<std::string, std::any> ListDirectoryTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string path = tool_utils::get_string(args, "path", ".");

    try {
        // Resolve absolute path
        fs::path abs_path = fs::absolute(path);

        // Check if directory exists
        if (!fs::exists(abs_path)) {
            result["error"] = std::string("directory not found: ") + path;
            result["success"] = false;
            return result;
        }

        if (!fs::is_directory(abs_path)) {
            result["error"] = std::string("path is not a directory");
            result["success"] = false;
            return result;
        }

        // List directory contents
        std::vector<std::map<std::string, std::any>> files;

        for (const auto& entry : fs::directory_iterator(abs_path)) {
            std::map<std::string, std::any> file_info;

            file_info["name"] = entry.path().filename().string();

            if (entry.is_directory()) {
                file_info["type"] = std::string("directory");
            } else if (entry.is_symlink()) {
                file_info["type"] = std::string("symlink");
            } else {
                file_info["type"] = std::string("file");
            }

            try {
                file_info["size"] = static_cast<int64_t>(entry.file_size());
            } catch (const std::exception&) {
                file_info["size"] = static_cast<int64_t>(0);
            }

            files.push_back(file_info);
        }

        result["path"] = abs_path.string();
        result["files"] = files;

        // Format the output as a readable string
        std::ostringstream content;
        content << "Here are the files in " << abs_path.string() << ":\n\n";
        for (const auto& file_info : files) {
            auto name_it = file_info.find("name");
            auto type_it = file_info.find("type");
            if (name_it != file_info.end() && type_it != file_info.end()) {
                try {
                    std::string name = std::any_cast<std::string>(name_it->second);
                    std::string type = std::any_cast<std::string>(type_it->second);
                    content << "- " << name << " (" << type << ")\n";
                } catch (const std::exception&) {
                    // Skip malformed entries
                }
            }
        }

        result["content"] = content.str();
        result["success"] = true;

    } catch (const std::exception& e) {
        result["error"] = std::string("error listing directory: ") + e.what();
        result["success"] = false;
    }

    return result;
}

// ListTool - alias for ListDirectoryTool
std::vector<ParameterDef> ListTool::get_parameters_schema() const {
    return {
        {"path", "string", "The directory path to list (defaults to current directory)", false, "."}
    };
}

std::map<std::string, std::any> ListTool::execute(const std::map<std::string, std::any>& args) {
    // Just delegate to ListDirectoryTool
    ListDirectoryTool list_dir;
    return list_dir.execute(args);
}

void register_filesystem_tools() {
    auto& registry = ToolRegistry::instance();

    registry.register_tool(std::make_unique<ReadFileTool>());
    registry.register_tool(std::make_unique<WriteFileTool>());
    registry.register_tool(std::make_unique<ListDirectoryTool>());
    registry.register_tool(std::make_unique<ListTool>());  // Register alias

    LOG_DEBUG("Registered filesystem tools: Read, Write, list_directory, List");
}