#include "shepherd.h"
#include "filesystem_tools.h"
#include "tools.h"
#include "../include/base64.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <set>

namespace fs = std::filesystem;

// Binary file detection helpers
static bool is_binary_extension(const std::string& ext) {
    static const std::set<std::string> binary_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib", ".bin",
        ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mkv", ".mov",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".db", ".sqlite", ".sqlite3",
        ".pyc", ".class", ".o", ".obj"
    };
    std::string lower_ext = ext;
    std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);
    return binary_exts.count(lower_ext) > 0;
}

static bool has_binary_content(const std::string& content, size_t check_bytes = 8192) {
    size_t to_check = std::min(content.size(), check_bytes);
    for (size_t i = 0; i < to_check; ++i) {
        unsigned char c = static_cast<unsigned char>(content[i]);
        // Null byte or other control characters (except tab, newline, carriage return)
        if (c == 0 || (c < 32 && c != 9 && c != 10 && c != 13)) {
            return true;
        }
    }
    return false;
}

std::vector<ParameterDef> ReadFileTool::get_parameters_schema() const {
    return {
        {"file_path", "string", "path to the file to be read", true, "", "", {}},
        {"offset", "number", "Optional line number to start reading from (1-indexed, text mode only)", false, "1", "", {}},
        {"limit", "number", "Optional number of lines to read (-1 for unlimited, text mode only)", false, "-1", "", {}},
        {"encoding", "string", "How to read the file: 'auto' (detect binary and use base64), 'text' (force text, fail on binary), 'base64' (force base64 encoding). Values: auto, text, base64", false, "auto", "", {}}
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

    // Get encoding parameter (default: auto)
    std::string encoding = tool_utils::get_string(args, "encoding", "auto");
    if (encoding != "auto" && encoding != "text" && encoding != "base64") {
        encoding = "auto";
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

        // Determine if file should be read as binary
        bool use_base64 = false;
        std::string file_ext = abs_path.extension().string();

        if (encoding == "base64") {
            use_base64 = true;
        } else if (encoding == "auto") {
            // Check extension first
            if (is_binary_extension(file_ext)) {
                use_base64 = true;
            }
        }
        // encoding == "text" means force text mode

        if (use_base64) {
            // Binary mode: read entire file and base64 encode
            std::ifstream file(abs_path, std::ios::binary);
            if (!file.is_open()) {
                result["error"] = std::string("error opening file: ") + path;
                result["success"] = false;
                return result;
            }

            // Read entire file into string
            std::ostringstream content_stream;
            content_stream << file.rdbuf();
            std::string raw_content = content_stream.str();

            // Base64 encode
            std::string encoded = base64::encode(raw_content);

            result["content"] = encoded;
            result["encoding"] = std::string("base64");
            result["path"] = abs_path.string();
            result["success"] = true;
            result["summary"] = std::string("Read ") + std::to_string(raw_content.size()) +
                               " bytes (base64 encoded: " + std::to_string(encoded.size()) + " chars)";

            dout(1) << "Read: Read " + std::to_string(raw_content.size()) + " bytes (base64) from " + abs_path.string() << std::endl;
        } else {
            // Text mode: read with line handling
            // Get optional line range parameters
            int start_line = 1;
            int limit = -1;

            auto offset_it = args.find("offset");
            if (offset_it != args.end()) {
                try {
                    start_line = std::any_cast<int>(offset_it->second);
                } catch (...) {
                    try {
                        start_line = std::stoi(std::any_cast<std::string>(offset_it->second));
                    } catch (...) {}
                }
            }

            auto limit_it = args.find("limit");
            if (limit_it != args.end()) {
                try {
                    limit = std::any_cast<int>(limit_it->second);
                } catch (...) {
                    try {
                        limit = std::stoi(std::any_cast<std::string>(limit_it->second));
                    } catch (...) {}
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

            while (std::getline(file, line)) {
                if (current_line >= start_line) {
                    content_stream << line << "\n";
                    lines_read++;

                    if (limit > 0 && lines_read >= limit) {
                        break;
                    }
                }
                current_line++;
            }

            std::string content = content_stream.str();

            // In auto mode, check if content has binary data
            if (encoding == "auto" && has_binary_content(content)) {
                // Re-read as binary and base64 encode
                file.close();
                std::ifstream bin_file(abs_path, std::ios::binary);
                std::ostringstream bin_stream;
                bin_stream << bin_file.rdbuf();
                std::string raw_content = bin_stream.str();
                std::string encoded = base64::encode(raw_content);

                result["content"] = encoded;
                result["encoding"] = std::string("base64");
                result["path"] = abs_path.string();
                result["success"] = true;
                result["summary"] = std::string("Read ") + std::to_string(raw_content.size()) +
                                   " bytes (binary detected, base64 encoded)";

                dout(1) << "Read: Binary content detected, base64 encoded " + abs_path.string() << std::endl;
                return result;
            }

            // Text mode with potential binary check failure
            if (encoding == "text" && has_binary_content(content)) {
                result["error"] = std::string("File contains binary data. Use encoding='base64' or encoding='auto' to read binary files.");
                result["success"] = false;
                return result;
            }

            result["content"] = content;
            result["encoding"] = std::string("text");
            result["path"] = abs_path.string();
            result["success"] = true;

            std::string summary;
            if (start_line > 1) {
                summary = "Read " + std::to_string(lines_read) + " line" + (lines_read != 1 ? "s" : "") +
                          " from offset " + std::to_string(start_line);
            } else {
                summary = "Read " + std::to_string(lines_read) + " line" + (lines_read != 1 ? "s" : "");
            }
            result["summary"] = summary;

            dout(1) << "Read: Successfully read " + abs_path.string() << std::endl;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error reading file: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> WriteFileTool::get_parameters_schema() const {
    return {
        {"file_path", "string", "path to the file to be written", true, "", "", {}},
        {"content", "string", "The content to write to the file", true, "", "", {}},
        {"encoding", "string", "How to write the file: 'text' (write content as-is), 'base64' (decode content from base64 and write as binary). Values: text, base64", false, "text", "", {}}
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

    // Get encoding parameter (default: text)
    std::string encoding = tool_utils::get_string(args, "encoding", "text");
    if (encoding != "text" && encoding != "base64") {
        encoding = "text";
    }

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

        if (encoding == "base64") {
            // Decode base64 and write as binary
            std::string decoded;
            try {
                decoded = base64::decode(content);
            } catch (const base64_error& e) {
                result["error"] = std::string("Invalid base64 content: ") + e.what();
                result["success"] = false;
                return result;
            }

            std::ofstream file(abs_path, std::ios::binary);
            if (!file.is_open()) {
                result["error"] = std::string("error creating file: ") + path;
                result["success"] = false;
                return result;
            }

            file.write(decoded.data(), decoded.size());
            file.close();

            result["status"] = std::string("success");
            result["path"] = abs_path.string();
            result["encoding"] = std::string("base64");
            result["summary"] = std::string("Wrote ") + std::to_string(decoded.size()) + " bytes (decoded from base64) to " + abs_path.filename().string();
            result["success"] = true;

            dout(1) << "Write: Successfully wrote " + std::to_string(decoded.size()) + " bytes (base64 decoded) to " + abs_path.string() << std::endl;
        } else {
            // Write as text
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
            result["encoding"] = std::string("text");
            result["summary"] = std::string("Wrote ") + std::to_string(content.length()) + " bytes to " + abs_path.filename().string();
            result["success"] = true;

            dout(1) << "Write: Successfully wrote " + std::to_string(content.length()) + " bytes to " + abs_path.string() << std::endl;
        }

    } catch (const std::exception& e) {
        result["error"] = std::string("error writing file: ") + e.what();
        result["success"] = false;
    }

    return result;
}

std::vector<ParameterDef> ListDirectoryTool::get_parameters_schema() const {
    return {
        {"path", "string", "The directory path to list (defaults to current directory)", false, ".", "", {}}
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
        result["summary"] = std::string("Listed ") + std::to_string(files.size()) + " item" + (files.size() != 1 ? "s" : "");
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
        {"path", "string", "The directory path to list (defaults to current directory)", false, ".", "", {}}
    };
}

std::map<std::string, std::any> ListTool::execute(const std::map<std::string, std::any>& args) {
    // Just delegate to ListDirectoryTool
    ListDirectoryTool list_dir;
    return list_dir.execute(args);
}

void register_filesystem_tools(Tools& tools) {
    tools.register_tool(std::make_unique<ReadFileTool>());
    tools.register_tool(std::make_unique<WriteFileTool>());
    tools.register_tool(std::make_unique<ListDirectoryTool>());
    tools.register_tool(std::make_unique<ListTool>());  // Register alias

    dout(1) << "Registered filesystem tools: Read, Write, list_directory, List" << std::endl;
}
