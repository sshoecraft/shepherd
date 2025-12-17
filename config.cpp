#include "shepherd.h"
#include "config.h"
#include "provider.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>
#include <algorithm>

// include the system prompt
#include "system_prompt.h"

using json = nlohmann::json;

Config::Config() {
    set_defaults();
}

void Config::set_max_db_size(const std::string& size_str) {
    max_db_size_str = size_str;
    max_db_size = parse_size_string(size_str);
}

void Config::set_defaults() {
	// Default system prompt (loaded from system_prompt.h)
	system_message = SYSTEM_PROMPT;

	// Default warmup message
	warmup_message = "I want you to respond with exactly 'Ready.' and absolutely nothing else one time only at the start.  **IMPORTANT:** DO NOT USE TOOLS!";
	warmup = false;  // Disable warmup by default
	calibration = false;  // Disable calibration by default (slow for thinking models)

    backend = "llamacpp";
    model = "";
    model_path = get_default_model_path();
    context_size = 0; // Auto-detect
    key = "none";
    api_base = "";  // Optional API base URL
    models_file = "";  // Optional models database file path
    system_prompt = "";  // Optional custom system prompt

    // RAG database defaults
    memory_database = "";  // Empty = use default from get_default_memory_db_path()
    max_db_size_str = "10G";
    max_db_size = parse_size_string(max_db_size_str);

    // Web search defaults (disabled by default)
    web_search_provider = "";
    web_search_api_key = "";
    web_search_instance_url = "";

	// Tool truncation limit, in tokens (0 = use 85% of available space)
	truncate_limit = 0;

	// Streaming enabled by default
	streaming = true;

	// Thinking/reasoning blocks hidden by default
	thinking = false;

	// Auto-switch provider on connection failure (disabled by default)
	auto_provider = false;

	// TUI mode disabled by default (classic scrolling terminal)
	tui = false;
	tui_history = 10000;  // TUI scrollback buffer size
}

size_t Config::parse_size_string(const std::string& size_str) {
    if (size_str.empty()) {
        throw ConfigError("Empty size string");
    }

    // Check if it's just a number (backward compatibility)
    bool all_digits = true;
    for (char c : size_str) {
        if (!isdigit(c)) {
            all_digits = false;
            break;
        }
    }
    if (all_digits) {
        return std::stoull(size_str);
    }

    // Parse number + suffix format
    size_t pos = 0;
    while (pos < size_str.length() && (isdigit(size_str[pos]) || size_str[pos] == '.')) {
        pos++;
    }

    if (pos == 0) {
        throw ConfigError("Invalid size string (no number): " + size_str);
    }

    double value = std::stod(size_str.substr(0, pos));
    std::string suffix = size_str.substr(pos);

    // Remove whitespace from suffix
    suffix.erase(std::remove_if(suffix.begin(), suffix.end(), ::isspace), suffix.end());

    // Convert to uppercase for case-insensitive matching
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);

    // Parse suffix
    size_t multiplier = 1;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1;
    } else if (suffix == "K" || suffix == "KB") {
        multiplier = 1024;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1024 * 1024;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1024ULL * 1024 * 1024;
    } else if (suffix == "T" || suffix == "TB") {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
    } else {
        throw ConfigError("Invalid size suffix: " + suffix + " (use K, M, G, T, KB, MB, GB, or TB");

    }

    return static_cast<size_t>(value * multiplier);
}

std::string Config::get_home_directory() {
    // Try HOME environment variable first (respects user's explicit setting)
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        return std::string(home);
    }

    // Fallback to system passwd database
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir) {
        return std::string(pw->pw_dir);
    }

    throw ConfigError("Unable to determine home directory");
}

std::string Config::get_default_config_path() {
    // Use XDG base directory
    std::string config_home;
    const char* xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        config_home = xdg_config;
    } else {
        config_home = get_home_directory() + "/.config";
    }
    return config_home + "/shepherd/config.json";
}

std::string Config::get_default_memory_db_path() {
    // Use XDG data directory
    std::string data_home;
    const char* xdg_data = getenv("XDG_DATA_HOME");
    if (xdg_data && xdg_data[0] != '\0') {
        data_home = xdg_data;
    } else {
        data_home = get_home_directory() + "/.local/share";
    }
    return data_home + "/shepherd/memory.db";
}

std::string Config::get_config_path() const {
    if (!custom_config_path_.empty()) {
        return custom_config_path_;
    }
    return get_default_config_path();
}

std::string Config::get_default_model_path() const {
    // Use XDG data directory
    const char* xdg_data = getenv("XDG_DATA_HOME");
    if (xdg_data && xdg_data[0] != '\0') {
        return std::string(xdg_data) + "/shepherd/models";
    }
    return get_home_directory() + "/.local/share/shepherd/models";
}

void Config::load() {
    std::string config_path = get_config_path();

    dout(1) << "Loading config from: " + config_path << std::endl;

    if (!std::filesystem::exists(config_path)) {
        dout(1) << "Config file not found, using defaults: " + config_path << std::endl;
        return;
    }

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to open config file: " + config_path);
        }

        file >> json;
        dout(1) << "Config file loaded, checking for mcp_servers key..." << std::endl;

        // Load values with fallbacks to defaults
        if (json.contains("backend")) {
            backend = json["backend"];
        }
        if (json.contains("model")) {
            model = json["model"];
        }
        if (json.contains("model_path") || json.contains("path")) {
            model_path = json.contains("model_path") ?
                json["model_path"].get<std::string>() :
                json["path"].get<std::string>();
        }
        if (json.contains("context_size")) {
            context_size = json["context_size"];
        }
        if (json.contains("key")) {
            key = json["key"];
        }
        if (json.contains("api_key")) {
            key = json["api_key"];
        }
        if (json.contains("api_base")) {
            api_base = json["api_base"];
        }
        if (json.contains("models_file")) {
            models_file = json["models_file"];
        }
        if (json.contains("system")) {
            system_prompt = json["system"];
        }
        if (json.contains("mcp_servers")) {
            // Store MCP servers as JSON string for MCPManager to parse
            mcp_config = json["mcp_servers"].dump();
            dout(1) << "Loaded MCP config: " + mcp_config << std::endl;
        } else {
            dout(1) << "No mcp_servers found in config file" << std::endl;
        }

        // Load web search configuration (optional)
        if (json.contains("web_search_provider")) {
            web_search_provider = json["web_search_provider"];
        }
        if (json.contains("web_search_api_key")) {
            web_search_api_key = json["web_search_api_key"];
        }
        if (json.contains("web_search_instance_url")) {
            web_search_instance_url = json["web_search_instance_url"];
        }

        // Load tool result truncation limit
        if (json.contains("truncate_limit")) {
            truncate_limit = json["truncate_limit"].get<int>();
        }

        // Load streaming flag
        if (json.contains("streaming")) {
            streaming = json["streaming"].get<bool>();
        }

        // Load warmup setting
        if (json.contains("warmup")) {
            warmup = json["warmup"].get<bool>();
        }

        // Load calibration setting
        if (json.contains("calibration")) {
            calibration = json["calibration"].get<bool>();
        }

        // Load thinking setting (show/hide reasoning blocks)
        if (json.contains("thinking")) {
            thinking = json["thinking"].get<bool>();
        }

        // Load auto_provider setting (auto-switch on connection failure)
        if (json.contains("auto_provider")) {
            auto_provider = json["auto_provider"].get<bool>();
        }

        // Load TUI mode setting
        if (json.contains("tui")) {
            tui = json["tui"].get<bool>();
        }

        // Load RAG memory database path (optional)
        if (json.contains("memory_database")) {
            memory_database = json["memory_database"];
        }

        // Load RAG database size limit (optional, supports both string and numeric formats)
        if (json.contains("max_db_size")) {
            if (json["max_db_size"].is_string()) {
                max_db_size_str = json["max_db_size"].get<std::string>();
                max_db_size = parse_size_string(max_db_size_str);
            } else if (json["max_db_size"].is_number()) {
                // Backward compatibility: numeric format
                max_db_size = json["max_db_size"].get<size_t>();
                // Convert to string format
                if (max_db_size >= 1024ULL * 1024 * 1024 * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024 * 1024 * 1024)) + "T";
                } else if (max_db_size >= 1024ULL * 1024 * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024 * 1024)) + "G";
                } else if (max_db_size >= 1024ULL * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024)) + "M";
                } else if (max_db_size >= 1024) {
                    max_db_size_str = std::to_string(max_db_size / 1024) + "K";
                } else {
                    max_db_size_str = std::to_string(max_db_size);
                }
            }
        }

        dout(1) << "Loaded configuration from: " + config_path << std::endl;

    } catch (const json::exception& e) {
        throw ConfigError("Invalid JSON in config file: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error loading config: " + std::string(e.what()));
    }
}

void Config::save() const {
    std::string config_path = get_config_path();

    // Create directory if it doesn't exist
    std::filesystem::path dir = std::filesystem::path(config_path).parent_path();
    std::filesystem::create_directories(dir);

    try {
        nlohmann::json save_json = {
            {"warmup", warmup},
            {"calibration", calibration},
            {"streaming", streaming},
            {"thinking", thinking},
            {"auto_provider", auto_provider},
            {"tui", tui},
            {"truncate_limit", truncate_limit},
            {"max_db_size", max_db_size_str}
        };

        // Optional fields
        if (!system_prompt.empty()) {
            save_json["system"] = system_prompt;
        }
        if (!mcp_config.empty()) {
            save_json["mcp_servers"] = nlohmann::json::parse(mcp_config);
        }
        if (!memory_database.empty()) {
            save_json["memory_database"] = memory_database;
        }
        if (!web_search_provider.empty()) {
            save_json["web_search_provider"] = web_search_provider;
        }
        if (!web_search_api_key.empty()) {
            save_json["web_search_api_key"] = web_search_api_key;
        }
        if (!web_search_instance_url.empty()) {
            save_json["web_search_instance_url"] = web_search_instance_url;
        }

        std::ofstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to create config file: " + config_path);
        }

        file << save_json.dump(4) << std::endl;
        dout(1) << "Saved configuration to: " + config_path << std::endl;

    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("Error creating JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error saving config: " + std::string(e.what()));
    }
}

void Config::set_backend(const std::string& backend_name) {
    if (!is_backend_available(backend_name)) {
        auto available = get_available_backends();
        std::string available_str;
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) available_str += ", ";
            available_str += available[i];
        }
        throw ConfigError("Backend '" + backend_name + "' is not available on this platform. Available backends: " + available_str);
    }
    backend = backend_name;
}

bool Config::is_backend_available(const std::string& backend) const {
    auto available = get_available_backends();
    return std::find(available.begin(), available.end(), backend) != available.end();
}

std::vector<std::string> Config::get_available_backends() {
    std::vector<std::string> backends;

    // Backend availability determined by CMake build configuration
#ifdef ENABLE_LLAMACPP
    backends.push_back("llamacpp");
#endif

#ifdef ENABLE_TENSORRT
    backends.push_back("tensorrt");
#endif

#ifdef ENABLE_API_BACKENDS
    backends.push_back("openai");
    backends.push_back("anthropic");
    backends.push_back("gemini");
    backends.push_back("ollama");
#endif
    backends.push_back("cli");

    return backends;
}

void Config::validate() const {
    // Validate global settings only
    // Memory database size
    if (max_db_size < 1024 * 1024) {  // Minimum 1MB
        throw ConfigError("max_db_size must be at least 1MB");
    }

    dout(1) << "Configuration validation passed" << std::endl;
}

// Common config command implementation
int handle_config_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback) {
    Config cfg;
    cfg.load();

    // Help
    if (!args.empty() && (args[0] == "help" || args[0] == "--help" || args[0] == "-h")) {
        callback("Usage: /config [subcommand]\n"
            "Subcommands:\n"
            "  show              - Show all config values\n"
            "  set <key> <value> - Set config value\n"
            "  (no args)         - Show all config values\n"
            "Keys: warmup, calibration, streaming, thinking, tui,\n"
            "      truncate_limit, max_db_size, web_search_provider,\n"
            "      web_search_api_key, web_search_instance_url, memory_database\n");
        return 0;
    }

    // No args or "show" displays configuration
    if (args.empty() || args[0] == "show") {
        callback("=== Shepherd Configuration ===\n");
        callback("warmup = " + std::string(cfg.warmup ? "true" : "false") + "\n");
        callback("calibration = " + std::string(cfg.calibration ? "true" : "false") + "\n");
        callback("streaming = " + std::string(cfg.streaming ? "true" : "false") + "\n");
        callback("thinking = " + std::string(cfg.thinking ? "true" : "false") + "\n");
        callback("tui = " + std::string(cfg.tui ? "true" : "false") + "\n");
        callback("truncate_limit = " + std::to_string(cfg.truncate_limit) + "\n");
        callback("max_db_size = " + cfg.max_db_size_str + "\n");
        callback("web_search_provider = " + cfg.web_search_provider + "\n");
        callback("web_search_api_key = " + std::string(cfg.web_search_api_key.empty() ? "" : "(set)") + "\n");
        callback("web_search_instance_url = " + cfg.web_search_instance_url + "\n");
        callback("memory_database = " + (cfg.memory_database.empty() ? Config::get_default_memory_db_path() : cfg.memory_database) + "\n");
        return 0;
    }

    if (args[0] == "set" && args.size() >= 3) {
        std::string key = args[1];
        std::string value = args[2];

        if (key == "warmup") {
            cfg.warmup = (value == "true" || value == "1" || value == "on");
        } else if (key == "calibration") {
            cfg.calibration = (value == "true" || value == "1" || value == "on");
        } else if (key == "streaming") {
            cfg.streaming = (value == "true" || value == "1" || value == "on");
        } else if (key == "thinking") {
            cfg.thinking = (value == "true" || value == "1" || value == "on");
        } else if (key == "tui") {
            cfg.tui = (value == "true" || value == "1" || value == "on");
        } else if (key == "truncate_limit") {
            cfg.truncate_limit = std::stoi(value);
        } else if (key == "max_db_size") {
            cfg.set_max_db_size(value);
        } else if (key == "web_search_provider") {
            cfg.web_search_provider = value;
        } else if (key == "web_search_api_key") {
            cfg.web_search_api_key = value;
        } else if (key == "web_search_instance_url") {
            cfg.web_search_instance_url = value;
        } else if (key == "memory_database") {
            cfg.memory_database = value;
        } else {
            callback("Unknown config key: " + key + "\n");
            return 1;
        }

        cfg.save();
        callback("Config updated: " + key + " = " + value + "\n");
        return 0;
    }

    callback("Unknown config subcommand: " + args[0] + "\n");
    callback("Available: show, set\n");
    return 1;
}
