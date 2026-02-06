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
	single_query_mode = false;  // Set to true when --prompt is specified

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

	// Max generation tokens: -1=max, 0=auto, >0=explicit
	max_tokens = 0;

	// Streaming enabled by default
	streaming = true;

	// Thinking/reasoning blocks hidden by default
	thinking = false;

	// Performance stats hidden by default
	stats = false;

	// Auto-switch provider on connection failure (disabled by default)
	auto_provider = false;

	// Raw output mode (disable channel parsing, like vLLM)
	raw_output = false;

	// TUI mode disabled by default (classic scrolling terminal)
	tui = false;
	tui_history = 10000;  // TUI scrollback buffer size

	// Server settings
	auth_mode = "none";
	server_tools = false;

	// Scheduler defaults
	scheduler_name = "default";
	schedulers_json = nlohmann::json::array();
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
        load_from_json(json);

        dout(1) << "Loaded configuration from: " + config_path << std::endl;

        // Migrate standalone schedule.json into config if no schedulers exist yet
        if (schedulers_json.empty()) {
            std::string schedule_path = std::filesystem::path(config_path).parent_path().string() + "/schedule.json";
            if (std::filesystem::exists(schedule_path)) {
                try {
                    std::ifstream sched_file(schedule_path);
                    nlohmann::json sched_json;
                    sched_file >> sched_json;

                    if (sched_json.contains("schedules") && sched_json["schedules"].is_array() && !sched_json["schedules"].empty()) {
                        nlohmann::json default_scheduler;
                        default_scheduler["name"] = "default";
                        default_scheduler["schedules"] = sched_json["schedules"];
                        schedulers_json.push_back(default_scheduler);

                        dout(1) << "Migrated " << sched_json["schedules"].size()
                                << " schedules from schedule.json to config.json" << std::endl;
                        save();
                        std::filesystem::rename(schedule_path, schedule_path + ".migrated");
                        dout(1) << "Renamed schedule.json to schedule.json.migrated" << std::endl;
                    }
                } catch (const std::exception& e) {
                    dout(1) << "Failed to migrate schedule.json: " << e.what() << std::endl;
                }
            }
        }

    } catch (const json::exception& e) {
        throw ConfigError("Invalid JSON in config file: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error loading config: " + std::string(e.what()));
    }
}

void Config::load_from_json_string(const std::string& json_str) {
    try {
        json = nlohmann::json::parse(json_str);
        load_from_json(json);
        dout(1) << "Loaded configuration from JSON string" << std::endl;
    } catch (const json::exception& e) {
        throw ConfigError("Invalid JSON: " + std::string(e.what()));
    }
}

void Config::load_from_json(const nlohmann::json& j) {
    // Load values with fallbacks to defaults
    if (j.contains("backend")) {
        backend = j["backend"];
    }
    if (j.contains("model")) {
        model = j["model"];
    }
    if (j.contains("model_path") || j.contains("path")) {
        model_path = j.contains("model_path") ?
            j["model_path"].get<std::string>() :
            j["path"].get<std::string>();
    }
    if (j.contains("context_size")) {
        context_size = j["context_size"];
    }
    if (j.contains("key")) {
        key = j["key"];
    }
    if (j.contains("api_key")) {
        key = j["api_key"];
    }
    if (j.contains("api_base")) {
        api_base = j["api_base"];
    }
    if (j.contains("models_file")) {
        models_file = j["models_file"];
    }
    if (j.contains("system")) {
        system_prompt = j["system"];
    }
    if (j.contains("mcp_servers")) {
        // Store MCP servers as JSON string for MCPManager to parse
        mcp_config = j["mcp_servers"].dump();
        dout(1) << "Loaded MCP config: " + mcp_config << std::endl;
    } else {
        dout(1) << "No mcp_servers found in config" << std::endl;
    }

    // Load web search configuration (optional)
    if (j.contains("web_search_provider")) {
        web_search_provider = j["web_search_provider"];
    }
    if (j.contains("web_search_api_key")) {
        web_search_api_key = j["web_search_api_key"];
    }
    if (j.contains("web_search_instance_url")) {
        web_search_instance_url = j["web_search_instance_url"];
    }

    // Load tool result truncation limit
    if (j.contains("truncate_limit")) {
        truncate_limit = j["truncate_limit"].get<int>();
    }

    // Load streaming flag
    if (j.contains("streaming")) {
        streaming = j["streaming"].get<bool>();
    }

    // Load warmup setting
    if (j.contains("warmup")) {
        warmup = j["warmup"].get<bool>();
    }

    // Load calibration setting
    if (j.contains("calibration")) {
        calibration = j["calibration"].get<bool>();
    }

    // Load thinking setting (show/hide reasoning blocks)
    if (j.contains("thinking")) {
        thinking = j["thinking"].get<bool>();
    }

    // Load auto_provider setting (auto-switch on connection failure)
    if (j.contains("auto_provider")) {
        auto_provider = j["auto_provider"].get<bool>();
    }

    // Load TUI mode setting
    if (j.contains("tui")) {
        tui = j["tui"].get<bool>();
    }

    // Load server settings
    if (j.contains("auth_mode")) {
        auth_mode = j["auth_mode"].get<std::string>();
    }
    if (j.contains("server_tools")) {
        server_tools = j["server_tools"].get<bool>();
    }

    // Load RAG memory database path (optional)
    if (j.contains("memory_database")) {
        memory_database = j["memory_database"];
    }

    // Load RAG database size limit (optional, supports both string and numeric formats)
    if (j.contains("max_db_size")) {
        if (j["max_db_size"].is_string()) {
            max_db_size_str = j["max_db_size"].get<std::string>();
            max_db_size = parse_size_string(max_db_size_str);
        } else if (j["max_db_size"].is_number()) {
            // Backward compatibility: numeric format
            max_db_size = j["max_db_size"].get<size_t>();
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

    // Load providers from unified config
    if (j.contains("providers") && j["providers"].is_array()) {
        providers_json.clear();
        for (const auto& p : j["providers"]) {
            providers_json.push_back(p);
        }
        dout(1) << "Loaded " << providers_json.size() << " providers from config" << std::endl;
    }

    // Load SMCP servers (store as JSON string like mcp_config)
    if (j.contains("smcp_servers") && j["smcp_servers"].is_array()) {
        smcp_config = j["smcp_servers"].dump();
        dout(1) << "Loaded SMCP config: " << smcp_config << std::endl;
    }

    // Load named schedulers
    if (j.contains("schedulers") && j["schedulers"].is_array()) {
        schedulers_json = j["schedulers"];
        dout(1) << "Loaded " << schedulers_json.size() << " scheduler(s) from config" << std::endl;
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
        if (!providers_json.empty()) {
            save_json["providers"] = providers_json;
        }
        if (!smcp_config.empty()) {
            save_json["smcp_servers"] = nlohmann::json::parse(smcp_config);
        }
        if (!schedulers_json.empty()) {
            save_json["schedulers"] = schedulers_json;
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

// Helper to get config value by key
static std::string get_config_value(const Config& cfg, const std::string& key) {
    if (key == "warmup") return cfg.warmup ? "true" : "false";
    if (key == "calibration") return cfg.calibration ? "true" : "false";
    if (key == "streaming") return cfg.streaming ? "true" : "false";
    if (key == "thinking") return cfg.thinking ? "true" : "false";
    if (key == "tui") return cfg.tui ? "true" : "false";
    if (key == "truncate_limit") return std::to_string(cfg.truncate_limit);
    if (key == "max_db_size") return cfg.max_db_size_str;
    if (key == "web_search_provider") return cfg.web_search_provider;
    if (key == "web_search_api_key") return cfg.web_search_api_key.empty() ? "" : "(set)";
    if (key == "web_search_instance_url") return cfg.web_search_instance_url;
    if (key == "memory_database") return cfg.memory_database.empty() ? Config::get_default_memory_db_path() : cfg.memory_database;
    if (key == "system_prompt") return cfg.system_prompt.empty() ? "(default)" : cfg.system_prompt;
    if (key == "auto_provider") return cfg.auto_provider ? "true" : "false";
    if (key == "auth_mode") return cfg.auth_mode;
    if (key == "server_tools") return cfg.server_tools ? "true" : "false";
    return "";
}

// Helper to set config value by key
static bool set_config_value(Config& cfg, const std::string& key, const std::string& value) {
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
    } else if (key == "system_prompt") {
        cfg.system_prompt = value;
    } else if (key == "auto_provider") {
        cfg.auto_provider = (value == "true" || value == "1" || value == "on");
    } else if (key == "auth_mode") {
        cfg.auth_mode = value;
    } else if (key == "server_tools") {
        cfg.server_tools = (value == "true" || value == "1" || value == "on");
    } else {
        return false;
    }
    return true;
}

// Valid config keys
static const std::vector<std::string> CONFIG_KEYS = {
    "warmup", "calibration", "streaming", "thinking", "tui",
    "truncate_limit", "max_db_size", "web_search_provider",
    "web_search_api_key", "web_search_instance_url", "memory_database",
    "system_prompt", "auto_provider", "auth_mode", "server_tools"
};

static bool is_config_key(const std::string& s) {
    return std::find(CONFIG_KEYS.begin(), CONFIG_KEYS.end(), s) != CONFIG_KEYS.end();
}

// Common config command implementation
int handle_config_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback) {
    Config cfg;
    cfg.load();

    // Help
    if (!args.empty() && (args[0] == "help" || args[0] == "--help" || args[0] == "-h")) {
        callback("Usage: shepherd config [key] [value]\n"
            "\nExamples:\n"
            "  shepherd config              - Show all config values\n"
            "  shepherd config show         - Show all config values\n"
            "  shepherd config streaming    - Show streaming value\n"
            "  shepherd config streaming true - Set streaming to true\n"
            "\nKeys: warmup, calibration, streaming, thinking, tui,\n"
            "      truncate_limit, max_db_size, web_search_provider,\n"
            "      web_search_api_key, web_search_instance_url, memory_database\n");
        return 0;
    }

    // No args or "show" displays all configuration
    if (args.empty() || args[0] == "show") {
        callback("=== Shepherd Configuration ===\n");
        for (const auto& key : CONFIG_KEYS) {
            callback(key + " = " + get_config_value(cfg, key) + "\n");
        }
        return 0;
    }

    // Check if first arg is a config key
    std::string key = args[0];
    if (is_config_key(key)) {
        if (args.size() == 1) {
            // Show single key value
            callback(key + " = " + get_config_value(cfg, key) + "\n");
            return 0;
        } else {
            // Set key value: config KEY VALUE
            if (config && config->is_read_only()) {
                callback("Error: Cannot modify config in read-only mode (config from Key Vault)\n");
                return 1;
            }

            std::string value = args[1];
            if (set_config_value(cfg, key, value)) {
                cfg.save();
                callback("Config updated: " + key + " = " + value + "\n");
                return 0;
            }
        }
    }

    // Legacy: config set KEY VALUE
    if (args[0] == "set" && args.size() >= 3) {
        if (config && config->is_read_only()) {
            callback("Error: Cannot modify config in read-only mode (config from Key Vault)\n");
            return 1;
        }

        key = args[1];
        std::string value = args[2];

        if (set_config_value(cfg, key, value)) {
            cfg.save();
            callback("Config updated: " + key + " = " + value + "\n");
            return 0;
        } else {
            callback("Unknown config key: " + key + "\n");
            return 1;
        }
    }

    callback("Unknown config key or command: " + args[0] + "\n");
    callback("Use 'shepherd config help' to see available keys\n");
    return 1;
}
