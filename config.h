#pragma once

#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include "nlohmann/json.hpp"

class ConfigError : public std::runtime_error {
public:
    explicit ConfigError(const std::string& message) : std::runtime_error(message) {}
};

class Config {
public:
    Config();

    // Load configuration from $HOME/.shepherd/config.json
    void load();

    // Save current configuration
    void save() const;

    // Validation
    void validate() const;

    // Get available backends for current platform
    static std::vector<std::string> get_available_backends();

    // Get user's home directory (tries getpwuid first, then HOME env var)
    static std::string get_home_directory();

    // Get default config path (XDG-compliant)
    static std::string get_default_config_path();

    // Get default memory database path (XDG-compliant)
    static std::string get_default_memory_db_path();


    // Set custom config file path (for command-line override)
    void set_config_path(const std::string& config_path) { custom_config_path_ = config_path; }

    // Set backend with validation
    void set_backend(const std::string& backend);

    // Set max DB size with parsing
    void set_max_db_size(const std::string& max_db_size_str);

    // Public configuration variables
    std::string system_message;
    std::string warmup_message;
    bool warmup;
    bool calibration;
    std::string system_prompt;
    std::string mcp_config;
    std::string memory_database;
    std::string max_db_size_str;
    size_t max_db_size;
    std::string web_search_provider;
    std::string web_search_api_key;
    std::string web_search_instance_url;
    int truncate_limit;
    bool streaming;
    nlohmann::json json;  // Parsed config JSON for backend-specific settings

    // Legacy/runtime fields (not saved to config, only used for command-line overrides)
    std::string backend;
    std::string model;
    std::string model_path;
    size_t context_size;
    std::string key;
    std::string api_base;
    std::string models_file;

private:
    // Internal helpers
    static size_t parse_size_string(const std::string& size_str);
    std::string get_config_path() const;
    std::string get_default_model_path() const;
    void set_defaults();
    bool is_backend_available(const std::string& backend) const;

    // Internal state
    std::string custom_config_path_;  // Custom config file path (optional)
};
