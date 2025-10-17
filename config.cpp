#include "config.h"
#include "logger.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>

using json = nlohmann::json;

Config::Config() {
    set_defaults();
}

void Config::set_defaults() {
    backend_ = "llamacpp";
    model_ = "";
    model_path_ = get_default_model_path();
    context_size_ = 0; // Auto-detect
    key_ = "none";
    api_base_ = "";  // Optional API base URL
    system_prompt_ = "";  // Optional custom system prompt

    // RAG database size limit (10 GB default)
    max_db_size_str_ = "10G";
    max_db_size_ = parse_size_string(max_db_size_str_);

    // Web search defaults (disabled by default)
    web_search_provider_ = "";
    web_search_api_key_ = "";
    web_search_instance_url_ = "";

    // Sampling parameters (llama.cpp defaults)
    temperature_ = 0.7f;
    top_p_ = 0.95f;
    top_k_ = 40;
    min_keep_ = 1;

    // Repetition penalty parameters (discourage repetitive behavior)
    penalty_repeat_ = 1.1f;   // Slight repetition penalty
    penalty_freq_ = 0.1f;     // Small frequency penalty
    penalty_present_ = 0.0f;  // No presence penalty
    penalty_last_n_ = 64;     // Apply penalty to last 64 tokens

    // GPU layers (for llamacpp backend with GPU support)
    gpu_layers_ = -1;  // -1 = auto (load all layers to GPU), 0 = CPU only, >0 = specific number
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
        throw ConfigError("Invalid size suffix: " + suffix + " (use K, M, G, T, KB, MB, GB, or TB)");
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

std::string Config::get_config_path() const {
    if (!custom_config_path_.empty()) {
        return custom_config_path_;
    }
    return get_home_directory() + "/.shepherd/config.json";
}

std::string Config::get_default_model_path() const {
    return get_home_directory() + "/.shepherd/models";
}

void Config::load() {
    std::string config_path = get_config_path();

    if (!std::filesystem::exists(config_path)) {
        LOG_INFO("Config file not found, using defaults: " + config_path);
        return;
    }

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to open config file: " + config_path);
        }

        json config_json;
        file >> config_json;

        // Load values with fallbacks to defaults
        if (config_json.contains("backend")) {
            backend_ = config_json["backend"];
        }
        if (config_json.contains("model")) {
            model_ = config_json["model"];
        }
        if (config_json.contains("model_path") || config_json.contains("path")) {
            model_path_ = config_json.contains("model_path") ?
                config_json["model_path"].get<std::string>() :
                config_json["path"].get<std::string>();
        }
        if (config_json.contains("context_size")) {
            context_size_ = config_json["context_size"];
        }
        if (config_json.contains("key")) {
            key_ = config_json["key"];
        }
        if (config_json.contains("api_base")) {
            api_base_ = config_json["api_base"];
        }
        if (config_json.contains("system")) {
            system_prompt_ = config_json["system"];
        }
        if (config_json.contains("mcp_servers")) {
            // Store MCP servers as JSON string for MCPManager to parse
            mcp_config_ = config_json["mcp_servers"].dump();
        }

        // Load web search configuration (optional)
        if (config_json.contains("web_search_provider")) {
            web_search_provider_ = config_json["web_search_provider"];
        }
        if (config_json.contains("web_search_api_key")) {
            web_search_api_key_ = config_json["web_search_api_key"];
        }
        if (config_json.contains("web_search_instance_url")) {
            web_search_instance_url_ = config_json["web_search_instance_url"];
        }

        // Load sampling parameters (optional, use defaults if not specified)
        if (config_json.contains("temperature")) {
            temperature_ = config_json["temperature"].get<float>();
        }
        if (config_json.contains("top_p")) {
            top_p_ = config_json["top_p"].get<float>();
        }
        if (config_json.contains("top_k")) {
            top_k_ = config_json["top_k"].get<int>();
        }
        if (config_json.contains("min_keep")) {
            min_keep_ = config_json["min_keep"].get<int>();
        }
        if (config_json.contains("penalty_repeat")) {
            penalty_repeat_ = config_json["penalty_repeat"].get<float>();
        }
        if (config_json.contains("penalty_freq")) {
            penalty_freq_ = config_json["penalty_freq"].get<float>();
        }
        if (config_json.contains("penalty_present")) {
            penalty_present_ = config_json["penalty_present"].get<float>();
        }
        if (config_json.contains("penalty_last_n")) {
            penalty_last_n_ = config_json["penalty_last_n"].get<int>();
        }
        if (config_json.contains("gpu_layers")) {
            gpu_layers_ = config_json["gpu_layers"].get<int>();
        }

        // Load RAG database size limit (optional, supports both string and numeric formats)
        if (config_json.contains("max_db_size")) {
            if (config_json["max_db_size"].is_string()) {
                max_db_size_str_ = config_json["max_db_size"].get<std::string>();
                max_db_size_ = parse_size_string(max_db_size_str_);
            } else if (config_json["max_db_size"].is_number()) {
                // Backward compatibility: numeric format
                max_db_size_ = config_json["max_db_size"].get<size_t>();
                // Convert to string format
                if (max_db_size_ >= 1024ULL * 1024 * 1024 * 1024) {
                    max_db_size_str_ = std::to_string(max_db_size_ / (1024ULL * 1024 * 1024 * 1024)) + "T";
                } else if (max_db_size_ >= 1024ULL * 1024 * 1024) {
                    max_db_size_str_ = std::to_string(max_db_size_ / (1024ULL * 1024 * 1024)) + "G";
                } else if (max_db_size_ >= 1024ULL * 1024) {
                    max_db_size_str_ = std::to_string(max_db_size_ / (1024ULL * 1024)) + "M";
                } else if (max_db_size_ >= 1024) {
                    max_db_size_str_ = std::to_string(max_db_size_ / 1024) + "K";
                } else {
                    max_db_size_str_ = std::to_string(max_db_size_);
                }
            }
        }

        LOG_INFO("Loaded configuration from: " + config_path);

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
        json config_json = {
            {"backend", backend_},
            {"model", model_},
            {"model_path", model_path_},
            {"context_size", context_size_},
            {"key", key_}
        };

        // Add optional api_base if set
        if (!api_base_.empty()) {
            config_json["api_base"] = api_base_;
        }

        // Add sampling parameters (always save them with defaults)
        config_json["temperature"] = temperature_;
        config_json["top_p"] = top_p_;
        config_json["top_k"] = top_k_;
        config_json["min_keep"] = min_keep_;
        config_json["penalty_repeat"] = penalty_repeat_;
        config_json["penalty_freq"] = penalty_freq_;
        config_json["penalty_present"] = penalty_present_;
        config_json["penalty_last_n"] = penalty_last_n_;
        config_json["gpu_layers"] = gpu_layers_;

        // Add RAG database size limit (as human-friendly string)
        config_json["max_db_size"] = max_db_size_str_;

        std::ofstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to create config file: " + config_path);
        }

        file << config_json.dump(4) << std::endl;
        LOG_INFO("Saved configuration to: " + config_path);

    } catch (const json::exception& e) {
        throw ConfigError("Error creating JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error saving config: " + std::string(e.what()));
    }
}

void Config::set_backend(const std::string& backend) {
    if (!is_backend_available(backend)) {
        auto available = get_available_backends();
        std::string available_str;
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) available_str += ", ";
            available_str += available[i];
        }
        throw ConfigError("Backend '" + backend + "' is not available on this platform. Available backends: " + available_str);
    }
    backend_ = backend;
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
    backends.push_back("grok");
    backends.push_back("ollama");
#endif

    return backends;
}

void Config::validate() const {
    // Validate backend
    if (!is_backend_available(backend_)) {
        auto available = get_available_backends();
        std::string available_str;
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) available_str += ", ";
            available_str += available[i];
        }
        throw ConfigError("Invalid backend '" + backend_ + "'. Available on this platform: " + available_str);
    }

    // Validate model for local backends
    if (backend_ == "llamacpp" || backend_ == "tensorrt") {
        if (model_.empty()) {
            throw ConfigError("Model name is required for backend: " + backend_);
        }

        // Check if model file exists (either as full path or relative to model_path)
        std::filesystem::path model_file;
        if (model_[0] == '/' || model_[0] == '~') {
            // Absolute path
            model_file = model_;
        } else {
            // Relative to model_path
            model_file = std::filesystem::path(model_path_) / model_;
        }

        if (!std::filesystem::exists(model_file)) {
            throw ConfigError("Model file not found: " + model_file.string());
        }
    }

    // Validate API key for cloud backends
    if (backend_ == "openai" || backend_ == "anthropic") {
        if (key_.empty()) {
            throw ConfigError("API key is required for backend: " + backend_);
        }
    }

    // Validate context size
    if (context_size_ > 0 && context_size_ < 512) {
        throw ConfigError("Context size must be at least 512 tokens if specified");
    }

    LOG_DEBUG("Configuration validation passed");
}