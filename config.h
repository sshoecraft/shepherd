#pragma once

#include <string>
#include <map>
#include <vector>
#include <functional>
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

    // Load configuration from a JSON string (for Key Vault, etc.)
    void load_from_json_string(const std::string& json_str);

    // Save current configuration
    void save() const;

    // Validation
    void validate() const;

    // Get available backends for current platform
    static std::vector<std::string> get_available_backends();

    // Get user's home directory (tries getpwuid first, then HOME env var)
    static std::string get_home_directory();

    // Get default config path (XDG-compliant), or override if set
    static std::string get_default_config_path();

    // Set a global config path override (used before subcommand dispatch)
    static void set_config_path_override(const std::string& path);


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
    std::string initial_prompt;  // Initial user prompt from --prompt / -e
    bool single_query_mode;      // True when --prompt was specified (even if empty)
    std::string warmup_message;
    bool warmup;
    bool calibration;
    std::string system_prompt;
    std::string mcp_config;
    std::string memory_database;
    std::string max_db_size_str;
    size_t max_db_size;
    bool rag_context_injection;       // Enable automatic RAG context injection before inference
    double rag_relevance_threshold;   // Minimum relevance score for RAG results (0.0-1.0)
    int rag_max_results;              // Maximum number of RAG results to inject
    std::string web_search_provider;
    std::string web_search_api_key;
    std::string web_search_instance_url;

    // User identity for multi-tenant RAG isolation
    // Set this to share the same memory pool across platforms (e.g., "steve")
    // If empty, falls back to hostname:username for CLI/TUI, API key name for API server
    std::string user_id;

    // Memory extraction settings (background fact extraction from conversations)
    bool memory_extraction;
    std::string memory_extraction_model;
    std::string memory_extraction_endpoint;
    std::string memory_extraction_api_key;
    int memory_extraction_max_tokens;
    double memory_extraction_temperature;
    int memory_extraction_min_turns;
    int memory_extraction_idle_timeout;
    int memory_extraction_max_turns;
    int memory_extraction_queue_limit;
    int truncate_limit;
    int max_tokens;       // Max generation tokens: -1=max, 0=auto, >0=explicit
    bool streaming;
    bool thinking;        // Show thinking/reasoning blocks in output
    bool stats;           // Show performance stats (prefill/decode speed, KV cache)
    bool auto_provider;   // Auto-switch to next provider on connection failure
    bool tui;             // Enable TUI mode (boxed input, status line)
    bool raw_output;      // Disable channel parsing, return raw model output (like vLLM)
    int tui_history;      // TUI scrollback buffer size (lines)
    std::string auth_mode;  // Server authentication mode: "none" or "json"
    bool server_tools;    // Enable server-side tool execution for authenticated users
    nlohmann::json json;  // Parsed config JSON for backend-specific settings

    // Unified config: providers loaded from config (Key Vault or file)
    std::vector<nlohmann::json> providers_json;

    // SMCP servers config (JSON string, like mcp_config)
    std::string smcp_config;

    // Named schedulers (persisted in config.json)
    std::string scheduler_name;              // Active scheduler name (from --scheduler flag)
    nlohmann::json schedulers_json;          // All schedulers data

    // Config source mode
    enum class SourceMode { LOCAL_FILE, KEY_VAULT, HASHICORP_VAULT };
    SourceMode source_mode = SourceMode::LOCAL_FILE;
    std::string keyvault_name;  // Azure Key Vault name (when source_mode == KEY_VAULT)

    // Check if config is read-only (vault modes)
    bool is_read_only() const { return source_mode != SourceMode::LOCAL_FILE; }

    // Legacy/runtime fields (not saved to config, only used for command-line overrides)
    std::string backend;
    std::string model;
    std::string model_path;
    size_t context_size;
    std::string key;
    std::string api_base;
    std::string models_file;

    // Sampling parameter overrides from command line (-1 = not overridden)
    float temperature_override = -1.0f;
    float top_p_override = -1.0f;
    int top_k_override = -1;
    float repeat_penalty_override = -1.0f;
    float frequency_penalty_override = -1.0f;
    float presence_penalty_override = -1.0f;

private:
    // Internal helpers
    static size_t parse_size_string(const std::string& size_str);
    std::string get_config_path() const;
    std::string get_default_model_path() const;
    void set_defaults();
    void load_from_json(const nlohmann::json& j);  // Shared JSON parsing logic
    bool is_backend_available(const std::string& backend) const;

    // Internal state
    std::string custom_config_path_;  // Custom config file path (optional)
};

// Common config command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// callback: function to emit output
int handle_config_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback);
