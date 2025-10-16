#pragma once

#include <string>
#include <map>
#include <vector>
#include <stdexcept>

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

    // Getters
    std::string get_backend() const { return backend_; }
    std::string get_model() const { return model_; }
    std::string get_model_path() const { return model_path_; }
    size_t get_context_size() const { return context_size_; }
    std::string get_key() const { return key_; }
    std::string get_api_base() const { return api_base_; }
    std::string get_system_prompt() const { return system_prompt_; }
    size_t get_max_db_size() const { return max_db_size_; }

    // Setters
    void set_backend(const std::string& backend);
    void set_model(const std::string& model) { model_ = model; }
    void set_model_path(const std::string& model_path) { model_path_ = model_path; }
    void set_context_size(size_t context_size) { context_size_ = context_size; }
    void set_key(const std::string& key) { key_ = key; }
    void set_api_base(const std::string& api_base) { api_base_ = api_base; }
    void set_config_path(const std::string& config_path) { custom_config_path_ = config_path; }
    void set_max_db_size(const std::string& max_db_size_str) {
        max_db_size_str_ = max_db_size_str;
        max_db_size_ = parse_size_string(max_db_size_str);
    }

    // Validation
    void validate() const;

    // Get available backends for current platform
    static std::vector<std::string> get_available_backends();

    // Get user's home directory (tries getpwuid first, then HOME env var) - public for edit-system command
    static std::string get_home_directory();

    // Get MCP servers configuration (if any)
    std::string get_mcp_config() const { return mcp_config_; }

    // Web search configuration
    std::string get_web_search_provider() const { return web_search_provider_; }
    std::string get_web_search_api_key() const { return web_search_api_key_; }
    std::string get_web_search_instance_url() const { return web_search_instance_url_; }

    // Sampling parameters (for llamacpp backend)
    float get_temperature() const { return temperature_; }
    float get_top_p() const { return top_p_; }
    int get_top_k() const { return top_k_; }
    int get_min_keep() const { return min_keep_; }
    float get_penalty_repeat() const { return penalty_repeat_; }
    float get_penalty_freq() const { return penalty_freq_; }
    float get_penalty_present() const { return penalty_present_; }
    int get_penalty_last_n() const { return penalty_last_n_; }

    void set_temperature(float temperature) { temperature_ = temperature; }
    void set_top_p(float top_p) { top_p_ = top_p; }
    void set_top_k(int top_k) { top_k_ = top_k; }
    void set_min_keep(int min_keep) { min_keep_ = min_keep; }
    void set_penalty_repeat(float penalty_repeat) { penalty_repeat_ = penalty_repeat; }
    void set_penalty_freq(float penalty_freq) { penalty_freq_ = penalty_freq; }
    void set_penalty_present(float penalty_present) { penalty_present_ = penalty_present; }
    void set_penalty_last_n(int penalty_last_n) { penalty_last_n_ = penalty_last_n; }

private:
    // Helper to parse size strings with suffixes (e.g., "10G", "500M")
    static size_t parse_size_string(const std::string& size_str);

    std::string backend_;
    std::string model_;
    std::string model_path_;
    size_t context_size_;
    std::string key_;
    std::string api_base_;  // API base URL for API backends (optional)
    std::string system_prompt_;  // Custom system prompt (optional)
    std::string mcp_config_;  // Raw JSON string for MCP servers
    std::string custom_config_path_;  // Custom config file path (optional)
    std::string max_db_size_str_;  // Maximum RAG database size string (e.g., "10G")
    size_t max_db_size_;  // Cached parsed size in bytes

    // Web search settings (optional)
    std::string web_search_provider_;  // "brave", "duckduckgo", "searxng", or empty
    std::string web_search_api_key_;  // API key (for Brave)
    std::string web_search_instance_url_;  // Instance URL (for SearXNG)

    // Sampling parameters (for llamacpp backend)
    float temperature_;
    float top_p_;
    int top_k_;
    int min_keep_;
    float penalty_repeat_;
    float penalty_freq_;
    float penalty_present_;
    int penalty_last_n_;

    std::string get_config_path() const;
    std::string get_default_model_path() const;
    void set_defaults();
    bool is_backend_available(const std::string& backend) const;
};
