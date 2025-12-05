#pragma once

#include <string>
#include <map>
#include <vector>
#include <optional>
#include <memory>
#include "nlohmann/json.hpp"

// Forward declarations
class Backend;
class Session;

// Base provider configuration
class ProviderConfig {
public:
    std::string type;           // Backend type: llamacpp, tensorrt, openai, etc.
    std::string name;           // User-friendly name
    std::string model;          // Model name or file
    int priority = 100;         // Priority for selection (lower = higher priority)
    int context_size = 0;       // Context window size (0=auto/default)

    // Rate limits (common to all providers)
    struct RateLimits {
        int requests_per_second = 0;     // 0 = unlimited
        int requests_per_minute = 0;
        int tokens_per_minute = 0;
        int tokens_per_day = 0;
        int tokens_per_month = 0;
        float max_cost_per_month = 0.0f;
        float warning_threshold = 0.85f;  // Warn at 85% by default
    } rate_limits;

    // Pricing (per million tokens)
    struct Pricing {
        float prompt_cost = 0.0f;
        float completion_cost = 0.0f;
        bool dynamic = false;  // If true, fetch pricing from API
    } pricing;

    virtual ~ProviderConfig() = default;

    // Pure virtual - each subclass implements its own serialization
    virtual nlohmann::json to_json() const = 0;

    // Factory method to deserialize based on type field
    static std::unique_ptr<ProviderConfig> from_json(const nlohmann::json& j);
};

// Llama.cpp provider configuration
class LlamaProviderConfig : public ProviderConfig {
public:
    std::string model_path = "~/models";
    int tp = 1;                     // Tensor parallelism
    int pp = 1;                     // Pipeline parallelism
    int gpu_layers = -1;            // -1=auto, 0=CPU only, >0=specific count
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int n_batch = 512;
    int n_threads = 0;              // 0=auto

    nlohmann::json to_json() const override;
    static LlamaProviderConfig from_json(const nlohmann::json& j);
};

// TensorRT-LLM provider configuration
class TensorRTProviderConfig : public ProviderConfig {
public:
    std::string model_path = "~/models";
    int tp = 1;                     // Tensor parallelism
    int pp = 1;                     // Pipeline parallelism
    int gpu_id = 0;
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;

    nlohmann::json to_json() const override;
    static TensorRTProviderConfig from_json(const nlohmann::json& j);
};

// API provider configuration (OpenAI, Anthropic, Gemini, Grok)
class ApiProviderConfig : public ProviderConfig {
public:
    std::string api_key;
    std::string base_url;           // Empty = use default
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 0;                  // 0=disabled
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int max_tokens = 0;             // 0=auto
    std::vector<std::string> stop_sequences;
    std::map<std::string, std::string> extra_headers;
    bool ssl_verify = true;         // SSL certificate verification
    std::string ca_bundle_path;     // Custom CA bundle path (empty = use system default)

    // OAuth 2.0 configuration (for Azure OpenAI and similar services)
    std::string client_id;          // OAuth client ID
    std::string client_secret;      // OAuth client secret
    std::string token_url;          // OAuth token endpoint
    std::string token_scope;        // OAuth scope (default: empty)

    // Azure OpenAI specific
    std::string deployment_name;    // Azure OpenAI deployment name
    std::string api_version;        // Azure OpenAI API version (e.g., "2024-06-01")

    nlohmann::json to_json() const override;
    static ApiProviderConfig from_json(const nlohmann::json& j);
};

// Ollama provider configuration
class OllamaProviderConfig : public ProviderConfig {
public:
    std::string base_url = "http://localhost:11434";
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    int num_ctx = 0;                // Context window (0=auto)
    int num_predict = -1;           // Max tokens (-1=unlimited)

    nlohmann::json to_json() const override;
    static OllamaProviderConfig from_json(const nlohmann::json& j);
};

class Provider {
public:
    Provider();
    ~Provider();

    // Load all providers from ~/.config/shepherd/providers/
    void load_providers();

    // Save provider to file
    void save_provider(const ProviderConfig& config);

    // Remove provider file
    void remove_provider(const std::string& name);

    // Get provider by name (returns raw pointer, Provider still owns it)
    ProviderConfig* get_provider(const std::string& name);

    // List all provider names
    std::vector<std::string> list_providers() const;

    // Get/set current provider
    std::string get_current_provider() const { return current_provider; }
    void set_current_provider(const std::string& name);

    // Get highest priority available provider (skips rate-limited ones)
    std::optional<std::string> get_highest_priority_provider() const;

    // Get next available provider (skips rate-limited ones)
    std::optional<std::string> get_next_provider() const;

    // Connect to next available provider - creates backend, initializes, returns it
    // Tries providers in priority order until one succeeds
    // Returns nullptr if all providers fail (errors logged)
    // On success, updates current_provider
    std::unique_ptr<Backend> connect_next_provider(Session& session, size_t context_size);

    // Connect to a specific provider by name
    // Returns nullptr on failure (error logged)
    // On success, updates current_provider
    std::unique_ptr<Backend> connect_provider(const std::string& name, Session& session, size_t context_size);

    // Check if provider is rate limited
    bool is_rate_limited(const std::string& name) const;

    // Get providers directory path
    static std::string get_providers_dir();

    // Interactive provider editor (returns true if saved)
    bool interactive_edit(ProviderConfig& config);

    // Parse command-line style arguments for provider config
    std::unique_ptr<ProviderConfig> parse_provider_args(const std::string& type, const std::vector<std::string>& args);

private:
    std::map<std::string, std::unique_ptr<ProviderConfig>> providers;
    std::string current_provider;
    std::string providers_dir;

    // Helper to get provider file path
    std::string get_provider_file(const std::string& name) const;

    // Load usage stats to check rate limits
    bool check_rate_limits(const ProviderConfig& config) const;

    // Type-specific interactive editors
    bool edit_llama_config(LlamaProviderConfig& cfg);
    bool edit_tensorrt_config(TensorRTProviderConfig& cfg);
    bool edit_api_config(ApiProviderConfig& cfg);
    bool edit_ollama_config(OllamaProviderConfig& cfg);
};

// Common provider command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// backend and session are optional - needed for "use" and "next" commands
int handle_provider_args(const std::vector<std::string>& args,
                         std::unique_ptr<Backend>* backend = nullptr,
                         Session* session = nullptr);

// Common model command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// backend is optional - needed for runtime model changes
int handle_model_args(const std::vector<std::string>& args,
                      std::unique_ptr<Backend>* backend = nullptr);