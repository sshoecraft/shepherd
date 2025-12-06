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

/// @brief Represents a single provider configuration
/// Unified class that holds all provider-type-specific fields
class Provider {
public:
    // Common fields
    std::string name;
    std::string type;           // llamacpp, tensorrt, openai, anthropic, gemini, ollama, cli
    std::string model;
    int priority = 100;         // Lower = higher priority (0 reserved for cmdline)
    size_t context_size = 0;    // 0 = auto/default

    // Rate limits
    struct RateLimits {
        int requests_per_second = 0;
        int requests_per_minute = 0;
        int tokens_per_minute = 0;
        int tokens_per_day = 0;
        int tokens_per_month = 0;
        float max_cost_per_month = 0.0f;
        float warning_threshold = 0.85f;
    } rate_limits;

    // Pricing (per million tokens)
    struct Pricing {
        float prompt_cost = 0.0f;
        float completion_cost = 0.0f;
        bool dynamic = false;
    } pricing;

    // Local backend fields (llamacpp, tensorrt)
    std::string model_path = "~/models";
    int tp = 1;                     // Tensor parallelism
    int pp = 1;                     // Pipeline parallelism
    int gpu_layers = -1;            // -1=auto, 0=CPU only
    int gpu_id = 0;                 // TensorRT GPU ID
    int n_batch = 512;
    int n_threads = 0;              // 0=auto

    // Sampling parameters (shared by most backends)
    float temperature = 0.7f;
    float top_p = 1.0f;
    int top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int max_tokens = 0;             // 0=auto
    std::vector<std::string> stop_sequences;

    // API backend fields
    std::string api_key;
    std::string base_url;
    std::map<std::string, std::string> extra_headers;
    bool ssl_verify = true;
    std::string ca_bundle_path;

    // OAuth 2.0 (Azure OpenAI, etc.)
    std::string client_id;
    std::string client_secret;
    std::string token_url;
    std::string token_scope;

    // Azure OpenAI specific
    std::string deployment_name;
    std::string api_version;

    // Ollama specific
    int num_ctx = 0;                // Context window (0=auto)
    int num_predict = -1;           // Max tokens (-1=unlimited)

    // Check if this is an API provider
    bool is_api() const;

    // Load all providers from ~/.config/shepherd/providers/
    static std::vector<Provider> load_providers();

    // Get providers directory path
    static std::string get_providers_dir();

    // Create backend from this provider config
    std::unique_ptr<Backend> connect(Session& session);

    // Serialization
    nlohmann::json to_json() const;
    static Provider from_json(const nlohmann::json& j);

    // Create provider from current global config state (for cmdline overrides)
    static Provider from_config();

    // Save this provider to disk
    void save() const;

    // Remove provider file from disk
    static void remove(const std::string& name);
};


// Common provider command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// backend and session are optional - needed for "use" and "next" commands
// providers is optional - if null, loads from disk; if provided, uses that list
// current_provider is optional - name of current provider for display
int handle_provider_args(const std::vector<std::string>& args,
                         std::unique_ptr<Backend>* backend = nullptr,
                         Session* session = nullptr,
                         std::vector<Provider>* providers = nullptr,
                         std::string* current_provider = nullptr);

// Common model command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// backend is optional - needed for runtime model changes
int handle_model_args(const std::vector<std::string>& args,
                      std::unique_ptr<Backend>* backend = nullptr,
                      std::vector<Provider>* providers = nullptr,
                      std::string* current_provider = nullptr);

