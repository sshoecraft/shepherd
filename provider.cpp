#include "provider.h"
#include "config.h"
#include "logger.h"
#include "backends/backend.h"
#include "backends/factory.h"
#include "session.h"
#include "shepherd.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Helper: Serialize common fields
static void serialize_common(json& j, const ProviderConfig& cfg) {
    j["type"] = cfg.type;
    j["name"] = cfg.name;
    j["model"] = cfg.model;
    j["priority"] = cfg.priority;
    j["context_size"] = cfg.context_size;

    // Rate limits
    if (cfg.rate_limits.requests_per_second > 0 ||
        cfg.rate_limits.requests_per_minute > 0 ||
        cfg.rate_limits.tokens_per_minute > 0 ||
        cfg.rate_limits.tokens_per_day > 0 ||
        cfg.rate_limits.tokens_per_month > 0 ||
        cfg.rate_limits.max_cost_per_month > 0) {

        json rl;
        if (cfg.rate_limits.requests_per_second > 0) rl["requests_per_second"] = cfg.rate_limits.requests_per_second;
        if (cfg.rate_limits.requests_per_minute > 0) rl["requests_per_minute"] = cfg.rate_limits.requests_per_minute;
        if (cfg.rate_limits.tokens_per_minute > 0) rl["tokens_per_minute"] = cfg.rate_limits.tokens_per_minute;
        if (cfg.rate_limits.tokens_per_day > 0) rl["tokens_per_day"] = cfg.rate_limits.tokens_per_day;
        if (cfg.rate_limits.tokens_per_month > 0) rl["tokens_per_month"] = cfg.rate_limits.tokens_per_month;
        if (cfg.rate_limits.max_cost_per_month > 0) rl["max_cost_per_month"] = cfg.rate_limits.max_cost_per_month;
        rl["warning_threshold"] = cfg.rate_limits.warning_threshold;
        j["rate_limits"] = rl;
    }

    // Pricing
    if (cfg.pricing.prompt_cost > 0 || cfg.pricing.completion_cost > 0 || cfg.pricing.dynamic) {
        json p;
        if (cfg.pricing.dynamic) {
            p["mode"] = "dynamic";
        } else {
            p["mode"] = "static";
            p["prompt_per_million"] = cfg.pricing.prompt_cost;
            p["completion_per_million"] = cfg.pricing.completion_cost;
        }
        j["pricing"] = p;
    }
}

// Helper: Deserialize common fields
static void deserialize_common(ProviderConfig& cfg, const json& j) {
    cfg.type = j.value("type", "");
    cfg.name = j.value("name", "");
    cfg.model = j.value("model", "");
    cfg.priority = j.value("priority", 100);
    cfg.context_size = j.value("context_size", 0);

    // Rate limits
    if (j.contains("rate_limits")) {
        const auto& rl = j["rate_limits"];
        cfg.rate_limits.requests_per_second = rl.value("requests_per_second", 0);
        cfg.rate_limits.requests_per_minute = rl.value("requests_per_minute", 0);
        cfg.rate_limits.tokens_per_minute = rl.value("tokens_per_minute", 0);
        cfg.rate_limits.tokens_per_day = rl.value("tokens_per_day", 0);
        cfg.rate_limits.tokens_per_month = rl.value("tokens_per_month", 0);
        cfg.rate_limits.max_cost_per_month = rl.value("max_cost_per_month", 0.0f);
        cfg.rate_limits.warning_threshold = rl.value("warning_threshold", 0.85f);
    }

    // Pricing
    if (j.contains("pricing")) {
        const auto& p = j["pricing"];
        std::string mode = p.value("mode", "static");
        cfg.pricing.dynamic = (mode == "dynamic");
        if (!cfg.pricing.dynamic) {
            cfg.pricing.prompt_cost = p.value("prompt_per_million", 0.0f);
            cfg.pricing.completion_cost = p.value("completion_per_million", 0.0f);
        }
    }
}

// LlamaProviderConfig serialization
json LlamaProviderConfig::to_json() const {
    json j;
    serialize_common(j, *this);

    j["model_path"] = model_path;
    j["tp"] = tp;
    j["pp"] = pp;
    j["gpu_layers"] = gpu_layers;
    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["top_k"] = top_k;
    j["repeat_penalty"] = repeat_penalty;
    j["n_batch"] = n_batch;
    j["n_threads"] = n_threads;

    return j;
}

LlamaProviderConfig LlamaProviderConfig::from_json(const json& j) {
    LlamaProviderConfig cfg;
    deserialize_common(cfg, j);

    cfg.model_path = j.value("model_path", "~/models");
    cfg.tp = j.value("tp", 1);
    cfg.pp = j.value("pp", 1);
    cfg.gpu_layers = j.value("gpu_layers", -1);
    cfg.temperature = j.value("temperature", 0.7f);
    cfg.top_p = j.value("top_p", 1.0f);
    cfg.top_k = j.value("top_k", 40);
    cfg.repeat_penalty = j.value("repeat_penalty", 1.1f);
    cfg.n_batch = j.value("n_batch", 512);
    cfg.n_threads = j.value("n_threads", 0);

    return cfg;
}

// TensorRTProviderConfig serialization
json TensorRTProviderConfig::to_json() const {
    json j;
    serialize_common(j, *this);

    j["model_path"] = model_path;
    j["tp"] = tp;
    j["pp"] = pp;
    j["gpu_id"] = gpu_id;
    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["top_k"] = top_k;
    j["repeat_penalty"] = repeat_penalty;
    j["frequency_penalty"] = frequency_penalty;
    j["presence_penalty"] = presence_penalty;

    return j;
}

TensorRTProviderConfig TensorRTProviderConfig::from_json(const json& j) {
    TensorRTProviderConfig cfg;
    deserialize_common(cfg, j);

    cfg.model_path = j.value("model_path", "~/models");
    cfg.tp = j.value("tp", 1);
    cfg.pp = j.value("pp", 1);
    cfg.gpu_id = j.value("gpu_id", 0);
    cfg.temperature = j.value("temperature", 0.7f);
    cfg.top_p = j.value("top_p", 1.0f);
    cfg.top_k = j.value("top_k", 40);
    cfg.repeat_penalty = j.value("repeat_penalty", 1.1f);
    cfg.frequency_penalty = j.value("frequency_penalty", 0.0f);
    cfg.presence_penalty = j.value("presence_penalty", 0.0f);

    return cfg;
}

// ApiProviderConfig serialization
json ApiProviderConfig::to_json() const {
    json j;
    serialize_common(j, *this);

    j["api_key"] = api_key;
    if (!base_url.empty()) {
        j["base_url"] = base_url;
    }
    j["temperature"] = temperature;
    j["top_p"] = top_p;
    if (top_k > 0) j["top_k"] = top_k;
    if (frequency_penalty != 0.0f) j["frequency_penalty"] = frequency_penalty;
    if (presence_penalty != 0.0f) j["presence_penalty"] = presence_penalty;
    if (max_tokens > 0) j["max_tokens"] = max_tokens;
    if (!stop_sequences.empty()) j["stop_sequences"] = stop_sequences;
    if (!extra_headers.empty()) j["extra_headers"] = extra_headers;
    if (!ssl_verify) j["ssl_verify"] = ssl_verify;  // Only write if false
    if (!ca_bundle_path.empty()) j["ca_bundle_path"] = ca_bundle_path;

    // OAuth 2.0 fields
    if (!client_id.empty()) j["client_id"] = client_id;
    if (!client_secret.empty()) j["client_secret"] = client_secret;
    if (!token_url.empty()) j["token_url"] = token_url;
    if (!token_scope.empty()) j["token_scope"] = token_scope;

    // Azure OpenAI fields
    if (!deployment_name.empty()) j["deployment_name"] = deployment_name;
    if (!api_version.empty()) j["api_version"] = api_version;

    return j;
}

ApiProviderConfig ApiProviderConfig::from_json(const json& j) {
    ApiProviderConfig cfg;
    deserialize_common(cfg, j);

    cfg.api_key = j.value("api_key", "");
    cfg.base_url = j.value("base_url", "");
    cfg.temperature = j.value("temperature", 0.7f);
    cfg.top_p = j.value("top_p", 1.0f);
    cfg.top_k = j.value("top_k", 0);
    cfg.frequency_penalty = j.value("frequency_penalty", 0.0f);
    cfg.presence_penalty = j.value("presence_penalty", 0.0f);
    cfg.max_tokens = j.value("max_tokens", 0);
    if (j.contains("stop_sequences")) {
        cfg.stop_sequences = j["stop_sequences"].get<std::vector<std::string>>();
    }
    if (j.contains("extra_headers")) {
        cfg.extra_headers = j["extra_headers"].get<std::map<std::string, std::string>>();
    }
    cfg.ssl_verify = j.value("ssl_verify", true);
    cfg.ca_bundle_path = j.value("ca_bundle_path", "");

    // OAuth 2.0 fields
    cfg.client_id = j.value("client_id", "");
    cfg.client_secret = j.value("client_secret", "");
    cfg.token_url = j.value("token_url", "");
    cfg.token_scope = j.value("token_scope", "");

    // Azure OpenAI fields
    cfg.deployment_name = j.value("deployment_name", "");
    cfg.api_version = j.value("api_version", "");

    return cfg;
}

// OllamaProviderConfig serialization
json OllamaProviderConfig::to_json() const {
    json j;
    serialize_common(j, *this);

    j["base_url"] = base_url;
    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["top_k"] = top_k;
    j["repeat_penalty"] = repeat_penalty;
    if (num_ctx > 0) j["num_ctx"] = num_ctx;
    if (num_predict != -1) j["num_predict"] = num_predict;

    return j;
}

OllamaProviderConfig OllamaProviderConfig::from_json(const json& j) {
    OllamaProviderConfig cfg;
    deserialize_common(cfg, j);

    cfg.base_url = j.value("base_url", "http://localhost:11434");
    cfg.temperature = j.value("temperature", 0.7f);
    cfg.top_p = j.value("top_p", 1.0f);
    cfg.top_k = j.value("top_k", 40);
    cfg.repeat_penalty = j.value("repeat_penalty", 1.1f);
    cfg.num_ctx = j.value("num_ctx", 0);
    cfg.num_predict = j.value("num_predict", -1);

    return cfg;
}

// ProviderConfig factory
std::unique_ptr<ProviderConfig> ProviderConfig::from_json(const json& j) {
    std::string type = j.value("type", "");

    if (type == "llamacpp") {
        return std::make_unique<LlamaProviderConfig>(LlamaProviderConfig::from_json(j));
    }
    else if (type == "tensorrt") {
        return std::make_unique<TensorRTProviderConfig>(TensorRTProviderConfig::from_json(j));
    }
    else if (type == "openai" || type == "anthropic" || type == "gemini" || type == "grok" || type == "cli") {
        return std::make_unique<ApiProviderConfig>(ApiProviderConfig::from_json(j));
    }
    else if (type == "ollama") {
        return std::make_unique<OllamaProviderConfig>(OllamaProviderConfig::from_json(j));
    }

    throw std::runtime_error("Unknown provider type: " + type);
}

// Provider implementation
Provider::Provider() {
    providers_dir = get_providers_dir();
    load_providers();
}

Provider::~Provider() {
}

std::string Provider::get_providers_dir() {
    std::string config_home;
    const char* xdg_config = getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        config_home = xdg_config;
    } else {
        config_home = Config::get_home_directory() + "/.config";
    }
    return config_home + "/shepherd/providers";
}

void Provider::load_providers() {
    providers.clear();

    // Create directory if it doesn't exist
    fs::create_directories(providers_dir);

    // Load all JSON files from providers directory
    for (const auto& entry : fs::directory_iterator(providers_dir)) {
        if (entry.path().extension() == ".json") {
            try {
                std::ifstream file(entry.path());
                json j;
                file >> j;

                auto config = ProviderConfig::from_json(j);
                if (config->name.empty()) {
                    // Use filename without extension as name
                    config->name = entry.path().stem().string();
                }

                std::string name = config->name;
                providers[name] = std::move(config);
                LOG_DEBUG("Loaded provider: " + name);

            } catch (const std::exception& e) {
                LOG_ERROR("Failed to load provider from " + entry.path().string() + ": " + e.what());
            }
        }
    }

    // current_provider will be set at startup by priority selection
    // Not loaded from config file
}

void Provider::save_provider(const ProviderConfig& config) {
    if (config.name.empty()) {
        throw std::runtime_error("Provider name cannot be empty");
    }

    fs::create_directories(providers_dir);

    std::string filename = get_provider_file(config.name);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create provider file: " + filename);
    }

    file << config.to_json().dump(4) << std::endl;
    LOG_INFO("Saved provider: " + config.name);

    // Reload providers to update in-memory map with correct type
    load_providers();
}

void Provider::remove_provider(const std::string& name) {
    std::string filename = get_provider_file(name);
    if (fs::exists(filename)) {
        fs::remove(filename);
        LOG_INFO("Removed provider: " + name);
    }

    providers.erase(name);

    if (current_provider == name) {
        current_provider.clear();
    }
}

ProviderConfig* Provider::get_provider(const std::string& name) {
    auto it = providers.find(name);
    if (it != providers.end()) {
        return it->second.get();
    }
    return nullptr;
}

std::vector<std::string> Provider::list_providers() const {
    std::vector<std::pair<int, std::string>> sorted_providers;
    for (const auto& [name, config] : providers) {
        sorted_providers.push_back({config->priority, name});
    }

    // Sort by priority (lower number = higher priority), then alphabetically
    std::sort(sorted_providers.begin(), sorted_providers.end());

    std::vector<std::string> names;
    for (const auto& [priority, name] : sorted_providers) {
        names.push_back(name);
    }
    return names;
}

void Provider::set_current_provider(const std::string& name) {
    if (providers.find(name) == providers.end()) {
        throw std::runtime_error("Provider not found: " + name);
    }

    current_provider = name;
    // current_provider is runtime state only, not persisted
}

std::optional<std::string> Provider::get_highest_priority_provider() const {
    if (providers.empty()) {
        return std::nullopt;
    }

    // Build sorted list of providers by priority
    std::vector<std::pair<int, std::string>> sorted_providers;
    for (const auto& [name, config] : providers) {
        sorted_providers.push_back({config->priority, name});
    }
    std::sort(sorted_providers.begin(), sorted_providers.end());

    // Find first available (not rate-limited) provider
    for (const auto& [priority, name] : sorted_providers) {
        if (!is_rate_limited(name)) {
            return name;
        }
    }

    return std::nullopt;  // All providers are rate limited
}

std::optional<std::string> Provider::get_next_provider() const {
    if (providers.empty()) {
        return std::nullopt;
    }

    // Build ordered list of providers
    std::vector<std::string> provider_names;
    for (const auto& [name, config] : providers) {
        provider_names.push_back(name);
    }
    std::sort(provider_names.begin(), provider_names.end());

    // Find current provider index
    size_t current_idx = 0;
    if (!current_provider.empty()) {
        auto it = std::find(provider_names.begin(), provider_names.end(), current_provider);
        if (it != provider_names.end()) {
            current_idx = std::distance(provider_names.begin(), it);
        }
    }

    // Try each provider starting from next index
    size_t attempts = 0;
    size_t idx = (current_idx + 1) % provider_names.size();

    while (attempts < provider_names.size()) {
        const std::string& name = provider_names[idx];
        if (!is_rate_limited(name)) {
            return name;
        }
        idx = (idx + 1) % provider_names.size();
        attempts++;
    }

    return std::nullopt;  // All providers are rate limited
}

bool Provider::is_rate_limited(const std::string& name) const {
    auto it = providers.find(name);
    if (it == providers.end()) {
        return false;
    }

    return check_rate_limits(*it->second);
}

std::string Provider::get_provider_file(const std::string& name) const {
    // Sanitize name for filesystem
    std::string safe_name = name;
    std::replace(safe_name.begin(), safe_name.end(), '/', '-');
    std::replace(safe_name.begin(), safe_name.end(), '\\', '-');
    return providers_dir + "/" + safe_name + ".json";
}

bool Provider::check_rate_limits(const ProviderConfig& config) const {
    // Rate limit checking implementation
    // Usage tracking files would be at: ~/.local/share/shepherd/usage/<provider-name>.json
    // For now, providers are never considered rate limited
    // This allows all configured providers to be used without implementing
    // complex usage file I/O. Can be enhanced later without API changes.
    return false;
}

bool Provider::interactive_edit(ProviderConfig& config) {
    std::cout << "\n=== Provider Configuration ===\n\n";

    std::string input;

    // Edit common fields
    std::cout << "Provider name [" << config.name << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) config.name = input;

    if (config.name.empty()) {
        std::cout << "Error: Provider name is required\n";
        return false;
    }

    std::cout << "Backend type [" << config.type << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) config.type = input;

    std::cout << "Model [" << config.model << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) config.model = input;

    std::cout << "Priority (lower = higher priority) [" << config.priority << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) config.priority = std::stoi(input);

    std::cout << "Context size (0=auto) [" << config.context_size << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) config.context_size = std::stoi(input);

    // Dispatch to type-specific editor
    bool is_api = false;
    if (auto* llama = dynamic_cast<LlamaProviderConfig*>(&config)) {
        if (!edit_llama_config(*llama)) return false;
    }
    else if (auto* tensorrt = dynamic_cast<TensorRTProviderConfig*>(&config)) {
        if (!edit_tensorrt_config(*tensorrt)) return false;
    }
    else if (auto* api = dynamic_cast<ApiProviderConfig*>(&config)) {
        if (!edit_api_config(*api)) return false;
        is_api = true;
    }
    else if (auto* ollama = dynamic_cast<OllamaProviderConfig*>(&config)) {
        if (!edit_ollama_config(*ollama)) return false;
    }

    // Rate limits and pricing only for API providers
    if (is_api) {
        // Rate limits
        std::cout << "\n--- Rate Limits (press Enter to skip) ---\n";
        std::cout << "Monthly token limit [" << config.rate_limits.tokens_per_month << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) config.rate_limits.tokens_per_month = std::stoi(input);

        std::cout << "Monthly cost limit ($) [" << config.rate_limits.max_cost_per_month << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) config.rate_limits.max_cost_per_month = std::stof(input);

        // Pricing
        std::cout << "\n--- Pricing ---\n";
        std::cout << "Use dynamic pricing? (y/n) [" << (config.pricing.dynamic ? "y" : "n") << "]: ";
        std::getline(std::cin, input);
        if (!input.empty()) {
            config.pricing.dynamic = (input[0] == 'y' || input[0] == 'Y');
        }

        if (!config.pricing.dynamic) {
            std::cout << "Prompt cost per million tokens [" << config.pricing.prompt_cost << "]: ";
            std::getline(std::cin, input);
            if (!input.empty()) config.pricing.prompt_cost = std::stof(input);

            std::cout << "Completion cost per million tokens [" << config.pricing.completion_cost << "]: ";
            std::getline(std::cin, input);
            if (!input.empty()) config.pricing.completion_cost = std::stof(input);
        }
    }

    // Confirm
    std::cout << "\nSave configuration? (y/n): ";
    std::getline(std::cin, input);
    return !input.empty() && (input[0] == 'y' || input[0] == 'Y');
}

bool Provider::edit_llama_config(LlamaProviderConfig& cfg) {
    std::cout << "\n--- Llama.cpp Settings ---\n";
    std::string input;

    std::cout << "Model path [" << cfg.model_path << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.model_path = input;

    std::cout << "Tensor parallelism (tp) [" << cfg.tp << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.tp = std::stoi(input);

    std::cout << "Pipeline parallelism (pp) [" << cfg.pp << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.pp = std::stoi(input);

    std::cout << "GPU layers (-1=auto, 0=CPU) [" << cfg.gpu_layers << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.gpu_layers = std::stoi(input);

    std::cout << "Context size (0=auto) [" << cfg.context_size << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.context_size = std::stoi(input);

    std::cout << "Temperature [" << cfg.temperature << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.temperature = std::stof(input);

    return true;
}

bool Provider::edit_tensorrt_config(TensorRTProviderConfig& cfg) {
    std::cout << "\n--- TensorRT-LLM Settings ---\n";
    std::string input;

    std::cout << "Model path [" << cfg.model_path << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.model_path = input;

    std::cout << "Tensor parallelism (tp) [" << cfg.tp << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.tp = std::stoi(input);

    std::cout << "Pipeline parallelism (pp) [" << cfg.pp << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.pp = std::stoi(input);

    std::cout << "GPU ID [" << cfg.gpu_id << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.gpu_id = std::stoi(input);

    std::cout << "Context size (0=auto) [" << cfg.context_size << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.context_size = std::stoi(input);

    std::cout << "Temperature [" << cfg.temperature << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.temperature = std::stof(input);

    return true;
}

bool Provider::edit_api_config(ApiProviderConfig& cfg) {
    std::cout << "\n--- API Settings ---\n";
    std::string input;

    std::cout << "API Key [" << (cfg.api_key.empty() ? "not set" : "****") << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.api_key = input;

    std::cout << "Base URL (leave empty for default) [" << cfg.base_url << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.base_url = input;

    std::cout << "Temperature [" << cfg.temperature << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.temperature = std::stof(input);

    std::cout << "Max tokens (0=auto) [" << cfg.max_tokens << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.max_tokens = std::stoi(input);

    return true;
}

bool Provider::edit_ollama_config(OllamaProviderConfig& cfg) {
    std::cout << "\n--- Ollama Settings ---\n";
    std::string input;

    std::cout << "Base URL [" << cfg.base_url << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.base_url = input;

    std::cout << "Temperature [" << cfg.temperature << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.temperature = std::stof(input);

    std::cout << "Context window (0=auto) [" << cfg.num_ctx << "]: ";
    std::getline(std::cin, input);
    if (!input.empty()) cfg.num_ctx = std::stoi(input);

    return true;
}

std::unique_ptr<ProviderConfig> Provider::parse_provider_args(const std::string& type, const std::vector<std::string>& args) {
    std::unique_ptr<ProviderConfig> config;

    // Create appropriate config type
    if (type == "llamacpp") {
        config = std::make_unique<LlamaProviderConfig>();
    }
    else if (type == "tensorrt") {
        config = std::make_unique<TensorRTProviderConfig>();
    }
    else if (type == "openai" || type == "anthropic" || type == "gemini" || type == "grok" || type == "cli") {
        config = std::make_unique<ApiProviderConfig>();
    }
    else if (type == "ollama") {
        config = std::make_unique<OllamaProviderConfig>();
    }
    else {
        throw std::runtime_error("Unknown provider type: " + type);
    }

    config->type = type;

    // Parse common arguments
    for (size_t i = 0; i < args.size(); i++) {
        const std::string& arg = args[i];

        if ((arg == "--name" || arg == "-n") && i + 1 < args.size()) {
            config->name = args[++i];
        }
        else if ((arg == "--model" || arg == "-m") && i + 1 < args.size()) {
            config->model = args[++i];
        }
        else if ((arg == "--priority" || arg == "-p") && i + 1 < args.size()) {
            config->priority = std::stoi(args[++i]);
        }
        else if (arg == "--tokens-per-month" && i + 1 < args.size()) {
            config->rate_limits.tokens_per_month = std::stoi(args[++i]);
        }
        else if (arg == "--max-cost" && i + 1 < args.size()) {
            config->rate_limits.max_cost_per_month = std::stof(args[++i]);
        }
        else if (arg == "--dynamic-pricing") {
            config->pricing.dynamic = true;
        }
        else if (arg == "--prompt-cost" && i + 1 < args.size()) {
            config->pricing.prompt_cost = std::stof(args[++i]);
        }
        else if (arg == "--completion-cost" && i + 1 < args.size()) {
            config->pricing.completion_cost = std::stof(args[++i]);
        }
    }

    // Parse type-specific arguments
    if (auto* llama = dynamic_cast<LlamaProviderConfig*>(config.get())) {
        for (size_t i = 0; i < args.size(); i++) {
            const std::string& arg = args[i];
            if (arg == "--tp" && i + 1 < args.size()) llama->tp = std::stoi(args[++i]);
            else if (arg == "--pp" && i + 1 < args.size()) llama->pp = std::stoi(args[++i]);
            else if (arg == "--gpu-layers" && i + 1 < args.size()) llama->gpu_layers = std::stoi(args[++i]);
        }
    }
    else if (auto* api = dynamic_cast<ApiProviderConfig*>(config.get())) {
        for (size_t i = 0; i < args.size(); i++) {
            const std::string& arg = args[i];
            if ((arg == "--key" || arg == "--api-key" || arg == "-k") && i + 1 < args.size()) {
                api->api_key = args[++i];
            }
            else if ((arg == "--base-url" || arg == "--url" || arg == "-u") && i + 1 < args.size()) {
                api->base_url = args[++i];
            }
        }
    }

    return config;
}

std::unique_ptr<Backend> Provider::connect_provider(const std::string& name, Session& session, size_t context_size) {
    auto provider_config = get_provider(name);
    if (!provider_config) {
        LOG_ERROR("Provider '" + name + "' not found");
        return nullptr;
    }

    // Use provider's context_size if set, otherwise use the passed-in value
    size_t effective_context_size = (provider_config->context_size > 0)
        ? provider_config->context_size
        : context_size;

    try {
        LOG_INFO("Connecting to provider: " + name + " (priority: " + std::to_string(provider_config->priority) + ")");
        auto backend = BackendFactory::create_from_provider(provider_config, effective_context_size);
        if (!backend) {
            LOG_ERROR("Failed to create backend for provider '" + name + "'");
            return nullptr;
        }

        backend->initialize(session);
        current_provider = name;
        LOG_INFO("Successfully connected to provider: " + name);
        return backend;

    } catch (const std::exception& e) {
        LOG_ERROR("Provider '" + name + "' failed: " + std::string(e.what()));
        return nullptr;
    }
}

std::unique_ptr<Backend> Provider::connect_next_provider(Session& session, size_t context_size) {
    auto provider_names = list_providers();
    if (provider_names.empty()) {
        LOG_ERROR("No providers configured");
        return nullptr;
    }

    for (const auto& name : provider_names) {
        auto backend = connect_provider(name, session, context_size);
        if (backend) {
            return backend;
        }
        // Failed - check if we should try next
        if (!config->auto_provider) {
            LOG_INFO("auto_provider disabled, not trying other providers");
            return nullptr;
        }
    }

    LOG_ERROR("All providers failed to connect");
    return nullptr;
}
// Common provider command implementation
int handle_provider_args(const std::vector<std::string>& args,
                         std::unique_ptr<Backend>* backend,
                         Session* session) {
	Provider provider_manager;
	provider_manager.load_providers();

	// No args shows current provider (interactive mode only)
	if (args.empty()) {
		std::string current = provider_manager.get_current_provider();
		if (current.empty()) {
			std::cout << "No provider configured\n";
		} else {
			auto prov = provider_manager.get_provider(current);
			if (prov) {
				std::cout << "Current provider: " << prov->name << "\n";
				std::cout << "  Type: " << prov->type << "\n";
				std::cout << "  Model: " << prov->model << "\n";
				if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
					if (!api->base_url.empty()) {
						std::cout << "  Base URL: " << api->base_url << "\n";
					}
				}
			}
		}
		return 0;
	}

	std::string subcmd = args[0];

	if (subcmd == "list") {
		auto providers = provider_manager.list_providers();
		if (providers.empty()) {
			std::cout << "No providers configured\n";
		} else {
			std::string current = provider_manager.get_current_provider();
			std::cout << "Available providers:\n";
			for (const auto& name : providers) {
				if (name == current) {
					std::cout << "  * " << name << " (current)\n";
				} else {
					std::cout << "    " << name << "\n";
				}
			}
		}
		return 0;
	}

	if (subcmd == "add") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider add <name> [--type <type>] [options...]\n";
			std::cerr << "Types: llamacpp, tensorrt, openai, anthropic, gemini, grok, ollama\n";
			return 1;
		}

		std::string name = args[1];
		std::vector<std::string> cmd_args(args.begin() + 2, args.end());

		// Check if provider already exists
		if (provider_manager.get_provider(name)) {
			std::cerr << "Provider '" << name << "' already exists. Use 'provider edit " << name << "' to modify.\n";
			return 1;
		}

		// If no additional args, create default and open interactive edit
		if (cmd_args.empty()) {
			// Create a default API provider config
			auto new_config = std::make_unique<ApiProviderConfig>();
			new_config->name = name;
			new_config->type = "openai";  // Default type, user can change in editor

			if (provider_manager.interactive_edit(*new_config)) {
				provider_manager.save_provider(*new_config);
				std::cout << "Provider '" << new_config->name << "' added successfully\n";
			} else {
				std::cout << "Add cancelled\n";
			}
			return 0;
		}

		// Parse --type from args
		std::string type = "openai";  // Default
		std::vector<std::string> remaining_args;
		for (size_t i = 0; i < cmd_args.size(); i++) {
			if (cmd_args[i] == "--type" && i + 1 < cmd_args.size()) {
				type = cmd_args[++i];
			} else {
				remaining_args.push_back(cmd_args[i]);
			}
		}

		try {
			auto new_config = provider_manager.parse_provider_args(type, remaining_args);
			new_config->name = name;

			provider_manager.save_provider(*new_config);
			std::cout << "Provider '" << new_config->name << "' added successfully\n";
			return 0;
		} catch (const std::exception& e) {
			std::cerr << "Error: " << e.what() << "\n";
			std::cerr << "Supported types: llamacpp, tensorrt, openai, anthropic, gemini, grok, ollama\n";
			return 1;
		}
	}

	if (subcmd == "show") {
		std::string name = (args.size() >= 2) ? args[1] : provider_manager.get_current_provider();
		if (name.empty()) {
			std::cout << "No provider specified\n";
			return 1;
		}

		auto prov = provider_manager.get_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		std::cout << "Provider: " << prov->name << "\n";
		std::cout << "  Type: " << prov->type << "\n";

		if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
			std::cout << "  API Key: " << (api->api_key.empty() ? "not set" : "****") << "\n";
			if (!api->base_url.empty()) {
				std::cout << "  Base URL: " << api->base_url << "\n";
			}
		} else if (auto* llama = dynamic_cast<LlamaProviderConfig*>(prov)) {
			std::cout << "  Model Path: " << llama->model_path << "\n";
			std::cout << "  TP: " << llama->tp << ", PP: " << llama->pp << "\n";
			std::cout << "  GPU Layers: " << llama->gpu_layers << "\n";
		}

		std::cout << "  Model: " << prov->model << "\n";

		if (prov->rate_limits.tokens_per_month > 0) {
			std::cout << "  Monthly token limit: " << prov->rate_limits.tokens_per_month << "\n";
		}
		if (prov->rate_limits.max_cost_per_month > 0) {
			std::cout << "  Monthly cost limit: $" << prov->rate_limits.max_cost_per_month << "\n";
		}

		if (prov->pricing.dynamic) {
			std::cout << "  Pricing: dynamic (from API)\n";
		} else if (prov->pricing.prompt_cost > 0 || prov->pricing.completion_cost > 0) {
			std::cout << "  Pricing: $" << prov->pricing.prompt_cost << " / $"
			          << prov->pricing.completion_cost << " per million tokens\n";
		}
		return 0;
	}

	if (subcmd == "edit") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider edit <name>\n";
			return 1;
		}

		std::string name = args[1];
		auto prov = provider_manager.get_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		if (provider_manager.interactive_edit(*prov)) {
			provider_manager.save_provider(*prov);
			std::cout << "Provider '" << prov->name << "' updated\n";
		} else {
			std::cout << "Edit cancelled\n";
		}
		return 0;
	}

	if (subcmd == "set") {
		if (args.size() < 3) {
			std::cerr << "Usage: provider set <name> <field> <value>\n";
			return 1;
		}

		std::string name = args[1];
		std::string field = args[2];
		std::string value = (args.size() >= 4) ? args[3] : "";

		auto prov = provider_manager.get_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		// Update specific field
		if (field == "model") {
			prov->model = value;
		} else if (field == "tokens_per_month") {
			prov->rate_limits.tokens_per_month = std::stoi(value);
		} else if (field == "max_cost") {
			prov->rate_limits.max_cost_per_month = std::stof(value);
		} else if (auto* api = dynamic_cast<ApiProviderConfig*>(prov)) {
			if (field == "api_key" || field == "key") {
				api->api_key = value;
			} else if (field == "base_url" || field == "url") {
				api->base_url = value;
			} else {
				std::cerr << "Unknown field: " << field << "\n";
				return 1;
			}
		} else {
			std::cerr << "Unknown field: " << field << "\n";
			return 1;
		}

		provider_manager.save_provider(*prov);
		std::cout << "Provider '" << name << "' updated\n";
		return 0;
	}

	if (subcmd == "remove") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider remove <name>\n";
			return 1;
		}

		std::string name = args[1];
		provider_manager.remove_provider(name);
		std::cout << "Provider '" << name << "' removed\n";
		return 0;
	}

	if (subcmd == "use") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider use <name>\n";
			return 1;
		}

		std::string name = args[1];
		auto prov = provider_manager.get_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Shutdown current backend first to free GPU memory
			size_t old_context_size = (*backend)->context_size;
			(*backend)->shutdown();

			// Connect to the specified provider (creates and initializes)
			auto new_backend = provider_manager.connect_provider(name, *session, old_context_size);
			if (!new_backend) {
				std::cerr << "Failed to connect to provider '" << name << "'\n";
				return 1;
			}

			// Update backend ownership and session pointer
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			std::cout << "Switched to provider '" << name << "' (" << prov->type << " / " << prov->model << ")\n";
		} else {
			// Command-line mode - just set current
			provider_manager.set_current_provider(name);
			std::cout << "Current provider set to: " << name << "\n";
		}
		return 0;
	}

	if (subcmd == "next") {
		auto next_name = provider_manager.get_next_provider();
		if (!next_name) {
			std::cout << "No available providers (all rate limited or none configured)\n";
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Shutdown current backend first to free GPU memory
			size_t old_context_size = (*backend)->context_size;
			(*backend)->shutdown();

			// Connect to the next provider (creates and initializes)
			auto new_backend = provider_manager.connect_provider(*next_name, *session, old_context_size);
			if (!new_backend) {
				std::cerr << "Failed to connect to provider '" << *next_name << "'\n";
				return 1;
			}

			// Get provider info for display
			auto prov = provider_manager.get_provider(*next_name);

			// Update backend ownership and session pointer
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			std::cout << "Switched to provider '" << *next_name << "'";
			if (prov) {
				std::cout << " (" << prov->type << " / " << prov->model << ")";
			}
			std::cout << "\n";
		} else {
			std::cout << "Next provider: " << *next_name << "\n";
		}
		return 0;
	}

	std::cerr << "Unknown provider subcommand: " << subcmd << "\n";
	std::cerr << "Available: list, add, show, edit, set, remove, use, next\n";
	return 1;
}

// Common model command implementation
int handle_model_args(const std::vector<std::string>& args,
                      std::unique_ptr<Backend>* backend) {
	Provider provider_manager;
	provider_manager.load_providers();

	// No args shows current model
	if (args.empty()) {
		std::string current_provider = provider_manager.get_current_provider();
		if (current_provider.empty()) {
			std::cout << "No provider configured\n";
		} else {
			auto prov = provider_manager.get_provider(current_provider);
			if (prov) {
				std::cout << "Current model: " << prov->model << "\n";
			}
		}
		return 0;
	}

	std::string subcmd = args[0];

	if (subcmd == "list") {
		std::cout << "Available models depend on your provider.\n";
		std::cout << "Refer to your provider's documentation for model list.\n";
		std::cout << "Common models:\n";
		std::cout << "  OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo\n";
		std::cout << "  Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku\n";
		std::cout << "  Google: gemini-pro, gemini-ultra\n";
		std::cout << "  OpenRouter: anthropic/claude-3.5-sonnet, google/gemini-pro\n";
		return 0;
	}

	if (subcmd == "set" && args.size() >= 2) {
		std::string model = args[1];
		std::string current_provider = provider_manager.get_current_provider();
		if (current_provider.empty()) {
			std::cout << "No provider configured\n";
			return 1;
		}

		auto prov = provider_manager.get_provider(current_provider);
		if (prov) {
			prov->model = model;
			provider_manager.save_provider(*prov);

			// Update backend's model if available
			if (backend && *backend) {
				(*backend)->model_name = model;
			}

			std::cout << "Model updated to: " << model << "\n";
			std::cout << "Note: Change takes effect on next message\n";
		}
		return 0;
	}

	std::cerr << "Unknown model subcommand: " << subcmd << "\n";
	std::cerr << "Available: list, set\n";
	return 1;
}
