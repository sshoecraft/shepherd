#include "provider.h"
#include "config.h"
#include "backend.h"
#include "backends/factory.h"
#include "session.h"
#include "shepherd.h"
#include "tools/tools.h"
#include "tools/api_tools.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ============================================================================
// New Provider class implementation
// ============================================================================

bool Provider::is_api() const {
    return type == "openai" || type == "anthropic" || type == "gemini" || type == "ollama";
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

std::vector<Provider> Provider::load_providers() {
    std::vector<Provider> result;
    std::string providers_dir = get_providers_dir();

    if (!fs::exists(providers_dir)) {
        fs::create_directories(providers_dir);
        return result;
    }

    for (const auto& entry : fs::directory_iterator(providers_dir)) {
        if (entry.path().extension() == ".json") {
            try {
                std::ifstream file(entry.path());
                json j;
                file >> j;

                Provider p = Provider::from_json(j);
                if (p.name.empty()) {
                    p.name = entry.path().stem().string();
                }
                result.push_back(std::move(p));
            } catch (const std::exception& e) {
                std::cerr << "Failed to load provider " + entry.path().string() + ": " + e.what() << std::endl;
            }
        }
    }

    // Sort by priority (lower = higher priority)
    std::sort(result.begin(), result.end(), [](const Provider& a, const Provider& b) {
        return a.priority < b.priority;
    });

    return result;
}

Provider Provider::from_json(const json& j) {
    Provider p;

    p.name = j.value("name", "");
    p.type = j.value("type", "");
    p.model = j.value("model", "");
    p.display_name = j.value("display_name", "");
    p.priority = j.value("priority", 100);
    p.context_size = j.value("context_size", 0);

    // Rate limits
    if (j.contains("rate_limits")) {
        auto& rl = j["rate_limits"];
        p.rate_limits.requests_per_second = rl.value("requests_per_second", 0);
        p.rate_limits.requests_per_minute = rl.value("requests_per_minute", 0);
        p.rate_limits.tokens_per_minute = rl.value("tokens_per_minute", 0);
        p.rate_limits.tokens_per_day = rl.value("tokens_per_day", 0);
        p.rate_limits.tokens_per_month = rl.value("tokens_per_month", 0);
        p.rate_limits.max_cost_per_month = rl.value("max_cost_per_month", 0.0f);
        p.rate_limits.warning_threshold = rl.value("warning_threshold", 0.85f);
    }

    // Pricing
    if (j.contains("pricing")) {
        auto& pr = j["pricing"];
        p.pricing.prompt_cost = pr.value("prompt_per_million", 0.0f);
        p.pricing.completion_cost = pr.value("completion_per_million", 0.0f);
        p.pricing.dynamic = pr.value("dynamic", false);
    }

    // Local backend fields
    p.model_path = j.value("model_path", "~/models");
    p.tp = j.value("tp", 1);
    p.pp = j.value("pp", 1);
    p.gpu_layers = j.value("gpu_layers", -1);
    p.gpu_id = j.value("gpu_id", 0);
    p.n_batch = j.value("n_batch", 512);
    p.ubatch = j.value("ubatch", 512);
    p.n_threads = j.value("n_threads", 0);
    p.cache_type = j.value("cache_type", "f16");

    // Sampling parameters
    p.temperature = j.value("temperature", 0.7f);
    p.top_p = j.value("top_p", 1.0f);
    p.top_k = j.value("top_k", 40);
    p.repeat_penalty = j.value("repeat_penalty", 1.1f);
    p.frequency_penalty = j.value("frequency_penalty", 0.0f);
    p.presence_penalty = j.value("presence_penalty", 0.0f);
    p.max_tokens = j.value("max_tokens", 0);

    if (j.contains("stop_sequences")) {
        p.stop_sequences = j["stop_sequences"].get<std::vector<std::string>>();
    }

    // API fields
    p.api_key = j.value("api_key", "");
    p.base_url = j.value("base_url", "");
    if (j.contains("extra_headers")) {
        p.extra_headers = j["extra_headers"].get<std::map<std::string, std::string>>();
    }
    p.ssl_verify = j.value("ssl_verify", true);
    p.ca_bundle_path = j.value("ca_bundle_path", "");

    // OAuth
    p.client_id = j.value("client_id", "");
    p.client_secret = j.value("client_secret", "");
    p.token_url = j.value("token_url", "");
    p.token_scope = j.value("token_scope", "");

    // Azure
    p.deployment_name = j.value("deployment_name", "");
    p.api_version = j.value("api_version", "");

    // Ollama
    p.num_ctx = j.value("num_ctx", 0);
    p.num_predict = j.value("num_predict", -1);

    return p;
}

json Provider::to_json() const {
    json j;

    j["name"] = name;
    j["type"] = type;
    j["model"] = model;
    if (!display_name.empty()) j["display_name"] = display_name;
    j["priority"] = priority;
    if (context_size > 0) j["context_size"] = context_size;

    // Rate limits
    if (rate_limits.requests_per_second > 0 || rate_limits.requests_per_minute > 0 ||
        rate_limits.tokens_per_minute > 0 || rate_limits.tokens_per_day > 0 ||
        rate_limits.tokens_per_month > 0 || rate_limits.max_cost_per_month > 0) {
        json rl;
        if (rate_limits.requests_per_second > 0) rl["requests_per_second"] = rate_limits.requests_per_second;
        if (rate_limits.requests_per_minute > 0) rl["requests_per_minute"] = rate_limits.requests_per_minute;
        if (rate_limits.tokens_per_minute > 0) rl["tokens_per_minute"] = rate_limits.tokens_per_minute;
        if (rate_limits.tokens_per_day > 0) rl["tokens_per_day"] = rate_limits.tokens_per_day;
        if (rate_limits.tokens_per_month > 0) rl["tokens_per_month"] = rate_limits.tokens_per_month;
        if (rate_limits.max_cost_per_month > 0) rl["max_cost_per_month"] = rate_limits.max_cost_per_month;
        rl["warning_threshold"] = rate_limits.warning_threshold;
        j["rate_limits"] = rl;
    }

    // Pricing
    if (pricing.prompt_cost > 0 || pricing.completion_cost > 0 || pricing.dynamic) {
        json pr;
        pr["prompt_per_million"] = pricing.prompt_cost;
        pr["completion_per_million"] = pricing.completion_cost;
        pr["dynamic"] = pricing.dynamic;
        j["pricing"] = pr;
    }

    // Type-specific fields
    if (type == "cli") {
        if (!base_url.empty()) j["base_url"] = base_url;
    }

    if (type == "llamacpp" || type == "tensorrt") {
        j["model_path"] = model_path;
        j["tp"] = tp;
        j["pp"] = pp;
        j["gpu_layers"] = gpu_layers;
        if (type == "tensorrt") j["gpu_id"] = gpu_id;
        if (type == "llamacpp") {
            j["n_batch"] = n_batch;
            j["ubatch"] = ubatch;
            if (n_threads > 0) j["n_threads"] = n_threads;
            j["cache_type"] = cache_type;
        }
    }

    if (is_api()) {
        if (!api_key.empty()) j["api_key"] = api_key;
        if (!base_url.empty()) j["base_url"] = base_url;
        if (!extra_headers.empty()) j["extra_headers"] = extra_headers;
        if (!ssl_verify) j["ssl_verify"] = ssl_verify;
        if (!ca_bundle_path.empty()) j["ca_bundle_path"] = ca_bundle_path;

        if (!client_id.empty()) j["client_id"] = client_id;
        if (!client_secret.empty()) j["client_secret"] = client_secret;
        if (!token_url.empty()) j["token_url"] = token_url;
        if (!token_scope.empty()) j["token_scope"] = token_scope;

        if (!deployment_name.empty()) j["deployment_name"] = deployment_name;
        if (!api_version.empty()) j["api_version"] = api_version;
    }

    if (type == "ollama") {
        if (num_ctx > 0) j["num_ctx"] = num_ctx;
        if (num_predict != -1) j["num_predict"] = num_predict;
    }

    // Sampling (for backends that use it)
    j["temperature"] = temperature;
    j["top_p"] = top_p;
    j["top_k"] = top_k;
    j["repeat_penalty"] = repeat_penalty;
    if (frequency_penalty != 0.0f) j["frequency_penalty"] = frequency_penalty;
    if (presence_penalty != 0.0f) j["presence_penalty"] = presence_penalty;
    if (max_tokens > 0) j["max_tokens"] = max_tokens;
    if (!stop_sequences.empty()) j["stop_sequences"] = stop_sequences;

    return j;
}

Provider Provider::from_config() {
    extern std::unique_ptr<Config> config;

    Provider p;
    p.type = config->backend;
    p.model = config->model;
    p.model_path = config->model_path;
    p.api_key = config->key;
    p.base_url = config->api_base;
    p.context_size = config->context_size;

    // Pull additional settings from config->json if present
    if (config->json.contains("temperature")) p.temperature = config->json["temperature"];
    if (config->json.contains("top_p")) p.top_p = config->json["top_p"];
    if (config->json.contains("top_k")) p.top_k = config->json["top_k"];
    if (config->json.contains("repeat_penalty")) p.repeat_penalty = config->json["repeat_penalty"];
    if (config->json.contains("frequency_penalty")) p.frequency_penalty = config->json["frequency_penalty"];
    if (config->json.contains("presence_penalty")) p.presence_penalty = config->json["presence_penalty"];
    if (config->json.contains("max_tokens")) p.max_tokens = config->json["max_tokens"];
    if (config->json.contains("tp")) p.tp = config->json["tp"];
    if (config->json.contains("pp")) p.pp = config->json["pp"];
    if (config->json.contains("gpu_layers")) p.gpu_layers = config->json["gpu_layers"];
    if (config->json.contains("gpu_id")) p.gpu_id = config->json["gpu_id"];
    if (config->json.contains("n_batch")) p.n_batch = config->json["n_batch"];
    if (config->json.contains("ubatch")) p.ubatch = config->json["ubatch"];
    if (config->json.contains("n_threads")) p.n_threads = config->json["n_threads"];
    if (config->json.contains("cache_type")) p.cache_type = config->json["cache_type"];
    if (config->json.contains("num_ctx")) p.num_ctx = config->json["num_ctx"];
    if (config->json.contains("num_predict")) p.num_predict = config->json["num_predict"];

    return p;
}

void Provider::save() const {
    std::string providers_dir = get_providers_dir();
    fs::create_directories(providers_dir);

    std::string filename = providers_dir + "/" + name + ".json";
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open provider file for writing: " + filename);
    }

    file << to_json().dump(4) << std::endl;
}

void Provider::remove(const std::string& name) {
    std::string providers_dir = get_providers_dir();
    std::string filename = providers_dir + "/" + name + ".json";

    if (fs::exists(filename)) {
        fs::remove(filename);
    }
}

std::unique_ptr<Backend> Provider::connect(Session& session, Backend::EventCallback callback) {
    extern std::unique_ptr<Config> config;

    dout(1) << "Connecting to provider: " + name + " (type: " + type + ")" << std::endl;

    // Print loading message (not for mpirun children)
    if (!getenv("OMPI_COMM_WORLD_SIZE")) {
        dout(1) << "Loading provider: " + name << std::endl;
    }

    // Set config globals from this provider
    config->model = model;
    config->backend = type;

    if (is_api()) {
        config->key = api_key;
        config->api_base = base_url;
        config->json["temperature"] = temperature;
        config->json["top_p"] = top_p;
        if (top_k > 0) config->json["top_k"] = top_k;
        config->json["repeat_penalty"] = repeat_penalty;
        config->json["frequency_penalty"] = frequency_penalty;
        config->json["presence_penalty"] = presence_penalty;
        if (max_tokens > 0) config->json["max_tokens"] = max_tokens;
        config->json["ssl_verify"] = ssl_verify;
        if (!ca_bundle_path.empty()) config->json["ca_bundle_path"] = ca_bundle_path;
        if (!client_id.empty()) config->json["client_id"] = client_id;
        if (!client_secret.empty()) config->json["client_secret"] = client_secret;
        if (!token_url.empty()) config->json["token_url"] = token_url;
        if (!token_scope.empty()) config->json["token_scope"] = token_scope;
        if (!deployment_name.empty()) config->json["deployment_name"] = deployment_name;
        if (!api_version.empty()) config->json["api_version"] = api_version;
    } else if (type == "llamacpp") {
        config->model_path = model_path;
        config->json["tp"] = tp;
        config->json["pp"] = pp;
        config->json["gpu_layers"] = gpu_layers;
        config->json["temperature"] = temperature;
        config->json["top_p"] = top_p;
        config->json["top_k"] = top_k;
        config->json["repeat_penalty"] = repeat_penalty;
        config->json["n_batch"] = n_batch;
        config->json["ubatch"] = ubatch;
        config->json["cache_type"] = cache_type;
        if (n_threads > 0) config->json["n_threads"] = n_threads;
    } else if (type == "tensorrt") {
        config->model_path = model_path;
        config->json["tp"] = tp;
        config->json["pp"] = pp;
        config->json["gpu_id"] = gpu_id;
        config->json["temperature"] = temperature;
        config->json["top_p"] = top_p;
        config->json["top_k"] = top_k;
        config->json["repeat_penalty"] = repeat_penalty;
        config->json["frequency_penalty"] = frequency_penalty;
        config->json["presence_penalty"] = presence_penalty;
    } else if (type == "ollama") {
        config->api_base = base_url.empty() ? "http://localhost:11434" : base_url;
        config->json["temperature"] = temperature;
        config->json["top_p"] = top_p;
        config->json["top_k"] = top_k;
        config->json["repeat_penalty"] = repeat_penalty;
        if (num_ctx > 0) config->json["num_ctx"] = num_ctx;
        if (num_predict != -1) config->json["num_predict"] = num_predict;
    } else if (type == "cli") {
        config->api_base = base_url.empty() ? "http://localhost:8000" : base_url;
    }

    // Create backend (constructor handles all initialization)
    // Command line --context-size takes precedence over provider config
    size_t ctx = (config->context_size > 0) ? config->context_size : context_size;
    auto backend = BackendFactory::create_backend(type, ctx, session, callback);

    // Set display name from provider config (for /v1/models API)
    if (backend) {
        backend->display_name = display_name;
    }

    return backend;
}

// Common provider command implementation
int handle_provider_args(const std::vector<std::string>& args,
                         std::function<void(const std::string&)> callback,
                         std::unique_ptr<Backend>* backend,
                         Session* session,
                         std::vector<Provider>* providers_ptr,
                         std::string* current_provider,
                         Tools* tools) {
	// Load providers if not passed in
	std::vector<Provider> local_providers;
	std::vector<Provider>& providers = providers_ptr ? *providers_ptr : local_providers;
	if (!providers_ptr) {
		local_providers = Provider::load_providers();
	}

	// Helper to find provider by name
	auto find_provider = [&providers](const std::string& name) -> Provider* {
		for (auto& p : providers) {
			if (p.name == name) return &p;
		}
		return nullptr;
	};

	// No args shows current provider (interactive mode only)
	if (args.empty()) {
		if (!current_provider || current_provider->empty()) {
			callback("No provider configured\n");
		} else {
			auto* prov = find_provider(*current_provider);
			if (prov) {
				callback("Current provider: " + prov->name + "\n");
				callback("  Type: " + prov->type + "\n");
				callback("  Model: " + prov->model + "\n");
				if (prov->is_api() && !prov->base_url.empty()) {
					callback("  Base URL: " + prov->base_url + "\n");
				}
			}
		}
		return 0;
	}

	std::string subcmd = args[0];

	if (subcmd == "help" || subcmd == "--help" || subcmd == "-h") {
		callback("Usage: /provider [subcommand]\n"
		    "Subcommands:\n"
		    "  list           - List all providers\n"
		    "  use <name>     - Switch to provider\n"
		    "  show <name>    - Show provider details\n"
		    "  add <name>     - Add new provider\n"
		    "  set <name>     - Modify provider settings\n"
		    "  edit <name>    - Edit provider JSON in $EDITOR\n"
		    "  remove <name>  - Remove provider\n"
		    "  next           - Try next provider\n"
		    "  (no args)      - Show current provider\n");
		return 0;
	}

	if (subcmd == "list") {
		if (providers.empty()) {
			callback("No providers configured\n");
		} else {
			callback("Available providers:\n");
			for (const auto& p : providers) {
				if (current_provider && p.name == *current_provider) {
					callback("  * " + p.name + " (current)\n");
				} else {
					callback("    " + p.name + "\n");
				}
			}
		}
		return 0;
	}

	if (subcmd == "add") {
		if (args.size() < 2) {
			callback("Usage: provider add <name> --type <type> [options...]\n");
			callback("Types: llamacpp, tensorrt, openai, anthropic, gemini, ollama, cli\n");
			return 1;
		}

		std::string name = args[1];
		std::vector<std::string> cmd_args(args.begin() + 2, args.end());

		// Check if provider already exists
		if (find_provider(name)) {
			callback("Provider '" + name + "' already exists. Use 'provider set' to modify.\n");
			return 1;
		}

		// Parse args
		Provider new_prov;
		new_prov.name = name;
		new_prov.type = "openai";  // Default

		for (size_t i = 0; i < cmd_args.size(); i++) {
			if (cmd_args[i] == "--type" && i + 1 < cmd_args.size()) {
				new_prov.type = cmd_args[++i];
			} else if (cmd_args[i] == "--model" && i + 1 < cmd_args.size()) {
				new_prov.model = cmd_args[++i];
			} else if (cmd_args[i] == "--api-key" && i + 1 < cmd_args.size()) {
				new_prov.api_key = cmd_args[++i];
			} else if (cmd_args[i] == "--base-url" && i + 1 < cmd_args.size()) {
				new_prov.base_url = cmd_args[++i];
			} else if (cmd_args[i] == "--model-path" && i + 1 < cmd_args.size()) {
				new_prov.model_path = cmd_args[++i];
			} else if (cmd_args[i] == "--priority" && i + 1 < cmd_args.size()) {
				new_prov.priority = std::stoi(cmd_args[++i]);
			}
		}

		new_prov.save();
		providers.push_back(new_prov);
		callback("Provider '" + name + "' added successfully\n");
		return 0;
	}

	if (subcmd == "show") {
		std::string name = (args.size() >= 2) ? args[1] : (current_provider ? *current_provider : "");
		if (name.empty()) {
			callback("No provider specified\n");
			return 1;
		}

		auto* prov = find_provider(name);
		if (!prov) {
			callback("Provider '" + name + "' not found\n");
			return 1;
		}

		callback("=== Provider: " + prov->name + " ===\n");

		// Core fields
		callback("type = " + prov->type + "\n");
		if (prov->type != "cli") {
			callback("model = " + prov->model + "\n");
		}
		callback("priority = " + std::to_string(prov->priority) + "\n");
		callback("context_size = " + std::to_string(prov->context_size) + (prov->context_size == 0 ? " (auto)" : "") + "\n");
		callback("display_name = " + (prov->display_name.empty() ? "(not set)" : prov->display_name) + "\n");

		// CLI backend fields
		if (prov->type == "cli") {
			callback("base_url = " + (prov->base_url.empty() ? "http://localhost:8000" : prov->base_url) + "\n");
		}

		// API backend fields
		if (prov->is_api()) {
			callback("api_key = " + std::string(prov->api_key.empty() ? "(not set)" : "****") + "\n");
			callback("base_url = " + (prov->base_url.empty() ? "(not set)" : prov->base_url) + "\n");
			callback("client_id = " + (prov->client_id.empty() ? "(not set)" : prov->client_id) + "\n");
			callback("client_secret = " + std::string(prov->client_secret.empty() ? "(not set)" : "****") + "\n");
			callback("token_url = " + (prov->token_url.empty() ? "(not set)" : prov->token_url) + "\n");
			callback("token_scope = " + (prov->token_scope.empty() ? "(not set)" : prov->token_scope) + "\n");
			callback("deployment_name = " + (prov->deployment_name.empty() ? "(not set)" : prov->deployment_name) + "\n");
			callback("api_version = " + (prov->api_version.empty() ? "(not set)" : prov->api_version) + "\n");
			callback("ssl_verify = " + std::string(prov->ssl_verify ? "true" : "false") + "\n");
			callback("ca_bundle_path = " + (prov->ca_bundle_path.empty() ? "(not set)" : prov->ca_bundle_path) + "\n");
		}

		// Local backend fields
		if (prov->type == "llamacpp" || prov->type == "tensorrt") {
			callback("model_path = " + prov->model_path + "\n");
			callback("tp = " + std::to_string(prov->tp) + "\n");
			callback("pp = " + std::to_string(prov->pp) + "\n");
			callback("gpu_layers = " + std::to_string(prov->gpu_layers) + "\n");
			if (prov->type == "tensorrt") {
				callback("gpu_id = " + std::to_string(prov->gpu_id) + "\n");
			}
			if (prov->type == "llamacpp") {
				callback("n_batch = " + std::to_string(prov->n_batch) + "\n");
				callback("ubatch = " + std::to_string(prov->ubatch) + "\n");
				callback("n_threads = " + std::to_string(prov->n_threads) + (prov->n_threads == 0 ? " (auto)" : "") + "\n");
				callback("cache_type = " + prov->cache_type + "\n");
			}
		}

		// Ollama specific
		if (prov->type == "ollama") {
			callback("num_ctx = " + std::to_string(prov->num_ctx) + (prov->num_ctx == 0 ? " (auto)" : "") + "\n");
			callback("num_predict = " + std::to_string(prov->num_predict) + (prov->num_predict == -1 ? " (auto)" : "") + "\n");
		}

		// Sampling parameters (not used by CLI backend)
		if (prov->type != "cli") {
			callback("temperature = " + std::to_string(prov->temperature) + "\n");
			callback("top_p = " + std::to_string(prov->top_p) + "\n");
			callback("top_k = " + std::to_string(prov->top_k) + "\n");
			callback("repeat_penalty = " + std::to_string(prov->repeat_penalty) + "\n");
			callback("frequency_penalty = " + std::to_string(prov->frequency_penalty) + "\n");
			callback("presence_penalty = " + std::to_string(prov->presence_penalty) + "\n");
			callback("max_tokens = " + std::to_string(prov->max_tokens) + (prov->max_tokens == 0 ? " (auto)" : "") + "\n");
			std::string stops = "stop_sequences = [";
			for (size_t i = 0; i < prov->stop_sequences.size(); i++) {
				if (i > 0) stops += ", ";
				stops += "\"" + prov->stop_sequences[i] + "\"";
			}
			stops += "]\n";
			callback(stops);
		}

		// Rate limits
		callback("requests_per_second = " + std::to_string(prov->rate_limits.requests_per_second) + (prov->rate_limits.requests_per_second == 0 ? " (unlimited)" : "") + "\n");
		callback("requests_per_minute = " + std::to_string(prov->rate_limits.requests_per_minute) + (prov->rate_limits.requests_per_minute == 0 ? " (unlimited)" : "") + "\n");
		callback("tokens_per_minute = " + std::to_string(prov->rate_limits.tokens_per_minute) + (prov->rate_limits.tokens_per_minute == 0 ? " (unlimited)" : "") + "\n");
		callback("tokens_per_day = " + std::to_string(prov->rate_limits.tokens_per_day) + (prov->rate_limits.tokens_per_day == 0 ? " (unlimited)" : "") + "\n");
		callback("tokens_per_month = " + std::to_string(prov->rate_limits.tokens_per_month) + (prov->rate_limits.tokens_per_month == 0 ? " (unlimited)" : "") + "\n");
		callback("max_cost_per_month = " + std::to_string(prov->rate_limits.max_cost_per_month) + (prov->rate_limits.max_cost_per_month == 0 ? " (unlimited)" : "") + "\n");

		// Pricing
		if (prov->pricing.dynamic) {
			callback("pricing_dynamic = true\n");
		}
		if (prov->pricing.prompt_cost > 0) {
			callback("prompt_cost_per_million = " + std::to_string(prov->pricing.prompt_cost) + "\n");
		}
		if (prov->pricing.completion_cost > 0) {
			callback("completion_cost_per_million = " + std::to_string(prov->pricing.completion_cost) + "\n");
		}

		callback("\nUse 'provider set " + prov->name + " <field> <value>' to modify\n");
		return 0;
	}

	if (subcmd == "edit") {
		if (args.size() < 2) {
			callback("Usage: provider edit <name>\n");
			return 1;
		}

		std::string name = args[1];
		auto* prov = find_provider(name);
		if (!prov) {
			callback("Provider '" + name + "' not found\n");
			return 1;
		}

		callback("Editing provider: " + prov->name + "\n");
		callback("Press Enter to keep current value, or type new value:\n\n");

		// Note: edit command requires interactive stdin - only works in CLI mode
		auto prompt = [&callback](const std::string& field, const std::string& current) -> std::string {
			callback(field + " [" + current + "]: ");
			std::string input;
			std::getline(std::cin, input);
			return input.empty() ? current : input;
		};

		auto prompt_int = [&prompt](const std::string& field, int current) -> int {
			std::string result = prompt(field, std::to_string(current));
			try {
				return std::stoi(result);
			} catch (...) {
				return current;
			}
		};

		auto prompt_float = [&prompt](const std::string& field, float current) -> float {
			std::string result = prompt(field, std::to_string(current));
			try {
				return std::stof(result);
			} catch (...) {
				return current;
			}
		};

		// Core fields
		prov->type = prompt("type", prov->type);
		prov->priority = prompt_int("priority", prov->priority);

		// Type-specific fields
		if (prov->type == "cli") {
			// CLI backend only needs base_url (host:port)
			callback("\n--- CLI Backend Settings ---\n");
			std::string default_url = prov->base_url.empty() ? "http://localhost:8000" : prov->base_url;
			prov->base_url = prompt("base_url", default_url);
		} else if (prov->is_api()) {
			callback("\n--- API Backend Settings ---\n");
			prov->model = prompt("model", prov->model);
			std::string masked_key = prov->api_key.empty() ? "" : "****";
			std::string new_key = prompt("api_key", masked_key);
			if (new_key != masked_key && !new_key.empty()) {
				prov->api_key = new_key;
			}
			// Always show base_url for API backends - needed for OpenAI-compatible APIs
			prov->base_url = prompt("base_url", prov->base_url);

			// Sampling parameters for API backends
			callback("\n--- Sampling Parameters ---\n");
			prov->temperature = prompt_float("temperature", prov->temperature);
			prov->top_p = prompt_float("top_p", prov->top_p);
			prov->top_k = prompt_int("top_k", prov->top_k);
			prov->repeat_penalty = prompt_float("repeat_penalty", prov->repeat_penalty);
			prov->frequency_penalty = prompt_float("frequency_penalty", prov->frequency_penalty);
			prov->presence_penalty = prompt_float("presence_penalty", prov->presence_penalty);
		} else if (prov->type == "llamacpp" || prov->type == "tensorrt") {
			callback("\n--- Local Backend Settings ---\n");
			prov->model = prompt("model", prov->model);
			prov->model_path = prompt("model_path", prov->model_path);
			prov->gpu_layers = prompt_int("gpu_layers", prov->gpu_layers);
			prov->tp = prompt_int("tp", prov->tp);
			prov->pp = prompt_int("pp", prov->pp);

			if (prov->type == "llamacpp") {
				prov->n_batch = prompt_int("n_batch", prov->n_batch);
				prov->ubatch = prompt_int("ubatch", prov->ubatch);
				prov->n_threads = prompt_int("n_threads", prov->n_threads);
				prov->cache_type = prompt("cache_type", prov->cache_type);
			} else if (prov->type == "tensorrt") {
				prov->gpu_id = prompt_int("gpu_id", prov->gpu_id);
			}

			// Sampling parameters
			callback("\n--- Sampling Parameters ---\n");
			prov->temperature = prompt_float("temperature", prov->temperature);
			prov->top_p = prompt_float("top_p", prov->top_p);
			prov->top_k = prompt_int("top_k", prov->top_k);
			prov->repeat_penalty = prompt_float("repeat_penalty", prov->repeat_penalty);
			prov->frequency_penalty = prompt_float("frequency_penalty", prov->frequency_penalty);
			prov->presence_penalty = prompt_float("presence_penalty", prov->presence_penalty);
		} else if (prov->type == "ollama") {
			callback("\n--- Ollama Settings ---\n");
			prov->model = prompt("model", prov->model);
			prov->base_url = prompt("base_url", prov->base_url.empty() ? "http://localhost:11434" : prov->base_url);
			prov->num_ctx = prompt_int("num_ctx", prov->num_ctx);
			prov->num_predict = prompt_int("num_predict", prov->num_predict);

			// Sampling parameters
			callback("\n--- Sampling Parameters ---\n");
			prov->temperature = prompt_float("temperature", prov->temperature);
			prov->top_p = prompt_float("top_p", prov->top_p);
			prov->top_k = prompt_int("top_k", prov->top_k);
			prov->repeat_penalty = prompt_float("repeat_penalty", prov->repeat_penalty);
			prov->frequency_penalty = prompt_float("frequency_penalty", prov->frequency_penalty);
			prov->presence_penalty = prompt_float("presence_penalty", prov->presence_penalty);
		}

		// Confirm save
		callback("\nSave changes? (y/n): ");
		std::string confirm;
		std::getline(std::cin, confirm);
		if (confirm == "y" || confirm == "Y" || confirm == "yes" || confirm == "Yes") {
			prov->save();
			callback("Provider '" + name + "' updated successfully\n");
		} else {
			callback("Edit cancelled\n");
		}
		return 0;
	}

	if (subcmd == "set") {
		if (args.size() < 4) {
			callback("Usage: provider set <name> <field> <value>\n"
			         "\nCommon fields:\n"
			         "  type, model, priority, context_size\n"
			         "\nAPI backends (openai, anthropic, gemini, ollama):\n"
			         "  api_key, base_url, client_id, client_secret, token_url, token_scope\n"
			         "  deployment_name, api_version, ssl_verify, ca_bundle_path\n"
			         "\nLocal backends (llamacpp, tensorrt):\n"
			         "  model_path, tp, pp, gpu_layers, gpu_id, n_batch, ubatch, n_threads, cache_type\n"
			         "\nOllama:\n"
			         "  num_ctx, num_predict\n"
			         "\nSampling:\n"
			         "  temperature, top_p, top_k, repeat_penalty, frequency_penalty,\n"
			         "  presence_penalty, max_tokens\n"
			         "\nRate limits:\n"
			         "  requests_per_second, requests_per_minute, tokens_per_minute,\n"
			         "  tokens_per_day, tokens_per_month, max_cost_per_month\n"
			         "\nPricing:\n"
			         "  pricing_dynamic, prompt_cost_per_million, completion_cost_per_million\n");
			return 1;
		}

		std::string name = args[1];
		std::string field = args[2];
		std::string value = args[3];

		auto* prov = find_provider(name);
		if (!prov) {
			callback("Provider '" + name + "' not found\n");
			return 1;
		}

		// Update specific field
		bool updated = true;
		try {
			// Core fields
			if (field == "type") {
				prov->type = value;
			} else if (field == "model") {
				prov->model = value;
			} else if (field == "priority") {
				prov->priority = std::stoi(value);
			} else if (field == "context_size") {
				prov->context_size = std::stoull(value);
			}
			// API fields
			else if (field == "api_key") {
				prov->api_key = value;
			} else if (field == "base_url") {
				prov->base_url = value;
			} else if (field == "client_id") {
				prov->client_id = value;
			} else if (field == "client_secret") {
				prov->client_secret = value;
			} else if (field == "token_url") {
				prov->token_url = value;
			} else if (field == "token_scope") {
				prov->token_scope = value;
			} else if (field == "deployment_name") {
				prov->deployment_name = value;
			} else if (field == "api_version") {
				prov->api_version = value;
			} else if (field == "ssl_verify") {
				prov->ssl_verify = (value == "true" || value == "1" || value == "yes");
			} else if (field == "ca_bundle_path") {
				prov->ca_bundle_path = value;
			}
			// Local backend fields
			else if (field == "model_path") {
				prov->model_path = value;
			} else if (field == "tp") {
				prov->tp = std::stoi(value);
			} else if (field == "pp") {
				prov->pp = std::stoi(value);
			} else if (field == "gpu_layers") {
				prov->gpu_layers = std::stoi(value);
			} else if (field == "gpu_id") {
				prov->gpu_id = std::stoi(value);
			} else if (field == "n_batch") {
				prov->n_batch = std::stoi(value);
			} else if (field == "ubatch") {
				prov->ubatch = std::stoi(value);
			} else if (field == "n_threads") {
				prov->n_threads = std::stoi(value);
			} else if (field == "cache_type") {
				if (value != "f16" && value != "f32" && value != "q8_0" && value != "q4_0") {
					callback("Error: cache_type must be one of: f16, f32, q8_0, q4_0\n");
					return 1;
				}
				prov->cache_type = value;
			}
			// Ollama fields
			else if (field == "num_ctx") {
				prov->num_ctx = std::stoi(value);
			} else if (field == "num_predict") {
				prov->num_predict = std::stoi(value);
			}
			// Sampling parameters
			else if (field == "temperature") {
				prov->temperature = std::stof(value);
			} else if (field == "top_p") {
				prov->top_p = std::stof(value);
			} else if (field == "top_k") {
				prov->top_k = std::stoi(value);
			} else if (field == "repeat_penalty") {
				prov->repeat_penalty = std::stof(value);
			} else if (field == "frequency_penalty") {
				prov->frequency_penalty = std::stof(value);
			} else if (field == "presence_penalty") {
				prov->presence_penalty = std::stof(value);
			} else if (field == "max_tokens") {
				prov->max_tokens = std::stoi(value);
			}
			// Rate limits
			else if (field == "requests_per_second") {
				prov->rate_limits.requests_per_second = std::stoi(value);
			} else if (field == "requests_per_minute") {
				prov->rate_limits.requests_per_minute = std::stoi(value);
			} else if (field == "tokens_per_minute") {
				prov->rate_limits.tokens_per_minute = std::stoi(value);
			} else if (field == "tokens_per_day") {
				prov->rate_limits.tokens_per_day = std::stoi(value);
			} else if (field == "tokens_per_month") {
				prov->rate_limits.tokens_per_month = std::stoi(value);
			} else if (field == "max_cost_per_month") {
				prov->rate_limits.max_cost_per_month = std::stof(value);
			}
			// Pricing
			else if (field == "pricing_dynamic") {
				prov->pricing.dynamic = (value == "true" || value == "1" || value == "yes");
			} else if (field == "prompt_cost_per_million") {
				prov->pricing.prompt_cost = std::stof(value);
			} else if (field == "completion_cost_per_million") {
				prov->pricing.completion_cost = std::stof(value);
			}
			else {
				updated = false;
				callback("Unknown field: " + field + "\n");
				callback("Use 'provider set' without arguments to see available fields\n");
				return 1;
			}
		} catch (const std::exception& e) {
			callback("Error setting field '" + field + "': " + e.what() + "\n");
			return 1;
		}

		if (updated) {
			prov->save();
			callback("Provider '" + name + "' updated: " + field + " = " + value + "\n");
		}
		return 0;
	}

	if (subcmd == "remove") {
		if (args.size() < 2) {
			callback("Usage: provider remove <name>\n");
			return 1;
		}

		std::string name = args[1];
		Provider::remove(name);

		// Remove from in-memory list
		auto it = std::remove_if(providers.begin(), providers.end(),
			[&name](const Provider& p) { return p.name == name; });
		providers.erase(it, providers.end());

		callback("Provider '" + name + "' removed\n");
		return 0;
	}

	if (subcmd == "use") {
		if (args.size() < 2) {
			callback("Usage: provider use <name>\n");
			return 1;
		}

		std::string name = args[1];
		auto* prov = find_provider(name);
		if (!prov) {
			callback("Provider '" + name + "' not found\n");
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Get event callback from current backend
			auto event_cb = (*backend)->callback;

			// Output loading message before attempting connection
			callback("Loading Provider: " + name + "\n");

			// Try to connect to new provider BEFORE shutting down old one
			std::unique_ptr<Backend> new_backend;
			try {
				new_backend = prov->connect(*session, event_cb);
			} catch (const std::exception& e) {
				// Emit error via callback and keep current backend
				if (event_cb) {
					event_cb(CallbackEvent::SYSTEM, std::string("Error switching provider: ") + e.what() + "\n", "", "");
				}
				return 1;
			}

			if (!new_backend) {
				if (event_cb) {
					event_cb(CallbackEvent::SYSTEM, "Failed to connect to provider '" + name + "'\n", "", "");
				}
				return 1;
			}

			// Success - now shutdown old and swap
			(*backend)->shutdown();
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			// Check if session needs context adjustment for new provider
			size_t new_ctx = (*backend)->context_size;
			if (new_ctx > 0 && session->total_tokens > 0) {
				// Reserve space for at least one response
				int reserved = session->desired_completion_tokens > 0 ? session->desired_completion_tokens : 2048;
				int available = static_cast<int>(new_ctx) - reserved;

				if (session->total_tokens > available) {
					int tokens_over = session->total_tokens - available;
					callback("Context adjustment: evicting " + std::to_string(tokens_over) +
					         " tokens to fit new provider's " + std::to_string(new_ctx) + " context\n");

					auto ranges = session->calculate_messages_to_evict(tokens_over);
					if (!ranges.empty()) {
						session->evict_messages(ranges);
					}
				}
			}

			if (current_provider) {
				*current_provider = name;
			}

			// Rebuild provider tools (excluding new active provider)
			// Only if native tools exist (tools not disabled via --notools)
			if (tools && tools->has_native_tools()) {
				register_provider_tools(*tools, name);
				tools->populate_session_tools(*session);
				// Update backend's valid tool names
				(*backend)->valid_tool_names.clear();
				for (const auto& tool : session->tools) {
					(*backend)->valid_tool_names.insert(tool.name);
				}
			}

			callback("Switched to provider '" + name + "' (" + prov->type + " / " + prov->model + ")\n");
		} else {
			callback("Provider '" + name + "' selected\n");
		}
		return 0;
	}

	if (subcmd == "next") {
		// Find next provider (skip current and priority 0 ephemeral providers)
		Provider* next_prov = nullptr;
		for (auto& p : providers) {
			if (p.priority == 0) continue;  // Skip ephemeral
			if (current_provider && p.name == *current_provider) continue;
			next_prov = &p;
			break;
		}

		if (!next_prov) {
			callback("No other providers available\n");
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Get event callback from current backend
			auto event_cb = (*backend)->callback;

			// Output loading message before attempting connection
			callback("Loading Provider: " + next_prov->name + "\n");

			// Try to connect to new provider BEFORE shutting down old one
			std::unique_ptr<Backend> new_backend;
			try {
				new_backend = next_prov->connect(*session, event_cb);
			} catch (const std::exception& e) {
				if (event_cb) {
					event_cb(CallbackEvent::SYSTEM, std::string("Error switching provider: ") + e.what() + "\n", "", "");
				}
				return 1;
			}

			if (!new_backend) {
				if (event_cb) {
					event_cb(CallbackEvent::SYSTEM, "Failed to connect to provider '" + next_prov->name + "'\n", "", "");
				}
				return 1;
			}

			// Success - now shutdown old and swap
			(*backend)->shutdown();
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			// Check if session needs context adjustment for new provider
			size_t new_ctx = (*backend)->context_size;
			if (new_ctx > 0 && session->total_tokens > 0) {
				// Reserve space for at least one response
				int reserved = session->desired_completion_tokens > 0 ? session->desired_completion_tokens : 2048;
				int available = static_cast<int>(new_ctx) - reserved;

				if (session->total_tokens > available) {
					int tokens_over = session->total_tokens - available;
					callback("Context adjustment: evicting " + std::to_string(tokens_over) +
					         " tokens to fit new provider's " + std::to_string(new_ctx) + " context\n");

					auto ranges = session->calculate_messages_to_evict(tokens_over);
					if (!ranges.empty()) {
						session->evict_messages(ranges);
					}
				}
			}

			if (current_provider) {
				*current_provider = next_prov->name;
			}

			// Rebuild provider tools (excluding new active provider)
			// Only if native tools exist (tools not disabled via --notools)
			if (tools && tools->has_native_tools()) {
				register_provider_tools(*tools, next_prov->name);
				tools->populate_session_tools(*session);
				// Update backend's valid tool names
				(*backend)->valid_tool_names.clear();
				for (const auto& tool : session->tools) {
					(*backend)->valid_tool_names.insert(tool.name);
				}
			}

			callback("Switched to provider '" + next_prov->name + "' (" + next_prov->type + " / " + next_prov->model + ")\n");
		} else {
			callback("Next provider: " + next_prov->name + "\n");
		}
		return 0;
	}

	callback("Unknown provider subcommand: " + subcmd + "\n");
	callback("Available: list, add, show, set, remove, use, next\n");
	return 1;
}

// Common model command implementation
int handle_model_args(const std::vector<std::string>& args,
                      std::function<void(const std::string&)> callback,
                      std::unique_ptr<Backend>* backend,
                      std::vector<Provider>* providers_ptr,
                      std::string* current_provider_name) {
	// Load providers if not passed in
	std::vector<Provider> local_providers;
	std::vector<Provider>& providers = providers_ptr ? *providers_ptr : local_providers;
	if (!providers_ptr) {
		local_providers = Provider::load_providers();
	}

	// Helper to find provider by name
	auto find_provider = [&providers](const std::string& name) -> Provider* {
		for (auto& p : providers) {
			if (p.name == name) return &p;
		}
		return nullptr;
	};

	// No args shows current model
	if (args.empty()) {
		if (!current_provider_name || current_provider_name->empty()) {
			callback("No provider configured\n");
		} else {
			auto* prov = find_provider(*current_provider_name);
			if (prov) {
				callback("Current model: " + prov->model + "\n");
			}
		}
		return 0;
	}

	std::string subcmd = args[0];

	if (subcmd == "help" || subcmd == "--help" || subcmd == "-h") {
		callback("Usage: /model [subcommand]\n"
		         "Subcommands:\n"
		         "  list           - Show common model names\n"
		         "  set <name>     - Set model for current provider\n"
		         "  (no args)      - Show current model\n");
		return 0;
	}

	if (subcmd == "list") {
		// Try to get models from current backend
		if (backend && *backend) {
			auto models = (*backend)->get_models();
			if (!models.empty()) {
				callback("Available models:\n");
				for (const auto& model : models) {
					callback("  " + model + "\n");
				}
				return 0;
			}
		}
		// Fallback for backends that don't support model listing
		callback("Model listing not available for this provider.\n");
		callback("Refer to your provider's documentation for available models.\n");
		return 0;
	}

	// Handle both "/model set <name>" and "/model <name>" as shortcuts
	std::string model;
	if (subcmd == "set" && args.size() >= 2) {
		model = args[1];
	} else if (subcmd != "list" && subcmd != "help" && subcmd != "--help" && subcmd != "-h") {
		// Treat as model name directly
		model = subcmd;
	}

	if (!model.empty()) {
		if (!current_provider_name || current_provider_name->empty()) {
			callback("No provider configured\n");
			return 1;
		}

		auto* prov = find_provider(*current_provider_name);
		if (prov) {
			prov->model = model;
			prov->save();

			// Update backend's model if available (also updates model_config)
			if (backend && *backend) {
				(*backend)->set_model(model);
			}

			callback("Model set to: " + model + "\n");
		}
		return 0;
	}

	callback("Unknown model subcommand: " + subcmd + "\n");
	callback("Available: list, set, or just provide model name\n");
	return 1;
}
