#include "provider.h"
#include "config.h"
#include "logger.h"
#include "terminal_io.h"
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
                LOG_ERROR("Failed to load provider " + entry.path().string() + ": " + e.what());
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
    p.n_threads = j.value("n_threads", 0);

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
    if (type == "llamacpp" || type == "tensorrt") {
        j["model_path"] = model_path;
        j["tp"] = tp;
        j["pp"] = pp;
        j["gpu_layers"] = gpu_layers;
        if (type == "tensorrt") j["gpu_id"] = gpu_id;
        if (type == "llamacpp") {
            j["n_batch"] = n_batch;
            if (n_threads > 0) j["n_threads"] = n_threads;
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
    if (config->json.contains("n_threads")) p.n_threads = config->json["n_threads"];
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

std::unique_ptr<Backend> Provider::connect(Session& session) {
    extern std::unique_ptr<Config> config;

    LOG_INFO("Connecting to provider: " + name + " (type: " + type + ")");

    // Print loading message (not for mpirun children, and route through TUI if active)
    if (!getenv("OMPI_COMM_WORLD_SIZE")) {
        std::string msg = "Loading provider: " + name + "\n";
        if (tio.tui_mode) {
            tio.write(msg.c_str(), msg.length(), Color::GRAY);
        } else {
            std::cerr << "Loading provider: " << name << std::endl;
        }
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
        if (frequency_penalty != 0.0f) config->json["frequency_penalty"] = frequency_penalty;
        if (presence_penalty != 0.0f) config->json["presence_penalty"] = presence_penalty;
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

    // Create and initialize backend
    size_t ctx = (context_size > 0) ? context_size : config->context_size;
    auto backend = BackendFactory::create_backend(type, ctx);

    if (backend) {
        backend->initialize(session);
    }

    return backend;
}

// Common provider command implementation
int handle_provider_args(const std::vector<std::string>& args,
                         std::unique_ptr<Backend>* backend,
                         Session* session,
                         std::vector<Provider>* providers_ptr,
                         std::string* current_provider) {
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
			std::cout << "No provider configured\n";
		} else {
			auto* prov = find_provider(*current_provider);
			if (prov) {
				std::cout << "Current provider: " << prov->name << "\n";
				std::cout << "  Type: " << prov->type << "\n";
				std::cout << "  Model: " << prov->model << "\n";
				if (prov->is_api() && !prov->base_url.empty()) {
					std::cout << "  Base URL: " << prov->base_url << "\n";
				}
			}
		}
		return 0;
	}

	std::string subcmd = args[0];

	if (subcmd == "list") {
		if (providers.empty()) {
			std::cout << "No providers configured\n";
		} else {
			std::cout << "Available providers:\n";
			for (const auto& p : providers) {
				if (current_provider && p.name == *current_provider) {
					std::cout << "  * " << p.name << " (current)\n";
				} else {
					std::cout << "    " << p.name << "\n";
				}
			}
		}
		return 0;
	}

	if (subcmd == "add") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider add <name> --type <type> [options...]\n";
			std::cerr << "Types: llamacpp, tensorrt, openai, anthropic, gemini, ollama, cli\n";
			return 1;
		}

		std::string name = args[1];
		std::vector<std::string> cmd_args(args.begin() + 2, args.end());

		// Check if provider already exists
		if (find_provider(name)) {
			std::cerr << "Provider '" << name << "' already exists. Use 'provider set' to modify.\n";
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
		std::cout << "Provider '" << name << "' added successfully\n";
		return 0;
	}

	if (subcmd == "show") {
		std::string name = (args.size() >= 2) ? args[1] : (current_provider ? *current_provider : "");
		if (name.empty()) {
			std::cout << "No provider specified\n";
			return 1;
		}

		auto* prov = find_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		std::cout << "Provider: " << prov->name << "\n";
		std::cout << "  Type: " << prov->type << "\n";

		if (prov->is_api()) {
			std::cout << "  API Key: " << (prov->api_key.empty() ? "not set" : "****") << "\n";
			if (!prov->base_url.empty()) {
				std::cout << "  Base URL: " << prov->base_url << "\n";
			}
		} else if (prov->type == "llamacpp" || prov->type == "tensorrt") {
			std::cout << "  Model Path: " << prov->model_path << "\n";
			std::cout << "  TP: " << prov->tp << ", PP: " << prov->pp << "\n";
			std::cout << "  GPU Layers: " << prov->gpu_layers << "\n";
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
		std::cerr << "Interactive edit not supported. Use 'provider set <name> <field> <value>'\n";
		return 1;
	}

	if (subcmd == "set") {
		if (args.size() < 4) {
			std::cerr << "Usage: provider set <name> <field> <value>\n";
			std::cerr << "Fields: model, api_key, base_url, model_path, priority\n";
			return 1;
		}

		std::string name = args[1];
		std::string field = args[2];
		std::string value = args[3];

		auto* prov = find_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		// Update specific field
		if (field == "model") {
			prov->model = value;
		} else if (field == "api_key" || field == "key") {
			prov->api_key = value;
		} else if (field == "base_url" || field == "url") {
			prov->base_url = value;
		} else if (field == "model_path") {
			prov->model_path = value;
		} else if (field == "priority") {
			prov->priority = std::stoi(value);
		} else if (field == "tokens_per_month") {
			prov->rate_limits.tokens_per_month = std::stoi(value);
		} else if (field == "max_cost") {
			prov->rate_limits.max_cost_per_month = std::stof(value);
		} else {
			std::cerr << "Unknown field: " << field << "\n";
			return 1;
		}

		prov->save();
		std::cout << "Provider '" << name << "' updated\n";
		return 0;
	}

	if (subcmd == "remove") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider remove <name>\n";
			return 1;
		}

		std::string name = args[1];
		Provider::remove(name);

		// Remove from in-memory list
		auto it = std::remove_if(providers.begin(), providers.end(),
			[&name](const Provider& p) { return p.name == name; });
		providers.erase(it, providers.end());

		std::cout << "Provider '" << name << "' removed\n";
		return 0;
	}

	if (subcmd == "use") {
		if (args.size() < 2) {
			std::cerr << "Usage: provider use <name>\n";
			return 1;
		}

		std::string name = args[1];
		auto* prov = find_provider(name);
		if (!prov) {
			std::cerr << "Provider '" << name << "' not found\n";
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Shutdown current backend first to free GPU memory
			(*backend)->shutdown();

			// Connect to the specified provider
			auto new_backend = prov->connect(*session);
			if (!new_backend) {
				std::cerr << "Failed to connect to provider '" << name << "'\n";
				return 1;
			}

			// Update backend ownership and session pointer
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			if (current_provider) {
				*current_provider = name;
			}

			std::cout << "Switched to provider '" << name << "' (" << prov->type << " / " << prov->model << ")\n";
		} else {
			std::cout << "Provider '" << name << "' selected\n";
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
			std::cout << "No other providers available\n";
			return 1;
		}

		// Interactive mode - switch backend
		if (backend && session) {
			// Shutdown current backend first to free GPU memory
			(*backend)->shutdown();

			// Connect to the next provider
			auto new_backend = next_prov->connect(*session);
			if (!new_backend) {
				std::cerr << "Failed to connect to provider '" << next_prov->name << "'\n";
				return 1;
			}

			// Update backend ownership and session pointer
			*backend = std::move(new_backend);
			session->backend = (*backend).get();

			if (current_provider) {
				*current_provider = next_prov->name;
			}

			std::cout << "Switched to provider '" << next_prov->name << "' (" << next_prov->type << " / " << next_prov->model << ")\n";
		} else {
			std::cout << "Next provider: " << next_prov->name << "\n";
		}
		return 0;
	}

	std::cerr << "Unknown provider subcommand: " << subcmd << "\n";
	std::cerr << "Available: list, add, show, set, remove, use, next\n";
	return 1;
}

// Common model command implementation
int handle_model_args(const std::vector<std::string>& args,
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
			std::cout << "No provider configured\n";
		} else {
			auto* prov = find_provider(*current_provider_name);
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
		if (!current_provider_name || current_provider_name->empty()) {
			std::cout << "No provider configured\n";
			return 1;
		}

		auto* prov = find_provider(*current_provider_name);
		if (prov) {
			prov->model = model;
			prov->save();

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
