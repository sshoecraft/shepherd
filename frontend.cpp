#include "frontend.h"
#include "cli.h"
#include "server/api_server.h"
#include "server/cli_server.h"
#include "logger.h"
#include "config.h"
#include <algorithm>

extern std::unique_ptr<Config> config;

Frontend::Frontend() {
}

Frontend::~Frontend() {
}

std::unique_ptr<Frontend> Frontend::create(const std::string& mode,
                                            const std::string& host,
                                            int port,
                                            Provider* cmdline_provider) {
    std::unique_ptr<Frontend> frontend;

    if (mode == "cli") {
        frontend = std::make_unique<CLI>();
    }
    else if (mode == "api-server") {
        frontend = std::make_unique<APIServer>(host, port);
    }
    else if (mode == "cli-server") {
        frontend = std::make_unique<CLIServer>(host, port);
    }
    else {
        throw std::runtime_error("Invalid frontend mode: " + mode);
    }

    // Load providers from disk
    frontend->providers = Provider::load_providers();

    // Add command-line provider at the front (highest priority)
    if (cmdline_provider) {
        frontend->providers.insert(frontend->providers.begin(), *cmdline_provider);
    }

    return frontend;
}

Provider* Frontend::get_provider(const std::string& name) {
    for (auto& p : providers) {
        if (p.name == name) {
            return &p;
        }
    }
    return nullptr;
}

std::vector<std::string> Frontend::list_providers() const {
    std::vector<std::string> names;
    for (const auto& p : providers) {
        names.push_back(p.name);
    }
    return names;
}

bool Frontend::connect_next_provider(Session& session) {
    if (providers.empty()) {
        LOG_ERROR("No providers configured");
        return false;
    }

    for (auto& p : providers) {
        try {
            backend = p.connect(session);
            if (backend) {
                current_provider = p.name;
                session.backend = backend.get();
                return true;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Provider '" + p.name + "' failed: " + std::string(e.what()));
            if (!config->auto_provider) {
                LOG_INFO("auto_provider disabled, not trying other providers");
                return false;
            }
        }
    }

    LOG_ERROR("All providers failed to connect");
    return false;
}

bool Frontend::connect_provider(const std::string& name, Session& session) {
    Provider* p = get_provider(name);
    if (!p) {
        LOG_ERROR("Provider '" + name + "' not found");
        return false;
    }

    try {
        backend = p->connect(session);
        if (backend) {
            current_provider = name;
            session.backend = backend.get();
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("Provider '" + name + "' failed: " + std::string(e.what()));
        return false;
    }
}
