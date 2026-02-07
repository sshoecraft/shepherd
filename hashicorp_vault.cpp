#include "hashicorp_vault.h"
#include "http_client.h"
#include "shepherd.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>

namespace hashicorp_vault {

static constexpr long VAULT_TIMEOUT_SECONDS = 30;

/// @brief Read token from file, trimming whitespace/newlines
static std::optional<std::string> read_token(const std::string& token_path) {
    std::ifstream file(token_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open Vault token file: " << token_path << std::endl;
        return std::nullopt;
    }

    std::string token;
    std::getline(file, token);

    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t\r\n"));
    token.erase(token.find_last_not_of(" \t\r\n") + 1);

    if (token.empty()) {
        std::cerr << "Vault token file is empty: " << token_path << std::endl;
        return std::nullopt;
    }

    return token;
}

std::optional<std::string> get_config(const std::string& vault_addr,
                                       const std::string& secret_path,
                                       const std::string& token_path) {
    // Read the pre-injected token
    auto token = read_token(token_path);
    if (!token) {
        return std::nullopt;
    }

    dout(1) << "Read Vault token from: " << token_path << std::endl;

    HttpClient client;
    client.set_timeout(VAULT_TIMEOUT_SECONDS);

    // Build Vault KV v2 URL
    // Strip trailing slash from vault_addr if present
    std::string addr = vault_addr;
    if (!addr.empty() && addr.back() == '/') {
        addr.pop_back();
    }

    std::string url = addr + "/v1/secret/data/" + secret_path;

    std::map<std::string, std::string> headers = {
        {"X-Vault-Token", *token}
    };

    dout(1) << "Fetching config from Vault: " << url << std::endl;

    HttpResponse response = client.get(url, headers);

    if (!response.is_success()) {
        std::cerr << "Failed to fetch config from Vault: HTTP " << response.status_code;
        if (!response.error_message.empty()) {
            std::cerr << " - " << response.error_message;
        }
        if (!response.body.empty() && response.body.length() < 500) {
            std::cerr << "\n" << response.body;
        }
        std::cerr << std::endl;
        return std::nullopt;
    }

    try {
        auto json = nlohmann::json::parse(response.body);

        // Vault KV v2 response format: { "data": { "data": { ... }, "metadata": { ... } } }
        if (!json.contains("data") || !json["data"].contains("data")) {
            std::cerr << "Unexpected Vault response format (missing data.data)" << std::endl;
            return std::nullopt;
        }

        auto config_data = json["data"]["data"];

        if (config_data.empty()) {
            std::cerr << "Vault secret is empty" << std::endl;
            return std::nullopt;
        }

        dout(1) << "Successfully retrieved config from Vault" << std::endl;
        return config_data.dump();

    } catch (const std::exception& e) {
        std::cerr << "Failed to parse Vault response: " << e.what() << std::endl;
        return std::nullopt;
    }
}

} // namespace hashicorp_vault
