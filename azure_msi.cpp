#include "azure_msi.h"
#include "http_client.h"
#include "shepherd.h"

#include <nlohmann/json.hpp>

namespace azure {

// Azure IMDS endpoint for managed identity tokens
static constexpr const char* IMDS_ENDPOINT = "http://169.254.169.254/metadata/identity/oauth2/token";
static constexpr const char* IMDS_API_VERSION = "2018-02-01";
static constexpr long IMDS_TIMEOUT_SECONDS = 5;

// Azure Key Vault API version
static constexpr const char* KEYVAULT_API_VERSION = "7.4";

MsiToken acquire_token(const std::string& resource) {
    MsiToken token;

    HttpClient client;
    client.set_timeout(IMDS_TIMEOUT_SECONDS);
    client.set_ssl_verify(false);  // IMDS is HTTP, not HTTPS

    // Build IMDS URL with query parameters
    std::string url = std::string(IMDS_ENDPOINT) +
                      "?api-version=" + IMDS_API_VERSION +
                      "&resource=" + resource;

    // IMDS requires the Metadata header
    std::map<std::string, std::string> headers = {
        {"Metadata", "true"}
    };

    dout(1) << "Acquiring MSI token for resource: " << resource << std::endl;

    HttpResponse response = client.get(url, headers);

    if (!response.is_success()) {
        std::cerr << "Failed to acquire MSI token: HTTP " << response.status_code;
        if (!response.error_message.empty()) {
            std::cerr << " - " << response.error_message;
        }
        std::cerr << std::endl;
        return token;
    }

    try {
        auto json = nlohmann::json::parse(response.body);

        token.access_token = json.value("access_token", "");
        token.token_type = json.value("token_type", "Bearer");

        // Parse expiration - IMDS returns expires_on as Unix timestamp string
        std::string expires_on = json.value("expires_on", "");
        if (!expires_on.empty()) {
            token.expires_at = std::stol(expires_on);
        }

        dout(1) << "MSI token acquired, expires at: " << token.expires_at << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to parse MSI token response: " << e.what() << std::endl;
        token.access_token.clear();
    }

    return token;
}

std::optional<std::string> get_keyvault_secret(const std::string& vault_name,
                                                const std::string& secret_name) {
    // First, acquire token for Key Vault
    MsiToken token = acquire_token("https://vault.azure.net");
    if (!token.is_valid()) {
        std::cerr << "Failed to acquire token for Key Vault access" << std::endl;
        return std::nullopt;
    }

    HttpClient client;
    client.set_timeout(30);

    // Build Key Vault URL
    std::string url = "https://" + vault_name + ".vault.azure.net/secrets/" +
                      secret_name + "?api-version=" + KEYVAULT_API_VERSION;

    std::map<std::string, std::string> headers = {
        {"Authorization", token.token_type + " " + token.access_token}
    };

    dout(1) << "Fetching secret from Key Vault: " << vault_name << "/" << secret_name << std::endl;

    HttpResponse response = client.get(url, headers);

    if (!response.is_success()) {
        std::cerr << "Failed to fetch secret from Key Vault: HTTP " << response.status_code;
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
        std::string value = json.value("value", "");

        if (value.empty()) {
            std::cerr << "Key Vault secret is empty" << std::endl;
            return std::nullopt;
        }

        dout(1) << "Successfully retrieved secret from Key Vault" << std::endl;
        return value;

    } catch (const std::exception& e) {
        std::cerr << "Failed to parse Key Vault response: " << e.what() << std::endl;
        return std::nullopt;
    }
}

} // namespace azure
