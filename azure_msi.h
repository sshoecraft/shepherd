#pragma once

#include <string>
#include <ctime>
#include <optional>

namespace azure {

/// @brief Token acquired from Azure Instance Metadata Service (IMDS)
struct MsiToken {
    std::string access_token;
    std::string token_type = "Bearer";
    time_t expires_at = 0;

    bool is_valid() const {
        return !access_token.empty() && time(nullptr) < expires_at;
    }
};

/// @brief Acquire OAuth token from Azure IMDS for a given resource
/// @param resource Token audience (e.g., "https://vault.azure.net")
/// @return Token structure, empty access_token on failure
MsiToken acquire_token(const std::string& resource);

/// @brief Fetch a secret from Azure Key Vault using Managed Identity
/// @param vault_name Azure Key Vault name (without .vault.azure.net)
/// @param secret_name Name of the secret to fetch
/// @return Secret value, or nullopt on failure
std::optional<std::string> get_keyvault_secret(const std::string& vault_name,
                                                const std::string& secret_name);

} // namespace azure
