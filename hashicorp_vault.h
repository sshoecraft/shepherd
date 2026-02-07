#pragma once

#include <string>
#include <optional>

namespace hashicorp_vault {

/// @brief Default path where Vault Agent Injector writes the token
static constexpr const char* DEFAULT_TOKEN_PATH = "/vault/secrets/token";

/// @brief Default secret path in Vault KV v2
static constexpr const char* DEFAULT_SECRET_PATH = "shepherd/config";

/// @brief Fetch config from HashiCorp Vault using a pre-injected token
/// @param vault_addr Vault server address (e.g., "http://vault:8200")
/// @param secret_path KV v2 secret path (default: "shepherd/config")
/// @param token_path Path to token file (default: "/vault/secrets/token")
/// @return Config JSON string, or nullopt on failure
std::optional<std::string> get_config(const std::string& vault_addr,
                                       const std::string& secret_path = DEFAULT_SECRET_PATH,
                                       const std::string& token_path = DEFAULT_TOKEN_PATH);

} // namespace hashicorp_vault
