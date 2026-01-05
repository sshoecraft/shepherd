#pragma once

#include "nlohmann/json.hpp"
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

/// @brief Entry for a single API key with metadata
struct ApiKeyEntry {
    std::string name;
    std::string notes;
    std::string created;
    nlohmann::json permissions;  // Reserved for future authorization
};

/// @brief Abstract interface for API key storage backends
class KeyStore {
public:
    virtual ~KeyStore() = default;

    /// @brief Validate an API key (constant-time comparison)
    /// @param key The API key to validate
    /// @return true if key is valid
    virtual bool validate_key(const std::string& key) = 0;

    /// @brief Check if authentication is enabled
    /// @return true if keys are configured and auth is required
    virtual bool is_enabled() = 0;

    /// @brief Factory method to create KeyStore from mode string
    /// @param mode "none", "json", "sqlite", "vault", "managed"
    /// @return Unique pointer to KeyStore implementation
    /// @throws std::runtime_error if mode is unknown
    static std::unique_ptr<KeyStore> create(const std::string& mode);
};

/// @brief No authentication (--auth-mode none)
class NoneKeyStore : public KeyStore {
public:
    bool validate_key(const std::string&) override { return true; }
    bool is_enabled() override { return false; }
};

/// @brief JSON file-based key store (--auth-mode json)
class JsonKeyStore : public KeyStore {
public:
    JsonKeyStore();

    bool validate_key(const std::string& key) override;
    bool is_enabled() override;

    /// @brief Get path to api_keys.json file
    /// @return Path like ~/.config/shepherd/api_keys.json
    static std::string get_keys_file_path();

    /// @brief Load all keys from disk
    /// @return Map of api_key -> ApiKeyEntry
    static std::map<std::string, ApiKeyEntry> load_keys();

    /// @brief Save keys to disk (sets 0600 permissions)
    /// @param keys Map of api_key -> ApiKeyEntry
    static void save_keys(const std::map<std::string, ApiKeyEntry>& keys);

    /// @brief Generate a new cryptographically secure API key
    /// @return Key in format sk-XXXXXXXX (35 chars total)
    static std::string generate_key();

private:
    std::map<std::string, ApiKeyEntry> keys;
};

/// @brief Handle 'shepherd keygen' CLI subcommand
/// @param args Arguments after "keygen"
/// @param callback Output callback function
/// @return 0 on success, non-zero on error
int handle_keygen_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback);
