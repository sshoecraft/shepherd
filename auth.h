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

    /// @brief Get entry for a validated key
    /// @param key The API key to look up
    /// @return Pointer to entry if found, nullptr otherwise
    virtual const ApiKeyEntry* get_entry(const std::string& key) const { (void)key; return nullptr; }

    /// @brief Load all keys from the store
    /// @return Map of api_key -> ApiKeyEntry
    virtual std::map<std::string, ApiKeyEntry> list_keys() { return {}; }

    /// @brief Add a key to the store
    /// @return true on success
    virtual bool add_key(const std::string& key, const ApiKeyEntry& entry) { (void)key; (void)entry; return false; }

    /// @brief Remove a key from the store
    /// @return true on success
    virtual bool remove_key(const std::string& key) { (void)key; return false; }

    /// @brief Update a key's metadata in the store
    /// @return true on success
    virtual bool update_key(const std::string& key, const ApiKeyEntry& entry) { (void)key; (void)entry; return false; }

    /// @brief Get a description of where keys are stored
    virtual std::string store_location() { return "(unknown)"; }

    /// @brief Factory method to create KeyStore from URI
    /// @param uri URI string: file://, postgresql://, msi://, or empty for no auth
    /// @return Unique pointer to KeyStore implementation
    /// @throws std::runtime_error if URI scheme is unknown
    static std::unique_ptr<KeyStore> create(const std::string& uri);
};

/// @brief No authentication (no apikey_store configured)
class NoneKeyStore : public KeyStore {
public:
    bool validate_key(const std::string&) override { return true; }
    bool is_enabled() override { return false; }
};

/// @brief JSON file-based key store (apikey_store: file://)
class JsonKeyStore : public KeyStore {
public:
    /// @brief Construct with optional custom path (empty = default path)
    JsonKeyStore(const std::string& path = "");

    bool validate_key(const std::string& key) override;
    bool is_enabled() override;

    /// @brief Get path to api_keys.json file
    /// @return Custom path if set, otherwise ~/.config/shepherd/api_keys.json
    std::string get_keys_file_path();

    /// @brief Load all keys from disk
    /// @return Map of api_key -> ApiKeyEntry
    std::map<std::string, ApiKeyEntry> load_keys();

    /// @brief Save keys to disk (sets 0600 permissions)
    /// @param keys Map of api_key -> ApiKeyEntry
    void save_keys(const std::map<std::string, ApiKeyEntry>& keys);

    /// @brief Generate a new cryptographically secure API key
    /// @return Key in format sk-XXXXXXXX (35 chars total)
    static std::string generate_key();

    /// @brief Get entry for a validated key (for session binding)
    /// @param key The API key to look up
    /// @return Pointer to entry if found, nullptr otherwise
    const ApiKeyEntry* get_entry(const std::string& key) const override;

    std::map<std::string, ApiKeyEntry> list_keys() override;
    bool add_key(const std::string& key, const ApiKeyEntry& entry) override;
    bool remove_key(const std::string& key) override;
    bool update_key(const std::string& key, const ApiKeyEntry& entry) override;
    std::string store_location() override;

private:
    std::map<std::string, ApiKeyEntry> keys;
    std::string custom_path;  // Custom file path from URI (empty = default)
};

/// @brief Azure Key Vault key store (apikey_store: msi://)
/// Retrieves API keys from Key Vault secret 'shepherd-keys' using Managed Identity
class MsiKeyStore : public KeyStore {
public:
    /// @brief Create MsiKeyStore using vault name from config
    MsiKeyStore();

    bool validate_key(const std::string& key) override;
    bool is_enabled() override;

    /// @brief Get entry for a validated key
    const ApiKeyEntry* get_entry(const std::string& key) const override;

private:
    std::map<std::string, ApiKeyEntry> keys;
    bool loaded = false;
    std::string error_message;

    /// @brief Load keys from Key Vault (lazy, on first use)
    void ensure_loaded();
};

/// @brief Factory function for PostgreSQL key store (defined in auth_pg.cpp)
#ifdef ENABLE_POSTGRESQL
std::unique_ptr<KeyStore> create_pg_keystore(const std::string& uri);
#endif

/// @brief Handle 'shepherd apikey' CLI subcommand
/// @param args Arguments after "apikey"
/// @param callback Output callback function
/// @param apikey_store_uri URI for the key store backend (empty = default file://)
/// @return 0 on success, non-zero on error
int handle_apikey_args(const std::vector<std::string>& args,
                       std::function<void(const std::string&)> callback,
                       const std::string& apikey_store_uri = "");
