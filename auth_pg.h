#pragma once

#include "auth.h"
#include <string>
#include <map>

/// @brief PostgreSQL-backed API key store (apikey_store: postgresql://)
/// Uses libpq directly, same patterns as rag/postgresql_backend.cpp
class PGKeyStore : public KeyStore {
public:
    /// @brief Construct with PostgreSQL connection URI
    PGKeyStore(const std::string& uri);
    ~PGKeyStore();

    bool validate_key(const std::string& key) override;
    bool is_enabled() override;
    const ApiKeyEntry* get_entry(const std::string& key) const override;

    std::map<std::string, ApiKeyEntry> list_keys() override;
    bool add_key(const std::string& key, const ApiKeyEntry& entry) override;
    bool remove_key(const std::string& key) override;
    bool update_key(const std::string& key, const ApiKeyEntry& entry) override;
    std::string store_location() override;

private:
    std::string connection_uri;
    std::string schema;  // Extracted from URI schema= parameter
    void* conn = nullptr;  // PGconn* (opaque to avoid libpq header in .h)
    std::map<std::string, ApiKeyEntry> keys;
    bool loaded = false;

    /// @brief Connect to database and create table if needed
    bool initialize();

    /// @brief Load all keys from database (lazy, on first use)
    void ensure_loaded();

    /// @brief Create api_keys table if it doesn't exist
    bool create_table();
};
