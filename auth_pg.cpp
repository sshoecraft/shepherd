#include "auth_pg.h"
#include "shepherd.h"
#include <libpq-fe.h>
#include <iostream>
#include <cstring>

// Factory function called from KeyStore::create()
std::unique_ptr<KeyStore> create_pg_keystore(const std::string& uri) {
    return std::make_unique<PGKeyStore>(uri);
}

PGKeyStore::PGKeyStore(const std::string& uri) : connection_uri(uri) {
    if (!initialize()) {
        std::cerr << "PGKeyStore: Failed to initialize PostgreSQL key store" << std::endl;
    }
}

PGKeyStore::~PGKeyStore() {
    if (conn) {
        PGconn* pg = static_cast<PGconn*>(conn);
        PQfinish(pg);
        conn = nullptr;
    }
}

bool PGKeyStore::initialize() {
    // Extract schema= parameter from URI before passing to libpq
    // (libpq doesn't understand schema=, same pattern as postgresql_backend.cpp)
    std::string clean_uri = connection_uri;
    auto qmark = clean_uri.find('?');
    if (qmark != std::string::npos) {
        std::string query = clean_uri.substr(qmark + 1);
        size_t schema_pos = query.find("schema=");
        if (schema_pos != std::string::npos) {
            size_t value_start = schema_pos + 7;
            size_t value_end = query.find('&', value_start);
            schema = (value_end == std::string::npos) ?
                query.substr(value_start) : query.substr(value_start, value_end - value_start);

            // Rebuild query string without schema=
            std::string new_query;
            size_t pos = 0;
            while (pos < query.size()) {
                size_t amp = query.find('&', pos);
                std::string param = (amp == std::string::npos) ?
                    query.substr(pos) : query.substr(pos, amp - pos);
                if (param.find("schema=") != 0) {
                    if (!new_query.empty()) new_query += '&';
                    new_query += param;
                }
                pos = (amp == std::string::npos) ? query.size() : amp + 1;
            }
            clean_uri = clean_uri.substr(0, qmark);
            if (!new_query.empty()) clean_uri += '?' + new_query;
        }
    }

    PGconn* pg = PQconnectdb(clean_uri.c_str());
    if (PQstatus(pg) != CONNECTION_OK) {
        std::cerr << "PGKeyStore: Connection failed: " << PQerrorMessage(pg) << std::endl;
        PQfinish(pg);
        return false;
    }
    conn = pg;

    // Suppress NOTICE messages (e.g. "relation already exists, skipping")
    PQexec(pg, "SET client_min_messages TO WARNING");

    // Set search_path if schema was specified
    if (!schema.empty()) {
        std::string sql = "CREATE SCHEMA IF NOT EXISTS " + schema;
        PGresult* res = PQexec(pg, sql.c_str());
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "PGKeyStore: Failed to create schema: " << PQerrorMessage(pg) << std::endl;
        }
        PQclear(res);

        sql = "SET search_path TO " + schema;
        res = PQexec(pg, sql.c_str());
        if (PQresultStatus(res) != PGRES_COMMAND_OK) {
            std::cerr << "PGKeyStore: Failed to set search_path: " << PQerrorMessage(pg) << std::endl;
            PQclear(res);
            return false;
        }
        PQclear(res);
    }

    if (!create_table()) {
        return false;
    }

    return true;
}

bool PGKeyStore::create_table() {
    PGconn* pg = static_cast<PGconn*>(conn);

    const char* sql =
        "CREATE TABLE IF NOT EXISTS api_keys ("
        "    key TEXT PRIMARY KEY,"
        "    name TEXT NOT NULL,"
        "    notes TEXT DEFAULT '',"
        "    created TEXT NOT NULL,"
        "    permissions JSONB DEFAULT '{}'"
        ")";

    PGresult* res = PQexec(pg, sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "PGKeyStore: Failed to create table: " << PQerrorMessage(pg) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);
    return true;
}

void PGKeyStore::ensure_loaded() {
    if (loaded) return;
    loaded = true;

    if (!conn) return;

    PGconn* pg = static_cast<PGconn*>(conn);

    PGresult* res = PQexec(pg, "SELECT key, name, notes, created, permissions::text FROM api_keys");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "PGKeyStore: Failed to load keys: " << PQerrorMessage(pg) << std::endl;
        PQclear(res);
        return;
    }

    int rows = PQntuples(res);
    for (int i = 0; i < rows; i++) {
        std::string key = PQgetvalue(res, i, 0);
        ApiKeyEntry entry;
        entry.name = PQgetvalue(res, i, 1);
        entry.notes = PQgetvalue(res, i, 2);
        entry.created = PQgetvalue(res, i, 3);

        std::string perms_str = PQgetvalue(res, i, 4);
        try {
            entry.permissions = nlohmann::json::parse(perms_str);
        } catch (...) {
            entry.permissions = nlohmann::json::object();
        }

        keys[key] = entry;
    }
    PQclear(res);

    dout(1) << "Loaded " << keys.size() << " API keys from PostgreSQL" << std::endl;
}

bool PGKeyStore::validate_key(const std::string& key) {
    ensure_loaded();

    if (key.empty() || keys.empty()) {
        return false;
    }

    // Constant-time comparison to prevent timing attacks
    bool found = false;
    for (const auto& [stored_key, entry] : keys) {
        if (stored_key.length() != key.length()) {
            continue;
        }

        int result = 0;
        for (size_t i = 0; i < stored_key.length(); i++) {
            result |= stored_key[i] ^ key[i];
        }

        if (result == 0) {
            found = true;
        }
    }

    return found;
}

bool PGKeyStore::is_enabled() {
    ensure_loaded();
    return !keys.empty();
}

const ApiKeyEntry* PGKeyStore::get_entry(const std::string& key) const {
    auto it = keys.find(key);
    return (it != keys.end()) ? &it->second : nullptr;
}

std::map<std::string, ApiKeyEntry> PGKeyStore::list_keys() {
    ensure_loaded();
    return keys;
}

bool PGKeyStore::add_key(const std::string& key, const ApiKeyEntry& entry) {
    if (!conn) return false;
    PGconn* pg = static_cast<PGconn*>(conn);

    const char* params[5] = {
        key.c_str(), entry.name.c_str(), entry.notes.c_str(),
        entry.created.c_str(), entry.permissions.dump().c_str()
    };

    PGresult* res = PQexecParams(pg,
        "INSERT INTO api_keys (key, name, notes, created, permissions) VALUES ($1, $2, $3, $4, $5::jsonb)",
        5, nullptr, params, nullptr, nullptr, 0);

    bool ok = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!ok) {
        std::cerr << "PGKeyStore: Failed to add key: " << PQerrorMessage(pg) << std::endl;
    } else {
        keys[key] = entry;
    }
    PQclear(res);
    return ok;
}

bool PGKeyStore::remove_key(const std::string& key) {
    if (!conn) return false;
    PGconn* pg = static_cast<PGconn*>(conn);

    const char* params[1] = { key.c_str() };
    PGresult* res = PQexecParams(pg, "DELETE FROM api_keys WHERE key = $1",
        1, nullptr, params, nullptr, nullptr, 0);

    bool ok = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!ok) {
        std::cerr << "PGKeyStore: Failed to remove key: " << PQerrorMessage(pg) << std::endl;
    } else {
        keys.erase(key);
    }
    PQclear(res);
    return ok;
}

bool PGKeyStore::update_key(const std::string& key, const ApiKeyEntry& entry) {
    if (!conn) return false;
    PGconn* pg = static_cast<PGconn*>(conn);

    const char* params[5] = {
        entry.name.c_str(), entry.notes.c_str(),
        entry.permissions.dump().c_str(), entry.created.c_str(),
        key.c_str()
    };

    PGresult* res = PQexecParams(pg,
        "UPDATE api_keys SET name = $1, notes = $2, permissions = $3::jsonb, created = $4 WHERE key = $5",
        5, nullptr, params, nullptr, nullptr, 0);

    bool ok = (PQresultStatus(res) == PGRES_COMMAND_OK);
    if (!ok) {
        std::cerr << "PGKeyStore: Failed to update key: " << PQerrorMessage(pg) << std::endl;
    } else {
        keys[key] = entry;
    }
    PQclear(res);
    return ok;
}

std::string PGKeyStore::store_location() {
    // Strip credentials from URI for display
    auto at_pos = connection_uri.find('@');
    if (at_pos != std::string::npos) {
        auto scheme_end = connection_uri.find("://");
        if (scheme_end != std::string::npos) {
            return connection_uri.substr(0, scheme_end + 3) + "***@" + connection_uri.substr(at_pos + 1);
        }
    }
    return connection_uri;
}
