#include "database_backend.h"
#include "sqlite_backend.h"
#ifdef ENABLE_POSTGRESQL
#include "postgresql_backend.h"
#endif

#include <algorithm>
#include <stdexcept>

std::string DatabaseBackendFactory::detect_backend_type(const std::string& connection_string) {
    // Empty string or file path -> SQLite
    if (connection_string.empty()) {
        return "sqlite";
    }

    // Check for PostgreSQL URI schemes
    std::string lower = connection_string;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.rfind("postgresql://", 0) == 0 || lower.rfind("postgres://", 0) == 0) {
        return "postgresql";
    }

    // Check for explicit sqlite:// scheme
    if (lower.rfind("sqlite://", 0) == 0) {
        return "sqlite";
    }

    // Default: treat as file path (SQLite)
    return "sqlite";
}

std::unique_ptr<DatabaseBackend> DatabaseBackendFactory::create(
    const std::string& connection_string,
    size_t max_db_size) {

    std::string backend_type = detect_backend_type(connection_string);

    if (backend_type == "sqlite") {
        // Remove sqlite:// prefix if present
        std::string db_path = connection_string;
        if (db_path.rfind("sqlite://", 0) == 0) {
            db_path = db_path.substr(9);  // Remove "sqlite://"
        }
        return std::make_unique<SQLiteBackend>(db_path, max_db_size);
    }

#ifdef ENABLE_POSTGRESQL
    if (backend_type == "postgresql") {
        return std::make_unique<PostgreSQLBackend>(connection_string);
    }
#else
    if (backend_type == "postgresql") {
        throw std::runtime_error("PostgreSQL backend not available (compile with -DENABLE_POSTGRESQL=ON)");
    }
#endif

    throw std::runtime_error("Unknown database backend type: " + backend_type);
}
