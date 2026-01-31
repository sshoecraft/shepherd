#pragma once

#include <string>
#include <vector>
#include <memory>

// Forward declarations (defined in rag.h)
struct SearchResult;
struct ConversationTurn;

/// @brief Abstract database backend interface for RAG storage
/// Implementations: SQLiteBackend, PostgreSQLBackend
class DatabaseBackend {
public:
    virtual ~DatabaseBackend() = default;

    /// @brief Initialize database connection and schema
    /// @return true on success
    virtual bool initialize() = 0;

    /// @brief Shutdown database connection
    virtual void shutdown() = 0;

    /// @brief Archive a conversation turn
    /// @param turn Conversation turn to archive
    virtual void archive_turn(const ConversationTurn& turn) = 0;

    /// @brief Search conversations using full-text search
    /// @param query Search query string
    /// @param max_results Maximum number of results
    /// @return Vector of search results with relevance scores
    virtual std::vector<SearchResult> search(const std::string& query, int max_results) = 0;

    /// @brief Get count of archived turns
    /// @return Number of archived conversation turns
    virtual size_t get_archived_turn_count() const = 0;

    /// @brief Store a question/answer pair to memory
    /// @param question The question text
    /// @param answer The answer text
    virtual void store_memory(const std::string& question, const std::string& answer) = 0;

    /// @brief Clear a memory by exact question match
    /// @param question The exact question to delete
    /// @return True if memory was deleted
    virtual bool clear_memory(const std::string& question) = 0;

    /// @brief Store a key-value fact
    /// @param key Unique identifier for the fact
    /// @param value The fact value to store
    virtual void set_fact(const std::string& key, const std::string& value) = 0;

    /// @brief Retrieve a fact by key
    /// @param key Unique identifier for the fact
    /// @return The stored value, or empty string if not found
    virtual std::string get_fact(const std::string& key) const = 0;

    /// @brief Check if a fact exists
    /// @param key Unique identifier for the fact
    /// @return True if fact exists
    virtual bool has_fact(const std::string& key) const = 0;

    /// @brief Delete a fact by key
    /// @param key Unique identifier for the fact
    /// @return True if fact was deleted, false if it didn't exist
    virtual bool clear_fact(const std::string& key) = 0;

    /// @brief Get database backend type identifier
    /// @return Backend type string ("sqlite", "postgresql")
    virtual std::string backend_type() const = 0;
};

/// @brief Factory for creating database backends
class DatabaseBackendFactory {
public:
    /// @brief Create a database backend based on connection string
    /// @param connection_string File path for SQLite, or postgresql:// URI for PostgreSQL
    /// @param max_db_size Maximum database size in bytes (SQLite only, ignored for PostgreSQL)
    /// @return Unique pointer to the created backend, or nullptr on error
    static std::unique_ptr<DatabaseBackend> create(
        const std::string& connection_string,
        size_t max_db_size = 10ULL * 1024 * 1024 * 1024);

    /// @brief Detect backend type from connection string
    /// @param connection_string Connection string to analyze
    /// @return "sqlite" for file paths, "postgresql" for postgres:// URIs
    static std::string detect_backend_type(const std::string& connection_string);
};
