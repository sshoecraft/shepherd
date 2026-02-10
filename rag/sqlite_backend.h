#pragma once

#include "database_backend.h"
#include <string>

/// @brief SQLite implementation of the RAG database backend
/// Uses FTS5 for full-text search with BM25 ranking
class SQLiteBackend : public DatabaseBackend {
public:
    /// @brief Create SQLite backend
    /// @param db_path Path to SQLite database file
    /// @param max_db_size Maximum database size in bytes (triggers pruning when exceeded)
    explicit SQLiteBackend(const std::string& db_path, size_t max_db_size = 10ULL * 1024 * 1024 * 1024);
    ~SQLiteBackend() override;

    // DatabaseBackend interface
    bool initialize() override;
    void shutdown() override;
    void archive_turn(const ConversationTurn& turn, const std::string& user_id) override;
    std::vector<SearchResult> search(const std::string& query, int max_results, const std::string& user_id) override;
    size_t get_archived_turn_count() const override;
    void store_memory(const std::string& question, const std::string& answer, const std::string& user_id) override;
    bool clear_memory(const std::string& question, const std::string& user_id) override;
    void set_fact(const std::string& key, const std::string& value) override;
    std::string get_fact(const std::string& key) const override;
    bool has_fact(const std::string& key) const override;
    bool clear_fact(const std::string& key) override;
    std::string backend_type() const override { return "sqlite"; }

private:
    bool create_tables();
    void check_and_prune_if_needed();

    std::string db_path_;
    void* db_;  // sqlite3* (void* to avoid header dependency)
    size_t max_db_size_;
};
