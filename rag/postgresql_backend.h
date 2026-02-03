#pragma once

#include "database_backend.h"
#include <string>

/// @brief PostgreSQL implementation of the RAG database backend
/// Uses tsvector/tsquery for full-text search with GIN index
class PostgreSQLBackend : public DatabaseBackend {
public:
    /// @brief Create PostgreSQL backend
    /// @param connection_string PostgreSQL connection URI (postgresql://user:pass@host:port/db)
    explicit PostgreSQLBackend(const std::string& connection_string);
    ~PostgreSQLBackend() override;

    // DatabaseBackend interface
    bool initialize() override;
    void shutdown() override;
    void archive_turn(const ConversationTurn& turn) override;
    std::vector<SearchResult> search(const std::string& query, int max_results) override;
    size_t get_archived_turn_count() const override;
    void store_memory(const std::string& question, const std::string& answer) override;
    bool clear_memory(const std::string& question) override;
    void set_fact(const std::string& key, const std::string& value) override;
    std::string get_fact(const std::string& key) const override;
    bool has_fact(const std::string& key) const override;
    bool clear_fact(const std::string& key) override;
    std::string backend_type() const override { return "postgresql"; }

private:
    bool create_tables();
    bool prepare_statements();
    bool set_schema();
    std::string compute_content_hash(const std::string& content);
    static std::string extract_schema(std::string& connection_string);

    std::string connection_string_;
    std::string schema_;  // Optional schema (search_path)
    void* conn_;  // PGconn* (void* to avoid header dependency)
};
