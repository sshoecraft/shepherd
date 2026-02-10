#include "postgresql_backend.h"
#include "../rag.h"
#include "../shepherd.h"

#include <libpq-fe.h>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <openssl/sha.h>

std::string PostgreSQLBackend::extract_schema(std::string& connection_string) {
    // Look for schema= parameter in query string and extract it
    // e.g., postgresql://...?sslmode=require&schema=acmai -> extracts "acmai"
    std::string schema;

    size_t query_start = connection_string.find('?');
    if (query_start == std::string::npos) {
        return schema;
    }

    // Find schema= in the query string
    std::string query = connection_string.substr(query_start + 1);
    size_t schema_pos = query.find("schema=");
    if (schema_pos == std::string::npos) {
        return schema;
    }

    // Extract schema value
    size_t value_start = schema_pos + 7;  // len("schema=")
    size_t value_end = query.find('&', value_start);
    if (value_end == std::string::npos) {
        schema = query.substr(value_start);
    } else {
        schema = query.substr(value_start, value_end - value_start);
    }

    // Remove schema= from connection string (libpq doesn't understand it)
    std::string new_query;
    size_t pos = 0;
    while (pos < query.length()) {
        size_t amp = query.find('&', pos);
        std::string param;
        if (amp == std::string::npos) {
            param = query.substr(pos);
            pos = query.length();
        } else {
            param = query.substr(pos, amp - pos);
            pos = amp + 1;
        }

        if (param.find("schema=") != 0) {
            if (!new_query.empty()) {
                new_query += '&';
            }
            new_query += param;
        }
    }

    // Rebuild connection string without schema parameter
    if (new_query.empty()) {
        connection_string = connection_string.substr(0, query_start);
    } else {
        connection_string = connection_string.substr(0, query_start + 1) + new_query;
    }

    return schema;
}

PostgreSQLBackend::PostgreSQLBackend(const std::string& connection_string)
    : connection_string_(connection_string), conn_(nullptr) {
    // Extract and remove schema parameter before passing to libpq
    schema_ = extract_schema(connection_string_);

    dout(1) << "PostgreSQLBackend created with connection string: " +
             connection_string_.substr(0, connection_string_.find('@')) + "@..." << std::endl;
    if (!schema_.empty()) {
        dout(1) << "PostgreSQLBackend using schema: " + schema_ << std::endl;
    }
}

PostgreSQLBackend::~PostgreSQLBackend() {
    dout(1) << "PostgreSQLBackend destructor" << std::endl;
    shutdown();
}

bool PostgreSQLBackend::initialize() {
    dout(1) << "Initializing PostgreSQL RAG database" << std::endl;

    PGconn* conn = PQconnectdb(connection_string_.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "PostgreSQL connection failed: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQfinish(conn);
        return false;
    }

    conn_ = conn;
    dout(1) << "Connected to PostgreSQL server" << std::endl;

    // Suppress NOTICE messages (e.g., "relation already exists, skipping")
    PQexec(conn, "SET client_min_messages TO WARNING");

    if (!set_schema()) {
        std::cerr << "Failed to set PostgreSQL schema" << std::endl;
        shutdown();
        return false;
    }

    if (!create_tables()) {
        std::cerr << "Failed to create PostgreSQL tables" << std::endl;
        shutdown();
        return false;
    }

    if (!prepare_statements()) {
        std::cerr << "Failed to prepare PostgreSQL statements" << std::endl;
        shutdown();
        return false;
    }

    dout(1) << "PostgreSQL RAG database initialized successfully" << std::endl;
    return true;
}

bool PostgreSQLBackend::set_schema() {
    if (schema_.empty()) {
        return true;  // No schema specified, use default
    }

    PGconn* conn = static_cast<PGconn*>(conn_);

    // Create schema if it doesn't exist
    std::string create_schema_sql = "CREATE SCHEMA IF NOT EXISTS " + schema_;
    PGresult* res = PQexec(conn, create_schema_sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create schema: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Set search_path to use the schema
    std::string set_path_sql = "SET search_path TO " + schema_;
    res = PQexec(conn, set_path_sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to set search_path: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    dout(1) << "Set PostgreSQL search_path to schema: " + schema_ << std::endl;
    return true;
}

bool PostgreSQLBackend::create_tables() {
    PGconn* conn = static_cast<PGconn*>(conn_);

    // Create conversations table with tsvector for full-text search
    const char* create_conversations_sql = R"(
        CREATE TABLE IF NOT EXISTS conversations (
            id BIGSERIAL PRIMARY KEY,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            content_hash CHAR(64) UNIQUE NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'local',
            search_vector TSVECTOR GENERATED ALWAYS AS (
                setweight(to_tsvector('english', user_message), 'A') ||
                setweight(to_tsvector('english', assistant_response), 'B')
            ) STORED
        );
    )";

    PGresult* res = PQexec(conn, create_conversations_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create conversations table: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Create GIN index for full-text search
    const char* create_fts_index_sql = R"(
        CREATE INDEX IF NOT EXISTS conversations_search_idx
        ON conversations USING GIN(search_vector);
    )";

    res = PQexec(conn, create_fts_index_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create FTS index: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Create timestamp index for efficient pruning
    const char* create_timestamp_index_sql = R"(
        CREATE INDEX IF NOT EXISTS conversations_timestamp_idx
        ON conversations(timestamp);
    )";

    res = PQexec(conn, create_timestamp_index_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create timestamp index: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Create user_id index
    const char* create_user_id_index_sql = R"(
        CREATE INDEX IF NOT EXISTS conversations_user_id_idx
        ON conversations(user_id);
    )";

    res = PQexec(conn, create_user_id_index_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create user_id index: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Migration: add user_id column to existing databases (idempotent)
    PQexec(conn, "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT 'local'");

    // Create facts table
    const char* create_facts_sql = R"(
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL,
            created_at BIGINT NOT NULL,
            updated_at BIGINT NOT NULL
        );
    )";

    res = PQexec(conn, create_facts_sql);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to create facts table: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    dout(1) << "PostgreSQL RAG database tables created successfully" << std::endl;
    return true;
}

bool PostgreSQLBackend::prepare_statements() {
    PGconn* conn = static_cast<PGconn*>(conn_);

    // Prepare archive_turn statement
    PGresult* res = PQprepare(conn, "archive_turn",
        "INSERT INTO conversations (user_message, assistant_response, timestamp, content_hash, user_id) "
        "VALUES ($1, $2, $3, $4, $5) ON CONFLICT (content_hash) DO NOTHING",
        5, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare archive_turn: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Prepare search statement
    res = PQprepare(conn, "search_memory",
        "SELECT user_message, assistant_response, timestamp, "
        "ts_rank_cd(search_vector, plainto_tsquery('english', $1)) as score "
        "FROM conversations "
        "WHERE search_vector @@ plainto_tsquery('english', $1) AND user_id = $3 "
        "ORDER BY score DESC "
        "LIMIT $2",
        3, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare search_memory: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Prepare set_fact statement
    res = PQprepare(conn, "set_fact",
        "INSERT INTO facts (key, value, created_at, updated_at) "
        "VALUES ($1, $2, $3, $3) "
        "ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = $3",
        3, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare set_fact: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Prepare get_fact statement
    res = PQprepare(conn, "get_fact",
        "SELECT value FROM facts WHERE key = $1",
        1, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare get_fact: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Prepare clear_fact statement
    res = PQprepare(conn, "clear_fact",
        "DELETE FROM facts WHERE key = $1",
        1, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare clear_fact: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    // Prepare clear_memory statement
    res = PQprepare(conn, "clear_memory",
        "DELETE FROM conversations WHERE id = (SELECT id FROM conversations WHERE user_message = $1 AND user_id = $2 LIMIT 1)",
        2, nullptr);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to prepare clear_memory: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }
    PQclear(res);

    dout(1) << "PostgreSQL prepared statements ready" << std::endl;
    return true;
}

std::string PostgreSQLBackend::compute_content_hash(const std::string& content) {
    unsigned char hash_bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(content.c_str()), content.length(), hash_bytes);

    std::ostringstream hash_stream;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        hash_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash_bytes[i]);
    }
    return hash_stream.str();
}

void PostgreSQLBackend::archive_turn(const ConversationTurn& turn, const std::string& user_id) {
    dout(1) << "Archiving conversation turn to PostgreSQL RAG database" << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);

    std::string combined = turn.user_message + turn.assistant_response;
    std::string content_hash = compute_content_hash(combined);
    std::string timestamp_str = std::to_string(turn.timestamp);

    const char* params[5] = {
        turn.user_message.c_str(),
        turn.assistant_response.c_str(),
        timestamp_str.c_str(),
        content_hash.c_str(),
        user_id.c_str()
    };

    PGresult* res = PQexecPrepared(conn, "archive_turn", 5, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to archive turn: " + std::string(PQerrorMessage(conn)) << std::endl;
    } else {
        int rows = atoi(PQcmdTuples(res));
        if (rows > 0) {
            dout(1) << "Archived conversation turn to PostgreSQL database" << std::endl;
        } else {
            dout(1) << "Skipped duplicate conversation (hash=" + content_hash.substr(0, 16) + "...)" << std::endl;
        }
    }
    PQclear(res);
}

std::vector<SearchResult> PostgreSQLBackend::search(const std::string& query, int max_results, const std::string& user_id) {
    dout(1) << "Searching PostgreSQL RAG database for: " + query << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);
    std::vector<SearchResult> results;

    std::string limit_str = std::to_string(max_results);
    const char* params[3] = { query.c_str(), limit_str.c_str(), user_id.c_str() };

    PGresult* res = PQexecPrepared(conn, "search_memory", 3, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Search query failed: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return results;
    }

    int rows = PQntuples(res);
    for (int i = 0; i < rows; i++) {
        std::string user_msg = PQgetvalue(res, i, 0);
        std::string assistant_msg = PQgetvalue(res, i, 1);
        int64_t timestamp = std::stoll(PQgetvalue(res, i, 2));
        double score = std::stod(PQgetvalue(res, i, 3));

        std::string combined = "User: " + user_msg + "\nAssistant: " + assistant_msg;

        // Normalize score to 0.0-1.0 range (ts_rank_cd returns small positive values)
        double normalized_score = std::min(1.0, score * 10.0);

        results.emplace_back(combined, normalized_score, timestamp);
    }

    PQclear(res);

    dout(1) << "Found " + std::to_string(results.size()) + " results for query: " + query << std::endl;
    return results;
}

size_t PostgreSQLBackend::get_archived_turn_count() const {
    if (!conn_) {
        return 0;
    }

    PGconn* conn = static_cast<PGconn*>(conn_);
    PGresult* res = PQexec(conn, "SELECT COUNT(*) FROM conversations");

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to get count: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return 0;
    }

    size_t count = std::stoull(PQgetvalue(res, 0, 0));
    PQclear(res);
    return count;
}

void PostgreSQLBackend::store_memory(const std::string& question, const std::string& answer, const std::string& user_id) {
    dout(1) << "Storing memory: " + question << std::endl;
    ConversationTurn turn(question, answer);
    archive_turn(turn, user_id);
    dout(1) << "Stored memory: question=" + question + ", answer=" + answer << std::endl;
}

bool PostgreSQLBackend::clear_memory(const std::string& question, const std::string& user_id) {
    if (!conn_) {
        return false;
    }

    dout(1) << "Clearing memory by question: " + question << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);
    const char* params[2] = { question.c_str(), user_id.c_str() };

    PGresult* res = PQexecPrepared(conn, "clear_memory", 2, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to clear memory: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = atoi(PQcmdTuples(res));
    PQclear(res);

    if (rows > 0) {
        dout(1) << "Cleared memory for question: " + question << std::endl;
        return true;
    } else {
        dout(1) << "Memory not found for question: " + question << std::endl;
        return false;
    }
}

void PostgreSQLBackend::set_fact(const std::string& key, const std::string& value) {
    dout(1) << "Setting fact: " + key << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);
    int64_t now = ConversationTurn::get_current_timestamp();
    std::string now_str = std::to_string(now);

    const char* params[3] = { key.c_str(), value.c_str(), now_str.c_str() };

    PGresult* res = PQexecPrepared(conn, "set_fact", 3, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to set fact: " + std::string(PQerrorMessage(conn)) << std::endl;
    } else {
        dout(1) << "Set fact: " + key + " = " + value << std::endl;
    }
    PQclear(res);
}

std::string PostgreSQLBackend::get_fact(const std::string& key) const {
    if (!conn_) {
        return "";
    }

    dout(1) << "Getting fact: " + key << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);
    const char* params[1] = { key.c_str() };

    PGresult* res = PQexecPrepared(conn, "get_fact", 1, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to get fact: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return "";
    }

    std::string value;
    if (PQntuples(res) > 0) {
        value = PQgetvalue(res, 0, 0);
        dout(1) << "Found fact: " + key + " = " + value << std::endl;
    } else {
        dout(1) << "Fact not found: " + key << std::endl;
    }

    PQclear(res);
    return value;
}

bool PostgreSQLBackend::has_fact(const std::string& key) const {
    if (!conn_) {
        return false;
    }

    PGconn* conn = static_cast<PGconn*>(conn_);
    const char* params[1] = { key.c_str() };

    PGresult* res = PQexecPrepared(conn, "get_fact", 1, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        PQclear(res);
        return false;
    }

    bool exists = (PQntuples(res) > 0);
    PQclear(res);
    return exists;
}

bool PostgreSQLBackend::clear_fact(const std::string& key) {
    if (!conn_) {
        return false;
    }

    dout(1) << "Clearing fact: " + key << std::endl;

    PGconn* conn = static_cast<PGconn*>(conn_);
    const char* params[1] = { key.c_str() };

    PGresult* res = PQexecPrepared(conn, "clear_fact", 1, params, nullptr, nullptr, 0);
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "Failed to clear fact: " + std::string(PQerrorMessage(conn)) << std::endl;
        PQclear(res);
        return false;
    }

    int rows = atoi(PQcmdTuples(res));
    PQclear(res);

    if (rows > 0) {
        dout(1) << "Cleared fact: " + key << std::endl;
        return true;
    } else {
        dout(1) << "Fact not found: " + key << std::endl;
        return false;
    }
}

void PostgreSQLBackend::shutdown() {
    dout(1) << "PostgreSQLBackend shutdown" << std::endl;
    if (conn_) {
        PGconn* conn = static_cast<PGconn*>(conn_);
        PQfinish(conn);
        conn_ = nullptr;
        dout(1) << "PostgreSQL RAG database connection closed" << std::endl;
    }
}
