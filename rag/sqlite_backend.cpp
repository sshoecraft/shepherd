#include "sqlite_backend.h"
#include "../rag.h"
#include "../shepherd.h"

#include <chrono>
#include <filesystem>
#include <sqlite3.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <openssl/sha.h>

SQLiteBackend::SQLiteBackend(const std::string& db_path, size_t max_db_size)
    : db_path_(db_path), db_(nullptr), max_db_size_(max_db_size) {
    dout(1) << "SQLiteBackend created with path: " + db_path + ", max size: " + std::to_string(max_db_size / (1024 * 1024)) + " MB" << std::endl;
}

SQLiteBackend::~SQLiteBackend() {
    dout(1) << "SQLiteBackend destructor" << std::endl;
    shutdown();
}

bool SQLiteBackend::initialize() {
    // Create directory if it doesn't exist
    std::filesystem::path db_file(db_path_);
    std::filesystem::path db_dir = db_file.parent_path();

    if (!db_dir.empty() && !std::filesystem::exists(db_dir)) {
        try {
            std::filesystem::create_directories(db_dir);
            dout(1) << "Created RAG database directory: " + db_dir.string() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to create RAG database directory: " + std::string(e.what()) << std::endl;
            return false;
        }
    }

    dout(1) << "Initializing SQLite RAG database: " + db_path_ << std::endl;

    // Open SQLite database
    sqlite3* db = nullptr;
    int rc = sqlite3_open(db_path_.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to open database: " + std::string(sqlite3_errmsg(db)) << std::endl;
        sqlite3_close(db);
        return false;
    }

    db_ = db;

    // Enable memory-mapped I/O for better performance
    std::string mmap_pragma = "PRAGMA mmap_size = " + std::to_string(max_db_size_);
    rc = sqlite3_exec(db, mmap_pragma.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        dout(1) << std::string("WARNING: ") + "Failed to enable memory mapping: " + std::string(sqlite3_errmsg(db)) << std::endl;
    } else {
        dout(1) << "Enabled memory-mapped I/O (mmap_size: " + std::to_string(max_db_size_ / (1024 * 1024)) + " MB)" << std::endl;
    }

    // Create tables
    if (!create_tables()) {
        std::cerr << "Failed to create database tables" << std::endl;
        shutdown();
        return false;
    }

    dout(1) << "SQLite RAG database initialized successfully" << std::endl;
    return true;
}

bool SQLiteBackend::create_tables() {
    sqlite3* db = static_cast<sqlite3*>(db_);
    char* err_msg = nullptr;

    // Create conversations table
    const char* create_table_sql = R"(
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            content_hash TEXT UNIQUE NOT NULL
        );
    )";

    int rc = sqlite3_exec(db, create_table_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create conversations table: " + std::string(err_msg) << std::endl;
        sqlite3_free(err_msg);
        return false;
    }

    // Create FTS5 virtual table for full-text search
    const char* create_fts_sql = R"(
        CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
            user_message,
            assistant_response,
            content='conversations',
            content_rowid='id'
        );
    )";

    rc = sqlite3_exec(db, create_fts_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create FTS table: " + std::string(err_msg) << std::endl;
        sqlite3_free(err_msg);
        return false;
    }

    // Create triggers to keep FTS index in sync
    const char* create_triggers_sql = R"(
        CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
            INSERT INTO conversations_fts(rowid, user_message, assistant_response)
            VALUES (new.id, new.user_message, new.assistant_response);
        END;

        CREATE TRIGGER IF NOT EXISTS conversations_ad AFTER DELETE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, user_message, assistant_response)
            VALUES('delete', old.id, old.user_message, old.assistant_response);
        END;

        CREATE TRIGGER IF NOT EXISTS conversations_au AFTER UPDATE ON conversations BEGIN
            INSERT INTO conversations_fts(conversations_fts, rowid, user_message, assistant_response)
            VALUES('delete', old.id, old.user_message, old.assistant_response);
            INSERT INTO conversations_fts(rowid, user_message, assistant_response)
            VALUES (new.id, new.user_message, new.assistant_response);
        END;
    )";

    rc = sqlite3_exec(db, create_triggers_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create triggers: " + std::string(err_msg) << std::endl;
        sqlite3_free(err_msg);
        return false;
    }

    // Create facts table
    const char* create_facts_sql = R"(
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );
    )";

    rc = sqlite3_exec(db, create_facts_sql, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to create facts table: " + std::string(err_msg) << std::endl;
        sqlite3_free(err_msg);
        return false;
    }

    dout(1) << "RAG database tables created successfully" << std::endl;
    return true;
}

void SQLiteBackend::archive_turn(const ConversationTurn& turn) {
    dout(1) << "Archiving conversation turn to SQLite RAG database" << std::endl;

    // Compute SHA256 hash of user_message + assistant_response to detect duplicates
    std::string combined = turn.user_message + turn.assistant_response;
    unsigned char hash_bytes[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(combined.c_str()), combined.length(), hash_bytes);

    // Convert to hex string
    std::ostringstream hash_stream;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        hash_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash_bytes[i]);
    }
    std::string content_hash = hash_stream.str();

    dout(1) << "Content hash: " + content_hash << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    const char* sql = "INSERT OR IGNORE INTO conversations (user_message, assistant_response, timestamp, content_hash) VALUES (?, ?, ?, ?)";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare archive statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return;
    }

    sqlite3_bind_text(stmt, 1, turn.user_message.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, turn.assistant_response.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 3, turn.timestamp);
    sqlite3_bind_text(stmt, 4, content_hash.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to archive turn: " + std::string(sqlite3_errmsg(db)) << std::endl;
    } else {
        int64_t row_id = sqlite3_last_insert_rowid(db);
        if (row_id > 0) {
            dout(1) << "Archived conversation turn to RAG database (id=" + std::to_string(row_id) + ")" << std::endl;
        } else {
            dout(1) << "Skipped duplicate conversation (hash=" + content_hash.substr(0, 16) + "...)" << std::endl;
        }
    }

    sqlite3_finalize(stmt);

    // Check size and prune if needed
    check_and_prune_if_needed();
}

std::vector<SearchResult> SQLiteBackend::search(const std::string& query, int max_results) {
    dout(1) << "Searching SQLite RAG database for: " + query << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    std::vector<SearchResult> results;

    // Convert natural language query to FTS5-compatible query
    std::vector<std::string> stop_words = {"and", "or", "the", "a", "an", "of", "to", "for", "in", "on", "at", "is", "are", "was", "were", "be", "been", "being"};
    std::string fts_query;
    std::istringstream iss(query);
    std::string word;
    std::vector<std::string> keywords;

    while (iss >> word) {
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);

        if (std::find(stop_words.begin(), stop_words.end(), lower_word) == stop_words.end()) {
            std::string escaped_word = word;
            size_t pos = 0;
            while ((pos = escaped_word.find('"', pos)) != std::string::npos) {
                escaped_word.replace(pos, 1, "\"\"");
                pos += 2;
            }
            keywords.push_back("\"" + escaped_word + "\"");
        }
    }

    if (!keywords.empty()) {
        fts_query = keywords[0];
        for (size_t i = 1; i < keywords.size(); i++) {
            fts_query += " OR " + keywords[i];
        }
    } else {
        std::string escaped_query = query;
        size_t pos = 0;
        while ((pos = escaped_query.find('"', pos)) != std::string::npos) {
            escaped_query.replace(pos, 1, "\"\"");
            pos += 2;
        }
        fts_query = "\"" + escaped_query + "\"";
    }

    dout(1) << "FTS5 query: " + fts_query << std::endl;

    const char* sql = R"(
        SELECT c.user_message, c.assistant_response, c.timestamp,
               bm25(conversations_fts) as score
        FROM conversations_fts
        JOIN conversations c ON conversations_fts.rowid = c.id
        WHERE conversations_fts MATCH ?
        ORDER BY score
        LIMIT ?
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare search statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return results;
    }

    sqlite3_bind_text(stmt, 1, fts_query.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, max_results);

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        std::string user_msg = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string assistant_msg = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        int64_t timestamp = sqlite3_column_int64(stmt, 2);
        double score = sqlite3_column_double(stmt, 3);

        std::string combined = "User: " + user_msg + "\nAssistant: " + assistant_msg;
        double normalized_score = 1.0 / (1.0 + std::exp(score / 10.0));

        results.emplace_back(combined, normalized_score, timestamp);
    }

    if (rc != SQLITE_DONE) {
        std::cerr << "Search query failed: " + std::string(sqlite3_errmsg(db)) << std::endl;
    }

    sqlite3_finalize(stmt);

    dout(1) << "Found " + std::to_string(results.size()) + " results for query: " + query << std::endl;
    return results;
}

size_t SQLiteBackend::get_archived_turn_count() const {
    if (!db_) {
        return 0;
    }

    sqlite3* db = static_cast<sqlite3*>(db_);
    const char* sql = "SELECT COUNT(*) FROM conversations";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare count statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return 0;
    }

    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

void SQLiteBackend::store_memory(const std::string& question, const std::string& answer) {
    dout(1) << "Storing memory: " + question << std::endl;
    ConversationTurn turn(question, answer);
    archive_turn(turn);
    dout(1) << "Stored memory: question=" + question + ", answer=" + answer << std::endl;
}

bool SQLiteBackend::clear_memory(const std::string& question) {
    if (!db_) {
        return false;
    }

    dout(1) << "Clearing memory by question: " + question << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    const char* sql = "DELETE FROM conversations WHERE id = (SELECT id FROM conversations WHERE user_message = ? LIMIT 1)";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare clear_memory statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, question.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to clear memory: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return false;
    }

    int changes = sqlite3_changes(db);
    if (changes > 0) {
        dout(1) << "Cleared memory for question: " + question << std::endl;
        return true;
    } else {
        dout(1) << "Memory not found for question: " + question << std::endl;
        return false;
    }
}

void SQLiteBackend::check_and_prune_if_needed() {
    if (!db_) {
        return;
    }

    try {
        if (!std::filesystem::exists(db_path_)) {
            return;
        }

        size_t current_size = std::filesystem::file_size(db_path_);

        if (current_size <= max_db_size_) {
            return;
        }

        dout(1) << "Database size (" + std::to_string(current_size / (1024 * 1024)) + " MB) exceeds limit (" +
                 std::to_string(max_db_size_ / (1024 * 1024)) + " MB), pruning oldest entries..." << std::endl;

        sqlite3* db = static_cast<sqlite3*>(db_);

        const char* count_sql = "SELECT COUNT(*) FROM conversations";
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, count_sql, -1, &stmt, nullptr);
        int total_entries = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            total_entries = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);

        int batch_size = std::max(1, total_entries / 10);
        int deleted_total = 0;

        while (current_size > max_db_size_ && total_entries > 0) {
            std::string delete_sql = "DELETE FROM conversations WHERE id IN "
                                    "(SELECT id FROM conversations ORDER BY timestamp ASC LIMIT " +
                                    std::to_string(batch_size) + ")";

            char* err_msg = nullptr;
            int rc = sqlite3_exec(db, delete_sql.c_str(), nullptr, nullptr, &err_msg);
            if (rc != SQLITE_OK) {
                std::cerr << "Failed to delete old entries: " + std::string(err_msg) << std::endl;
                sqlite3_free(err_msg);
                break;
            }

            int deleted = sqlite3_changes(db);
            deleted_total += deleted;
            total_entries -= deleted;

            dout(1) << "Running VACUUM to reclaim space..." << std::endl;
            rc = sqlite3_exec(db, "VACUUM", nullptr, nullptr, &err_msg);
            if (rc != SQLITE_OK) {
                dout(1) << std::string("WARNING: ") + "VACUUM failed: " + std::string(err_msg) << std::endl;
                sqlite3_free(err_msg);
            }

            current_size = std::filesystem::file_size(db_path_);

            dout(1) << "Deleted " + std::to_string(deleted) + " entries, new size: " +
                     std::to_string(current_size / (1024 * 1024)) + " MB" << std::endl;

            if (deleted == 0) {
                dout(1) << std::string("WARNING: ") + "No entries deleted in batch, stopping pruning" << std::endl;
                break;
            }
        }

        dout(1) << "Pruning complete: deleted " + std::to_string(deleted_total) + " old entries, " +
                 "final size: " + std::to_string(current_size / (1024 * 1024)) + " MB" << std::endl;

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error during size check: " + std::string(e.what()) << std::endl;
    }
}

void SQLiteBackend::shutdown() {
    dout(1) << "SQLiteBackend shutdown" << std::endl;
    if (db_) {
        sqlite3* db = static_cast<sqlite3*>(db_);
        sqlite3_close(db);
        db_ = nullptr;
        dout(1) << "SQLite RAG database connection closed" << std::endl;
    }
}

void SQLiteBackend::set_fact(const std::string& key, const std::string& value) {
    dout(1) << "Setting fact: " + key << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    int64_t now = ConversationTurn::get_current_timestamp();

    const char* sql = R"(
        INSERT OR REPLACE INTO facts (key, value, created_at, updated_at)
        VALUES (?, ?,
            COALESCE((SELECT created_at FROM facts WHERE key = ?), ?),
            ?
        )
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare set_fact statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return;
    }

    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, value.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, key.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 4, now);
    sqlite3_bind_int64(stmt, 5, now);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to set fact: " + std::string(sqlite3_errmsg(db)) << std::endl;
    } else {
        dout(1) << "Set fact: " + key + " = " + value << std::endl;
    }

    sqlite3_finalize(stmt);
}

std::string SQLiteBackend::get_fact(const std::string& key) const {
    if (!db_) {
        return "";
    }

    dout(1) << "Getting fact: " + key << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);

    const char* sql = "SELECT value FROM facts WHERE key = ?";
    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare get_fact statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return "";
    }

    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

    std::string value;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        dout(1) << "Found fact: " + key + " = " + value << std::endl;
        sqlite3_finalize(stmt);
        return value;
    }
    sqlite3_finalize(stmt);

    // Try common variations
    std::vector<std::string> variations;

    std::string no_underscore = key;
    no_underscore.erase(std::remove(no_underscore.begin(), no_underscore.end(), '_'), no_underscore.end());
    if (no_underscore != key) {
        variations.push_back(no_underscore);
    }

    std::string with_underscore;
    for (size_t i = 0; i < key.length(); i++) {
        if (i > 0 && std::isupper(key[i]) && std::islower(key[i-1])) {
            with_underscore += '_';
        }
        with_underscore += std::tolower(key[i]);
    }
    if (with_underscore != key) {
        variations.push_back(with_underscore);
    }

    for (const auto& variant : variations) {
        rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
        if (rc != SQLITE_OK) continue;

        sqlite3_bind_text(stmt, 1, variant.c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(stmt) == SQLITE_ROW) {
            value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            dout(1) << "Found fact with variation '" + variant + "': " + value << std::endl;
            sqlite3_finalize(stmt);
            return value;
        }
        sqlite3_finalize(stmt);
    }

    dout(1) << "Fact not found: " + key << std::endl;
    return "";
}

bool SQLiteBackend::has_fact(const std::string& key) const {
    if (!db_) {
        return false;
    }

    sqlite3* db = static_cast<sqlite3*>(db_);
    const char* sql = "SELECT 1 FROM facts WHERE key = ? LIMIT 1";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare has_fact statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

    bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return exists;
}

bool SQLiteBackend::clear_fact(const std::string& key) {
    if (!db_) {
        return false;
    }

    dout(1) << "Clearing fact: " + key << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    const char* sql = "DELETE FROM facts WHERE key = ?";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare clear_fact statement: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, key.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    if (rc != SQLITE_DONE) {
        std::cerr << "Failed to clear fact: " + std::string(sqlite3_errmsg(db)) << std::endl;
        return false;
    }

    int changes = sqlite3_changes(db);
    if (changes > 0) {
        dout(1) << "Cleared fact: " + key << std::endl;
        return true;
    } else {
        dout(1) << "Fact not found: " + key << std::endl;
        return false;
    }
}
