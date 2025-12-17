#include "shepherd.h"
#include "rag.h"
#include "config.h"
#include <thread>
#include <mutex>

#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <sqlite3.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <openssl/sha.h>

// ConversationTurn implementation
ConversationTurn::ConversationTurn(const std::string& user, const std::string& assistant, int64_t ts)
    : user_message(user), assistant_response(assistant),
      timestamp(ts == 0 ? get_current_timestamp() : ts) {}

int64_t ConversationTurn::get_current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// RAGDatabase implementation
RAGDatabase::RAGDatabase(const std::string& db_path, size_t max_db_size)
    : db_path_(db_path), db_(nullptr), max_db_size_(max_db_size) {
    dout(1) << "RAGDatabase created with path: " + db_path + ", max size: " + std::to_string(max_db_size / (1024 * 1024)) + " MB" << std::endl;
}

RAGDatabase::~RAGDatabase() {
    dout(1) << "RAGDatabase destructor" << std::endl;
    shutdown();
}

bool RAGDatabase::initialize() {
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

    dout(1) << "Initializing RAG database: " + db_path_ << std::endl;

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
    // Set mmap_size to max_db_size (SQLite will memory-map the database file)
    std::string mmap_pragma = "PRAGMA mmap_size = " + std::to_string(max_db_size_);
    rc = sqlite3_exec(db, mmap_pragma.c_str(), nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        dout(1) << std::string("WARNING: ") +"Failed to enable memory mapping: " + std::string(sqlite3_errmsg(db)) << std::endl;
        // Continue anyway - not critical
    } else {
        dout(1) << "Enabled memory-mapped I/O (mmap_size: " + std::to_string(max_db_size_ / (1024 * 1024)) + " MB)" << std::endl;
    }

    // Create tables
    if (!create_tables()) {
        std::cerr << "Failed to create database tables" << std::endl;
        shutdown();
        return false;
    }

    dout(1) << "RAG database initialized successfully" << std::endl;
    return true;
}

bool RAGDatabase::create_tables() {
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

void RAGDatabase::archive_turn(const ConversationTurn& turn) {
    dout(1) << "Archiving conversation turn to RAG database" << std::endl;

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
    // Use INSERT OR IGNORE to skip duplicate conversations (UNIQUE constraint on content_hash)
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

std::vector<SearchResult> RAGDatabase::search(const std::string& query, int max_results) {
    dout(1) << "Searching RAG database for: " + query << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    std::vector<SearchResult> results;

    // Convert natural language query to FTS5-compatible query
    // Remove common stop words and extract meaningful terms
    std::vector<std::string> stop_words = {"and", "or", "the", "a", "an", "of", "to", "for", "in", "on", "at", "is", "are", "was", "were", "be", "been", "being"};
    std::string fts_query;
    std::istringstream iss(query);
    std::string word;
    std::vector<std::string> keywords;

    while (iss >> word) {
        // Convert to lowercase for stop word check
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);

        // Skip stop words
        if (std::find(stop_words.begin(), stop_words.end(), lower_word) == stop_words.end()) {
            // Escape double quotes in the word and wrap in quotes for FTS5
            std::string escaped_word = word;
            size_t pos = 0;
            while ((pos = escaped_word.find('"', pos)) != std::string::npos) {
                escaped_word.replace(pos, 1, "\"\"");
                pos += 2;
            }
            keywords.push_back("\"" + escaped_word + "\"");
        }
    }

    // Join keywords with OR for broader matching
    if (!keywords.empty()) {
        fts_query = keywords[0];
        for (size_t i = 1; i < keywords.size(); i++) {
            fts_query += " OR " + keywords[i];
        }
    } else {
        // Fallback: escape and quote the entire original query
        std::string escaped_query = query;
        size_t pos = 0;
        while ((pos = escaped_query.find('"', pos)) != std::string::npos) {
            escaped_query.replace(pos, 1, "\"\"");
            pos += 2;
        }
        fts_query = "\"" + escaped_query + "\"";
    }

    dout(1) << "FTS5 query: " + fts_query << std::endl;

    // Use FTS5 search with BM25 ranking
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

        // Combine user and assistant messages for the result
        std::string combined = "User: " + user_msg + "\nAssistant: " + assistant_msg;

        // Normalize BM25 score to 0.0-1.0 range (BM25 can be negative, higher is better)
        // We'll use a simple sigmoid-like transformation
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

size_t RAGDatabase::get_archived_turn_count() const {
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

void RAGDatabase::store_memory(const std::string& question, const std::string& answer) {
    dout(1) << "Storing memory: " + question << std::endl;

    // Create a ConversationTurn and archive it
    ConversationTurn turn(question, answer);
    archive_turn(turn);

    dout(1) << "Stored memory: question=" + question + ", answer=" + answer << std::endl;

    // Note: archive_turn() already calls check_and_prune_if_needed()
}

bool RAGDatabase::clear_memory(const std::string& question) {
    if (!db_) {
        return false;
    }

    dout(1) << "Clearing memory by question: " + question << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);

    // Delete by exact match on user_message, limit to 1 row
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

void RAGDatabase::check_and_prune_if_needed() {
    if (!db_) {
        return;
    }

    // Check current database file size
    try {
        if (!std::filesystem::exists(db_path_)) {
            return;  // DB file doesn't exist yet
        }

        size_t current_size = std::filesystem::file_size(db_path_);

        // If we're under the limit, nothing to do
        if (current_size <= max_db_size_) {
            return;
        }

        dout(1) << "Database size (" + std::to_string(current_size / (1024 * 1024)) + " MB) exceeds limit (" +
                 std::to_string(max_db_size_ / (1024 * 1024)) + " MB), pruning oldest entries..." << std::endl;

        sqlite3* db = static_cast<sqlite3*>(db_);

        // Count total entries before pruning
        const char* count_sql = "SELECT COUNT(*) FROM conversations";
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, count_sql, -1, &stmt, nullptr);
        int total_entries = 0;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            total_entries = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);

        // Delete oldest 10% of entries at a time until we're under the limit
        int batch_size = std::max(1, total_entries / 10);
        int deleted_total = 0;

        while (current_size > max_db_size_ && total_entries > 0) {
            // Delete oldest batch (by timestamp)
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

            // Run VACUUM to reclaim space (this is expensive but necessary)
            dout(1) << "Running VACUUM to reclaim space..." << std::endl;
            rc = sqlite3_exec(db, "VACUUM", nullptr, nullptr, &err_msg);
            if (rc != SQLITE_OK) {
                dout(1) << std::string("WARNING: ") +"VACUUM failed: " + std::string(err_msg) << std::endl;
                sqlite3_free(err_msg);
            }

            // Check new size
            current_size = std::filesystem::file_size(db_path_);

            dout(1) << "Deleted " + std::to_string(deleted) + " entries, new size: " +
                     std::to_string(current_size / (1024 * 1024)) + " MB" << std::endl;

            // Safety check: if we deleted a batch but size didn't decrease much, stop
            if (deleted == 0) {
                dout(1) << std::string("WARNING: ") +"No entries deleted in batch, stopping pruning" << std::endl;
                break;
            }
        }

        dout(1) << "Pruning complete: deleted " + std::to_string(deleted_total) + " old entries, " +
                 "final size: " + std::to_string(current_size / (1024 * 1024)) + " MB" << std::endl;

    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error during size check: " + std::string(e.what()) << std::endl;
    }
}

void RAGDatabase::shutdown() {
    dout(1) << "RAGDatabase shutdown" << std::endl;
    if (db_) {
        sqlite3* db = static_cast<sqlite3*>(db_);
        sqlite3_close(db);
        db_ = nullptr;
        dout(1) << "RAG database connection closed" << std::endl;
    }
}

// Fact storage implementation
void RAGDatabase::set_fact(const std::string& key, const std::string& value) {
    dout(1) << "Setting fact: " + key << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);
    int64_t now = ConversationTurn::get_current_timestamp();

    // Use INSERT OR REPLACE to update if key exists
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

std::string RAGDatabase::get_fact(const std::string& key) const {
    if (!db_) {
        return "";
    }

    dout(1) << "Getting fact: " + key << std::endl;

    sqlite3* db = static_cast<sqlite3*>(db_);

    // Try the exact key first
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

    // If not found, try common variations
    std::vector<std::string> variations;

    // Remove underscores: user_name -> username
    std::string no_underscore = key;
    no_underscore.erase(std::remove(no_underscore.begin(), no_underscore.end(), '_'), no_underscore.end());
    if (no_underscore != key) {
        variations.push_back(no_underscore);
    }

    // Add underscores before capitals: userName -> user_name
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

    // Try each variation
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

bool RAGDatabase::has_fact(const std::string& key) const {
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

bool RAGDatabase::clear_fact(const std::string& key) {
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

// RAGManager static implementation
std::unique_ptr<RAGDatabase> RAGManager::instance_ = nullptr;

bool RAGManager::initialize(const std::string& db_path, size_t max_db_size) {
    if (instance_) {
        dout(1) << std::string("WARNING: ") +"RAGManager already initialized" << std::endl;
        return true;
    }

    std::string rag_path = db_path;
    if (rag_path.empty()) {
        // Generate default path using XDG spec
        try {
            rag_path = Config::get_default_memory_db_path();
        } catch (const ConfigError& e) {
            std::cerr << "Failed to determine memory database path: " + std::string(e.what()) << std::endl;
            return false;
        }
    }

    dout(1) << "Initializing RAG system with database: " + rag_path + ", max size: " + std::to_string(max_db_size / (1024 * 1024)) + " MB" << std::endl;
    instance_ = std::make_unique<RAGDatabase>(rag_path, max_db_size);

    if (!instance_->initialize()) {
        std::cerr << "Failed to initialize RAG database" << std::endl;
        instance_.reset();
        return false;
    }

    dout(1) << "RAG system initialized successfully" << std::endl;
    return true;
}

void RAGManager::shutdown() {
    if (instance_) {
        instance_->shutdown();
        instance_.reset();
        dout(1) << "RAG system shutdown complete" << std::endl;
    }
}

void RAGManager::archive_turn(const ConversationTurn& turn) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot archive turn" << std::endl;
        return;
    }
    instance_->archive_turn(turn);
}

std::vector<SearchResult> RAGManager::search_memory(const std::string& query, int max_results) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot search" << std::endl;
        return {};
    }
    return instance_->search(query, max_results);
}

size_t RAGManager::get_archived_turn_count() {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot get count" << std::endl;
        return 0;
    }
    return instance_->get_archived_turn_count();
}

bool RAGManager::is_initialized() {
    return instance_ != nullptr;
}

// Fact storage wrapper methods
void RAGManager::set_fact(const std::string& key, const std::string& value) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot set fact" << std::endl;
        return;
    }
    instance_->set_fact(key, value);
}

std::string RAGManager::get_fact(const std::string& key) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot get fact" << std::endl;
        return "";
    }
    return instance_->get_fact(key);
}

bool RAGManager::has_fact(const std::string& key) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot check fact" << std::endl;
        return false;
    }
    return instance_->has_fact(key);
}

bool RAGManager::clear_fact(const std::string& key) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot clear fact" << std::endl;
        return false;
    }
    return instance_->clear_fact(key);
}

// Core tool interface implementation
std::string RAGManager::get_search_tool_name() {
    return "search_memory";
}

std::string RAGManager::get_search_tool_description() {
    return "Search historical conversation context for relevant information that may have slid out of the current context window";
}

std::string RAGManager::get_search_tool_parameters() {
    return "query=\"search_query\", max_results=\"5\"";
}

std::string RAGManager::execute_search_tool(const std::string& query, int max_results) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute search tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (query.empty()) {
        std::cerr << "Search tool called with empty query" << std::endl;
        return "Error: Query parameter is required";
    }

    dout(1) << "SEARCH_MEMORY tool called with query: '" + query + "', max_results: " + std::to_string(max_results) << std::endl;

    try {
        auto search_results = search_memory(query, max_results);

        if (search_results.empty()) {
            dout(1) << "SEARCH_MEMORY: No results found" << std::endl;
            return "No archived conversations found matching: " + query;
        }

        std::ostringstream oss;
        oss << "Found " << search_results.size() << " archived conversation(s):\n\n";

        dout(1) << "SEARCH_MEMORY: Found " + std::to_string(search_results.size()) + " results:" << std::endl;
        for (size_t i = 0; i < search_results.size(); i++) {
            const auto& sr = search_results[i];
            oss << "Result " << (i + 1) << " [Relevance: "
                << std::fixed << std::setprecision(2) << sr.relevance_score << "]:\n"
                << sr.content << "\n\n";

            // Log each result in debug
            std::string preview = sr.content.length() > 100 ? sr.content.substr(0, 100) + "..." : sr.content;
            dout(1) << "  [" + std::to_string(i + 1) + "] Score: " + std::to_string(sr.relevance_score) +
                      ", Content preview: " + preview << std::endl;
        }

        dout(1) << "SEARCH_MEMORY: Returning " + std::to_string(search_results.size()) + " results to model" << std::endl;
        return oss.str();

    } catch (const std::exception& e) {
        std::cerr << "Error executing search tool: " + std::string(e.what()) << std::endl;
        return "Error: Search failed - " + std::string(e.what());
    }
}

// set_fact tool interface
std::string RAGManager::get_set_fact_tool_name() {
    return "set_fact";
}

std::string RAGManager::get_set_fact_tool_description() {
    return "Store a specific piece of information for later retrieval. Use this to remember important facts like user preferences, names, project details, or any information you'll need to recall. Facts are stored permanently and don't depend on context window.";
}

std::string RAGManager::get_set_fact_tool_parameters() {
    return "key=\"fact_identifier\", value=\"fact_content\"";
}

std::string RAGManager::execute_set_fact_tool(const std::string& key, const std::string& value) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute set_fact tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (key.empty()) {
        std::cerr << "set_fact called with empty key" << std::endl;
        return "Error: Key parameter is required";
    }

    if (value.empty()) {
        std::cerr << "set_fact called with empty value" << std::endl;
        return "Error: Value parameter is required";
    }

    dout(1) << "SET_FACT tool called with key: '" + key + "', value: '" + value + "'" << std::endl;

    try {
        set_fact(key, value);
        dout(1) << "SET_FACT: Stored fact '" + key + "'" << std::endl;
        return "Successfully stored fact: " + key;
    } catch (const std::exception& e) {
        std::cerr << "Error executing set_fact tool: " + std::string(e.what()) << std::endl;
        return "Error: Failed to store fact - " + std::string(e.what());
    }
}

// get_fact tool interface
std::string RAGManager::get_get_fact_tool_name() {
    return "get_fact";
}

std::string RAGManager::get_get_fact_tool_description() {
    return "Retrieve a specific piece of information that was previously stored. Use this to recall facts like user preferences, names, or other important details you've learned. Returns the stored value or indicates if the fact doesn't exist.";
}

std::string RAGManager::get_get_fact_tool_parameters() {
    return "key=\"fact_identifier\"";
}

std::string RAGManager::execute_get_fact_tool(const std::string& key) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute get_fact tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (key.empty()) {
        std::cerr << "get_fact called with empty key" << std::endl;
        return "Error: Key parameter is required";
    }

    dout(1) << "GET_FACT tool called with key: '" + key + "'" << std::endl;

    try {
        std::string value = get_fact(key);

        if (value.empty()) {
            dout(1) << "GET_FACT: Fact '" + key + "' not found" << std::endl;
            return "Fact not found: " + key;
        }

        dout(1) << "GET_FACT: Retrieved fact '" + key + "' = '" + value + "'" << std::endl;
        return value;
    } catch (const std::exception& e) {
        std::cerr << "Error executing get_fact tool: " + std::string(e.what()) << std::endl;
        return "Error: Failed to retrieve fact - " + std::string(e.what());
    }
}

// clear_fact tool interface
std::string RAGManager::get_clear_fact_tool_name() {
    return "clear_fact";
}

std::string RAGManager::get_clear_fact_tool_description() {
    return "Delete a specific piece of information that was previously stored. Use this to remove facts that are no longer needed, outdated, or incorrect.";
}

std::string RAGManager::get_clear_fact_tool_parameters() {
    return "key=\"fact_identifier\"";
}

std::string RAGManager::execute_clear_fact_tool(const std::string& key) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute clear_fact tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (key.empty()) {
        std::cerr << "clear_fact called with empty key" << std::endl;
        return "Error: Key parameter is required";
    }

    dout(1) << "CLEAR_FACT tool called with key: '" + key + "'" << std::endl;

    try {
        bool deleted = clear_fact(key);

        if (deleted) {
            dout(1) << "CLEAR_FACT: Deleted fact '" + key + "'" << std::endl;
            return "Successfully deleted fact: " + key;
        } else {
            dout(1) << "CLEAR_FACT: Fact '" + key + "' not found" << std::endl;
            return "Fact not found: " + key;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error executing clear_fact tool: " + std::string(e.what()) << std::endl;
        return "Error: Failed to delete fact - " + std::string(e.what());
    }
}

// Memory management wrapper methods
void RAGManager::store_memory(const std::string& question, const std::string& answer) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot store memory" << std::endl;
        return;
    }
    instance_->store_memory(question, answer);
}

bool RAGManager::clear_memory(const std::string& question) {
    if (!instance_) {
        std::cerr << "RAGManager not initialized - cannot clear memory" << std::endl;
        return false;
    }
    return instance_->clear_memory(question);
}

// store_memory tool interface
std::string RAGManager::get_store_memory_tool_name() {
    return "store_memory";
}

std::string RAGManager::get_store_memory_tool_description() {
    return "Store a question/answer pair directly to long-term memory. Use this to save important information that should be remembered across sessions, even if it hasn't been part of a natural conversation yet.";
}

std::string RAGManager::get_store_memory_tool_parameters() {
    return "question=\"the_question\", answer=\"the_answer\"";
}

std::string RAGManager::execute_store_memory_tool(const std::string& question, const std::string& answer) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute store_memory tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (question.empty()) {
        std::cerr << "store_memory called with empty question" << std::endl;
        return "Error: Question parameter is required";
    }

    if (answer.empty()) {
        std::cerr << "store_memory called with empty answer" << std::endl;
        return "Error: Answer parameter is required";
    }

    dout(1) << "STORE_MEMORY tool called with question: '" + question + "', answer: '" + answer + "'" << std::endl;

    try {
        store_memory(question, answer);
        dout(1) << "STORE_MEMORY: Stored Q/A pair" << std::endl;
        return "Successfully stored memory: " + question;
    } catch (const std::exception& e) {
        std::cerr << "Error executing store_memory tool: " + std::string(e.what()) << std::endl;
        return "Error: Failed to store memory - " + std::string(e.what());
    }
}

// clear_memory tool interface
std::string RAGManager::get_clear_memory_tool_name() {
    return "clear_memory";
}

std::string RAGManager::get_clear_memory_tool_description() {
    return "Delete a specific memory from long-term storage by exact question match. Use this to remove outdated, incorrect, or no-longer-needed information from memory.";
}

std::string RAGManager::get_clear_memory_tool_parameters() {
    return "question=\"exact_question_to_delete\"";
}

std::string RAGManager::execute_clear_memory_tool(const std::string& question) {
    if (!is_initialized()) {
        std::cerr << "RAGManager not initialized - cannot execute clear_memory tool" << std::endl;
        return "Error: RAG system not initialized";
    }

    if (question.empty()) {
        std::cerr << "clear_memory called with empty question" << std::endl;
        return "Error: Question parameter is required";
    }

    dout(1) << "CLEAR_MEMORY tool called with question: '" + question + "'" << std::endl;

    try {
        bool deleted = clear_memory(question);

        if (deleted) {
            dout(1) << "CLEAR_MEMORY: Deleted memory for question '" + question + "'" << std::endl;
            return "Successfully deleted memory: " + question;
        } else {
            dout(1) << "CLEAR_MEMORY: Memory not found for question '" + question + "'" << std::endl;
            return "Memory not found: " + question;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error executing clear_memory tool: " + std::string(e.what()) << std::endl;
        return "Error: Failed to delete memory - " + std::string(e.what());
    }
}