#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

// Forward declaration
class DatabaseBackend;

/// @brief Search result from RAG database
struct SearchResult {
    std::string content;        // Content of the result
    double relevance_score;     // Relevance score (0.0 to 1.0)
    int64_t timestamp;          // When the conversation occurred

    SearchResult(const std::string& content, double score, int64_t ts = 0)
        : content(content), relevance_score(score), timestamp(ts) {}
};

/// @brief Conversation turn for archiving
struct ConversationTurn {
    std::string user_message;
    std::string assistant_response;
    int64_t timestamp;

    ConversationTurn(const std::string& user, const std::string& assistant, int64_t ts = 0);

    static int64_t get_current_timestamp();
};

/// @brief Global RAG manager for conversation memory
class RAGManager {
public:
    /// @brief Initialize the global RAG system
    /// @param db_path Path to memory database (empty for default)
    /// @param max_db_size Maximum database size in bytes (default 10GB)
    /// @return True if initialization successful
    static bool initialize(const std::string& db_path = "", size_t max_db_size = 10ULL * 1024 * 1024 * 1024);

    /// @brief Shutdown the global RAG system
    static void shutdown();

    /// @brief Archive a conversation turn to memory
    /// @param turn Conversation turn to archive
    /// @param user_id User identifier for multi-tenant isolation
    static void archive_turn(const ConversationTurn& turn, const std::string& user_id);

    /// @brief Search archived conversations
    /// @param query Search query
    /// @param max_results Maximum number of results to return
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Vector of search results
    static std::vector<SearchResult> search_memory(const std::string& query, int max_results, const std::string& user_id);

    /// @brief Get count of archived conversation turns
    /// @return Number of archived turns
    static size_t get_archived_turn_count();

    /// @brief Check if RAG system is initialized
    /// @return True if initialized
    static bool is_initialized();

    // Fact storage
    /// @brief Store a specific fact for later retrieval
    /// @param key Unique identifier for the fact
    /// @param value The fact to store
    /// @param user_id User identifier for multi-tenant isolation
    static void set_fact(const std::string& key, const std::string& value, const std::string& user_id);

    /// @brief Retrieve a specific fact by key
    /// @param key Unique identifier for the fact
    /// @param user_id User identifier for multi-tenant isolation
    /// @return The stored fact, or empty string if not found
    static std::string get_fact(const std::string& key, const std::string& user_id);

    /// @brief Check if a fact exists
    /// @param key Unique identifier for the fact
    /// @param user_id User identifier for multi-tenant isolation
    /// @return True if fact exists
    static bool has_fact(const std::string& key, const std::string& user_id);

    /// @brief Delete a fact by key
    /// @param key Unique identifier for the fact
    /// @param user_id User identifier for multi-tenant isolation
    /// @return True if fact was deleted, false if it didn't exist
    static bool clear_fact(const std::string& key, const std::string& user_id);

    /// @brief Get all facts for a user
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Vector of (key, value) pairs
    static std::vector<std::pair<std::string, std::string>> get_all_facts(const std::string& user_id);

    // Core tool interfaces
    /// @brief Get search memory tool name
    static std::string get_search_tool_name();

    /// @brief Get search memory tool description
    static std::string get_search_tool_description();

    /// @brief Get search memory tool parameters schema
    static std::string get_search_tool_parameters();

    /// @brief Execute search memory as a tool
    /// @param query Search query string
    /// @param max_results Maximum number of results (default 5)
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Formatted search results as string
    static std::string execute_search_tool(const std::string& query, int max_results, const std::string& user_id);

    /// @brief Get set_fact tool name
    static std::string get_set_fact_tool_name();

    /// @brief Get set_fact tool description
    static std::string get_set_fact_tool_description();

    /// @brief Get set_fact tool parameters schema
    static std::string get_set_fact_tool_parameters();

    /// @brief Execute set_fact as a tool
    /// @param key Fact identifier
    /// @param value Fact value
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Success message
    static std::string execute_set_fact_tool(const std::string& key, const std::string& value, const std::string& user_id);

    /// @brief Get get_fact tool name
    static std::string get_get_fact_tool_name();

    /// @brief Get get_fact tool description
    static std::string get_get_fact_tool_description();

    /// @brief Get get_fact tool parameters schema
    static std::string get_get_fact_tool_parameters();

    /// @brief Execute get_fact as a tool
    /// @param key Fact identifier
    /// @param user_id User identifier for multi-tenant isolation
    /// @return The fact value or error message
    static std::string execute_get_fact_tool(const std::string& key, const std::string& user_id);

    /// @brief Get the name of the clear_fact tool
    static std::string get_clear_fact_tool_name();

    /// @brief Get the description of the clear_fact tool
    static std::string get_clear_fact_tool_description();

    /// @brief Get the parameters of the clear_fact tool
    static std::string get_clear_fact_tool_parameters();

    /// @brief Execute clear_fact as a tool
    /// @param key Fact identifier to delete
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Success/error message
    static std::string execute_clear_fact_tool(const std::string& key, const std::string& user_id);

    // Memory management
    /// @brief Store a question/answer pair directly to memory
    /// @param question The question text
    /// @param answer The answer text
    /// @param user_id User identifier for multi-tenant isolation
    static void store_memory(const std::string& question, const std::string& answer, const std::string& user_id);

    /// @brief Clear a memory by exact question match
    /// @param question The exact question to delete
    /// @param user_id User identifier for multi-tenant isolation
    /// @return True if memory was deleted
    static bool clear_memory(const std::string& question, const std::string& user_id);

    /// @brief Get the name of the store_memory tool
    static std::string get_store_memory_tool_name();

    /// @brief Get the description of the store_memory tool
    static std::string get_store_memory_tool_description();

    /// @brief Get the parameters of the store_memory tool
    static std::string get_store_memory_tool_parameters();

    /// @brief Execute store_memory as a tool
    /// @param question The question text
    /// @param answer The answer text
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Success/error message
    static std::string execute_store_memory_tool(const std::string& question, const std::string& answer, const std::string& user_id);

    /// @brief Get the name of the clear_memory tool
    static std::string get_clear_memory_tool_name();

    /// @brief Get the description of the clear_memory tool
    static std::string get_clear_memory_tool_description();

    /// @brief Get the parameters of the clear_memory tool
    static std::string get_clear_memory_tool_parameters();

    /// @brief Execute clear_memory as a tool
    /// @param question The exact question to delete
    /// @param user_id User identifier for multi-tenant isolation
    /// @return Success/error message
    static std::string execute_clear_memory_tool(const std::string& question, const std::string& user_id);

private:
    static std::unique_ptr<DatabaseBackend> instance_;
};
