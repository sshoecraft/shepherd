#include "shepherd.h"
#include "rag.h"
#include "rag/database_backend.h"
#include "config.h"
#include <chrono>
#include <sstream>
#include <iomanip>

// ConversationTurn implementation
ConversationTurn::ConversationTurn(const std::string& user, const std::string& assistant, int64_t ts)
    : user_message(user), assistant_response(assistant),
      timestamp(ts == 0 ? get_current_timestamp() : ts) {}

int64_t ConversationTurn::get_current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// RAGManager static implementation
std::unique_ptr<DatabaseBackend> RAGManager::instance_ = nullptr;

bool RAGManager::initialize(const std::string& db_path, size_t max_db_size) {
    if (instance_) {
        dout(1) << std::string("WARNING: ") + "RAGManager already initialized" << std::endl;
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

    try {
        instance_ = DatabaseBackendFactory::create(rag_path, max_db_size);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create database backend: " + std::string(e.what()) << std::endl;
        return false;
    }

    if (!instance_->initialize()) {
        std::cerr << "Failed to initialize RAG database" << std::endl;
        instance_.reset();
        return false;
    }

    dout(1) << "RAG system initialized successfully (backend: " + instance_->backend_type() + ")" << std::endl;
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
