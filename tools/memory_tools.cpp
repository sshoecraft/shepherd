#if 1
#include "memory_tools.h"
#include "tools.h"
#include "../rag.h"
#include "../logger.h"
#include <sstream>
#include <iomanip>

std::string SearchMemoryTool::unsanitized_name() const {
    return "search_memory";
}

std::string SearchMemoryTool::description() const {
    return "Search historical conversation context for relevant information that may have slid out of the current context window";
}

std::string SearchMemoryTool::parameters() const {
    return "query=\"search_query\", max_results=\"5\"";
}

std::vector<ParameterDef> SearchMemoryTool::get_parameters_schema() const {
    return {
        {"query", "string", "The search query to find relevant information", true, "", "", {}},
        {"max_results", "string", "Maximum number of results to return", false, "5", "", {}}
    };
}

std::map<std::string, std::any> SearchMemoryTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    if (!RAGManager::is_initialized()) {
        result["error"] = std::string("RAG system not initialized");
        return result;
    }

    std::string query = tool_utils::get_string(args, "query");
    int max_results = tool_utils::get_int(args, "max_results", 5);

    if (query.empty()) {
        result["error"] = std::string("Query parameter is required");
        return result;
    }

    LOG_DEBUG("SEARCH_MEMORY called with query: '" + query + "', max_results: " + std::to_string(max_results));

    try {
        auto search_results = RAGManager::search_memory(query, max_results);

        if (search_results.empty()) {
            LOG_DEBUG("SEARCH_MEMORY: No results found");
            result["output"] = std::string("No archived conversations found matching: " + query);
            return result;
        }

        std::ostringstream oss;
        oss << "Found " << search_results.size() << " archived conversation(s):\n\n";

        // Hard limit: each search result should be max 2000 chars to prevent huge results
        const size_t MAX_RESULT_CHARS = 2000;
        size_t total_chars = 0;
        const size_t MAX_TOTAL_CHARS = 10000; // Cap total output at 10K chars

        LOG_DEBUG("SEARCH_MEMORY: Found " + std::to_string(search_results.size()) + " results:");
        for (size_t i = 0; i < search_results.size(); i++) {
            const auto& sr = search_results[i];

            // Sanitize content: validate UTF-8 and remove invalid sequences
            std::string content;
            content.reserve(sr.content.length());

            for (size_t j = 0; j < sr.content.length(); ) {
                unsigned char c = sr.content[j];

                // Single-byte ASCII (0x00-0x7F)
                if (c < 0x80) {
                    // Replace control characters (except tab, newline, carriage return) with space
                    if ((c >= 32 && c <= 126) || c == '\t' || c == '\n' || c == '\r') {
                        content += c;
                    } else {
                        content += ' ';
                    }
                    j++;
                }
                // Multi-byte UTF-8 sequence
                else if ((c & 0xE0) == 0xC0) {
                    // 2-byte sequence (110xxxxx 10xxxxxx)
                    if (j + 1 < sr.content.length() && (sr.content[j+1] & 0xC0) == 0x80) {
                        content += sr.content[j];
                        content += sr.content[j+1];
                        j += 2;
                    } else {
                        content += '?';  // Invalid sequence
                        j++;
                    }
                }
                else if ((c & 0xF0) == 0xE0) {
                    // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
                    if (j + 2 < sr.content.length() &&
                        (sr.content[j+1] & 0xC0) == 0x80 &&
                        (sr.content[j+2] & 0xC0) == 0x80) {
                        content += sr.content[j];
                        content += sr.content[j+1];
                        content += sr.content[j+2];
                        j += 3;
                    } else {
                        content += '?';  // Invalid sequence
                        j++;
                    }
                }
                else if ((c & 0xF8) == 0xF0) {
                    // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
                    if (j + 3 < sr.content.length() &&
                        (sr.content[j+1] & 0xC0) == 0x80 &&
                        (sr.content[j+2] & 0xC0) == 0x80 &&
                        (sr.content[j+3] & 0xC0) == 0x80) {
                        content += sr.content[j];
                        content += sr.content[j+1];
                        content += sr.content[j+2];
                        content += sr.content[j+3];
                        j += 4;
                    } else {
                        content += '?';  // Invalid sequence
                        j++;
                    }
                }
                else {
                    // Invalid UTF-8 lead byte
                    content += '?';
                    j++;
                }
            }

            // Truncate individual result if needed
            bool truncated = false;
            if (content.length() > MAX_RESULT_CHARS) {
                content = content.substr(0, MAX_RESULT_CHARS);
                truncated = true;
            }

            // Check total size limit
            if (total_chars + content.length() > MAX_TOTAL_CHARS) {
                oss << "\n[Additional results truncated - total output limit reached]\n";
                break;
            }

            oss << "Result " << (i + 1) << " [Relevance: "
                << std::fixed << std::setprecision(2) << sr.relevance_score << "]:\n"
                << content;

            if (truncated) {
                oss << "\n[... truncated to " << MAX_RESULT_CHARS << " chars]";
            }

            oss << "\n\n";
            total_chars += content.length();

            // Log each result in debug
            std::string preview = sr.content.length() > 100 ? sr.content.substr(0, 100) + "..." : sr.content;
            LOG_DEBUG("  [" + std::to_string(i + 1) + "] Score: " + std::to_string(sr.relevance_score) +
                      ", Content preview: " + preview);
        }

        result["output"] = oss.str();

        LOG_DEBUG("SEARCH_MEMORY: Returning " + std::to_string(search_results.size()) + " results to model");

    } catch (const std::exception& e) {
        LOG_ERROR("Error searching memory: " + std::string(e.what()));
        result["error"] = std::string("Search failed: ") + e.what();
    }

    return result;
}

// SetFactTool implementation
std::string SetFactTool::unsanitized_name() const {
    return RAGManager::get_set_fact_tool_name();
}

std::string SetFactTool::description() const {
    return RAGManager::get_set_fact_tool_description();
}

std::string SetFactTool::parameters() const {
    return RAGManager::get_set_fact_tool_parameters();
}

std::vector<ParameterDef> SetFactTool::get_parameters_schema() const {
    return {
        {"key", "string", "Identifier for the fact to store", true, "", "", {}},
        {"value", "string", "Content of the fact to store", true, "", "", {}}
    };
}

std::map<std::string, std::any> SetFactTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string key = tool_utils::get_string(args, "key");
    std::string value = tool_utils::get_string(args, "value");

    std::string response = RAGManager::execute_set_fact_tool(key, value);

    if (response.find("Error:") == 0) {
        result["error"] = response;
    } else {
        result["output"] = response;
    }

    return result;
}

// GetFactTool implementation
std::string GetFactTool::unsanitized_name() const {
    return RAGManager::get_get_fact_tool_name();
}

std::string GetFactTool::description() const {
    return RAGManager::get_get_fact_tool_description();
}

std::string GetFactTool::parameters() const {
    return RAGManager::get_get_fact_tool_parameters();
}

std::vector<ParameterDef> GetFactTool::get_parameters_schema() const {
    return {
        {"key", "string", "Identifier of the fact to retrieve", true, "", "", {}}
    };
}

std::map<std::string, std::any> GetFactTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string key = tool_utils::get_string(args, "key");

    std::string response = RAGManager::execute_get_fact_tool(key);

    // "Fact not found" is a successful tool execution, just with no result
    if (response.find("Error:") == 0 && response.find("Fact not found:") != 0) {
        result["error"] = response;
    } else {
        result["output"] = response;
    }

    return result;
}

// ClearFactTool implementation
std::string ClearFactTool::unsanitized_name() const {
    return RAGManager::get_clear_fact_tool_name();
}

std::string ClearFactTool::description() const {
    return RAGManager::get_clear_fact_tool_description();
}

std::string ClearFactTool::parameters() const {
    return RAGManager::get_clear_fact_tool_parameters();
}

std::vector<ParameterDef> ClearFactTool::get_parameters_schema() const {
    return {
        {"key", "string", "Identifier of the fact to clear", true, "", "", {}}
    };
}

std::map<std::string, std::any> ClearFactTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string key = tool_utils::get_string(args, "key");

    std::string response = RAGManager::execute_clear_fact_tool(key);

    // "Fact not found" is a successful tool execution, just with no deletion
    if (response.find("Error:") == 0 && response.find("Fact not found:") != 0) {
        result["error"] = response;
    } else {
        result["output"] = response;
    }

    return result;
}

// StoreMemoryTool implementation
std::string StoreMemoryTool::unsanitized_name() const {
    return RAGManager::get_store_memory_tool_name();
}

std::string StoreMemoryTool::description() const {
    return RAGManager::get_store_memory_tool_description();
}

std::string StoreMemoryTool::parameters() const {
    return RAGManager::get_store_memory_tool_parameters();
}

std::vector<ParameterDef> StoreMemoryTool::get_parameters_schema() const {
    return {
        {"question", "string", "The question text to store", true, "", "", {}},
        {"answer", "string", "The answer text to store", true, "", "", {}}
    };
}

std::map<std::string, std::any> StoreMemoryTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string question = tool_utils::get_string(args, "question");
    std::string answer = tool_utils::get_string(args, "answer");

    std::string response = RAGManager::execute_store_memory_tool(question, answer);

    if (response.find("Error:") == 0) {
        result["error"] = response;
    } else {
        result["output"] = response;
    }

    return result;
}

// ClearMemoryTool implementation
std::string ClearMemoryTool::unsanitized_name() const {
    return RAGManager::get_clear_memory_tool_name();
}

std::string ClearMemoryTool::description() const {
    return RAGManager::get_clear_memory_tool_description();
}

std::string ClearMemoryTool::parameters() const {
    return RAGManager::get_clear_memory_tool_parameters();
}

std::vector<ParameterDef> ClearMemoryTool::get_parameters_schema() const {
    return {
        {"question", "string", "The exact question to delete from memory", true, "", "", {}}
    };
}

std::map<std::string, std::any> ClearMemoryTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string question = tool_utils::get_string(args, "question");

    std::string response = RAGManager::execute_clear_memory_tool(question);

    // "Memory not found" is a successful tool execution, just with no deletion
    if (response.find("Error:") == 0 && response.find("Memory not found:") != 0) {
        result["error"] = response;
    } else {
        result["output"] = response;
    }

    return result;
}

void register_memory_tools(Tools& tools) {
    // Register memory tools FIRST so they appear at top of tool list
    tools.register_tool(std::make_unique<SearchMemoryTool>());
    tools.register_tool(std::make_unique<StoreMemoryTool>());
    tools.register_tool(std::make_unique<ClearMemoryTool>());
    tools.register_tool(std::make_unique<SetFactTool>());
    tools.register_tool(std::make_unique<GetFactTool>());
    tools.register_tool(std::make_unique<ClearFactTool>());
    LOG_DEBUG("Registered memory tools: search_memory, store_memory, clear_memory, set_fact, get_fact, clear_fact");
}
#endif
