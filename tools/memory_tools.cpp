#include "shepherd.h"
#if 1
#include "memory_tools.h"
#include "tools.h"
#include "../rag.h"

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

    std::string user_id = tool_utils::get_string(args, "_user_id");

    dout(1) << "SEARCH_MEMORY called with query: '" + query + "', max_results: " + std::to_string(max_results) << std::endl;

    try {
        auto search_results = RAGManager::search_memory(query, max_results, user_id);

        if (search_results.empty()) {
            dout(1) << "SEARCH_MEMORY: No results found" << std::endl;
            result["output"] = std::string("No archived conversations found matching: " + query);
            return result;
        }

        std::ostringstream oss;
        oss << "Found " << search_results.size() << " archived conversation(s):\n\n";

        // No limits here - truncation handled by CLI based on context window

        dout(1) << "SEARCH_MEMORY: Found " + std::to_string(search_results.size()) + " results:" << std::endl;
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

            oss << "Result " << (i + 1) << " [Relevance: "
                << std::fixed << std::setprecision(2) << sr.relevance_score << "]:\n"
                << content << "\n\n";

            // Log each result in debug
            std::string preview = sr.content.length() > 100 ? sr.content.substr(0, 100) + "..." : sr.content;
            dout(1) << "  [" + std::to_string(i + 1) + "] Score: " + std::to_string(sr.relevance_score) +
                      ", Content preview: " + preview << std::endl;
        }

        result["output"] = oss.str();
        result["content"] = oss.str();
        result["summary"] = std::string("Found ") + std::to_string(search_results.size()) + " memor" + (search_results.size() != 1 ? "ies" : "y");
        result["success"] = true;

        dout(1) << "SEARCH_MEMORY: Returning " + std::to_string(search_results.size()) + " results to model" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error searching memory: " + std::string(e.what()) << std::endl;
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
    std::string user_id = tool_utils::get_string(args, "_user_id");

    std::string response = RAGManager::execute_set_fact_tool(key, value, user_id);

    if (response.find("Error:") == 0) {
        result["error"] = response;
        result["success"] = false;
    } else {
        result["output"] = response;
        result["content"] = response;
        result["summary"] = std::string("Fact stored");
        result["success"] = true;
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
    std::string user_id = tool_utils::get_string(args, "_user_id");

    std::string response = RAGManager::execute_get_fact_tool(key, user_id);

    // "Fact not found" is a successful tool execution, just with no result
    if (response.find("Error:") == 0 && response.find("Fact not found:") != 0) {
        result["error"] = response;
        result["success"] = false;
    } else {
        result["output"] = response;
        result["content"] = response;
        // Build summary: key=value (truncated)
        std::string summary = key + "=" + (response.length() > 40 ? response.substr(0, 37) + "..." : response);
        if (response.find("Fact not found") != std::string::npos) {
            summary = "Fact not found";
        }
        result["summary"] = summary;
        result["success"] = true;
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
    std::string user_id = tool_utils::get_string(args, "_user_id");

    std::string response = RAGManager::execute_clear_fact_tool(key, user_id);

    // "Fact not found" is a successful tool execution, just with no deletion
    if (response.find("Error:") == 0 && response.find("Fact not found:") != 0) {
        result["error"] = response;
        result["success"] = false;
    } else {
        result["output"] = response;
        result["content"] = response;
        result["summary"] = std::string("Fact cleared");
        result["success"] = true;
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

    // Accept "key" as alias for "question"
    std::string question = tool_utils::get_string(args, "question");
    if (question.empty()) {
        question = tool_utils::get_string(args, "key");
    }
    std::string answer = tool_utils::get_string(args, "answer");

    std::string user_id = tool_utils::get_string(args, "_user_id");
    std::string response = RAGManager::execute_store_memory_tool(question, answer, user_id);

    if (response.find("Error:") == 0) {
        result["error"] = response;
        result["success"] = false;
    } else {
        result["output"] = response;
        result["content"] = response;
        result["summary"] = std::string("Memory stored");
        result["success"] = true;
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

    // Accept "key" as alias for "question"
    std::string question = tool_utils::get_string(args, "question");
    if (question.empty()) {
        question = tool_utils::get_string(args, "key");
    }

    std::string user_id = tool_utils::get_string(args, "_user_id");
    std::string response = RAGManager::execute_clear_memory_tool(question, user_id);

    // "Memory not found" is a successful tool execution, just with no deletion
    if (response.find("Error:") == 0 && response.find("Memory not found:") != 0) {
        result["error"] = response;
        result["success"] = false;
    } else {
        result["output"] = response;
        result["content"] = response;
        result["summary"] = std::string("Memory cleared");
        result["success"] = true;
    }

    return result;
}

void register_memory_tools(Tools& tools, bool enable) {
    if (!enable) {
        dout(1) << "Memory tools disabled (use --memtools to enable)" << std::endl;
        return;
    }

    tools.register_tool(std::make_unique<SearchMemoryTool>());
    tools.register_tool(std::make_unique<StoreMemoryTool>());
    tools.register_tool(std::make_unique<ClearMemoryTool>());
    tools.register_tool(std::make_unique<SetFactTool>());
    tools.register_tool(std::make_unique<GetFactTool>());
    tools.register_tool(std::make_unique<ClearFactTool>());
    dout(1) << "Registered memory tools: search_memory, store_memory, clear_memory, set_fact, get_fact, clear_fact" << std::endl;
}
#endif
