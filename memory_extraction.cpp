#include "memory_extraction.h"
#include "shepherd.h"
#include "config.h"
#include "rag.h"
#include "nlohmann/json.hpp"

#ifdef ENABLE_API_BACKENDS
#include "http_client.h"
#endif

#include <chrono>
#include <sstream>
#include <regex>

extern std::unique_ptr<Config> config;

static const char* EXTRACTION_SYSTEM_PROMPT = R"(You are a memory extraction system. Your job is to read conversation segments and extract two types of information:

1. **Facts**: Durable key-value pairs about the user that will still be true weeks or months from now. Examples: name, location, preferences, job title, project names, technical choices.
2. **Context**: Question/answer pairs capturing useful conversation details that don't fit as simple facts. Examples: decisions made, procedures discussed, problems solved.

Rules:
- Extract only concrete information explicitly stated or confirmed BY THE USER
- Do NOT extract facts about the assistant itself -- its name, identity, capabilities, or behavior are not user-provided information
- NEVER extract something where the value is unknown, not provided, or not mentioned -- the absence of information is not worth storing
- Only store DURABLE information -- not transient data like current readings, prices, or timestamps
- Do NOT store the results of tool calls or API responses
- Ignore small talk, greetings, and filler
- Ignore dead ends and corrections -- only keep final conclusions
- Keep everything concise

Output format: JSON object with two fields:
- "facts": object of key-value pairs (use lowercase_snake_case keys)
- "context": array of {"q": "...", "a": "..."} objects

Example:
{"facts": {"name": "Steve", "location": "Texas"}, "context": [{"q": "What IDE does the user prefer?", "a": "VS Code with vim keybindings"}]}

If nothing to extract: {"facts": {}, "context": []})";

MemoryExtractionThread::MemoryExtractionThread() {
}

MemoryExtractionThread::~MemoryExtractionThread() {
    stop();
}

void MemoryExtractionThread::start() {
    if (running) return;
    running = true;
    worker = std::thread(&MemoryExtractionThread::worker_loop, this);
    dout(1) << "MemoryExtractionThread started" << std::endl;
}

void MemoryExtractionThread::stop() {
    if (!running) return;
    running = false;
    if (worker.joinable()) {
        worker.join();
    }
    dout(1) << "MemoryExtractionThread stopped" << std::endl;
}

void MemoryExtractionThread::queue(ExtractionWorkItem item) {
    // Enforce queue limit if configured
    if (config && config->memory_extraction_queue_limit > 0) {
        while (work_queue.size() >= static_cast<size_t>(config->memory_extraction_queue_limit)) {
            work_queue.pop();  // Drop oldest
        }
    }
    work_queue.push(std::move(item));
}

void MemoryExtractionThread::flush() {
    // Wait for queue to drain (for shutdown)
    int wait_count = 0;
    while (!work_queue.empty() && wait_count < 300) {  // Max 30 seconds
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
}

void MemoryExtractionThread::worker_loop() {
    dout(1) << "MemoryExtractionThread worker_loop started" << std::endl;

    while (running) {
        auto item = work_queue.wait_for_and_pop(std::chrono::seconds(1));
        if (!item.has_value()) continue;
        if (!running) break;

        try {
            // Build conversation text from messages
            std::string conversation_text = build_conversation_text(item->messages, item->start_index);
            if (conversation_text.empty()) {
                dout(1) << "MemoryExtraction: no content to extract from" << std::endl;
                continue;
            }

            dout(1) << "MemoryExtraction: processing " << item->messages.size() - item->start_index
                     << " messages for session " << item->session_id << std::endl;

            // Call extraction API with retry
            std::string response;
            while (running) {
                response = call_extraction_api(conversation_text, item->user_id);
                if (!response.empty()) break;

                int interval = config ? config->memory_extraction_retry_interval : 5;
                dout(1) << "MemoryExtraction: API call failed, retrying in "
                         << interval << "s" << std::endl;

                // Sleep in 1-second increments for responsive shutdown
                for (int i = 0; i < interval && running; i++) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            if (!running) break;

            // Parse and store extracted facts
            parse_and_store_facts(response, item->user_id);

            // Update high-water mark
            last_processed_index = static_cast<int>(item->messages.size()) - 1;

        } catch (const std::exception& e) {
            dout(1) << "MemoryExtraction error: " + std::string(e.what()) << std::endl;
        } catch (...) {
            dout(1) << "MemoryExtraction: unknown error" << std::endl;
        }
    }

    dout(1) << "MemoryExtractionThread worker_loop ended" << std::endl;
}

std::string MemoryExtractionThread::build_conversation_text(const std::vector<Message>& messages, int start_index) {
    std::ostringstream oss;
    int turn_count = 0;
    int max_turns = config ? config->memory_extraction_max_turns : 20;

    for (size_t i = static_cast<size_t>(std::max(0, start_index)); i < messages.size(); i++) {
        const Message& msg = messages[i];

        // Only include USER and ASSISTANT messages
        if (msg.role != Message::USER && msg.role != Message::ASSISTANT) {
            continue;
        }

        std::string content = msg.content;

        // Strip injected [facts: ...] and [context: ...] postfix from user messages
        if (msg.role == Message::USER) {
            size_t facts_pos = content.find("\n\n[facts: ");
            size_t ctx_pos = content.find("\n\n[context: ");
            size_t strip_pos = std::string::npos;
            if (facts_pos != std::string::npos && ctx_pos != std::string::npos) {
                strip_pos = std::min(facts_pos, ctx_pos);
            } else if (facts_pos != std::string::npos) {
                strip_pos = facts_pos;
            } else if (ctx_pos != std::string::npos) {
                strip_pos = ctx_pos;
            }
            if (strip_pos != std::string::npos) {
                content = content.substr(0, strip_pos);
            }
        }

        if (content.empty()) continue;

        if (msg.role == Message::USER) {
            oss << "User: " << content << "\n";
        } else {
            oss << "Assistant: " << content << "\n";
        }

        turn_count++;
        if (turn_count >= max_turns) break;
    }

    return oss.str();
}

std::string MemoryExtractionThread::call_extraction_api(const std::string& conversation_text, const std::string& user_id) {
#ifdef ENABLE_API_BACKENDS
    if (!config) return "";

    HttpClient client;
    client.set_timeout(60);

    std::map<std::string, std::string> headers;
    headers["Content-Type"] = "application/json";
    if (!config->memory_extraction_api_key.empty()) {
        headers["Authorization"] = "Bearer " + config->memory_extraction_api_key;
    }

    nlohmann::json request_body = {
        {"model", config->memory_extraction_model},
        {"messages", nlohmann::json::array({
            {{"role", "system"}, {"content", EXTRACTION_SYSTEM_PROMPT}},
            {{"role", "user"}, {"content", "Extract facts from this conversation:\n\n" + conversation_text}}
        })},
        {"max_tokens", config->memory_extraction_max_tokens},
        {"temperature", config->memory_extraction_temperature},
        {"stream", false}
    };

    std::string url = config->memory_extraction_endpoint;
    // Ensure endpoint ends with /v1/chat/completions
    if (url.back() == '/') url.pop_back();
    if (url.find("/v1/chat/completions") == std::string::npos) {
        if (url.find("/v1") == std::string::npos) {
            url += "/v1";
        }
        url += "/chat/completions";
    }

    dout(1) << "MemoryExtraction: calling " << url << " with model " << config->memory_extraction_model << std::endl;

    HttpResponse response = client.post(url, request_body.dump(), headers);

    if (!response.is_success()) {
        dout(1) << "MemoryExtraction: API call failed with status " << response.status_code
                << ": " << response.error_message << std::endl;
        return "";
    }

    // Parse OpenAI-format response
    try {
        auto json_response = nlohmann::json::parse(response.body);
        if (json_response.contains("choices") && !json_response["choices"].empty()) {
            std::string content = json_response["choices"][0]["message"]["content"].get<std::string>();
            dout(1) << "MemoryExtraction: received response (" << content.length() << " chars)" << std::endl;
            return content;
        }
    } catch (const std::exception& e) {
        dout(1) << "MemoryExtraction: failed to parse API response: " + std::string(e.what()) << std::endl;
    }

    return "";
#else
    dout(1) << "MemoryExtraction: API backends not enabled at compile time" << std::endl;
    return "";
#endif
}

void MemoryExtractionThread::parse_and_store_facts(const std::string& response, const std::string& user_id) {
    // Trim whitespace
    std::string trimmed = response;
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return;
    trimmed = trimmed.substr(start);

    if (!RAGManager::is_initialized()) {
        dout(1) << "MemoryExtraction: RAGManager not initialized, skipping storage" << std::endl;
        return;
    }

    // Extract JSON from response (handle markdown code blocks)
    std::string json_str = trimmed;
    if (json_str.find("```") != std::string::npos) {
        size_t json_start = json_str.find('{');
        size_t json_end = json_str.rfind('}');
        if (json_start != std::string::npos && json_end != std::string::npos && json_end > json_start) {
            json_str = json_str.substr(json_start, json_end - json_start + 1);
        }
    }

    // Parse JSON response
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(json_str);
    } catch (const std::exception& e) {
        dout(1) << "MemoryExtraction: failed to parse JSON response: " << e.what() << std::endl;
        dout(1) << "MemoryExtraction: raw response: " << response.substr(0, 200) << std::endl;
        return;
    }

    int fact_count = 0;
    int context_count = 0;

    // Process facts: key-value pairs → set_fact (INSERT OR REPLACE)
    if (parsed.contains("facts") && parsed["facts"].is_object()) {
        for (auto& [key, value] : parsed["facts"].items()) {
            if (value.is_string()) {
                std::string val = value.get<std::string>();
                if (!val.empty()) {
                    RAGManager::set_fact(key, val, user_id);
                    fact_count++;
                    dout(1) << "MemoryExtraction: stored fact: " << key << " = " << val << std::endl;
                }
            }
        }
    }

    // Process context: Q/A pairs → store_memory (into context table)
    if (parsed.contains("context") && parsed["context"].is_array()) {
        for (const auto& item : parsed["context"]) {
            if (item.contains("q") && item.contains("a") &&
                item["q"].is_string() && item["a"].is_string()) {
                std::string q = item["q"].get<std::string>();
                std::string a = item["a"].get<std::string>();
                if (!q.empty() && !a.empty()) {
                    RAGManager::store_memory(q, a, user_id);
                    context_count++;
                    dout(1) << "MemoryExtraction: stored context: Q=" << q.substr(0, 50) << std::endl;
                }
            }
        }
    }

    dout(1) << "MemoryExtraction: stored " << fact_count << " facts, "
            << context_count << " context entries" << std::endl;
}
