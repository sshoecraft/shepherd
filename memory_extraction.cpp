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

static const char* EXTRACTION_SYSTEM_PROMPT = R"(You are a memory extraction system. Your job is to read conversation segments and extract useful facts, decisions, preferences, and technical details that would be valuable to recall in future conversations.

Rules:
- Extract only concrete, factual information
- Ignore small talk, greetings, thank-yous, and filler
- Ignore dead ends, corrections, and wrong answers -- only keep final conclusions
- Each extracted fact should be a self-contained question/answer pair
- Keep facts concise -- one concept per entry
- If the conversation contains no useful extractable information, respond with NONE

Output format (one per line):
Q: <natural language question that this fact answers>
A: <concise factual answer>)";

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
            // Set user_id for multi-tenant storage
            RAGManager::set_current_user_id(item->user_id);

            // Build conversation text from messages
            std::string conversation_text = build_conversation_text(item->messages, item->start_index);
            if (conversation_text.empty()) {
                dout(1) << "MemoryExtraction: no content to extract from" << std::endl;
                continue;
            }

            dout(1) << "MemoryExtraction: processing " << item->messages.size() - item->start_index
                     << " messages for session " << item->session_id << std::endl;

            // Call extraction API
            std::string response = call_extraction_api(conversation_text);
            if (response.empty()) {
                continue;  // Error already logged
            }

            // Parse and store extracted facts
            parse_and_store_facts(response);

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

        // Strip [context: ...] postfix from user messages
        if (msg.role == Message::USER) {
            size_t ctx_pos = content.find("\n\n[context: ");
            if (ctx_pos != std::string::npos) {
                content = content.substr(0, ctx_pos);
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

std::string MemoryExtractionThread::call_extraction_api(const std::string& conversation_text) {
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

void MemoryExtractionThread::parse_and_store_facts(const std::string& response) {
    // Check for NONE response
    std::string trimmed = response;
    // Trim whitespace
    size_t start = trimmed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return;
    trimmed = trimmed.substr(start);

    if (trimmed == "NONE" || trimmed == "NONE\n") {
        dout(1) << "MemoryExtraction: no facts to extract (NONE)" << std::endl;
        return;
    }

    if (!RAGManager::is_initialized()) {
        dout(1) << "MemoryExtraction: RAGManager not initialized, skipping storage" << std::endl;
        return;
    }

    // Parse Q/A pairs line by line
    std::istringstream iss(response);
    std::string line;
    std::string current_question;
    int stored_count = 0;

    while (std::getline(iss, line)) {
        // Trim leading whitespace
        size_t line_start = line.find_first_not_of(" \t");
        if (line_start == std::string::npos) continue;
        line = line.substr(line_start);

        if (line.substr(0, 3) == "Q: ") {
            current_question = line.substr(3);
        } else if (line.substr(0, 3) == "A: " && !current_question.empty()) {
            std::string answer = line.substr(3);
            if (!answer.empty()) {
                RAGManager::store_memory(current_question, answer);
                stored_count++;
                dout(1) << "MemoryExtraction: stored fact: Q=" << current_question.substr(0, 50) << std::endl;
            }
            current_question.clear();
        }
    }

    dout(1) << "MemoryExtraction: stored " << stored_count << " facts" << std::endl;
}
