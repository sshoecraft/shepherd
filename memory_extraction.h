#pragma once

#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include "thread_queue.h"
#include "message.h"

struct ExtractionWorkItem {
    std::vector<Message> messages;   // Deep copy of session messages
    std::string session_id;          // Session identifier
    std::string user_id;             // User identifier for multi-tenant isolation
    int64_t timestamp;               // When queued
    int start_index;                 // First message index to process
};

class MemoryExtractionThread {
public:
    MemoryExtractionThread();
    ~MemoryExtractionThread();

    void start();
    void stop();
    void queue(ExtractionWorkItem item);
    void flush();

    bool is_running() const { return running.load(); }
    size_t pending_count() const { return work_queue.size(); }

    ThreadQueue<ExtractionWorkItem> work_queue;
    std::thread worker;
    std::atomic<bool> running{false};
    int last_processed_index{-1};

private:
    void worker_loop();
    std::string build_conversation_text(const std::vector<Message>& messages, int start_index);
    std::string call_extraction_api(const std::string& conversation_text);
    void parse_and_store_facts(const std::string& response);
};
