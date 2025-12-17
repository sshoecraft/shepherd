#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <string>
#include <functional>
#include "thread_queue.h"
#include "backend.h"
#include "message.h"

// Forward declarations
class Session;

// Event callback type for generation (same signature as Backend::EventCallback)
using EventCallback = std::function<bool(CallbackEvent event,
                                         const std::string& content,
                                         const std::string& name,
                                         const std::string& id)>;

// Request to generate a response
struct GenerationRequest {
    Message::Role role = Message::USER;
    std::string content;
    std::string tool_name;
    std::string tool_id;
    int max_tokens = 0;
    EventCallback callback;  // Event callback for streaming (CONTENT, TOOL_CALL, ERROR, STOP)
};

// GenerationThread - Runs session.add_message() in a separate thread
// This is the ONLY blocking operation - everything else stays in main thread
class GenerationThread {
public:
    GenerationThread();
    ~GenerationThread();

    // Initialize with session pointer (backend accessed via session)
    void init(Session* session);

    // Start/stop the worker thread
    void start();
    void stop();

    // Submit a generation request (non-blocking)
    // Returns immediately; poll is_complete() and get last_response
    void submit(const GenerationRequest& request);

    // Check if generation is complete
    bool is_complete() const { return complete.load(); }

    // Check if currently generating
    bool is_busy() const { return busy.load(); }

    // Get the response (only valid after is_complete() returns true)
    // Caller should copy this before calling reset()
    Response last_response;

    // Reset for next request
    void reset() { complete = false; }

private:
    void worker_loop();

    Session* session;

    std::thread worker;
    std::atomic<bool> running{false};
    std::atomic<bool> busy{false};
    std::atomic<bool> complete{false};

    // Request signaling
    std::mutex request_mutex;
    std::condition_variable request_cv;
    GenerationRequest current_request;
    bool has_request{false};
};

// Global instance (created/destroyed by CLI)
extern GenerationThread* g_generation_thread;
