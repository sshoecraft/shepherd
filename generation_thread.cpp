#include "generation_thread.h"
#include "session.h"
#include "shepherd.h"

// Global instance
GenerationThread* g_generation_thread = nullptr;

GenerationThread::GenerationThread()
    : session(nullptr) {
}

GenerationThread::~GenerationThread() {
    stop();
}

void GenerationThread::init(Session* s) {
    session = s;
}

void GenerationThread::start() {
    if (running) return;

    running = true;
    worker = std::thread(&GenerationThread::worker_loop, this);
    LOG_DEBUG("GenerationThread started");
}

void GenerationThread::stop() {
    if (!running) return;

    running = false;

    // Wake up worker if waiting
    {
        std::lock_guard<std::mutex> lock(request_mutex);
        has_request = true;
    }
    request_cv.notify_one();

    if (worker.joinable()) {
        worker.join();
    }

    LOG_DEBUG("GenerationThread stopped");
}

void GenerationThread::submit(const GenerationRequest& request) {
    {
        std::lock_guard<std::mutex> lock(request_mutex);
        current_request = request;
        has_request = true;
        complete = false;
    }
    request_cv.notify_one();
    LOG_DEBUG("GenerationThread: request submitted");
}

void GenerationThread::worker_loop() {
    LOG_DEBUG("GenerationThread worker_loop started");

    while (running) {
        // Wait for a request
        {
            std::unique_lock<std::mutex> lock(request_mutex);
            request_cv.wait(lock, [this] { return has_request || !running; });

            if (!running) break;
            if (!has_request) continue;

            has_request = false;
        }

        // Process the request
        busy = true;
        LOG_DEBUG("GenerationThread: starting generation, type=" + std::to_string(static_cast<int>(current_request.type)));

        try {
            // This is the blocking call - session.add_message()
            // Tokens stream to g_output_queue via backend callbacks
            last_response = session->add_message(
                current_request.type,
                current_request.content,
                current_request.tool_name,
                current_request.tool_id,
                current_request.prompt_tokens,
                current_request.max_tokens
            );

            LOG_DEBUG("GenerationThread: generation complete, success=" +
                      std::string(last_response.success ? "true" : "false") +
                      ", finish_reason=" + last_response.finish_reason);

        } catch (const std::exception& e) {
            last_response = Response{};
            last_response.success = false;
            last_response.error = e.what();
            LOG_ERROR("GenerationThread exception: " + std::string(e.what()));
        }

        busy = false;
        complete = true;
    }

    LOG_DEBUG("GenerationThread worker_loop ended");
}
