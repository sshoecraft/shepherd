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
    dout(1) << "GenerationThread started" << std::endl;
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

    dout(1) << "GenerationThread stopped" << std::endl;
}

void GenerationThread::submit(const GenerationRequest& request) {
    {
        std::lock_guard<std::mutex> lock(request_mutex);
        current_request = request;
        has_request = true;
        complete = false;
    }
    request_cv.notify_one();
    dout(1) << "GenerationThread: request submitted" << std::endl;
}

void GenerationThread::worker_loop() {
    dout(1) << "GenerationThread worker_loop started" << std::endl;

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
        dout(1) << "GenerationThread: starting generation, role=" + std::to_string(static_cast<int>(current_request.role)) << std::endl;

        try {
            // Call session->add_message - all output flows through backend callback
            // (CONTENT, TOOL_CALL, ERROR, STOP events)
            session->add_message(
                current_request.role,
                current_request.content,
                current_request.tool_name,
                current_request.tool_id,
                current_request.max_tokens
            );

            dout(1) << "GenerationThread: generation complete" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "GenerationThread exception: " + std::string(e.what()) << std::endl;
        }

        busy = false;
        complete = true;
    }

    dout(1) << "GenerationThread worker_loop ended" << std::endl;
}
