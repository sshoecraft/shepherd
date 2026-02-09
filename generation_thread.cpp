#include "generation_thread.h"
#include "session.h"
#include "frontend.h"
#include "shepherd.h"


// Global instance
GenerationThread* g_generation_thread = nullptr;

GenerationThread::GenerationThread()
    : frontend(nullptr) {
}

GenerationThread::~GenerationThread() {
    stop();
}

void GenerationThread::init(Frontend* f) {
    frontend = f;
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
            // Add message to session and generate response using unified Frontend path
            frontend->add_message_to_session(
                current_request.role,
                current_request.content,
                current_request.tool_name,
                current_request.tool_id
            );
            if (current_request.role == Message::USER) {
                frontend->enrich_with_rag_context(frontend->session);
            }
            frontend->generate_response(current_request.max_tokens);

            // Queue memory extraction after generation completes
            frontend->queue_memory_extraction();

            dout(1) << "GenerationThread: generation complete" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "GenerationThread exception: " + std::string(e.what()) << std::endl;
        }

        busy = false;
        complete = true;
    }

    dout(1) << "GenerationThread worker_loop ended" << std::endl;
}
