#include "backend_manager.h"
#include "backends/llamacpp.h"
#include "backends/tensorrt.h"
#include "backends/openai.h"
#include "backends/anthropic.h"
#include "backends/gemini.h"
#include "backends/grok.h"
#include "backends/ollama.h"
#include "logger.h"

BackendManager::BackendManager(size_t max_context_tokens)
    : max_context_size_(max_context_tokens) {
    LOG_DEBUG("BackendManager base constructor with " + std::to_string(max_context_tokens) + " max tokens");
}

std::unique_ptr<BackendManager> BackendFactory::create_backend(
    const std::string& backend,
    const std::string& model_path_or_name,
    size_t max_context_tokens,
    const std::string& api_key) {

    LOG_DEBUG("Creating backend: " + backend + " with " + std::to_string(max_context_tokens) + " max tokens");

    std::unique_ptr<BackendManager> backend_manager;

    if (backend == "llamacpp") {
#ifdef ENABLE_LLAMACPP
        backend_manager = std::make_unique<LlamaCppBackend>(max_context_tokens);
#else
        throw BackendManagerError("LlamaCpp backend not available (not compiled in)");
#endif
    }
    else if (backend == "tensorrt") {
#ifdef ENABLE_TENSORRT
        backend_manager = std::make_unique<TensorRTBackend>(max_context_tokens);
#else
        throw BackendManagerError("TensorRT backend not available (not compiled in or not on Linux)");
#endif
    }
    else if (backend == "openai") {
#ifdef ENABLE_API_BACKENDS
        backend_manager = std::make_unique<OpenAIBackend>(max_context_tokens);
#else
        throw BackendManagerError("OpenAI backend not available (API backends not compiled in)");
#endif
    }
    else if (backend == "anthropic") {
#ifdef ENABLE_API_BACKENDS
        backend_manager = std::make_unique<AnthropicBackend>(max_context_tokens);
#else
        throw BackendManagerError("Anthropic backend not available (API backends not compiled in)");
#endif
    }
    else if (backend == "gemini") {
#ifdef ENABLE_API_BACKENDS
        backend_manager = std::make_unique<GeminiBackend>(max_context_tokens);
#else
        throw BackendManagerError("Gemini backend not available (API backends not compiled in)");
#endif
    }
    else if (backend == "grok") {
#ifdef ENABLE_API_BACKENDS
        backend_manager = std::make_unique<GrokBackend>(max_context_tokens);
#else
        throw BackendManagerError("Grok backend not available (API backends not compiled in)");
#endif
    }
    else if (backend == "ollama") {
#ifdef ENABLE_API_BACKENDS
        backend_manager = std::make_unique<OllamaBackend>(max_context_tokens);
#else
        throw BackendManagerError("Ollama backend not available (API backends not compiled in)");
#endif
    }
    else {
        throw BackendManagerError("Unknown backend: " + backend);
    }

    LOG_INFO("Successfully created " + backend + " backend");
    return backend_manager;
}

std::vector<std::string> BackendFactory::get_available_backends() {
    std::vector<std::string> backends;

#ifdef ENABLE_LLAMACPP
    backends.push_back("llamacpp");
#endif

#ifdef ENABLE_TENSORRT
    backends.push_back("tensorrt");
#endif

#ifdef ENABLE_API_BACKENDS
    backends.push_back("openai");
    backends.push_back("anthropic");
    backends.push_back("gemini");
    backends.push_back("grok");
    backends.push_back("ollama");
#endif

    return backends;
}

bool BackendFactory::is_backend_available(const std::string& backend) {
    auto available = get_available_backends();
    return std::find(available.begin(), available.end(), backend) != available.end();
}

void BackendManager::update_token_counts_from_api(int prompt_tokens, int completion_tokens, int estimated_prompt_tokens) {
    // Update user message with actual prompt token count if different
    if (prompt_tokens != estimated_prompt_tokens) {
        auto& messages = context_manager_->get_messages();
        if (!messages.empty() && messages.back().type == Message::USER) {
            messages.back().token_count = prompt_tokens;
            // Recalculate total token count since we changed a message
            context_manager_->recalculate_total_tokens();
        }
    }

    // Note: completion tokens are handled when creating the assistant message
    // This method just handles updating the prompt token count
}