
#include "backends/factory.h"
#include "backends/openai.h"
#include "backends/ollama.h"

#include "backends/llamacpp.h"
#if 0
#include "backends/tensorrt.h"
#include "backends/anthropic.h"
#include "backends/gemini.h"
#include "backends/grok.h"
#endif

std::unique_ptr<Backend> BackendFactory::create_backend(std::string &name, size_t context_size) {

    LOG_DEBUG("Creating backend: " + name + " with " + std::to_string(context_size) + " context_size");

    std::unique_ptr<Backend> backend;

    if (name == "llamacpp") {
#ifdef ENABLE_LLAMACPP
        backend = std::make_unique<LlamaCppBackend>(context_size);
#else
        throw BackendError("LlamaCpp backend not available (not compiled in)");
#endif
    }
    else if (name == "tensorrt") {
#ifdef ENABLE_TENSORRT
        backend = std::make_unique<TensorRTBackend>(context_size);
#else
        throw BackendError("TensorRT backend not available (not compiled in or not on Linux)");
#endif
    }
    else if (name == "openai") {
#ifdef ENABLE_API_BACKENDS
        backend = std::make_unique<OpenAIBackend>(context_size);
#else
        throw std::runtime_error("OpenAI backend not available (API backends not compiled in)");
#endif
    }
    else if (name == "ollama") {
#ifdef ENABLE_API_BACKENDS
        backend = std::make_unique<OllamaBackend>(context_size);
#else
        throw std::runtime_error("Ollama backend not available (API backends not compiled in)");
#endif
    }
    else {
        throw std::runtime_error("Unknown backend: " + name);
    }

    LOG_INFO("Successfully created " + name + " backend");
    return backend;
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

#if 0
bool BackendFactory::is_backend_available(const std::string& backend) {
    auto available = get_available_backends();
    return std::find(available.begin(), available.end(), backend) != available.end();
}
#endif
