
#include "backends/factory.h"
#include "backends/llamacpp.h"
#ifdef ENABLE_TENSORRT
#include "backends/tensorrt.h"
#endif
#include "backends/openai.h"
#include "backends/ollama.h"
#include "backends/anthropic.h"
#include "backends/gemini.h"

std::unique_ptr<Backend> BackendFactory::create_from_provider(ProviderConfig* provider, size_t context_size) {
    LOG_DEBUG("Creating backend from provider: " + provider->name + " (type: " + provider->type + ")");

    // Print loading message - but only for the initial process (not mpirun children)
    // This ensures "Loading provider" prints exactly once, even for TensorRT multi-GPU
    if (!getenv("OMPI_COMM_WORLD_SIZE")) {
        std::cerr << "Loading provider: " << provider->name << std::endl;
    }

    // Set config globals from provider (for now, until backends are fully refactored)
    extern std::unique_ptr<Config> config;
    config->model = provider->model;

    // Type-specific setup
    if (auto* api = dynamic_cast<ApiProviderConfig*>(provider)) {
        config->key = api->api_key;
        config->api_base = api->base_url;

        // Set API-specific parameters in JSON
        config->json["temperature"] = api->temperature;
        config->json["top_p"] = api->top_p;
        if (api->top_k > 0) config->json["top_k"] = api->top_k;
        if (api->frequency_penalty != 0.0f) config->json["frequency_penalty"] = api->frequency_penalty;
        if (api->presence_penalty != 0.0f) config->json["presence_penalty"] = api->presence_penalty;
        if (api->max_tokens > 0) config->json["max_tokens"] = api->max_tokens;
    }
    else if (auto* llama = dynamic_cast<LlamaProviderConfig*>(provider)) {
        config->model_path = llama->model_path;

        // Set llama-specific parameters in JSON for parse_backend_config()
        config->json["tp"] = llama->tp;
        config->json["pp"] = llama->pp;
        config->json["gpu_layers"] = llama->gpu_layers;
        if (llama->context_size > 0) config->json["context_size"] = llama->context_size;
        config->json["temperature"] = llama->temperature;
        config->json["top_p"] = llama->top_p;
        config->json["top_k"] = llama->top_k;
        config->json["repeat_penalty"] = llama->repeat_penalty;
        config->json["n_batch"] = llama->n_batch;
        if (llama->n_threads > 0) config->json["n_threads"] = llama->n_threads;
    }
    else if (auto* tensorrt = dynamic_cast<TensorRTProviderConfig*>(provider)) {
        config->model_path = tensorrt->model_path;

        // Set tensorrt-specific parameters
        config->json["tp"] = tensorrt->tp;
        config->json["pp"] = tensorrt->pp;
        config->json["gpu_id"] = tensorrt->gpu_id;
        if (tensorrt->context_size > 0) config->json["context_size"] = tensorrt->context_size;
        config->json["temperature"] = tensorrt->temperature;
        config->json["top_p"] = tensorrt->top_p;
        config->json["top_k"] = tensorrt->top_k;
        config->json["repeat_penalty"] = tensorrt->repeat_penalty;
        config->json["frequency_penalty"] = tensorrt->frequency_penalty;
        config->json["presence_penalty"] = tensorrt->presence_penalty;
    }
    else if (auto* ollama = dynamic_cast<OllamaProviderConfig*>(provider)) {
        config->api_base = ollama->base_url;

        // Set ollama-specific parameters
        config->json["temperature"] = ollama->temperature;
        config->json["top_p"] = ollama->top_p;
        config->json["top_k"] = ollama->top_k;
        config->json["repeat_penalty"] = ollama->repeat_penalty;
        if (ollama->num_ctx > 0) config->json["num_ctx"] = ollama->num_ctx;
        if (ollama->num_predict != -1) config->json["num_predict"] = ollama->num_predict;
    }

    // Create backend by type
    std::string type = provider->type;
    auto backend = create_backend(type, context_size);

    return backend;
}

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
    else if (name == "anthropic") {
#ifdef ENABLE_API_BACKENDS
        backend = std::make_unique<AnthropicBackend>(context_size);
#else
        throw std::runtime_error("Anthropic backend not available (API backends not compiled in)");
#endif
    }
    else if (name == "gemini") {
#ifdef ENABLE_API_BACKENDS
        backend = std::make_unique<GeminiBackend>(context_size);
#else
        throw std::runtime_error("Gemini backend not available (API backends not compiled in)");
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
