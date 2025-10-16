#include "inference_engine.h"
#include <iostream>
#include <stdexcept>

InferenceEngine::InferenceEngine()
    : initialized_(false), context_size_(4096) {
#ifdef USE_TENSORRT_LLM
    backend_name_ = "TensorRT-LLM";
#elif defined(USE_LLAMA_CPP)
    backend_name_ = "llama.cpp";
    model_ = nullptr;
    context_ = nullptr;
#endif
    initialize_backend();
}

InferenceEngine::~InferenceEngine() {
    cleanup();
}

void InferenceEngine::initialize_backend() {
#ifdef USE_TENSORRT_LLM
    // TensorRT-LLM initialization
    // TODO: Implement TensorRT-LLM setup
    std::cout << "Initializing TensorRT-LLM backend..." << std::endl;

#elif defined(USE_LLAMA_CPP)
    // llama.cpp initialization
    std::cout << "Initializing llama.cpp backend..." << std::endl;
    llama_backend_init();

    // Set default context parameters
    ctx_params_ = llama_context_default_params();
    ctx_params_.n_ctx = context_size_;
    ctx_params_.n_batch = 512;
    ctx_params_.n_threads = -1; // Use all available threads

#endif
    initialized_ = true;
}

bool InferenceEngine::load_model(const std::string& model_path) {
    if (!initialized_) {
        std::cerr << "Backend not initialized" << std::endl;
        return false;
    }

#ifdef USE_TENSORRT_LLM
    // TensorRT-LLM model loading
    // TODO: Implement TensorRT-LLM model loading
    std::cout << "Loading model with TensorRT-LLM: " << model_path << std::endl;
    return false; // Not implemented yet

#elif defined(USE_LLAMA_CPP)
    // llama.cpp model loading
    std::cout << "Loading model with llama.cpp: " << model_path << std::endl;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1; // Use all GPU layers if available

    model_ = llama_load_model_from_file(model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load model" << std::endl;
        return false;
    }

    context_ = llama_new_context_with_model(model_, ctx_params_);
    if (!context_) {
        std::cerr << "Failed to create context" << std::endl;
        llama_free_model(model_);
        model_ = nullptr;
        return false;
    }

    context_size_ = llama_n_ctx(context_);
    std::cout << "Model loaded successfully, context size: " << context_size_ << std::endl;
    return true;

#endif
    return false;
}

std::string InferenceEngine::generate(const std::string& prompt, int max_tokens) {
#ifdef USE_TENSORRT_LLM
    // TensorRT-LLM generation
    // TODO: Implement TensorRT-LLM generation
    return "TensorRT-LLM generation not implemented yet";

#elif defined(USE_LLAMA_CPP)
    // llama.cpp generation
    if (!model_ || !context_) {
        return "Error: Model not loaded";
    }

    // Tokenize prompt
    std::vector<llama_token> tokens;
    tokens.resize(prompt.length() + 1);
    int n_tokens = llama_tokenize(model_, prompt.c_str(), prompt.length(),
                                  tokens.data(), tokens.size(), true, true);
    tokens.resize(n_tokens);

    if (n_tokens < 0) {
        return "Error: Failed to tokenize prompt";
    }

    // Create batch
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    // Process prompt
    if (llama_decode(context_, batch) != 0) {
        llama_batch_free(batch);
        return "Error: Failed to process prompt";
    }

    // Generate response
    std::string response;
    for (int i = 0; i < max_tokens; ++i) {
        auto logits = llama_get_logits_ith(context_, batch.n_tokens - 1);
        auto n_vocab = llama_n_vocab(model_);

        // Simple sampling (greedy)
        llama_token next_token = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; ++j) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_token = j;
            }
        }

        // Check for EOS
        if (llama_token_is_eog(model_, next_token)) {
            break;
        }

        // Convert token to text
        char buf[256];
        int len = llama_token_to_piece(model_, next_token, buf, sizeof(buf), false);
        if (len > 0) {
            response.append(buf, len);
        }

        // Prepare next iteration
        llama_batch_clear(batch);
        llama_batch_add(batch, next_token, tokens.size() + i, {0}, true);

        if (llama_decode(context_, batch) != 0) {
            break;
        }
    }

    llama_batch_free(batch);
    return response;

#endif
    return "Unknown backend";
}

std::string InferenceEngine::get_backend_name() const {
    return backend_name_;
}

size_t InferenceEngine::get_context_size() const {
    return context_size_;
}

void InferenceEngine::cleanup() {
#ifdef USE_TENSORRT_LLM
    // TensorRT-LLM cleanup
    session_.reset();

#elif defined(USE_LLAMA_CPP)
    // llama.cpp cleanup
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
    if (model_) {
        llama_free_model(model_);
        model_ = nullptr;
    }
    if (initialized_) {
        llama_backend_free();
    }
#endif
    initialized_ = false;
}