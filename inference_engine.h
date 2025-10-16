#pragma once

#include <string>
#include <vector>
#include <memory>

#ifdef USE_TENSORRT_LLM
#include <tensorrt_llm/runtime/session.h>
#elif defined(USE_LLAMA_CPP)
#include <llama.h>
#endif

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    bool load_model(const std::string& model_path);
    std::string generate(const std::string& prompt, int max_tokens = 512);
    std::string get_backend_name() const;
    size_t get_context_size() const;

private:
    void initialize_backend();
    void cleanup();

#ifdef USE_TENSORRT_LLM
    // TensorRT-LLM specific members
    std::unique_ptr<tensorrt_llm::runtime::Session> session_;
#elif defined(USE_LLAMA_CPP)
    // llama.cpp specific members
    llama_model* model_;
    llama_context* context_;
    llama_context_params ctx_params_;
#endif

    bool initialized_;
    size_t context_size_;
    std::string backend_name_;
};