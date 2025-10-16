#include "gemini.h"
#include "../logger.h"
#include <sstream>
#include <algorithm>

// GeminiTokenizer implementation
GeminiTokenizer::GeminiTokenizer(const std::string& model_name)
    : model_name_(model_name) {
    LOG_DEBUG("Gemini tokenizer initialized for model: " + model_name);
}

int GeminiTokenizer::count_tokens(const std::string& text) {
    // TODO: Implement SentencePiece tokenization for Gemini
    // For now, use approximation (roughly 4 chars per token)
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> GeminiTokenizer::encode(const std::string& text) {
    // TODO: Implement SentencePiece encoding
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string GeminiTokenizer::decode(const std::vector<int>& tokens) {
    // TODO: Implement SentencePiece decoding
    return "TODO: Implement Gemini decode";
}

std::string GeminiTokenizer::get_tokenizer_name() const {
    return "sentencepiece-" + model_name_;
}

// GeminiBackend implementation
GeminiBackend::GeminiBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    tokenizer_ = std::make_unique<GeminiTokenizer>("gemini-pro"); // Default model
    LOG_DEBUG("GeminiBackend created");
}

GeminiBackend::~GeminiBackend() {
    shutdown();
}

bool GeminiBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("GeminiBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("Gemini API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "gemini-pro" : model_name;
    api_key_ = api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<GeminiTokenizer>(model_name_);

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();

    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for Gemini backend");
        return false;
    }

    // Query the actual context size for this model
    size_t actual_context_size = query_model_context_size(model_name_);
    if (actual_context_size > 0) {
        max_context_size_ = actual_context_size;
        LOG_INFO("Gemini model " + model_name_ + " context size: " + std::to_string(actual_context_size));
    } else {
        LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(max_context_size_));
    }

    // Create the shared context manager with actual model context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens");

    LOG_INFO("GeminiBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string GeminiBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Gemini backend not initialized");
    }

    // Get current context for API call
    std::string context_json = context_manager_->get_context_for_inference();
    LOG_DEBUG("Gemini generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // TODO: Implement actual Gemini API call
    std::string response = "Gemini skeleton response";

    // Return response directly - main will add it to context
    return response;
}

std::string GeminiBackend::get_backend_name() const {
    return "gemini";
}

std::string GeminiBackend::get_model_name() const {
    return model_name_;
}

size_t GeminiBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return max_context_size_;
#else
    return 4096;
#endif
}

bool GeminiBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && curl_ && !api_key_.empty();
#else
    return false;
#endif
}

void GeminiBackend::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef ENABLE_API_BACKENDS
    if (curl_) {
        curl_easy_cleanup(curl_);
        curl_ = nullptr;
    }
    curl_global_cleanup();
#endif

    initialized_ = false;
    LOG_DEBUG("GeminiBackend shutdown complete");
}

std::string GeminiBackend::make_api_request(const std::string& json_payload) {
    // TODO: Implement actual HTTP request with CURL
    // Set headers (x-goog-api-key: api_key_, Content-Type: application/json)
    // POST to api_endpoint_/model_name_:generateContent
    // Return response body
    return "TODO: Implement HTTP request";
}

std::string GeminiBackend::make_get_request(const std::string& endpoint) {
    // TODO: Implement actual HTTP GET request with CURL
    // Set headers (x-goog-api-key: api_key_)
    // GET to https://generativelanguage.googleapis.com/v1beta + endpoint
    // Return response body
    return "TODO: Implement HTTP GET request";
}

size_t GeminiBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    if (!is_ready()) {
        LOG_ERROR("Gemini backend not ready for model query");
        return 0;
    }

    // TODO: Query Gemini API for model info using /models/{model_name}
    // For now, return known context sizes based on model name
    if (model_name.find("gemini-1.5") != std::string::npos) {
        if (model_name.find("pro") != std::string::npos) {
            return 2000000; // Gemini 1.5 Pro (2M tokens)
        } else if (model_name.find("flash") != std::string::npos) {
            return 1000000; // Gemini 1.5 Flash (1M tokens)
        }
    } else if (model_name.find("gemini-pro") != std::string::npos) {
        return 32000; // Gemini Pro (32k tokens)
    } else if (model_name.find("gemini-2") != std::string::npos) {
        return 1000000; // Gemini 2.0 Flash (1M tokens)
    }

    // Default fallback
    LOG_WARN("Unknown Gemini model: " + model_name + ", using default context size");
    return 32000;
#else
    return 32000;
#endif
}

std::string GeminiBackend::parse_gemini_response(const std::string& response_json) {
    // TODO: Parse Gemini response format
    // Extract candidates[0].content.parts[0].text
    return "TODO: Parse response";
}