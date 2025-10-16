#include "grok.h"
#include "../logger.h"
#include <sstream>
#include <algorithm>

// GrokTokenizer implementation
GrokTokenizer::GrokTokenizer(const std::string& model_name)
    : model_name_(model_name) {
    LOG_DEBUG("Grok tokenizer initialized for model: " + model_name);
}

int GrokTokenizer::count_tokens(const std::string& text) {
    // TODO: Integrate tiktoken library for accurate token counting (same as OpenAI)
    // For now, use the same approximation as OpenAI
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> GrokTokenizer::encode(const std::string& text) {
    // TODO: Implement tiktoken encoding (same as OpenAI)
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string GrokTokenizer::decode(const std::vector<int>& tokens) {
    // TODO: Implement tiktoken decoding (same as OpenAI)
    return "TODO: Implement tiktoken decode";
}

std::string GrokTokenizer::get_tokenizer_name() const {
    return "tiktoken-" + model_name_;
}

// GrokBackend implementation
GrokBackend::GrokBackend(size_t max_context_tokens)
    : ApiBackend(max_context_tokens) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    tokenizer_ = std::make_unique<GrokTokenizer>("grok-1"); // Default model
    LOG_DEBUG("GrokBackend created");
}

GrokBackend::~GrokBackend() {
    shutdown();
}

bool GrokBackend::initialize(const std::string& model_name, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_API_BACKENDS
    if (initialized_) {
        LOG_WARN("GrokBackend already initialized");
        return true;
    }

    if (api_key.empty()) {
        LOG_ERROR("Grok API key is required");
        return false;
    }

    model_name_ = model_name.empty() ? "grok-1" : model_name;
    api_key_ = api_key;

    // Update tokenizer with correct model name
    tokenizer_ = std::make_unique<GrokTokenizer>(model_name_);

    // Initialize curl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();

    if (!curl_) {
        LOG_ERROR("Failed to initialize CURL for Grok backend");
        return false;
    }

    // Query the actual context size for this model
    size_t actual_context_size = query_model_context_size(model_name_);
    if (actual_context_size > 0) {
        max_context_size_ = actual_context_size;
        LOG_INFO("Grok model " + model_name_ + " context size: " + std::to_string(actual_context_size));
    } else {
        LOG_WARN("Failed to query context size for " + model_name_ + ", using default: " + std::to_string(max_context_size_));
    }

    // Create the shared context manager with actual model context size
    context_manager_ = std::make_unique<ApiContextManager>(max_context_size_);
    LOG_DEBUG("Created ApiContextManager with " + std::to_string(max_context_size_) + " tokens");

    LOG_INFO("GrokBackend initialized with model: " + model_name_);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("API backends not compiled in");
    return false;
#endif
}

std::string GrokBackend::generate(int max_tokens) {
    if (!is_ready()) {
        throw BackendManagerError("Grok backend not initialized");
    }

    // Get current context for API call
    std::string context_json = context_manager_->get_context_for_inference();
    LOG_DEBUG("Grok generate called with " + std::to_string(context_manager_->get_message_count()) + " messages");

    // TODO: Implement actual Grok API call
    std::string response = "Grok skeleton response";

    // Return response directly - main will add it to context
    return response;
}

std::string GrokBackend::get_backend_name() const {
    return "grok";
}

std::string GrokBackend::get_model_name() const {
    return model_name_;
}

size_t GrokBackend::get_max_context_size() const {
#ifdef ENABLE_API_BACKENDS
    return max_context_size_;
#else
    return 4096;
#endif
}

bool GrokBackend::is_ready() const {
#ifdef ENABLE_API_BACKENDS
    return initialized_ && curl_ && !api_key_.empty();
#else
    return false;
#endif
}

void GrokBackend::shutdown() {
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
    LOG_DEBUG("GrokBackend shutdown complete");
}

std::string GrokBackend::make_api_request(const std::string& json_payload) {
    // TODO: Implement actual HTTP request with CURL
    // Set headers (Authorization: Bearer api_key_, Content-Type: application/json)
    // POST to api_endpoint_
    // Return response body
    return "TODO: Implement HTTP request";
}

std::string GrokBackend::make_get_request(const std::string& endpoint) {
    // TODO: Implement actual HTTP GET request with CURL
    // Set headers (Authorization: Bearer api_key_)
    // GET to https://api.x.ai/v1 + endpoint
    // Return response body
    return "TODO: Implement HTTP GET request";
}

size_t GrokBackend::query_model_context_size(const std::string& model_name) {
#ifdef ENABLE_API_BACKENDS
    if (!is_ready()) {
        LOG_ERROR("Grok backend not ready for model query");
        return 0;
    }

    // TODO: Query Grok API for model info (OpenAI-compatible /models endpoint)
    // For now, return known context sizes based on model name
    if (model_name.find("grok-1") != std::string::npos) {
        return 131072; // Grok-1 (128k tokens)
    } else if (model_name.find("grok-2") != std::string::npos) {
        return 131072; // Grok-2 (128k tokens)
    } else if (model_name.find("grok") != std::string::npos) {
        return 131072; // Default Grok model
    }

    // Default fallback
    LOG_WARN("Unknown Grok model: " + model_name + ", using default context size");
    return 131072;
#else
    return 131072;
#endif
}

std::string GrokBackend::parse_grok_response(const std::string& response_json) {
    // TODO: Parse Grok response format (OpenAI-compatible)
    // Extract choices[0].message.content
    return "TODO: Parse response";
}