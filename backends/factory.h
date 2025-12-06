
#pragma once

#include "shepherd.h"
#include "config.h"
#include "backends/backend.h"
#include "provider.h"

/// @brief Factory for creating backend managers
class BackendFactory {
public:
    /// @brief Create backend from provider configuration
    /// @param provider Provider configuration with type, api_key, model, etc.
    /// @param context_size Maximum context window size (0 = auto)
    /// @return Unique pointer to backend instance
    static std::unique_ptr<Backend> create_from_provider(ProviderConfig* provider, size_t context_size);

    /// @brief Create backend manager by backend name (legacy)
    /// @param backend Backend name ("llamacpp", "tensorrt", "openai", "anthropic", "gemini", "ollama", "cli")
    /// @param model_path_or_name Model file path or name
    /// @param max_context_tokens Maximum context window size
    /// @param api_key API key for cloud providers
    /// @return Unique pointer to backend instance
    static std::unique_ptr<Backend> create_backend(std::string &name, size_t context_size);

    /// @brief Get list of available backends
    /// @return Vector of backend names
    static std::vector<std::string> get_available_backends();

    /// @brief Check if backend is available (compiled in)
    /// @param backend Backend name to check
    /// @return True if available
    static bool is_backend_available(const std::string& backend);
};
