
#pragma once

#include "shepherd.h"
#include "config.h"
#include "backends/backend.h"

/// @brief Factory for creating backend instances
class BackendFactory {
public:
    /// @brief Create backend by type name
    /// @param type Backend type ("llamacpp", "tensorrt", "openai", "anthropic", "gemini", "ollama", "cli")
    /// @param context_size Maximum context window size (0 = auto)
    /// @return Unique pointer to backend instance (not initialized - call initialize() after)
    static std::unique_ptr<Backend> create_backend(std::string &type, size_t context_size);

    /// @brief Get list of available backends (compiled in)
    /// @return Vector of backend type names
    static std::vector<std::string> get_available_backends();
};
