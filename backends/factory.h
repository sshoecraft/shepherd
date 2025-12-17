
#pragma once

#include "shepherd.h"
#include "config.h"
#include "backend.h"

/// @brief Factory for creating backend instances
class BackendFactory {
public:
    /// @brief Create backend by type name
    /// @param type Backend type ("llamacpp", "tensorrt", "openai", "anthropic", "gemini", "ollama", "cli")
    /// @param context_size Maximum context window size (0 = auto)
    /// @param session Session for initialization
    /// @param callback Frontend callback for streaming output
    /// @return Unique pointer to backend instance
    static std::unique_ptr<Backend> create_backend(std::string &type, size_t context_size, Session& session, Backend::EventCallback callback);

    /// @brief Get list of available backends (compiled in)
    /// @return Vector of backend type names
    static std::vector<std::string> get_available_backends();
};
