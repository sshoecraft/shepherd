#pragma once

#include "model_config.h"
#include <string>
#include <map>

// Forward declarations
class Tool;
class ToolRegistry;

/// @brief Manager for model detection and configuration
/// Centralizes model family detection logic that was previously scattered across backends
class ModelManager {
public:
    /// @brief Detect model family from chat template content
    /// Primary detection method - analyzes Jinja template for model-specific patterns
    /// @param template_text Chat template content (Jinja format)
    /// @param model_path Optional model path for additional hints
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_chat_template(const std::string& template_text,
                                                  const std::string& model_path = "");

    /// @brief Detect model family from model filename/path
    /// Fallback detection method - analyzes filename for model name patterns
    /// @param model_path Full path to model file
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_model_path(const std::string& model_path);

    /// @brief Detect model family from API model name
    /// For API backends (OpenAI, Anthropic, etc.)
    /// @param model_name Model identifier from API provider
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_model_name(const std::string& model_name);

    /// @brief Format system message with tools for specific model family
    /// Each model family has different tool formatting requirements
    /// @param config Model configuration (family, version, etc.)
    /// @param custom_system_prompt User's custom system prompt from config
    /// @param registry Tool registry to get available tools
    /// @param template_node Minja template node (void* to shared_ptr<TemplateNode>*), can be nullptr
    /// @return Fully formatted system message content with tools embedded
    static std::string format_system_message(const ModelConfig& config,
                                             const std::string& custom_system_prompt,
                                             ToolRegistry& registry,
                                             void* template_node = nullptr);

private:
    /// @brief Helper to detect from template text (primary method)
    static ModelConfig detect_from_template_content(const std::string& template_text,
                                                     const std::string& model_path);

    /// @brief Helper to detect from path analysis (secondary method)
    static ModelConfig detect_from_path_analysis(const std::string& model_path);

    /// @brief Helper to extract version from model path
    static std::string extract_version_from_path(const std::string& model_path,
                                                  const std::string& pattern);
};
