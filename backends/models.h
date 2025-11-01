#pragma once

#include <string>
#include <map>

/// @brief Model family classification for prompt format detection
enum class ModelFamily {
    LLAMA_3_X,      // Llama 3.0, 3.1, 3.2, 3.3 - Uses ipython role, <|eom_id|>, <|python_tag|>
    QWEN_2_X,       // Qwen 2.x series - Uses <|im_start|>, tool role
    GLM_4,          // GLM-4, 4.5, 4.6 - Uses <|observation|>, <think> tags
    MISTRAL,        // Mistral models - Uses [INST], tool role
    GEMMA,          // Google Gemma - Uses <start_of_turn>, tool role
    DEEPSEEK,       // DeepSeek models (including R1) - Various formats
    COMMAND_R,      // Cohere Command-R - Uses tool_results role
    PHI_3,          // Microsoft Phi-3.x - Uses <|system|>, tool role
    FUNCTIONARY,    // Meetkai Functionary series - Uses ipython, python_tag
    GENERIC         // Unknown/generic models - Uses tool role
};

/// @brief Configuration for model-specific prompt behavior
struct ModelConfig {
    ModelFamily family;
    std::string version;              // e.g., "3.1", "4.5", "2.5"

    // Role mapping for different message types
    std::string tool_result_role;     // "ipython", "tool", "observation", "tool_results"

    // Special token handling
    bool uses_eom_token;              // Uses <|eom_id|> for continued tool calls
    bool uses_python_tag;             // Expects <|python_tag|> format for built-in tools
    bool uses_builtin_tools_array;    // Set builtin_tools = ["ipython"] in template context

    // GLM-specific features
    bool supports_thinking_mode;      // GLM-4.5+ thinking mode
    bool uses_observation_role;       // GLM-4.x uses <|observation|> instead of tool

    // Tool call format
    std::string tool_call_format;     // "json", "python_tag", "xml"

    // Message format tags (populated by Models from chat template)
    std::string assistant_start_tag;  // e.g., "<|im_start|>assistant\n" or "<|start_header_id|>assistant<|end_header_id|>\n\n"
    std::string assistant_end_tag;    // e.g., "<|im_end|>\n" or "<|eot_id|>"

    /// @brief Create default config for generic models
    static ModelConfig create_generic() {
        return ModelConfig{
            .family = ModelFamily::GENERIC,
            .version = "",
            .tool_result_role = "tool",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = false,
            .uses_observation_role = false,
            .tool_call_format = "json",
            .assistant_start_tag = "assistant: ",
            .assistant_end_tag = "\n"
        };
    }

    /// @brief Create config for Llama 3.x family
    static ModelConfig create_llama_3x(const std::string& version = "3.1") {
        return ModelConfig{
            .family = ModelFamily::LLAMA_3_X,
            .version = version,
            .tool_result_role = "ipython",
            .uses_eom_token = true,
            .uses_python_tag = true,
            .uses_builtin_tools_array = true,
            .supports_thinking_mode = false,
            .uses_observation_role = false,
            .tool_call_format = "json",  // Can also use python_tag for built-ins
            .assistant_start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n",
            .assistant_end_tag = "<|eot_id|>"
        };
    }

    /// @brief Create config for GLM-4.x family
    static ModelConfig create_glm_4(const std::string& version = "4") {
        bool has_thinking = (version == "4.5" || version == "4.6");
        return ModelConfig{
            .family = ModelFamily::GLM_4,
            .version = version,
            .tool_result_role = "observation",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = has_thinking,
            .uses_observation_role = true,
            .tool_call_format = "xml",  // Uses <tool_call> tags
            .assistant_start_tag = "<|assistant|>\n",
            .assistant_end_tag = ""  // GLM doesn't use explicit end tag
        };
    }

    /// @brief Create config for Qwen 2.x family
    static ModelConfig create_qwen_2x(const std::string& version = "2.5") {
        return ModelConfig{
            .family = ModelFamily::QWEN_2_X,
            .version = version,
            .tool_result_role = "tool",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = false,
            .uses_observation_role = false,
            .tool_call_format = "json",
            .assistant_start_tag = "<|im_start|>assistant\n",
            .assistant_end_tag = "<|im_end|>\n"
        };
    }
};

// Forward declarations
class ToolRegistry;

/// @brief Model detection and configuration
/// Centralizes model family detection logic that was previously scattered across backends
class Models {
public:
    /// @brief Detect model family from chat template content
    /// Primary detection method - analyzes Jinja template for model-specific patterns
    /// @param template_text Chat template content (Jinja format)
    /// @param model_path Optional model path for additional hints
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_chat_template(const std::string& template_text, const std::string& model_path = "");

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
    static std::string format_system_message(const ModelConfig& config, const std::string& custom_system_prompt, ToolRegistry& registry, void* template_node = nullptr);

private:
    /// @brief Helper to detect from template text (primary method)
    static ModelConfig detect_from_template_content(const std::string& template_text, const std::string& model_path);

    /// @brief Helper to detect from path analysis (secondary method)
    static ModelConfig detect_from_path_analysis(const std::string& model_path);

    /// @brief Helper to extract version from model path
    static std::string extract_version_from_path(const std::string& model_path, const std::string& pattern);
};
