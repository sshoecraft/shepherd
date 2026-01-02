#pragma once

#include <string>
#include <map>
#include <vector>

/// @brief Model family classification for prompt format detection
enum class ModelFamily {
    LLAMA_2_X,      // Llama 1.x, 2.x, TinyLlama - Uses [INST] format
    LLAMA_3_X,      // Llama 3.0, 3.1, 3.2, 3.3 - Uses ipython role, <|eom_id|>, <|python_tag|>
    QWEN_2_X,       // Qwen 2.x series - Uses <|im_start|>, tool role
    QWEN_3_X,       // Qwen 3.x series (includes MindLink) - Uses <|im_start|>, tool role, enhanced capabilities
    GLM_4,          // GLM-4, 4.5, 4.6 - Uses <|observation|>, <think> tags
    MISTRAL,        // Mistral models - Uses [INST], tool role
    GEMMA,          // Google Gemma - Uses <start_of_turn>, tool role
    DEEPSEEK,       // DeepSeek models (including R1) - Various formats
    COMMAND_R,      // Cohere Command-R - Uses tool_results role
    PHI_3,          // Microsoft Phi-3.x - Uses <|system|>, tool role
    FUNCTIONARY,    // Meetkai Functionary series - Uses ipython, python_tag
    GPT_OSS,        // OpenAI GPT-OSS - Uses <|channel|> markers (analysis, commentary, final)
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

    // Output markers for ModelOutput processing
    std::vector<std::string> tool_call_start_markers;    // e.g., {"<tool_call>"}
    std::vector<std::string> tool_call_end_markers;      // e.g., {"</tool_call>"}
    std::vector<std::string> thinking_start_markers;     // e.g., {"<think>", "<thinking>"}
    std::vector<std::string> thinking_end_markers;       // e.g., {"</think>", "</thinking>"}

    // Content extraction markers (for models like GPT-OSS that use channels)
    // When set, output before content_extract_marker is hidden, content after is shown
    std::string content_extract_marker;  // e.g., "<|channel|>final<|message|>" - extract content after this
    std::string content_end_marker;      // e.g., "<|end|>" - stop extraction at this marker

    // Message format tags (populated by Models from chat template)
    std::string assistant_start_tag;  // e.g., "<|im_start|>assistant\n" or "<|start_header_id|>assistant<|end_header_id|>\n\n"
    std::string assistant_end_tag;    // e.g., "<|im_end|>\n" or "<|eot_id|>"

    // API model metadata (for OpenAI, Anthropic, Gemini, etc.)
    std::string provider;             // "openai", "anthropic", "gemini", "local", ""
    std::string model_name;           // Full model identifier
    size_t context_window;            // Maximum input tokens
    int max_output_tokens;            // Maximum completion tokens
    int max_cot_tokens;               // Maximum chain-of-thought/reasoning tokens (0 = not applicable)
    std::string max_tokens_param_name; // Parameter name to use: "max_tokens", "max_completion_tokens", "maxOutputTokens"
    std::vector<std::string> supported_endpoints; // Compatible API endpoints
    std::map<std::string, std::string> special_headers; // Provider-specific HTTP headers
    std::vector<std::string> aliases;  // Alternative names for this model

    // API capabilities
    bool supports_temperature;        // Can use temperature parameter
    bool supports_streaming;          // Supports streaming responses
    bool vision_support;              // Can process images
    bool audio_support;               // Can process audio
    bool function_calling_support;    // Supports function/tool calling
    bool realtime_capable;            // Supports realtime API
    bool fine_tunable;                // Can be fine-tuned

    // Metadata
    std::string training_cutoff_date; // Knowledge cutoff date
    bool deprecated;                  // Model is deprecated
    std::string replacement_model;    // Suggested replacement if deprecated
    std::string notes;                // Additional information

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
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "assistant: ",
            .assistant_end_tag = "\n",
            .provider = "",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
        };
    }

    /// @brief Create config for Llama 2.x family (includes Llama 1.x, TinyLlama)
    static ModelConfig create_llama_2x(const std::string& version = "2") {
        return ModelConfig{
            .family = ModelFamily::LLAMA_2_X,
            .version = version,
            .tool_result_role = "tool",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = false,
            .uses_observation_role = false,
            .tool_call_format = "json",
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "[/INST] ",
            .assistant_end_tag = "</s>",
            .provider = "local",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = false,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
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
            .tool_call_format = "json",
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n",
            .assistant_end_tag = "<|eot_id|>",
            .provider = "local",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
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
            .tool_call_format = "xml",
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "<|assistant|>\n",
            .assistant_end_tag = "",
            .provider = "local",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
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
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "<|im_start|>assistant\n",
            .assistant_end_tag = "<|im_end|>\n",
            .provider = "local",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
        };
    }

    /// @brief Create config for Qwen 3.x family (includes MindLink)
    static ModelConfig create_qwen_3x(const std::string& version = "3", bool is_thinking = false) {
        ModelConfig config{
            .family = ModelFamily::QWEN_3_X,
            .version = version,
            .tool_result_role = "tool",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = is_thinking,
            .uses_observation_role = false,
            .tool_call_format = "json",
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",
            .content_end_marker = "",
            .assistant_start_tag = "<|im_start|>assistant\n",
            .assistant_end_tag = "<|im_end|>\n",
            .provider = "local",
            .model_name = "",
            .context_window = 0,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "",
            .deprecated = false,
            .replacement_model = "",
            .notes = ""
        };

        // Add thinking markers for thinking models
        if (is_thinking) {
            config.thinking_start_markers = {"<think>"};
            config.thinking_end_markers = {"</think>"};
        }

        return config;
    }

    /// @brief Create config for OpenAI GPT-OSS family
    /// Channel-based output format detected dynamically from template
    static ModelConfig create_gpt_oss() {
        return ModelConfig{
            .family = ModelFamily::GPT_OSS,
            .version = "",
            .tool_result_role = "tool",
            .uses_eom_token = false,
            .uses_python_tag = false,
            .uses_builtin_tools_array = false,
            .supports_thinking_mode = true,  // Has analysis channel for reasoning
            .uses_observation_role = false,
            .tool_call_format = "json",
            .tool_call_start_markers = {},
            .tool_call_end_markers = {},
            .thinking_start_markers = {},
            .thinking_end_markers = {},
            .content_extract_marker = "",  // Detected dynamically from template
            .content_end_marker = "",      // Detected dynamically from template
            .assistant_start_tag = "<|start|>assistant",
            .assistant_end_tag = "<|end|>",
            .provider = "local",
            .model_name = "",
            .context_window = 131072,
            .max_output_tokens = 0,
            .max_cot_tokens = 0,
            .max_tokens_param_name = "max_tokens",
            .supported_endpoints = {},
            .special_headers = {},
            .aliases = {},
            .supports_temperature = true,
            .supports_streaming = true,
            .vision_support = false,
            .audio_support = false,
            .function_calling_support = true,
            .realtime_capable = false,
            .fine_tunable = false,
            .training_cutoff_date = "2024-06",
            .deprecated = false,
            .replacement_model = "",
            .notes = "OpenAI GPT-OSS with channel-based output format"
        };
    }
};

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

    /// @brief Detect model family from config.json file
    /// Reads architecture/model_type from HuggingFace or TensorRT-LLM config format
    /// @param model_dir Path to model/engine directory containing config.json
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_config_file(const std::string& model_dir);

    /// @brief Detect model family from API model name
    /// For API backends (OpenAI, Anthropic, etc.)
    /// @param model_name Model identifier from API provider
    /// @return ModelConfig with detected family and settings
    static ModelConfig detect_from_model_name(const std::string& model_name);

    /// @brief Detect API model configuration from provider and model name
    /// Primary method for API backends - looks up model metadata from database
    /// @param provider Provider name: "openai", "anthropic", "gemini"
    /// @param model_name Model identifier from API provider
    /// @return ModelConfig with full API metadata (context window, capabilities, etc.)
    static ModelConfig detect_from_api_model(const std::string& provider, const std::string& model_name);

    /// @brief Get default chat template for a model family
    /// Returns a standard Jinja2 chat template when tokenizer_config.json is missing one
    /// @param family The detected model family
    /// @return Default chat template string, or empty string if no default available
    static std::string get_default_chat_template(ModelFamily family);

    /// @brief Detect model family from tokenizer class name
    /// Used as fallback when chat_template is missing from tokenizer_config.json
    /// @param tokenizer_class The tokenizer_class field from tokenizer_config.json
    /// @return Detected ModelFamily based on tokenizer class
    static ModelFamily detect_from_tokenizer_class(const std::string& tokenizer_class);

    /// @brief Load generation_config.json from model directory
    /// Reads sampling parameters (temperature, top_p, top_k) from generation_config.json
    /// @param model_dir_path Path to model directory
    /// @param temperature Output parameter for temperature (unchanged if not found)
    /// @param top_p Output parameter for top_p (unchanged if not found)
    /// @param top_k Output parameter for top_k (unchanged if not found)
    /// @return true if file was read successfully, false otherwise
    static bool load_generation_config(const std::string& model_dir_path,
                                       float& temperature,
                                       float& top_p,
                                       int& top_k);

    /// @brief Check if model supports a specific API endpoint
    /// @param model_name Model identifier
    /// @param endpoint_path API endpoint path (e.g., "/v1/chat/completions")
    /// @return true if model supports the endpoint
    static bool supports_endpoint(const std::string& model_name, const std::string& endpoint_path);

    /// @brief Get list of models compatible with an endpoint
    /// @param endpoint_path API endpoint path
    /// @return List of compatible model names
    static std::vector<std::string> get_compatible_models(const std::string& endpoint_path);

    /// @brief Initialize models database from JSON file or embedded defaults
    /// Called automatically on first use, can be called explicitly to override
    /// @param custom_path Optional custom path to models.json
    static void init(const std::string& custom_path = "");

    /// @brief Load models database from JSON file
    /// @param path Path to models.json file
    /// @return true if loaded successfully
    static bool load_models_database(const std::string& path);

    /// @brief Format system message with tools for specific model family
private:
    /// @brief Helper to detect from template text (primary method)
    static ModelConfig detect_from_template_content(const std::string& template_text, const std::string& model_path);

    /// @brief Helper to detect from path analysis (secondary method)
    static ModelConfig detect_from_path_analysis(const std::string& model_path);

    /// @brief Helper to extract version from model path
    static std::string extract_version_from_path(const std::string& model_path, const std::string& pattern);

    /// @brief Ensure models database is initialized
    static void ensure_initialized();

    /// @brief Get embedded default models JSON as string
    static std::string get_embedded_models_json();

    /// @brief Parse JSON and populate model database
    static bool parse_models_json(const std::string& json_content);

    /// @brief Match model name against pattern
    static bool matches_pattern(const std::string& model_name, const std::string& pattern);

    /// @brief Get provider default configuration
    static ModelConfig get_provider_default(const std::string& provider);

    /// @brief Static database of model configurations (exact matches)
    static std::map<std::string, ModelConfig> model_database;

    /// @brief Static database of model patterns for wildcard matching
    static std::map<std::string, ModelConfig> pattern_database;

    /// @brief Static database of provider defaults
    static std::map<std::string, ModelConfig> provider_defaults;

    /// @brief Flag tracking initialization state
    static bool initialized;

    /// @brief Custom models file path (if specified)
    static std::string custom_models_path;
};
