#include <gtest/gtest.h>
#include "backends/models.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class ModelsTest : public ::testing::Test {
protected:
    std::string test_dir;

    void SetUp() override {
        // Create unique test directory in /tmp
        test_dir = "/tmp/shepherd_models_test_" + std::to_string(getpid());
        fs::create_directories(test_dir);

        // Initialize the models database
        Models::init();
    }

    void TearDown() override {
        // Clean up test directory
        if (fs::exists(test_dir)) {
            fs::remove_all(test_dir);
        }
    }

    // Helper to create a test config.json file
    std::string create_config_json(const std::string& content) {
        std::string model_dir = test_dir + "/model";
        fs::create_directories(model_dir);
        std::string config_path = model_dir + "/config.json";
        std::ofstream file(config_path);
        file << content;
        file.close();
        return model_dir;
    }
};

// ============================================================================
// MDL-001: Detect Qwen from template
// ============================================================================

TEST_F(ModelsTest, MDL001_DetectQwen2FromTemplate) {
    // Qwen 2.x uses <|im_start|> and <|im_end|> tokens
    std::string qwen2_template = R"(
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
)";

    ModelConfig config = Models::detect_from_chat_template(qwen2_template, "qwen2-7b-instruct");

    EXPECT_EQ(config.family, ModelFamily::QWEN_2_X)
        << "Should detect Qwen 2.x from ChatML template with qwen2 path";
    EXPECT_EQ(config.tool_result_role, "tool")
        << "Qwen 2.x should use 'tool' role for tool results";
}

TEST_F(ModelsTest, MDL001_DetectQwen3FromTemplate) {
    // Qwen 3.x also uses ChatML but we detect from path
    std::string qwen3_template = R"(
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
)";

    ModelConfig config = Models::detect_from_chat_template(qwen3_template, "qwen3-8b-instruct");

    EXPECT_EQ(config.family, ModelFamily::QWEN_3_X)
        << "Should detect Qwen 3.x from ChatML template with qwen3 path";
}

TEST_F(ModelsTest, MDL001_DetectQwen3Thinking) {
    // Qwen3 Thinking model has <think> tags
    std::string qwen3_thinking_template = R"(
{% for message in messages %}
<|im_start|>{{ message['role'] }}
<think>{{ message.thinking }}</think>
{{ message['content'] }}<|im_end|>
{% endfor %}
)";

    ModelConfig config = Models::detect_from_chat_template(qwen3_thinking_template, "qwen3-32b-thinking");

    EXPECT_EQ(config.family, ModelFamily::QWEN_3_X)
        << "Should detect Qwen 3.x thinking model";
    EXPECT_TRUE(config.supports_thinking_mode)
        << "Thinking model should have thinking mode support";
}

// ============================================================================
// MDL-002: Detect Llama3 from template
// ============================================================================

TEST_F(ModelsTest, MDL002_DetectLlama3FromTemplate) {
    // Llama 3.x has "Environment: ipython" and <|eom_id|>
    std::string llama3_template = R"(
{% for message in messages %}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
{{ message['content'] }}{{ '<|eot_id|>' }}
{% endfor %}
{% if 'ipython' in tools %}
Environment: ipython
{% endif %}
<|eom_id|>
)";

    ModelConfig config = Models::detect_from_chat_template(llama3_template, "llama-3.1-8b-instruct");

    EXPECT_EQ(config.family, ModelFamily::LLAMA_3_X)
        << "Should detect Llama 3.x from template with ipython environment";
    EXPECT_EQ(config.tool_result_role, "ipython")
        << "Llama 3.x should use 'ipython' role for tool results";
    EXPECT_TRUE(config.uses_eom_token)
        << "Llama 3.x should use <|eom_id|> token";
    EXPECT_TRUE(config.uses_python_tag)
        << "Llama 3.x should use python_tag";
}

TEST_F(ModelsTest, MDL002_DetectLlama3Version) {
    std::string llama3_template = R"(
Environment: ipython
<|eom_id|>
)";

    // Test version extraction from path
    ModelConfig config1 = Models::detect_from_chat_template(llama3_template, "llama-3.1-70b");
    EXPECT_EQ(config1.version, "3.1");

    ModelConfig config2 = Models::detect_from_chat_template(llama3_template, "llama-3.2-90b-vision");
    EXPECT_EQ(config2.version, "3.2");
}

// ============================================================================
// MDL-003: Detect GLM4 from template
// ============================================================================

TEST_F(ModelsTest, MDL003_DetectGLM4FromTemplate) {
    // GLM-4.x uses <|observation|> role
    std::string glm4_template = R"(
{% for message in messages %}
{% if message['role'] == 'observation' %}
<|observation|>{{ message['content'] }}
{% else %}
<|{{ message['role'] }}|>{{ message['content'] }}
{% endif %}
{% endfor %}
)";

    ModelConfig config = Models::detect_from_chat_template(glm4_template, "glm-4-9b");

    EXPECT_EQ(config.family, ModelFamily::GLM_4)
        << "Should detect GLM-4 from template with <|observation|>";
    EXPECT_EQ(config.tool_result_role, "observation")
        << "GLM-4 should use 'observation' role for tool results";
    EXPECT_TRUE(config.uses_observation_role)
        << "GLM-4 should use observation role flag";
}

TEST_F(ModelsTest, MDL003_DetectGLM4Version) {
    std::string glm4_template = "<|observation|>";

    ModelConfig config = Models::detect_from_chat_template(glm4_template, "glm-4.5-9b-chat");
    EXPECT_EQ(config.version, "4.5");

    // GLM 4.5+ supports thinking mode
    EXPECT_TRUE(config.supports_thinking_mode)
        << "GLM 4.5+ should support thinking mode";
}

// ============================================================================
// MDL-004: Detect from config.json
// ============================================================================

TEST_F(ModelsTest, MDL004_DetectFromConfigJsonArchitecture) {
    // Create a config.json with LlamaForCausalLM architecture
    std::string model_dir = create_config_json(R"({
        "architecture": "LlamaForCausalLM",
        "model_type": "llama"
    })");

    ModelConfig config = Models::detect_from_config_file(model_dir);

    EXPECT_EQ(config.family, ModelFamily::LLAMA_2_X)
        << "Should detect Llama from architecture field";
}

TEST_F(ModelsTest, MDL004_DetectFromConfigJsonQwen2) {
    std::string model_dir = create_config_json(R"({
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2"
    })");

    ModelConfig config = Models::detect_from_config_file(model_dir);

    EXPECT_EQ(config.family, ModelFamily::QWEN_2_X)
        << "Should detect Qwen 2.x from architecture field";
}

TEST_F(ModelsTest, MDL004_DetectFromConfigJsonGLM) {
    std::string model_dir = create_config_json(R"({
        "architecture": "ChatGLMForConditionalGeneration",
        "model_type": "chatglm"
    })");

    ModelConfig config = Models::detect_from_config_file(model_dir);

    EXPECT_EQ(config.family, ModelFamily::GLM_4)
        << "Should detect GLM from architecture field";
}

TEST_F(ModelsTest, MDL004_DetectFromConfigJsonModelType) {
    std::string model_dir = create_config_json(R"({
        "model_type": "mistral"
    })");

    ModelConfig config = Models::detect_from_config_file(model_dir);

    // Mistral uses similar format to Llama 2
    EXPECT_EQ(config.family, ModelFamily::LLAMA_2_X)
        << "Should detect Mistral from model_type (maps to LLAMA_2_X)";
}

TEST_F(ModelsTest, MDL004_DetectFromConfigJsonMissing) {
    // Non-existent directory
    ModelConfig config = Models::detect_from_config_file("/nonexistent/path");

    EXPECT_EQ(config.family, ModelFamily::GENERIC)
        << "Should return GENERIC when config.json is missing";
}

TEST_F(ModelsTest, MDL004_DetectLlama3FromConfigJson) {
    // Llama 3.x detected by special tokens in config
    std::string model_dir = create_config_json(R"({
        "architecture": "LlamaForCausalLM",
        "model_type": "llama",
        "added_tokens_decoder": {
            "128000": {"content": "<|begin_of_text|>"},
            "128009": {"content": "<|eom_id|>"}
        }
    })");

    ModelConfig config = Models::detect_from_config_file(model_dir);

    EXPECT_EQ(config.family, ModelFamily::LLAMA_3_X)
        << "Should detect Llama 3.x from config.json with special tokens";
}

// ============================================================================
// MDL-005: Detect from model path
// ============================================================================

TEST_F(ModelsTest, MDL005_DetectFromPathLlama3) {
    ModelConfig config = Models::detect_from_model_path("/models/llama-3.1-8b-instruct.gguf");
    EXPECT_EQ(config.family, ModelFamily::LLAMA_3_X);
    EXPECT_EQ(config.version, "3.1");
}

TEST_F(ModelsTest, MDL005_DetectFromPathQwen2) {
    ModelConfig config = Models::detect_from_model_path("/models/qwen2.5-7b-instruct.gguf");
    EXPECT_EQ(config.family, ModelFamily::QWEN_2_X);
    EXPECT_EQ(config.version, "2.5");
}

TEST_F(ModelsTest, MDL005_DetectFromPathQwen3) {
    ModelConfig config = Models::detect_from_model_path("/models/qwen3-32b-instruct.gguf");
    EXPECT_EQ(config.family, ModelFamily::QWEN_3_X);
}

TEST_F(ModelsTest, MDL005_DetectFromPathGLM4) {
    ModelConfig config = Models::detect_from_model_path("/models/glm-4-9b-chat.gguf");
    EXPECT_EQ(config.family, ModelFamily::GLM_4);
}

TEST_F(ModelsTest, MDL005_DetectFromPathMindLink) {
    // MindLink is a Qwen 3.x derivative
    ModelConfig config = Models::detect_from_model_path("/models/mindlink-32b-instruct.gguf");
    EXPECT_EQ(config.family, ModelFamily::QWEN_3_X)
        << "MindLink should be detected as Qwen 3.x family";
}

TEST_F(ModelsTest, MDL005_DetectFromPathGPTOSS) {
    ModelConfig config = Models::detect_from_model_path("/models/gpt-oss-1.gguf");
    EXPECT_EQ(config.family, ModelFamily::GPT_OSS)
        << "GPT-OSS should be detected from path";
}

TEST_F(ModelsTest, MDL005_DetectFromPathCaseInsensitive) {
    // Path detection should be case-insensitive
    ModelConfig config1 = Models::detect_from_model_path("/models/LLAMA-3-8B.gguf");
    EXPECT_EQ(config1.family, ModelFamily::LLAMA_3_X);

    ModelConfig config2 = Models::detect_from_model_path("/models/Qwen2.5-7B-Instruct.gguf");
    EXPECT_EQ(config2.family, ModelFamily::QWEN_2_X);
}

TEST_F(ModelsTest, MDL005_DetectFromPathUnknown) {
    ModelConfig config = Models::detect_from_model_path("/models/unknown-model.gguf");
    EXPECT_EQ(config.family, ModelFamily::GENERIC)
        << "Unknown model paths should return GENERIC";
}

TEST_F(ModelsTest, MDL005_DetectFromPathEmpty) {
    ModelConfig config = Models::detect_from_model_path("");
    EXPECT_EQ(config.family, ModelFamily::GENERIC)
        << "Empty path should return GENERIC";
}

// ============================================================================
// MDL-006: API model lookup - context size
// ============================================================================

TEST_F(ModelsTest, MDL006_APIModelLookupGPT4o) {
    ModelConfig config = Models::detect_from_api_model("openai", "gpt-4o");

    EXPECT_EQ(config.provider, "openai");
    EXPECT_EQ(config.context_window, 128000)
        << "GPT-4o should have 128k context window";
    EXPECT_EQ(config.max_output_tokens, 16384)
        << "GPT-4o should have 16k max output tokens";
    EXPECT_TRUE(config.vision_support)
        << "GPT-4o should support vision";
}

TEST_F(ModelsTest, MDL006_APIModelLookupGPT4oMini) {
    ModelConfig config = Models::detect_from_api_model("openai", "gpt-4o-mini");

    EXPECT_EQ(config.context_window, 128000);
    EXPECT_EQ(config.max_output_tokens, 16384);
    EXPECT_TRUE(config.fine_tunable)
        << "GPT-4o-mini should be fine-tunable";
}

TEST_F(ModelsTest, MDL006_APIModelLookupO1) {
    ModelConfig config = Models::detect_from_api_model("openai", "o1");

    EXPECT_EQ(config.context_window, 200000)
        << "o1 should have 200k context window";
    EXPECT_EQ(config.max_output_tokens, 100000);
    EXPECT_EQ(config.max_cot_tokens, 32768)
        << "o1 should have max_cot_tokens set";
    EXPECT_EQ(config.max_tokens_param_name, "max_completion_tokens")
        << "o1 models use max_completion_tokens parameter";
}

TEST_F(ModelsTest, MDL006_APIModelLookupClaude) {
    ModelConfig config = Models::detect_from_api_model("anthropic", "claude-3-5-sonnet-20240620");

    EXPECT_EQ(config.provider, "anthropic");
    EXPECT_EQ(config.context_window, 200000)
        << "Claude should have 200k context window";
    EXPECT_TRUE(config.vision_support)
        << "Claude 3.5 Sonnet should support vision";
}

TEST_F(ModelsTest, MDL006_APIModelLookupGemini) {
    ModelConfig config = Models::detect_from_api_model("gemini", "gemini-1.5-pro");

    EXPECT_EQ(config.provider, "gemini");
    EXPECT_EQ(config.context_window, 2000000)
        << "Gemini 1.5 Pro should have 2M context window";
    EXPECT_EQ(config.max_tokens_param_name, "maxOutputTokens")
        << "Gemini uses maxOutputTokens parameter";
}

TEST_F(ModelsTest, MDL006_APIModelLookupPatternMatch) {
    // Pattern matching: gpt-4* should match gpt-4-0613
    ModelConfig config = Models::detect_from_api_model("openai", "gpt-4-0613");

    EXPECT_EQ(config.context_window, 8192)
        << "GPT-4-0613 should have 8k context window";
}

TEST_F(ModelsTest, MDL006_APIModelLookupUnknown) {
    // Unknown model should use provider defaults
    ModelConfig config = Models::detect_from_api_model("openai", "unknown-model-xyz");

    EXPECT_EQ(config.provider, "openai");
    EXPECT_GT(config.context_window, 0)
        << "Unknown model should get provider default context window";
}

TEST_F(ModelsTest, MDL006_APIModelLookupSupportsEndpoint) {
    // Check endpoint support
    EXPECT_TRUE(Models::supports_endpoint("gpt-4o", "/v1/chat/completions"))
        << "GPT-4o should support chat completions endpoint";
    EXPECT_TRUE(Models::supports_endpoint("gpt-4o", "/v1/assistants"))
        << "GPT-4o should support assistants endpoint";
}

// ============================================================================
// Additional Model Tests
// ============================================================================

TEST_F(ModelsTest, DefaultChatTemplates) {
    // Test that default templates are available
    std::string llama3_template = Models::get_default_chat_template(ModelFamily::LLAMA_3_X);
    EXPECT_FALSE(llama3_template.empty());
    EXPECT_TRUE(llama3_template.find("start_header_id") != std::string::npos)
        << "Llama 3 template should use header tags";

    std::string qwen_template = Models::get_default_chat_template(ModelFamily::QWEN_2_X);
    EXPECT_FALSE(qwen_template.empty());
    EXPECT_TRUE(qwen_template.find("im_start") != std::string::npos)
        << "Qwen template should use ChatML format";

    std::string generic_template = Models::get_default_chat_template(ModelFamily::GENERIC);
    EXPECT_FALSE(generic_template.empty())
        << "Generic family should have a default template";
}

TEST_F(ModelsTest, DetectFromTokenizerClass) {
    EXPECT_EQ(Models::detect_from_tokenizer_class("LlamaTokenizerFast"),
              ModelFamily::LLAMA_3_X);
    EXPECT_EQ(Models::detect_from_tokenizer_class("Qwen2Tokenizer"),
              ModelFamily::QWEN_2_X);
    EXPECT_EQ(Models::detect_from_tokenizer_class("QwenTokenizer"),
              ModelFamily::QWEN_3_X);
    EXPECT_EQ(Models::detect_from_tokenizer_class("MistralTokenizer"),
              ModelFamily::MISTRAL);
    EXPECT_EQ(Models::detect_from_tokenizer_class("UnknownTokenizer"),
              ModelFamily::GENERIC);
}

TEST_F(ModelsTest, ModelConfigFactories) {
    // Test factory methods create correct configs
    ModelConfig generic = ModelConfig::create_generic();
    EXPECT_EQ(generic.family, ModelFamily::GENERIC);
    EXPECT_EQ(generic.tool_result_role, "tool");

    ModelConfig llama2 = ModelConfig::create_llama_2x("2");
    EXPECT_EQ(llama2.family, ModelFamily::LLAMA_2_X);
    EXPECT_EQ(llama2.version, "2");

    ModelConfig llama3 = ModelConfig::create_llama_3x("3.1");
    EXPECT_EQ(llama3.family, ModelFamily::LLAMA_3_X);
    EXPECT_EQ(llama3.version, "3.1");
    EXPECT_TRUE(llama3.uses_eom_token);
    EXPECT_TRUE(llama3.uses_python_tag);

    ModelConfig glm = ModelConfig::create_glm_4("4.5");
    EXPECT_EQ(glm.family, ModelFamily::GLM_4);
    EXPECT_TRUE(glm.supports_thinking_mode);
    EXPECT_TRUE(glm.uses_observation_role);

    ModelConfig qwen2 = ModelConfig::create_qwen_2x("2.5");
    EXPECT_EQ(qwen2.family, ModelFamily::QWEN_2_X);
    EXPECT_EQ(qwen2.assistant_start_tag, "<|im_start|>assistant\n");

    ModelConfig qwen3_think = ModelConfig::create_qwen_3x("3", true);
    EXPECT_EQ(qwen3_think.family, ModelFamily::QWEN_3_X);
    EXPECT_TRUE(qwen3_think.supports_thinking_mode);
    EXPECT_FALSE(qwen3_think.thinking_start_markers.empty());

    ModelConfig gpt_oss = ModelConfig::create_gpt_oss();
    EXPECT_EQ(gpt_oss.family, ModelFamily::GPT_OSS);
    EXPECT_TRUE(gpt_oss.supports_thinking_mode);
}

TEST_F(ModelsTest, LoadGenerationConfig) {
    // Create a generation_config.json
    std::string model_dir = test_dir + "/model";
    fs::create_directories(model_dir);
    std::string gen_config_path = model_dir + "/generation_config.json";
    std::ofstream file(gen_config_path);
    file << R"({
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    })";
    file.close();

    float temperature = 1.0;
    float top_p = 1.0;
    int top_k = 50;

    bool loaded = Models::load_generation_config(model_dir, temperature, top_p, top_k);

    EXPECT_TRUE(loaded);
    EXPECT_FLOAT_EQ(temperature, 0.7f);
    EXPECT_FLOAT_EQ(top_p, 0.9f);
    EXPECT_EQ(top_k, 40);
}

TEST_F(ModelsTest, LoadGenerationConfigMissing) {
    float temperature = 1.0;
    float top_p = 1.0;
    int top_k = 50;

    bool loaded = Models::load_generation_config("/nonexistent/path", temperature, top_p, top_k);

    EXPECT_FALSE(loaded);
    // Values should be unchanged
    EXPECT_FLOAT_EQ(temperature, 1.0f);
    EXPECT_FLOAT_EQ(top_p, 1.0f);
    EXPECT_EQ(top_k, 50);
}

TEST_F(ModelsTest, GetCompatibleModels) {
    std::vector<std::string> chat_models = Models::get_compatible_models("/v1/chat/completions");

    EXPECT_GT(chat_models.size(), 0)
        << "Should return models compatible with chat completions endpoint";

    // Check that known chat-capable models are in the list
    bool has_gpt4o = std::find(chat_models.begin(), chat_models.end(), "gpt-4o") != chat_models.end();
    EXPECT_TRUE(has_gpt4o) << "gpt-4o should be compatible with chat completions";
}
