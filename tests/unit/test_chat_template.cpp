#include <gtest/gtest.h>
#include "backends/chat_template.h"
#include "backends/models.h"
#include "message.h"
#include <memory>

using namespace ChatTemplates;

// ============================================================================
// Test Fixtures
// ============================================================================

class ChatTemplateTest : public ::testing::Test {
protected:
    std::unique_ptr<ChatMLTemplate> chatml;
    std::unique_ptr<Llama2Template> llama2;
    std::unique_ptr<Llama3Template> llama3;
    std::unique_ptr<GLM4Template> glm4;

    void SetUp() override {
        chatml = std::make_unique<ChatMLTemplate>();
        llama2 = std::make_unique<Llama2Template>();
        llama3 = std::make_unique<Llama3Template>();
        glm4 = std::make_unique<GLM4Template>();
    }

    // Helper to create Session::Tool
    Session::Tool create_tool(const std::string& name, const std::string& description) {
        Session::Tool tool;
        tool.name = name;
        tool.description = description;
        tool.parameters = nlohmann::json::object();
        tool.parameters["type"] = "object";
        tool.parameters["properties"] = nlohmann::json::object();
        return tool;
    }
};

// ============================================================================
// TPL-001: ChatML format_message
// ============================================================================

TEST_F(ChatTemplateTest, TPL001_ChatMLFormatUserMessage) {
    Message msg(Message::USER, "Hello, how are you?", 0);

    std::string formatted = chatml->format_message(msg);

    EXPECT_TRUE(formatted.find("<|im_start|>user") != std::string::npos)
        << "ChatML user message should have <|im_start|>user tag";
    EXPECT_TRUE(formatted.find("Hello, how are you?") != std::string::npos)
        << "Message content should be present";
    EXPECT_TRUE(formatted.find("<|im_end|>") != std::string::npos)
        << "ChatML message should end with <|im_end|>";
}

TEST_F(ChatTemplateTest, TPL001_ChatMLFormatAssistantMessage) {
    Message msg(Message::ASSISTANT, "I'm doing well, thank you!", 0);

    std::string formatted = chatml->format_message(msg);

    EXPECT_TRUE(formatted.find("<|im_start|>assistant") != std::string::npos)
        << "ChatML assistant message should have <|im_start|>assistant tag";
    EXPECT_TRUE(formatted.find("I'm doing well") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|im_end|>") != std::string::npos);
}

TEST_F(ChatTemplateTest, TPL001_ChatMLFormatSystemMessage) {
    std::string formatted = chatml->format_system_message("You are a helpful assistant.", {});

    EXPECT_TRUE(formatted.find("<|im_start|>system") != std::string::npos)
        << "ChatML system message should have <|im_start|>system tag";
    EXPECT_TRUE(formatted.find("You are a helpful assistant.") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|im_end|>") != std::string::npos);
}

TEST_F(ChatTemplateTest, TPL001_ChatMLFormatToolResponseMessage) {
    Message msg(Message::TOOL_RESPONSE, "Result: 42", 0);
    msg.tool_name = "calculator";
    msg.tool_call_id = "call_123";

    std::string formatted = chatml->format_message(msg);

    // Tool responses use "tool" role in ChatML
    EXPECT_TRUE(formatted.find("<|im_start|>tool") != std::string::npos)
        << "ChatML tool response should use 'tool' role";
    EXPECT_TRUE(formatted.find("Result: 42") != std::string::npos);
}

// ============================================================================
// TPL-002: Llama3 format_message
// ============================================================================

TEST_F(ChatTemplateTest, TPL002_Llama3FormatUserMessage) {
    Message msg(Message::USER, "What is the capital of France?", 0);

    std::string formatted = llama3->format_message(msg);

    EXPECT_TRUE(formatted.find("<|start_header_id|>user<|end_header_id|>") != std::string::npos)
        << "Llama3 user message should have header tags";
    EXPECT_TRUE(formatted.find("What is the capital of France?") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|eot_id|>") != std::string::npos)
        << "Llama3 message should end with <|eot_id|>";
}

TEST_F(ChatTemplateTest, TPL002_Llama3FormatAssistantMessage) {
    Message msg(Message::ASSISTANT, "The capital of France is Paris.", 0);

    std::string formatted = llama3->format_message(msg);

    EXPECT_TRUE(formatted.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos);
    EXPECT_TRUE(formatted.find("Paris") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|eot_id|>") != std::string::npos);
}

TEST_F(ChatTemplateTest, TPL002_Llama3FormatToolResponse) {
    Message msg(Message::TOOL_RESPONSE, "Tool output here", 0);
    msg.tool_name = "test_tool";

    std::string formatted = llama3->format_message(msg);

    // Llama3 uses "ipython" role for tool responses
    EXPECT_TRUE(formatted.find("<|start_header_id|>ipython<|end_header_id|>") != std::string::npos)
        << "Llama3 tool response should use 'ipython' role";
    EXPECT_TRUE(formatted.find("Tool output here") != std::string::npos);
}

TEST_F(ChatTemplateTest, TPL002_Llama3FormatSystemMessage) {
    std::string formatted = llama3->format_system_message("You are an expert.", {});

    EXPECT_TRUE(formatted.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos);
    EXPECT_TRUE(formatted.find("You are an expert.") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|eot_id|>") != std::string::npos);
}

// ============================================================================
// TPL-003: System message with tools
// ============================================================================

TEST_F(ChatTemplateTest, TPL003_ChatMLSystemWithTools) {
    Session::Tool read_file = create_tool("read_file", "Read contents of a file");
    Session::Tool write_file = create_tool("write_file", "Write content to a file");
    std::vector<Session::Tool> tools = {read_file, write_file};

    std::string formatted = chatml->format_system_message("You are a helpful assistant.", tools);

    // ChatML should include tools section
    EXPECT_TRUE(formatted.find("<tools>") != std::string::npos)
        << "ChatML system with tools should have <tools> tag";
    EXPECT_TRUE(formatted.find("</tools>") != std::string::npos);
    EXPECT_TRUE(formatted.find("read_file") != std::string::npos)
        << "Tool name should be present";
    EXPECT_TRUE(formatted.find("write_file") != std::string::npos);
    EXPECT_TRUE(formatted.find("<tool_call>") != std::string::npos)
        << "Should include tool_call instruction";
}

TEST_F(ChatTemplateTest, TPL003_Llama3SystemWithTools) {
    Session::Tool calculator = create_tool("calculator", "Perform calculations");
    std::vector<Session::Tool> tools = {calculator};

    std::string formatted = llama3->format_system_message("You are a helpful assistant.", tools);

    // Llama3 uses BFCL-style format
    EXPECT_TRUE(formatted.find("calculator") != std::string::npos)
        << "Tool name should be present in Llama3 system message";
    EXPECT_TRUE(formatted.find("JSON object") != std::string::npos ||
                formatted.find("JSON") != std::string::npos)
        << "Should mention JSON for tool calls";
}

TEST_F(ChatTemplateTest, TPL003_GLM4SystemWithTools) {
    Session::Tool search = create_tool("web_search", "Search the web");
    std::vector<Session::Tool> tools = {search};

    std::string formatted = glm4->format_system_message("", tools);

    // GLM4 uses Chinese format
    EXPECT_TRUE(formatted.find("<|system|>") != std::string::npos);
    EXPECT_TRUE(formatted.find("web_search") != std::string::npos);
}

TEST_F(ChatTemplateTest, TPL003_EmptyTools) {
    // System message with no tools
    std::string formatted = chatml->format_system_message("You are a helpful assistant.", {});

    EXPECT_TRUE(formatted.find("<|im_start|>system") != std::string::npos);
    EXPECT_TRUE(formatted.find("<tools>") == std::string::npos)
        << "No tools section when tools list is empty";
}

// ============================================================================
// TPL-004: Generation prompt
// ============================================================================

TEST_F(ChatTemplateTest, TPL004_ChatMLGenerationPrompt) {
    std::string prompt = chatml->get_generation_prompt();

    EXPECT_EQ(prompt, "<|im_start|>assistant\n")
        << "ChatML generation prompt should be '<|im_start|>assistant\\n'";
}

TEST_F(ChatTemplateTest, TPL004_Llama3GenerationPrompt) {
    std::string prompt = llama3->get_generation_prompt();

    EXPECT_EQ(prompt, "<|start_header_id|>assistant<|end_header_id|>\n\n")
        << "Llama3 generation prompt should use header tags";
}

TEST_F(ChatTemplateTest, TPL004_Llama2GenerationPrompt) {
    std::string prompt = llama2->get_generation_prompt();

    // Llama2 has no explicit generation prompt - it follows directly after [/INST]
    EXPECT_TRUE(prompt.empty())
        << "Llama2 generation prompt should be empty";
}

TEST_F(ChatTemplateTest, TPL004_GLM4GenerationPrompt) {
    std::string prompt = glm4->get_generation_prompt();

    EXPECT_EQ(prompt, "<|assistant|>\n")
        << "GLM4 generation prompt should be '<|assistant|>\\n'";
}

// ============================================================================
// TPL-005: Assistant end tag
// ============================================================================

TEST_F(ChatTemplateTest, TPL005_ChatMLAssistantEndTag) {
    std::string end_tag = chatml->get_assistant_end_tag();

    EXPECT_EQ(end_tag, "<|im_end|>\n")
        << "ChatML assistant end tag should be '<|im_end|>\\n'";
}

TEST_F(ChatTemplateTest, TPL005_Llama3AssistantEndTag) {
    std::string end_tag = llama3->get_assistant_end_tag();

    EXPECT_EQ(end_tag, "<|eot_id|>")
        << "Llama3 assistant end tag should be '<|eot_id|>'";
}

TEST_F(ChatTemplateTest, TPL005_Llama2AssistantEndTag) {
    std::string end_tag = llama2->get_assistant_end_tag();

    EXPECT_EQ(end_tag, "</s>")
        << "Llama2 assistant end tag should be '</s>'";
}

TEST_F(ChatTemplateTest, TPL005_GLM4AssistantEndTag) {
    std::string end_tag = glm4->get_assistant_end_tag();

    // GLM4 doesn't use an end tag
    EXPECT_TRUE(end_tag.empty())
        << "GLM4 assistant end tag should be empty";
}

// ============================================================================
// TPL-006: Conversation formatting
// ============================================================================

TEST_F(ChatTemplateTest, TPL006_ChatMLFormatConversation) {
    std::vector<Message> messages = {
        Message(Message::SYSTEM, "You are a helpful assistant.", 0),
        Message(Message::USER, "Hello!", 0),
        Message(Message::ASSISTANT, "Hi! How can I help?", 0),
        Message(Message::USER, "What's the weather?", 0)
    };

    std::string formatted = chatml->format_conversation(messages, {}, true);

    // Check order and structure
    EXPECT_TRUE(formatted.find("<|im_start|>system") != std::string::npos);
    EXPECT_TRUE(formatted.find("Hello!") != std::string::npos);
    EXPECT_TRUE(formatted.find("Hi! How can I help?") != std::string::npos);
    EXPECT_TRUE(formatted.find("What's the weather?") != std::string::npos);

    // Check generation prompt is added
    size_t last_prompt_pos = formatted.rfind("<|im_start|>assistant\n");
    EXPECT_NE(last_prompt_pos, std::string::npos)
        << "Generation prompt should be at the end";
}

TEST_F(ChatTemplateTest, TPL006_Llama3FormatConversation) {
    std::vector<Message> messages = {
        Message(Message::SYSTEM, "You are an AI.", 0),
        Message(Message::USER, "Hi there", 0),
    };

    std::string formatted = llama3->format_conversation(messages, {}, true);

    EXPECT_TRUE(formatted.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|start_header_id|>user<|end_header_id|>") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos)
        << "Generation prompt should be added at end";
}

TEST_F(ChatTemplateTest, TPL006_NoGenerationPrompt) {
    std::vector<Message> messages = {
        Message(Message::USER, "Test", 0)
    };

    std::string with_prompt = chatml->format_conversation(messages, {}, true);
    std::string without_prompt = chatml->format_conversation(messages, {}, false);

    EXPECT_TRUE(with_prompt.find("<|im_start|>assistant\n") != std::string::npos);
    EXPECT_TRUE(without_prompt.find("<|im_start|>assistant\n") == std::string::npos
                || without_prompt.rfind("<|im_start|>assistant") < without_prompt.rfind("<|im_end|>"))
        << "Without generation prompt, no trailing assistant prompt";
}

// ============================================================================
// TPL-007: Incremental rendering
// ============================================================================

TEST_F(ChatTemplateTest, TPL007_IncrementalRendering) {
    std::vector<Message> messages = {
        Message(Message::SYSTEM, "System prompt", 0),
        Message(Message::USER, "First user message", 0),
        Message(Message::ASSISTANT, "First assistant response", 0),
        Message(Message::USER, "Second user message", 0)
    };

    // Render incrementally - should return just the new message
    std::string incremental = chatml->format_message_incremental(messages, 3, {}, true);

    // Should contain the new user message and generation prompt
    EXPECT_TRUE(incremental.find("Second user message") != std::string::npos)
        << "Incremental render should contain the new message";
    EXPECT_TRUE(incremental.find("<|im_start|>assistant\n") != std::string::npos)
        << "Incremental render should include generation prompt";

    // Should NOT contain previous messages (they're in the prefix)
    // This is implementation-dependent - some templates may include more context
    // but the key is that incremental is shorter than full conversation
    std::string full = chatml->format_conversation(messages, {}, true);
    EXPECT_LT(incremental.length(), full.length())
        << "Incremental render should be shorter than full conversation";
}

TEST_F(ChatTemplateTest, TPL007_IncrementalFirstMessage) {
    std::vector<Message> messages = {
        Message(Message::USER, "First message", 0)
    };

    std::string incremental = chatml->format_message_incremental(messages, 0, {}, false);

    EXPECT_TRUE(incremental.find("First message") != std::string::npos);
}

// ============================================================================
// Model Family Tests
// ============================================================================

TEST_F(ChatTemplateTest, GetFamilyChatML) {
    EXPECT_EQ(chatml->get_family(), ModelFamily::QWEN_2_X);
}

TEST_F(ChatTemplateTest, GetFamilyLlama2) {
    EXPECT_EQ(llama2->get_family(), ModelFamily::LLAMA_2_X);
}

TEST_F(ChatTemplateTest, GetFamilyLlama3) {
    EXPECT_EQ(llama3->get_family(), ModelFamily::LLAMA_3_X);
}

TEST_F(ChatTemplateTest, GetFamilyGLM4) {
    EXPECT_EQ(glm4->get_family(), ModelFamily::GLM_4);
}

// ============================================================================
// Llama2 Specific Tests
// ============================================================================

TEST_F(ChatTemplateTest, Llama2FormatUserMessage) {
    Message msg(Message::USER, "Tell me a joke", 0);

    std::string formatted = llama2->format_message(msg);

    EXPECT_TRUE(formatted.find("[INST]") != std::string::npos)
        << "Llama2 user message should have [INST] tag";
    EXPECT_TRUE(formatted.find("[/INST]") != std::string::npos)
        << "Llama2 user message should have [/INST] tag";
    EXPECT_TRUE(formatted.find("Tell me a joke") != std::string::npos);
}

TEST_F(ChatTemplateTest, Llama2FormatAssistantMessage) {
    Message msg(Message::ASSISTANT, "Why did the chicken cross the road?", 0);

    std::string formatted = llama2->format_message(msg);

    EXPECT_TRUE(formatted.find("</s>") != std::string::npos)
        << "Llama2 assistant message should end with </s>";
    EXPECT_TRUE(formatted.find("chicken") != std::string::npos);
}

TEST_F(ChatTemplateTest, Llama2FormatSystemMessage) {
    std::string formatted = llama2->format_system_message("You are a comedian.", {});

    EXPECT_TRUE(formatted.find("<<SYS>>") != std::string::npos)
        << "Llama2 system should use <<SYS>> tags";
    EXPECT_TRUE(formatted.find("<</SYS>>") != std::string::npos);
    EXPECT_TRUE(formatted.find("You are a comedian.") != std::string::npos);
}

// ============================================================================
// GLM4 Specific Tests
// ============================================================================

TEST_F(ChatTemplateTest, GLM4FormatUserMessage) {
    Message msg(Message::USER, "What is 2+2?", 0);

    std::string formatted = glm4->format_message(msg);

    EXPECT_TRUE(formatted.find("<|user|>") != std::string::npos)
        << "GLM4 user message should have <|user|> tag";
    EXPECT_TRUE(formatted.find("2+2") != std::string::npos);
}

TEST_F(ChatTemplateTest, GLM4FormatToolResponse) {
    Message msg(Message::TOOL_RESPONSE, "Calculation result: 4", 0);
    msg.tool_name = "calculator";

    std::string formatted = glm4->format_message(msg);

    // GLM4 uses "observation" role for tool responses
    EXPECT_TRUE(formatted.find("<|observation|>") != std::string::npos)
        << "GLM4 tool response should use 'observation' role";
    EXPECT_TRUE(formatted.find("Calculation result: 4") != std::string::npos);
}

// ============================================================================
// ChatTemplateFactory Tests
// ============================================================================

TEST_F(ChatTemplateTest, FactoryCreateChatMLForQwen2) {
    ModelConfig config = ModelConfig::create_qwen_2x();

    auto template_ptr = ChatTemplateFactory::create("", config, nullptr, "", "");

    ASSERT_NE(template_ptr, nullptr);
    EXPECT_EQ(template_ptr->get_family(), ModelFamily::QWEN_2_X);
}

TEST_F(ChatTemplateTest, FactoryCreateLlama2) {
    ModelConfig config = ModelConfig::create_llama_2x();

    auto template_ptr = ChatTemplateFactory::create("", config, nullptr, "", "");

    ASSERT_NE(template_ptr, nullptr);
    EXPECT_EQ(template_ptr->get_family(), ModelFamily::LLAMA_2_X);
}

TEST_F(ChatTemplateTest, FactoryCreateLlama3) {
    ModelConfig config = ModelConfig::create_llama_3x();

    auto template_ptr = ChatTemplateFactory::create("", config, nullptr, "", "");

    ASSERT_NE(template_ptr, nullptr);
    EXPECT_EQ(template_ptr->get_family(), ModelFamily::LLAMA_3_X);
}

TEST_F(ChatTemplateTest, FactoryCreateGLM4) {
    ModelConfig config = ModelConfig::create_glm_4();

    auto template_ptr = ChatTemplateFactory::create("", config, nullptr, "", "");

    ASSERT_NE(template_ptr, nullptr);
    EXPECT_EQ(template_ptr->get_family(), ModelFamily::GLM_4);
}

// ============================================================================
// Capability Probing Tests (hardcoded templates)
// ============================================================================

TEST_F(ChatTemplateTest, DefaultCapabilities) {
    // Hardcoded templates have default capabilities
    chatml->probe_capabilities();
    auto caps = chatml->get_capabilities();

    EXPECT_TRUE(caps.probed) << "Capabilities should be marked as probed";
    EXPECT_TRUE(caps.supports_system_role) << "Default should support system role";
}

// ============================================================================
// Content Extraction Tests
// ============================================================================

TEST_F(ChatTemplateTest, ExtractContentNoChannels) {
    // For non-channel templates, extract_content should pass through
    std::string raw_output = "This is the model's response.";

    std::string extracted = chatml->extract_content(raw_output);

    EXPECT_EQ(extracted, raw_output)
        << "Non-channel templates should pass through content unchanged";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ChatTemplateTest, EmptyMessage) {
    Message msg(Message::USER, "", 0);

    std::string formatted = chatml->format_message(msg);

    // Should still have the structure even with empty content
    EXPECT_TRUE(formatted.find("<|im_start|>user") != std::string::npos);
    EXPECT_TRUE(formatted.find("<|im_end|>") != std::string::npos);
}

TEST_F(ChatTemplateTest, EmptySystemMessage) {
    std::string formatted = chatml->format_system_message("", {});

    // ChatML provides a default system message
    EXPECT_TRUE(formatted.find("Qwen") != std::string::npos ||
                formatted.find("assistant") != std::string::npos)
        << "Empty system message should get default content";
}

TEST_F(ChatTemplateTest, MessageWithNewlines) {
    Message msg(Message::USER, "Line 1\nLine 2\nLine 3", 0);

    std::string formatted = chatml->format_message(msg);

    EXPECT_TRUE(formatted.find("Line 1\nLine 2\nLine 3") != std::string::npos)
        << "Newlines should be preserved in message content";
}

TEST_F(ChatTemplateTest, MessageWithSpecialCharacters) {
    Message msg(Message::USER, "Test with <tags> and \"quotes\" and 'apostrophes'", 0);

    std::string formatted = chatml->format_message(msg);

    // Special characters should be preserved
    EXPECT_TRUE(formatted.find("<tags>") != std::string::npos);
    EXPECT_TRUE(formatted.find("\"quotes\"") != std::string::npos);
}
