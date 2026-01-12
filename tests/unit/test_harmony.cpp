#include <gtest/gtest.h>
#include "backends/harmony.h"
#include <vector>
#include <string>

using namespace Harmony;

class HarmonyParserTest : public ::testing::Test {
protected:
    Parser parser;

    void SetUp() override {
        parser.reset();
    }
};

// =============================================================================
// Basic parsing tests
// =============================================================================

TEST_F(HarmonyParserTest, ParseSimpleFinalChannel) {
    std::string input = "<|start|>assistant<|channel|>final<|message|>Hello World<|end|>";

    auto result = parser.parse(input);

    EXPECT_EQ(result.content, "Hello World");
    EXPECT_TRUE(result.reasoning_content.empty());
    EXPECT_TRUE(result.tool_calls.empty());
}

TEST_F(HarmonyParserTest, ParseAnalysisThenFinal) {
    std::string input =
        "<|start|>assistant<|channel|>analysis<|message|>Thinking...<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Answer<|end|>";

    auto result = parser.parse(input);

    EXPECT_EQ(result.reasoning_content, "Thinking...");
    EXPECT_EQ(result.content, "Answer");
}

TEST_F(HarmonyParserTest, ParseWithoutChannelMarker) {
    // Messages without explicit channel should be parsed as content
    std::string input = "<|start|>assistant<|message|>Simple response<|end|>";

    auto result = parser.parse(input);

    // Without channel marker, content should be captured after the message marker
    EXPECT_FALSE(result.content.empty() || result.reasoning_content.empty());
}

// =============================================================================
// Streaming tests
// =============================================================================

TEST_F(HarmonyParserTest, StreamingFeed) {
    parser.reset();

    parser.feed("<|start|>assistant<|channel|>final<|message|>");
    parser.feed("Hello ");
    parser.feed("World");
    parser.feed("<|end|>");

    auto result = parser.get_partial_result();
    EXPECT_EQ(result.content, "Hello World");
}

TEST_F(HarmonyParserTest, StreamingContentDelta) {
    parser.reset();

    parser.feed("<|start|>assistant<|channel|>final<|message|>Hello");

    std::string delta1 = parser.consume_content_delta();
    EXPECT_EQ(delta1, "Hello");

    parser.feed(" World");
    std::string delta2 = parser.consume_content_delta();
    EXPECT_EQ(delta2, " World");

    parser.feed("<|end|>");
    std::string delta3 = parser.consume_content_delta();
    EXPECT_TRUE(delta3.empty());
}

// =============================================================================
// Tool call tests
// =============================================================================

TEST_F(HarmonyParserTest, ParseToolCall) {
    std::string input =
        "<|start|>assistant to=functions.my_tool<|channel|>commentary<|message|>"
        "{\"arg\": \"value\"}<|end|>";

    auto result = parser.parse(input);

    ASSERT_EQ(result.tool_calls.size(), 1);
    EXPECT_EQ(result.tool_calls[0].name, "my_tool");
    EXPECT_TRUE(result.tool_calls[0].arguments.find("\"arg\"") != std::string::npos);
}

TEST_F(HarmonyParserTest, ParseToolCallPattern2) {
    // Pattern 2: <|channel|>... to=functions.X
    std::string input =
        "<|start|>assistant<|channel|>commentary to=functions.search<|message|>"
        "{\"query\": \"test\"}<|end|>";

    auto result = parser.parse(input);

    ASSERT_EQ(result.tool_calls.size(), 1);
    EXPECT_EQ(result.tool_calls[0].name, "search");
}

TEST_F(HarmonyParserTest, ParseCommentaryWithoutRecipient) {
    std::string input =
        "<|start|>assistant<|channel|>commentary<|message|>Some preamble<|end|>";

    auto result = parser.parse(input);

    // Commentary without recipient should go to content
    EXPECT_EQ(result.content, "Some preamble");
    EXPECT_TRUE(result.tool_calls.empty());
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_F(HarmonyParserTest, EmptyInput) {
    auto result = parser.parse("");

    EXPECT_TRUE(result.content.empty());
    EXPECT_TRUE(result.reasoning_content.empty());
    EXPECT_TRUE(result.tool_calls.empty());
}

TEST_F(HarmonyParserTest, ContentWithNewlines) {
    std::string input =
        "<|start|>assistant<|channel|>final<|message|>Line1\nLine2\nLine3<|end|>";

    auto result = parser.parse(input);

    EXPECT_EQ(result.content, "Line1\nLine2\nLine3");
}

TEST_F(HarmonyParserTest, ContentWithSpecialCharacters) {
    std::string input =
        "<|start|>assistant<|channel|>final<|message|>Test <tag> & \"quotes\"<|end|>";

    auto result = parser.parse(input);

    EXPECT_EQ(result.content, "Test <tag> & \"quotes\"");
}

// =============================================================================
// Partial parsing
// =============================================================================

TEST_F(HarmonyParserTest, PartialParsing) {
    std::string incomplete =
        "<|start|>assistant<|channel|>final<|message|>Hello";

    ParseOptions opts;
    opts.is_partial = true;

    auto result = parser.parse(incomplete, opts);

    // Partial parsing should capture what's available
    EXPECT_EQ(result.content, "Hello");
}

// =============================================================================
// Real-world GPT-OSS patterns
// =============================================================================

TEST_F(HarmonyParserTest, RealWorldGptOssPattern) {
    std::string model_output =
        "<|start|>assistant<|channel|>analysis<|message|>"
        "We need to evaluate the options carefully. Option A seems correct.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>A<|end|>";

    auto result = parser.parse(model_output);

    EXPECT_EQ(result.reasoning_content, "We need to evaluate the options carefully. Option A seems correct.");
    EXPECT_EQ(result.content, "A");
}

TEST_F(HarmonyParserTest, RealWorldWithToolCall) {
    std::string model_output =
        "<|start|>assistant<|channel|>analysis<|message|>I need to search for this.<|end|>"
        "<|start|>assistant to=functions.web_search<|channel|>commentary<|message|>"
        "{\"query\": \"test query\"}<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Here are the results.<|end|>";

    auto result = parser.parse(model_output);

    EXPECT_EQ(result.reasoning_content, "I need to search for this.");
    EXPECT_EQ(result.content, "Here are the results.");
    ASSERT_EQ(result.tool_calls.size(), 1);
    EXPECT_EQ(result.tool_calls[0].name, "web_search");
}

TEST_F(HarmonyParserTest, ResetParser) {
    std::string input1 = "<|start|>assistant<|channel|>final<|message|>First<|end|>";
    auto result1 = parser.parse(input1);
    EXPECT_EQ(result1.content, "First");

    parser.reset();

    std::string input2 = "<|start|>assistant<|channel|>final<|message|>Second<|end|>";
    auto result2 = parser.parse(input2);
    EXPECT_EQ(result2.content, "Second");
}
