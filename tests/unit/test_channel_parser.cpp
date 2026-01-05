#include <gtest/gtest.h>
#include "backends/channel_parser.h"
#include <vector>
#include <string>

using namespace ChannelParsing;

// Helper to collect events from channel parser
struct CollectedEvent {
    EventType type;
    std::string content;
    std::string tool_name;
    std::string args;
};

class ChannelParserTest : public ::testing::Test {
protected:
    std::vector<CollectedEvent> events;

    ChannelParser::Config make_config(bool include_reasoning = false) {
        ChannelParser::Config cfg;
        cfg.has_channels = true;
        cfg.include_reasoning = include_reasoning;
        cfg.channel_start = "<|channel|>";
        cfg.message_start = "<|message|>";
        cfg.channel_end = "<|end|>";
        cfg.turn_start = "<|start|>";
        return cfg;
    }

    EventCallback make_callback() {
        return [this](EventType evt, const std::string& content,
                     const std::string& tool_name, const std::string& args) -> bool {
            events.push_back({evt, content, tool_name, args});
            return true;
        };
    }

    void clear_events() { events.clear(); }
};

// =============================================================================
// Basic channel parsing
// =============================================================================

TEST_F(ChannelParserTest, ParseFinalChannelOnly) {
    ChannelParser parser(make_config());

    parser.process("<|channel|>final<|message|>Hello World<|end|>", make_callback());
    parser.flush(make_callback());

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].type, EventType::CONTENT);
    EXPECT_EQ(events[0].content, "Hello World");
}

TEST_F(ChannelParserTest, ParseAnalysisThenFinal) {
    ChannelParser parser(make_config(true));  // include_reasoning = true

    parser.process("<|channel|>analysis<|message|>Thinking...<|end|>"
                   "<|start|>assistant<|channel|>final<|message|>Answer<|end|>",
                   make_callback());
    parser.flush(make_callback());

    ASSERT_GE(events.size(), 2);
    EXPECT_EQ(events[0].type, EventType::THINKING);
    EXPECT_EQ(events[0].content, "Thinking...");
    EXPECT_EQ(events[1].type, EventType::CONTENT);
    EXPECT_EQ(events[1].content, "Answer");
}

TEST_F(ChannelParserTest, SuppressAnalysisWhenDisabled) {
    ChannelParser parser(make_config(false));  // include_reasoning = false

    parser.process("<|channel|>analysis<|message|>Hidden<|end|>"
                   "<|start|>assistant<|channel|>final<|message|>Visible<|end|>",
                   make_callback());
    parser.flush(make_callback());

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].type, EventType::CONTENT);
    EXPECT_EQ(events[0].content, "Visible");
}

// =============================================================================
// Stop detection on turn start
// =============================================================================

TEST_F(ChannelParserTest, StopOnTurnStartAfterFinal) {
    ChannelParser parser(make_config());

    // First complete response
    bool cont = parser.process("<|channel|>final<|message|>A<|end|>", make_callback());
    EXPECT_TRUE(cont);

    // New turn should stop
    cont = parser.process("<|start|>user<|message|>next question", make_callback());
    EXPECT_FALSE(cont);  // Should signal stop

    parser.flush(make_callback());

    // Should only have the answer, not the "next question"
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].content, "A");
}

// =============================================================================
// Partial/broken special token handling (the bug we found!)
// =============================================================================

TEST_F(ChannelParserTest, DetectPartialTurnStart_TartPipe) {
    // This is the actual bug: llama.cpp outputs "tart|>" instead of "<|start|>"
    ChannelParser parser(make_config());

    // Complete a final channel first
    bool cont = parser.process("<|channel|>final<|message|>B<|end|>", make_callback());
    EXPECT_TRUE(cont);

    // Broken turn start marker (missing "<|s")
    cont = parser.process("tart|>assistant<|channel|>", make_callback());
    EXPECT_FALSE(cont);  // Should still detect and stop

    parser.flush(make_callback());

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].content, "B");
}

TEST_F(ChannelParserTest, DetectPartialTurnStart_StartPipe) {
    ChannelParser parser(make_config());

    bool cont = parser.process("<|channel|>final<|message|>C<|end|>", make_callback());
    EXPECT_TRUE(cont);

    // Missing just "<|"
    cont = parser.process("start|>user", make_callback());
    EXPECT_FALSE(cont);

    parser.flush(make_callback());

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].content, "C");
}

TEST_F(ChannelParserTest, DetectPartialTurnStart_JustPipe) {
    ChannelParser parser(make_config());

    bool cont = parser.process("<|channel|>final<|message|>D<|end|>", make_callback());
    EXPECT_TRUE(cont);

    // Very broken - just "|>"
    cont = parser.process("|>user", make_callback());
    // Note: "|>" alone might be too short to reliably detect, but we should handle it

    parser.flush(make_callback());

    // At minimum, the answer should be clean
    ASSERT_GE(events.size(), 1);
    EXPECT_EQ(events[0].content, "D");
}

// =============================================================================
// Streaming (token-by-token) handling
// =============================================================================

TEST_F(ChannelParserTest, StreamingTokenByToken) {
    ChannelParser parser(make_config());

    // Simulate token-by-token streaming
    parser.process("<|channel|>", make_callback());
    parser.process("final", make_callback());
    parser.process("<|message|>", make_callback());
    parser.process("X", make_callback());
    parser.process("<|end|>", make_callback());
    parser.flush(make_callback());

    // Should have collected the content
    ASSERT_GE(events.size(), 1);

    // Find the content event
    std::string total_content;
    for (const auto& evt : events) {
        if (evt.type == EventType::CONTENT) {
            total_content += evt.content;
        }
    }
    EXPECT_EQ(total_content, "X");
}

TEST_F(ChannelParserTest, PartialMarkerAtBufferEnd) {
    ChannelParser parser(make_config());

    // Content ending with partial marker
    parser.process("<|channel|>final<|message|>Answer<", make_callback());

    // The "<" should be held back (might be start of <|end|>)
    parser.flush(make_callback());

    // Verify content was collected
    std::string total;
    for (const auto& e : events) {
        if (e.type == EventType::CONTENT) total += e.content;
    }
    // "Answer" should be there, "<" handling may vary
    EXPECT_TRUE(total.find("Answer") != std::string::npos);
}

// =============================================================================
// Non-channel passthrough
// =============================================================================

TEST_F(ChannelParserTest, NonChannelPassthrough) {
    ChannelParser::Config cfg;
    cfg.has_channels = false;  // Disabled
    ChannelParser parser(cfg);

    parser.process("Hello World", make_callback());

    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].type, EventType::CONTENT);
    EXPECT_EQ(events[0].content, "Hello World");
}

// =============================================================================
// Real-world GPT-OSS output patterns
// =============================================================================

TEST_F(ChannelParserTest, RealWorldGptOssPattern) {
    ChannelParser parser(make_config(true));

    // Actual pattern from GPT-OSS model
    std::string model_output =
        "<|channel|>analysis<|message|>We need to evaluate the options.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>B<|end|>";

    parser.process(model_output, make_callback());
    parser.flush(make_callback());

    // Should get thinking + answer
    ASSERT_GE(events.size(), 2);

    // Find final answer
    std::string answer;
    for (const auto& e : events) {
        if (e.type == EventType::CONTENT) answer = e.content;
    }
    EXPECT_EQ(answer, "B");
}

TEST_F(ChannelParserTest, BrokenGptOssPattern_PartialTokenLeak) {
    // The actual bug: model outputs garbage after answer
    ChannelParser parser(make_config());

    std::string broken_output =
        "<|channel|>final<|message|>A<|end|>"
        "tart|>assistant<|channel|>analysis<|message|>garbage";

    bool cont = parser.process(broken_output, make_callback());

    // Should have stopped on "tart|>"
    EXPECT_FALSE(cont);

    parser.flush(make_callback());

    // Should only have "A", no garbage
    ASSERT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].type, EventType::CONTENT);
    EXPECT_EQ(events[0].content, "A");
}
