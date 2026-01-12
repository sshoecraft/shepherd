#include <gtest/gtest.h>
#include "session.h"

// =============================================================================
// calculate_truncation_scale tests
// =============================================================================

TEST(SessionTest, TruncationScaleSmallContext) {
    // 8k context should be around 0.33-0.36
    double scale = calculate_truncation_scale(8192);
    EXPECT_GE(scale, 0.33);
    EXPECT_LE(scale, 0.40);
}

TEST(SessionTest, TruncationScaleMediumContext) {
    // 32k context should be around 0.43-0.46
    double scale = calculate_truncation_scale(32768);
    EXPECT_GE(scale, 0.40);
    EXPECT_LE(scale, 0.50);
}

TEST(SessionTest, TruncationScaleLargeContext) {
    // 64k context should be around 0.54-0.60
    double scale = calculate_truncation_scale(65536);
    EXPECT_GE(scale, 0.50);
    EXPECT_LE(scale, 0.60);
}

TEST(SessionTest, TruncationScaleMinimum) {
    // Very small context should not go below 0.33
    double scale = calculate_truncation_scale(1000);
    EXPECT_GE(scale, 0.33);
}

TEST(SessionTest, TruncationScaleMaximum) {
    // Very large context should not exceed 0.60
    double scale = calculate_truncation_scale(200000);
    EXPECT_LE(scale, 0.60);
}

// =============================================================================
// calculate_desired_completion_tokens tests
// =============================================================================

TEST(SessionTest, DesiredCompletionTokensZeroContext) {
    int tokens = calculate_desired_completion_tokens(0, 4096);
    EXPECT_EQ(tokens, 0);
}

TEST(SessionTest, DesiredCompletionTokensSmallContext) {
    // 8k context, 4096 max output
    int tokens = calculate_desired_completion_tokens(8192, 4096);
    EXPECT_GT(tokens, 1000);
    EXPECT_LE(tokens, 4096);
}

TEST(SessionTest, DesiredCompletionTokensMediumContext) {
    // 16k context
    int tokens = calculate_desired_completion_tokens(16384, 4096);
    EXPECT_GT(tokens, 2000);
    EXPECT_LE(tokens, 4096);
}

TEST(SessionTest, DesiredCompletionTokensLargeContext) {
    // 32k+ context should cap at model limit
    int tokens = calculate_desired_completion_tokens(65536, 4096);
    EXPECT_EQ(tokens, 4096);  // Capped at max_output_tokens
}

TEST(SessionTest, DesiredCompletionTokensNoLimit) {
    // No max output limit
    int tokens = calculate_desired_completion_tokens(32768, 0);
    EXPECT_GT(tokens, 0);
    // Without cap, it should be calculated value
}

// =============================================================================
// Session basic operations tests
// =============================================================================

TEST(SessionTest, DefaultState) {
    Session session;

    EXPECT_TRUE(session.messages.empty());
    EXPECT_TRUE(session.system_message.empty());
    EXPECT_EQ(session.system_message_tokens, 0);
    EXPECT_EQ(session.total_tokens, 0);
    EXPECT_EQ(session.last_prompt_tokens, 0);
    EXPECT_EQ(session.last_user_message_index, -1);
    EXPECT_EQ(session.last_assistant_message_index, -1);
    EXPECT_FALSE(session.auto_evict);
    EXPECT_EQ(session.desired_completion_tokens, 0);
    EXPECT_TRUE(session.tools.empty());
}

TEST(SessionTest, Clear) {
    Session session;

    // Add some state
    session.messages.push_back(Message(Message::USER, "hello", 10));
    session.messages.push_back(Message(Message::ASSISTANT, "world", 10));
    session.total_tokens = 100;
    session.last_prompt_tokens = 50;
    session.last_user_message_index = 0;
    session.last_assistant_message_index = 1;

    session.clear();

    EXPECT_TRUE(session.messages.empty());
    EXPECT_EQ(session.total_tokens, 0);
    EXPECT_EQ(session.last_prompt_tokens, 0);
    EXPECT_EQ(session.last_user_message_index, -1);
    EXPECT_EQ(session.last_assistant_message_index, -1);
}

// =============================================================================
// Session::Tool tests
// =============================================================================

TEST(SessionToolTest, ParametersJson) {
    Session::Tool tool;
    tool.name = "test_tool";
    tool.description = "A test tool";
    tool.parameters = nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    })");

    std::string json_str = tool.parameters_json();
    EXPECT_TRUE(json_str.find("type") != std::string::npos);
    EXPECT_TRUE(json_str.find("object") != std::string::npos);
}

TEST(SessionToolTest, ParametersText) {
    Session::Tool tool;
    tool.name = "search";
    tool.description = "Search for something";
    tool.parameters = nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"}
        }
    })");

    std::string text = tool.parameters_text();
    EXPECT_TRUE(text.find("query") != std::string::npos);
    EXPECT_TRUE(text.find("string") != std::string::npos);
    EXPECT_TRUE(text.find("count") != std::string::npos);
    EXPECT_TRUE(text.find("integer") != std::string::npos);
}

TEST(SessionToolTest, ParametersTextEmpty) {
    Session::Tool tool;
    tool.name = "simple";
    tool.parameters = nlohmann::json::object();

    std::string text = tool.parameters_text();
    EXPECT_EQ(text, "none");
}

TEST(SessionToolTest, ParametersTextNoType) {
    Session::Tool tool;
    tool.parameters = nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
            "value": {}
        }
    })");

    std::string text = tool.parameters_text();
    EXPECT_TRUE(text.find("any") != std::string::npos);
}

// =============================================================================
// Sampling parameter defaults
// =============================================================================

TEST(SessionTest, SamplingParameterDefaults) {
    Session session;

    // All sampling parameters should be "unset" (negative values)
    EXPECT_LT(session.sampling.temperature, 0);
    EXPECT_LT(session.sampling.top_p, 0);
    EXPECT_LT(session.sampling.top_k, 0);
    EXPECT_LT(session.sampling.min_p, 0);
    EXPECT_LT(session.sampling.repetition_penalty, 0);
    EXPECT_EQ(session.sampling.presence_penalty, -999.0f);
    EXPECT_EQ(session.sampling.frequency_penalty, -999.0f);
    EXPECT_EQ(session.sampling.length_penalty, -999.0f);
    EXPECT_LT(session.sampling.no_repeat_ngram_size, 0);
}

// =============================================================================
// Message queue tests (deque operations)
// =============================================================================

TEST(SessionTest, MessageDequeOperations) {
    Session session;

    // Add messages
    session.messages.push_back(Message(Message::USER, "first", 5));
    session.messages.push_back(Message(Message::ASSISTANT, "second", 10));
    session.messages.push_back(Message(Message::USER, "third", 7));

    EXPECT_EQ(session.messages.size(), 3u);

    // Front and back
    EXPECT_EQ(session.messages.front().content, "first");
    EXPECT_EQ(session.messages.back().content, "third");

    // Index access
    EXPECT_EQ(session.messages[1].content, "second");
    EXPECT_EQ(session.messages[1].tokens, 10);
}

TEST(SessionTest, MessageTokenSum) {
    Session session;

    session.messages.push_back(Message(Message::USER, "msg1", 10));
    session.messages.push_back(Message(Message::ASSISTANT, "msg2", 20));
    session.messages.push_back(Message(Message::USER, "msg3", 15));

    // Calculate total manually
    int sum = 0;
    for (const auto& msg : session.messages) {
        sum += msg.tokens;
    }
    EXPECT_EQ(sum, 45);
}

// =============================================================================
// Eviction calculation tests (without backend)
// Note: Full eviction tests require a backend, these test the calculation logic
// =============================================================================

TEST(SessionTest, CalculateEvictionEmptySession) {
    Session session;

    // Empty session should return empty ranges
    auto ranges = session.calculate_messages_to_evict(100);
    EXPECT_TRUE(ranges.empty());
}

TEST(SessionTest, CalculateEvictionNoProtectedMessages) {
    Session session;

    // Add messages without setting protected indices
    session.messages.push_back(Message(Message::USER, "old1", 50));
    session.messages.push_back(Message(Message::ASSISTANT, "old2", 50));
    session.messages.push_back(Message(Message::USER, "new1", 50));
    session.messages.push_back(Message(Message::ASSISTANT, "new2", 50));

    // Don't set last_user/assistant indices - all can be evicted
    // Request to free 100 tokens
    auto ranges = session.calculate_messages_to_evict(100);

    // Should return ranges covering at least 100 tokens worth
    if (!ranges.empty()) {
        int evicted_tokens = 0;
        for (const auto& range : ranges) {
            for (int i = range.first; i <= range.second; i++) {
                evicted_tokens += session.messages[i].tokens;
            }
        }
        EXPECT_GE(evicted_tokens, 100);
    }
}

// =============================================================================
// System message tests
// =============================================================================

TEST(SessionTest, SystemMessage) {
    Session session;

    session.system_message = "You are a helpful assistant.";
    session.system_message_tokens = 10;

    EXPECT_EQ(session.system_message, "You are a helpful assistant.");
    EXPECT_EQ(session.system_message_tokens, 10);
}

// =============================================================================
// Tools vector tests
// =============================================================================

TEST(SessionTest, ToolsVector) {
    Session session;

    Session::Tool tool1;
    tool1.name = "read_file";
    tool1.description = "Read file contents";
    tool1.parameters = nlohmann::json::object();

    Session::Tool tool2;
    tool2.name = "write_file";
    tool2.description = "Write to file";
    tool2.parameters = nlohmann::json::object();

    session.tools.push_back(tool1);
    session.tools.push_back(tool2);

    EXPECT_EQ(session.tools.size(), 2u);
    EXPECT_EQ(session.tools[0].name, "read_file");
    EXPECT_EQ(session.tools[1].name, "write_file");
}
