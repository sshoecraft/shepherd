#include <gtest/gtest.h>
#include "message.h"
#include <sstream>

// =============================================================================
// Role conversion tests
// =============================================================================

TEST(MessageTest, StringToRoleSystem) {
    EXPECT_EQ(Message::stringToRole("system"), Message::SYSTEM);
}

TEST(MessageTest, StringToRoleUser) {
    EXPECT_EQ(Message::stringToRole("user"), Message::USER);
}

TEST(MessageTest, StringToRoleAssistant) {
    EXPECT_EQ(Message::stringToRole("assistant"), Message::ASSISTANT);
}

TEST(MessageTest, StringToRoleTool) {
    EXPECT_EQ(Message::stringToRole("tool"), Message::TOOL_RESPONSE);
}

TEST(MessageTest, StringToRoleFunction) {
    EXPECT_EQ(Message::stringToRole("function"), Message::FUNCTION);
}

TEST(MessageTest, StringToRoleUnknown) {
    // Unknown role defaults to USER
    EXPECT_EQ(Message::stringToRole("unknown"), Message::USER);
    EXPECT_EQ(Message::stringToRole(""), Message::USER);
    EXPECT_EQ(Message::stringToRole("invalid"), Message::USER);
}

// =============================================================================
// get_role tests
// =============================================================================

TEST(MessageTest, GetRoleSystem) {
    Message msg(Message::SYSTEM, "system message");
    EXPECT_EQ(msg.get_role(), "system");
}

TEST(MessageTest, GetRoleUser) {
    Message msg(Message::USER, "user message");
    EXPECT_EQ(msg.get_role(), "user");
}

TEST(MessageTest, GetRoleAssistant) {
    Message msg(Message::ASSISTANT, "assistant message");
    EXPECT_EQ(msg.get_role(), "assistant");
}

TEST(MessageTest, GetRoleToolResponse) {
    Message msg(Message::TOOL_RESPONSE, "tool result");
    EXPECT_EQ(msg.get_role(), "tool");
}

TEST(MessageTest, GetRoleFunction) {
    Message msg(Message::FUNCTION, "function result");
    EXPECT_EQ(msg.get_role(), "function");
}

// =============================================================================
// is_tool_response tests
// =============================================================================

TEST(MessageTest, IsToolResponseTrue) {
    Message tool_msg(Message::TOOL_RESPONSE, "result");
    EXPECT_TRUE(tool_msg.is_tool_response());

    Message func_msg(Message::FUNCTION, "result");
    EXPECT_TRUE(func_msg.is_tool_response());
}

TEST(MessageTest, IsToolResponseFalse) {
    Message user_msg(Message::USER, "hello");
    EXPECT_FALSE(user_msg.is_tool_response());

    Message assistant_msg(Message::ASSISTANT, "response");
    EXPECT_FALSE(assistant_msg.is_tool_response());

    Message system_msg(Message::SYSTEM, "prompt");
    EXPECT_FALSE(system_msg.is_tool_response());
}

// =============================================================================
// Constructor tests
// =============================================================================

TEST(MessageTest, ConstructorBasic) {
    Message msg(Message::USER, "hello world");

    EXPECT_EQ(msg.role, Message::USER);
    EXPECT_EQ(msg.content, "hello world");
    EXPECT_EQ(msg.tokens, 0);  // Default
    EXPECT_TRUE(msg.tool_name.empty());
    EXPECT_TRUE(msg.tool_call_id.empty());
    EXPECT_TRUE(msg.tool_calls_json.empty());
}

TEST(MessageTest, ConstructorWithTokens) {
    Message msg(Message::ASSISTANT, "response", 50);

    EXPECT_EQ(msg.role, Message::ASSISTANT);
    EXPECT_EQ(msg.content, "response");
    EXPECT_EQ(msg.tokens, 50);
}

// =============================================================================
// Optional fields tests
// =============================================================================

TEST(MessageTest, ToolFields) {
    Message msg(Message::TOOL_RESPONSE, "result", 10);
    msg.tool_name = "read_file";
    msg.tool_call_id = "call_123";

    EXPECT_EQ(msg.tool_name, "read_file");
    EXPECT_EQ(msg.tool_call_id, "call_123");
}

TEST(MessageTest, ToolCallsJson) {
    Message msg(Message::ASSISTANT, "I'll read that file");
    msg.tool_calls_json = R"([{"id":"call_1","function":{"name":"read_file"}}])";

    EXPECT_FALSE(msg.tool_calls_json.empty());
}

// =============================================================================
// Stream operator tests
// =============================================================================

TEST(MessageTest, StreamOperatorBasic) {
    Message msg(Message::USER, "hello", 5);

    std::ostringstream oss;
    oss << msg;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("user") != std::string::npos);
    EXPECT_TRUE(output.find("5 tokens") != std::string::npos);
    EXPECT_TRUE(output.find("hello") != std::string::npos);
}

TEST(MessageTest, StreamOperatorLongContent) {
    // Content longer than 100 chars should be truncated
    std::string long_content(150, 'x');
    Message msg(Message::ASSISTANT, long_content, 100);

    std::ostringstream oss;
    oss << msg;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("...") != std::string::npos);  // Truncation indicator
}

TEST(MessageTest, StreamOperatorWithToolName) {
    Message msg(Message::TOOL_RESPONSE, "result", 10);
    msg.tool_name = "read_file";

    std::ostringstream oss;
    oss << msg;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("[tool: read_file]") != std::string::npos);
}

TEST(MessageTest, StreamOperatorWithToolCallId) {
    Message msg(Message::TOOL_RESPONSE, "result", 10);
    msg.tool_call_id = "call_abc123";

    std::ostringstream oss;
    oss << msg;

    std::string output = oss.str();
    EXPECT_TRUE(output.find("[tool_call_id: call_abc123]") != std::string::npos);
}

// =============================================================================
// Roundtrip tests (stringToRole <-> get_role)
// =============================================================================

TEST(MessageTest, RoleRoundtrip) {
    // All roles should roundtrip correctly
    std::vector<std::pair<Message::Role, std::string>> cases = {
        {Message::SYSTEM, "system"},
        {Message::USER, "user"},
        {Message::ASSISTANT, "assistant"},
        {Message::TOOL_RESPONSE, "tool"},
        {Message::FUNCTION, "function"}
    };

    for (const auto& [role, str] : cases) {
        Message msg(role, "content");
        std::string role_str = msg.get_role();
        Message::Role parsed = Message::stringToRole(role_str);

        // Note: TOOL_RESPONSE and FUNCTION both map to their own strings,
        // but stringToRole maps "tool" to TOOL_RESPONSE and "function" to FUNCTION
        if (role == Message::TOOL_RESPONSE) {
            EXPECT_EQ(parsed, Message::TOOL_RESPONSE);
        } else if (role == Message::FUNCTION) {
            EXPECT_EQ(parsed, Message::FUNCTION);
        } else {
            EXPECT_EQ(parsed, role);
        }
    }
}
