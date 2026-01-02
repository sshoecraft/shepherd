#include <gtest/gtest.h>
#include "sse_parser.h"
#include <vector>

// =============================================================================
// Basic parsing tests
// =============================================================================

TEST(SSEParserTest, ParseDataOnly) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        received_data = data;
        return true;
    };

    bool result = parser.process_chunk("data: hello world\n\n", callback);
    EXPECT_TRUE(result);
    EXPECT_EQ(received_data, "hello world");
}

TEST(SSEParserTest, ParseEventAndData) {
    SSEParser parser;
    std::string received_event;
    std::string received_data;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        received_event = event;
        received_data = data;
        return true;
    };

    parser.process_chunk("event: message\ndata: test content\n\n", callback);
    EXPECT_EQ(received_event, "message");
    EXPECT_EQ(received_data, "test content");
}

TEST(SSEParserTest, ParseWithId) {
    SSEParser parser;
    std::string received_id;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        received_id = id;
        return true;
    };

    parser.process_chunk("id: 123\ndata: test\n\n", callback);
    EXPECT_EQ(received_id, "123");
}

// =============================================================================
// Multi-line data tests
// =============================================================================

TEST(SSEParserTest, ParseMultiLineData) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        received_data = data;
        return true;
    };

    parser.process_chunk("data: line 1\ndata: line 2\n\n", callback);
    EXPECT_EQ(received_data, "line 1\nline 2");
}

TEST(SSEParserTest, ParseThreeLineData) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        received_data = data;
        return true;
    };

    parser.process_chunk("data: a\ndata: b\ndata: c\n\n", callback);
    EXPECT_EQ(received_data, "a\nb\nc");
}

// =============================================================================
// Chunked data tests
// =============================================================================

TEST(SSEParserTest, ParseSplitAcrossChunks) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        events.push_back(data);
        return true;
    };

    // First chunk - incomplete
    parser.process_chunk("data: hel", callback);
    EXPECT_TRUE(events.empty());
    EXPECT_TRUE(parser.has_buffered_data());

    // Second chunk - completes the event
    parser.process_chunk("lo\n\n", callback);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "hello");
}

TEST(SSEParserTest, ParseMultipleEventsInOneChunk) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        events.push_back(data);
        return true;
    };

    parser.process_chunk("data: first\n\ndata: second\n\n", callback);
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0], "first");
    EXPECT_EQ(events[1], "second");
}

TEST(SSEParserTest, ParseManyEventsInOneChunk) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string& event, const std::string& data,
                       const std::string& id) {
        events.push_back(data);
        return true;
    };

    parser.process_chunk("data: 1\n\ndata: 2\n\ndata: 3\n\ndata: 4\n\n", callback);
    ASSERT_EQ(events.size(), 4u);
    EXPECT_EQ(events[0], "1");
    EXPECT_EQ(events[1], "2");
    EXPECT_EQ(events[2], "3");
    EXPECT_EQ(events[3], "4");
}

// =============================================================================
// Reset tests
// =============================================================================

TEST(SSEParserTest, Reset) {
    SSEParser parser;
    int count = 0;

    auto callback = [&](const std::string&, const std::string&, const std::string&) {
        count++;
        return true;
    };

    // Add partial data
    parser.process_chunk("data: partial", callback);
    EXPECT_TRUE(parser.has_buffered_data());

    // Reset should clear buffer
    parser.reset();
    EXPECT_FALSE(parser.has_buffered_data());
}

TEST(SSEParserTest, ResetAndContinue) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        events.push_back(data);
        return true;
    };

    // Partial event
    parser.process_chunk("data: will_be_lost", callback);
    EXPECT_TRUE(events.empty());

    // Reset
    parser.reset();

    // New complete event
    parser.process_chunk("data: new_event\n\n", callback);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "new_event");
}

// =============================================================================
// Callback cancellation tests
// =============================================================================

TEST(SSEParserTest, CallbackCancellation) {
    SSEParser parser;
    int count = 0;

    auto callback = [&](const std::string&, const std::string&, const std::string&) {
        count++;
        return count < 2;  // Stop after second event
    };

    bool result = parser.process_chunk("data: 1\n\ndata: 2\n\ndata: 3\n\n", callback);
    EXPECT_FALSE(result);  // Parsing was stopped
    EXPECT_EQ(count, 2);
}

TEST(SSEParserTest, CallbackCancellationFirstEvent) {
    SSEParser parser;
    int count = 0;

    auto callback = [&](const std::string&, const std::string&, const std::string&) {
        count++;
        return false;  // Stop immediately
    };

    bool result = parser.process_chunk("data: 1\n\ndata: 2\n\n", callback);
    EXPECT_FALSE(result);
    EXPECT_EQ(count, 1);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST(SSEParserTest, EmptyData) {
    SSEParser parser;
    std::string received_data = "not_set";

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        received_data = data;
        return true;
    };

    parser.process_chunk("data:\n\n", callback);
    EXPECT_EQ(received_data, "");
}

TEST(SSEParserTest, DataWithColon) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        received_data = data;
        return true;
    };

    parser.process_chunk("data: key: value\n\n", callback);
    EXPECT_EQ(received_data, "key: value");
}

TEST(SSEParserTest, DataWithLeadingSpace) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        received_data = data;
        return true;
    };

    // SSE spec: first space after colon is optional separator
    parser.process_chunk("data:  hello\n\n", callback);
    EXPECT_EQ(received_data, " hello");  // Second space preserved
}

TEST(SSEParserTest, CommentLine) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        events.push_back(data);
        return true;
    };

    // Comments (lines starting with :) should be ignored
    parser.process_chunk(": this is a comment\ndata: actual data\n\n", callback);
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events[0], "actual data");
}

TEST(SSEParserTest, UnknownField) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        received_data = data;
        return true;
    };

    // Unknown fields should be ignored
    parser.process_chunk("unknown: ignored\ndata: kept\n\n", callback);
    EXPECT_EQ(received_data, "kept");
}

TEST(SSEParserTest, CarriageReturnHandling) {
    SSEParser parser;
    std::string received_data;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        received_data = data;
        return true;
    };

    // Should handle \r\n line endings
    parser.process_chunk("data: test\r\n\r\n", callback);
    EXPECT_EQ(received_data, "test");
}

TEST(SSEParserTest, OpenAIFormat) {
    SSEParser parser;
    std::vector<std::string> events;

    auto callback = [&](const std::string&, const std::string& data, const std::string&) {
        events.push_back(data);
        return true;
    };

    // Simulate OpenAI streaming format
    std::string openai_stream =
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n"
        "data: [DONE]\n\n";

    parser.process_chunk(openai_stream, callback);
    ASSERT_EQ(events.size(), 3u);
    EXPECT_TRUE(events[0].find("Hello") != std::string::npos);
    EXPECT_TRUE(events[1].find("world") != std::string::npos);
    EXPECT_EQ(events[2], "[DONE]");
}

// =============================================================================
// has_buffered_data tests
// =============================================================================

TEST(SSEParserTest, HasBufferedDataInitial) {
    SSEParser parser;
    EXPECT_FALSE(parser.has_buffered_data());
}

TEST(SSEParserTest, HasBufferedDataAfterIncomplete) {
    SSEParser parser;

    auto callback = [](const std::string&, const std::string&, const std::string&) {
        return true;
    };

    parser.process_chunk("data: incomplete", callback);
    EXPECT_TRUE(parser.has_buffered_data());
}

TEST(SSEParserTest, HasBufferedDataAfterComplete) {
    SSEParser parser;

    auto callback = [](const std::string&, const std::string&, const std::string&) {
        return true;
    };

    parser.process_chunk("data: complete\n\n", callback);
    EXPECT_FALSE(parser.has_buffered_data());
}
