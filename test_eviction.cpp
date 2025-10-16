#include "context_manager.h"
#include "backends/api_backend.h"
#include "logger.h"
#include <iostream>
#include <string>
#include <cassert>

/// @brief Simple test context manager that doesn't need a full backend
class TestContextManager : public ApiContextManager {
public:
    explicit TestContextManager(size_t max_tokens) : ApiContextManager(max_tokens) {}
};

/// @brief Helper to print test header
void print_test(const std::string& name) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << "========================================\n";
}

/// @brief Helper to print context state
void print_context_state(TestContextManager& ctx) {
    std::cout << "Messages: " << ctx.get_message_count() << "\n";
    std::cout << "Tokens: " << ctx.get_total_tokens() << "/" << ctx.get_max_context_tokens() << "\n";
    std::cout << "Utilization: " << (ctx.get_context_utilization() * 100) << "%\n";

    // Print message types
    auto& messages = ctx.get_messages();
    std::cout << "Message sequence: ";
    for (const auto& msg : messages) {
        std::cout << msg.get_role()[0] << " ";  // Print first letter (s/u/a/t/f)
    }
    std::cout << "\n";
}

/// @brief Test 1: Normal eviction with multiple turns
void test_normal_eviction() {
    print_test("Normal Eviction - Multiple Turns");

    // Small context window to trigger evictions easily
    TestContextManager ctx(500);  // ~2000 characters

    // Add small system message
    Message sys(Message::SYSTEM, "You are a helpful assistant.",
                ctx.count_tokens("You are a helpful assistant."));
    ctx.add_message(sys);

    print_context_state(ctx);

    // Add multiple conversation turns
    for (int i = 0; i < 5; i++) {
        std::string user_text = "This is user message number " + std::to_string(i) +
                                ". It contains some text to take up space in the context window.";
        Message user(Message::USER, user_text, ctx.count_tokens(user_text));
        ctx.add_message(user);

        std::string assistant_text = "This is assistant response number " + std::to_string(i) +
                                      ". It also contains text to fill up the context window with meaningful content.";
        Message assistant(Message::ASSISTANT, assistant_text, ctx.count_tokens(assistant_text));
        ctx.add_message(assistant);

        std::cout << "\nAfter turn " << i << ":\n";
        print_context_state(ctx);
    }

    std::cout << "\n✓ Test completed - eviction should have occurred\n";
}

/// @brief Test 2: Edge case - only system + single user message
void test_system_user_only() {
    print_test("Edge Case - System + Single User Only");

    // Very small context window
    TestContextManager ctx(100);  // ~400 characters

    // Add large system message that takes most of the space
    std::string sys_text = "You are a helpful assistant. This is a large system prompt that takes up most of the available context window space. It contains important instructions.";
    Message sys(Message::SYSTEM, sys_text, ctx.count_tokens(sys_text));
    ctx.add_message(sys);

    std::cout << "After system message:\n";
    print_context_state(ctx);

    // Add user message that exceeds remaining space
    std::string user_text = "This is a user message that will cause the context to be full with only system and user messages present.";
    Message user(Message::USER, user_text, ctx.count_tokens(user_text));

    std::cout << "\nAdding user message that will trigger eviction...\n";
    ctx.add_message(user);

    std::cout << "\nAfter user message:\n";
    print_context_state(ctx);

    std::cout << "\n✓ This should show the 'Cannot evict' warning - expected behavior\n";
}

/// @brief Test 3: Single request fills remaining context after eviction
void test_large_request_after_eviction() {
    print_test("Large Request After Eviction");

    TestContextManager ctx(300);  // ~1200 characters

    // Add system message
    Message sys(Message::SYSTEM, "You are a helpful assistant.",
                ctx.count_tokens("You are a helpful assistant."));
    ctx.add_message(sys);

    // Add two conversation turns
    for (int i = 0; i < 2; i++) {
        std::string user_text = "User message " + std::to_string(i);
        Message user(Message::USER, user_text, ctx.count_tokens(user_text));
        ctx.add_message(user);

        std::string assistant_text = "Assistant response " + std::to_string(i) + " with some content";
        Message assistant(Message::ASSISTANT, assistant_text, ctx.count_tokens(assistant_text));
        ctx.add_message(assistant);
    }

    std::cout << "After 2 turns:\n";
    print_context_state(ctx);

    // Now add a very large user message that should trigger eviction
    std::string large_text;
    for (int i = 0; i < 50; i++) {
        large_text += "This is a very large user message that will require eviction of old content. ";
    }

    Message large_user(Message::USER, large_text, ctx.count_tokens(large_text));

    std::cout << "\nAdding large user message (" << large_user.token_count << " tokens)...\n";
    ctx.add_message(large_user);

    std::cout << "\nAfter large message:\n";
    print_context_state(ctx);

    std::cout << "\n✓ Eviction should have freed space for large message\n";
}

/// @brief Test 4: Tool call mini-turns eviction
void test_tool_mini_turns() {
    print_test("Tool Mini-Turns Eviction");

    TestContextManager ctx(400);

    // System message
    Message sys(Message::SYSTEM, "You are a helpful assistant.",
                ctx.count_tokens("You are a helpful assistant."));
    ctx.add_message(sys);

    // First turn with tool calls
    std::string user1 = "What's the weather?";
    ctx.add_message(Message(Message::USER, user1, ctx.count_tokens(user1)));

    // Tool call mini-turn 1
    std::string tool_call1 = "Calling weather API";
    Message asst1(Message::ASSISTANT, tool_call1, ctx.count_tokens(tool_call1));
    ctx.add_message(asst1);

    std::string tool_result1 = "Weather: sunny, 75F";
    Message tool1(Message::TOOL, tool_result1, ctx.count_tokens(tool_result1));
    tool1.tool_name = "get_weather";
    ctx.add_message(tool1);

    // Tool call mini-turn 2
    std::string tool_call2 = "Calling forecast API";
    Message asst2(Message::ASSISTANT, tool_call2, ctx.count_tokens(tool_call2));
    ctx.add_message(asst2);

    std::string tool_result2 = "Forecast: clear skies for 3 days";
    Message tool2(Message::TOOL, tool_result2, ctx.count_tokens(tool_result2));
    tool2.tool_name = "get_forecast";
    ctx.add_message(tool2);

    // Final assistant response
    std::string final_resp = "The weather is sunny and 75F. Forecast shows clear skies for the next 3 days.";
    ctx.add_message(Message(Message::ASSISTANT, final_resp, ctx.count_tokens(final_resp)));

    std::cout << "After first turn with tools:\n";
    print_context_state(ctx);

    // Add more turns to trigger eviction
    for (int i = 0; i < 3; i++) {
        std::string user_text = "Follow up question number " + std::to_string(i);
        ctx.add_message(Message(Message::USER, user_text, ctx.count_tokens(user_text)));

        std::string resp_text = "Response to follow up " + std::to_string(i);
        ctx.add_message(Message(Message::ASSISTANT, resp_text, ctx.count_tokens(resp_text)));
    }

    std::cout << "\nAfter additional turns (should trigger eviction):\n";
    print_context_state(ctx);

    std::cout << "\n✓ Tool mini-turns should be handled correctly\n";
}

/// @brief Test 5: Calculate eviction explicitly
void test_calculate_eviction() {
    print_test("Calculate Eviction Explicitly");

    TestContextManager ctx(500);

    // System message
    Message sys(Message::SYSTEM, "System prompt", ctx.count_tokens("System prompt"));
    ctx.add_message(sys);

    // Add 3 conversation turns
    for (int i = 0; i < 3; i++) {
        std::string user_text = "User message " + std::to_string(i) + " with some content to take space";
        ctx.add_message(Message(Message::USER, user_text, ctx.count_tokens(user_text)));

        std::string resp_text = "Assistant response " + std::to_string(i) + " with more content";
        ctx.add_message(Message(Message::ASSISTANT, resp_text, ctx.count_tokens(resp_text)));
    }

    std::cout << "Current state:\n";
    print_context_state(ctx);

    // Try to calculate eviction for different token amounts
    std::vector<int> test_amounts = {50, 100, 200, 500};

    for (int tokens_needed : test_amounts) {
        std::cout << "\nCalculating eviction for " << tokens_needed << " tokens:\n";
        auto [start, end] = ctx.calculate_messages_to_evict(tokens_needed);

        if (start != -1 && end != -1) {
            std::cout << "  Would evict messages [" << start << ", " << end << "]\n";
            int evicted_msgs = end - start + 1;
            std::cout << "  Total messages to evict: " << evicted_msgs << "\n";
        } else {
            std::cout << "  Cannot evict enough messages\n";
        }
    }

    std::cout << "\n✓ Eviction calculation test completed\n";
}

/// @brief Test 6: Manual eviction by index
void test_manual_eviction() {
    print_test("Manual Eviction By Index");

    TestContextManager ctx(500);

    // System message
    Message sys(Message::SYSTEM, "System", ctx.count_tokens("System"));
    ctx.add_message(sys);

    // Add turns
    for (int i = 0; i < 4; i++) {
        std::string user_text = "User " + std::to_string(i);
        ctx.add_message(Message(Message::USER, user_text, ctx.count_tokens(user_text)));

        std::string resp_text = "Response " + std::to_string(i);
        ctx.add_message(Message(Message::ASSISTANT, resp_text, ctx.count_tokens(resp_text)));
    }

    std::cout << "Before eviction:\n";
    print_context_state(ctx);

    // Manually evict first user-assistant pair (indices 1-2)
    std::cout << "\nManually evicting messages [1, 2]...\n";
    bool success = ctx.evict_messages_by_index(1, 2);

    std::cout << "Eviction " << (success ? "succeeded" : "failed") << "\n";
    std::cout << "\nAfter eviction:\n";
    print_context_state(ctx);

    std::cout << "\n✓ Manual eviction test completed\n";
}

int main(int argc, char* argv[]) {
    std::cout << "===========================================\n";
    std::cout << "Context Manager Eviction Test Suite\n";
    std::cout << "===========================================\n";

    try {
        // Run all tests
        test_normal_eviction();
        test_system_user_only();
        test_large_request_after_eviction();
        test_tool_mini_turns();
        test_calculate_eviction();
        test_manual_eviction();

        std::cout << "\n===========================================\n";
        std::cout << "All tests completed!\n";
        std::cout << "===========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
