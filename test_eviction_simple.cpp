#include <iostream>
#include <string>
#include <deque>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <chrono>

// Minimal logger stubs
#define LOG_DEBUG(x) std::cout << "[DEBUG] " << x << std::endl
#define LOG_INFO(x) std::cout << "[INFO] " << x << std::endl
#define LOG_WARN(x) std::cout << "[WARN] " << x << std::endl
#define LOG_ERROR(x) std::cout << "[ERROR] " << x << std::endl

// Minimal RAG stub
struct ConversationTurn {
    std::string user_message;
    std::string assistant_message;

    ConversationTurn(const std::string& u, const std::string& a)
        : user_message(u), assistant_message(a) {}
};

class RAGManager {
public:
    static void archive_turn(const ConversationTurn& turn) {
        LOG_INFO("RAG: Archived turn: " + turn.user_message.substr(0, 50) + "... -> " + turn.assistant_message.substr(0, 50) + "...");
    }
};

// Copy the Message struct and ContextManager implementation directly
struct Message {
    enum Type {
        SYSTEM,
        USER,
        ASSISTANT,
        TOOL,
        FUNCTION
    };

    Type type;
    std::string content;
    int token_count;
    int64_t timestamp;
    bool in_kv_cache;

    std::string tool_name;
    std::string tool_call_id;

    Message(Type t, const std::string& c, int tokens = 0)
        : type(t), content(c), token_count(tokens), timestamp(get_current_timestamp()), in_kv_cache(false) {}

    std::string get_role() const {
        switch (type) {
            case SYSTEM: return "system";
            case USER: return "user";
            case ASSISTANT: return "assistant";
            case TOOL: return "tool";
            case FUNCTION: return "function";
            default: return "user";
        }
    }

private:
    static int64_t get_current_timestamp() {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

class ContextManagerError : public std::runtime_error {
public:
    explicit ContextManagerError(const std::string& message)
        : std::runtime_error("ContextManager: " + message) {}
};

class SimpleContextManager {
public:
    explicit SimpleContextManager(size_t max_context_tokens)
        : max_context_tokens_(max_context_tokens) {
        if (max_context_tokens_ < 512) {
            throw ContextManagerError("Context size too small");
        }
        LOG_DEBUG("ContextManager created with " + std::to_string(max_context_tokens_) + " max tokens");
    }

    void add_message(const Message& message) {
        // Check if we need to evict BEFORE adding
        if (needs_eviction(message.token_count)) {
            // Calculate how many tokens we need to free
            int tokens_over = get_total_tokens() + message.token_count - static_cast<int>(max_context_tokens_);

            // Try to calculate if eviction is possible
            auto [start_msg, end_msg] = calculate_messages_to_evict(tokens_over);

            if (start_msg == -1 || end_msg == -1) {
                // Cannot evict enough space for this message
                throw ContextManagerError(
                    "Cannot add message: would exceed context limit (" +
                    std::to_string(get_total_tokens() + message.token_count) + " > " +
                    std::to_string(max_context_tokens_) + " tokens) and no messages available for eviction. " +
                    "This usually happens when system messages + current user message exceed context size.");
            }

            // Eviction is possible, perform it directly using the calculated indices
            if (!evict_messages_by_index(start_msg, end_msg)) {
                throw ContextManagerError(
                    "Eviction failed unexpectedly when trying to free space for message");
            }

            // Verify we actually freed enough space
            if (needs_eviction(message.token_count)) {
                throw ContextManagerError(
                    "Eviction failed to free enough space for message (" +
                    std::to_string(message.token_count) + " tokens needed, " +
                    std::to_string(max_context_tokens_ - get_total_tokens()) + " tokens available)");
            }
        }

        // Add the message
        messages_.push_back(message);
        current_token_count_ += message.token_count;

        if (message.type == Message::SYSTEM && messages_.size() <= system_message_count_ + 1) {
            system_message_count_++;
        }

        LOG_DEBUG("Added " + message.get_role() + " message with " +
                  std::to_string(message.token_count) + " tokens");
        LOG_DEBUG("Total tokens: " + std::to_string(current_token_count_) + "/" +
                  std::to_string(max_context_tokens_));
    }

    size_t get_message_count() const { return messages_.size(); }
    size_t get_max_context_tokens() const { return max_context_tokens_; }
    int get_total_tokens() const { return current_token_count_ + calculate_json_overhead(); }
    double get_context_utilization() const {
        if (max_context_tokens_ == 0) return 0.0;
        return static_cast<double>(get_total_tokens()) / static_cast<double>(max_context_tokens_);
    }

    std::deque<Message>& get_messages() { return messages_; }

    std::pair<int, int> calculate_messages_to_evict(int tokens_needed, size_t max_evict_index = SIZE_MAX) const {
        int tokens_freed = 0;
        int start_index = -1;
        int end_index = -1;

        size_t evict_start = system_message_count_;

        // Find last USER message
        size_t last_user_index = evict_start;
        for (int i = static_cast<int>(messages_.size()) - 1; i >= static_cast<int>(evict_start); i--) {
            if (messages_[i].type == Message::USER) {
                last_user_index = i;
                break;
            }
        }

        size_t evict_boundary = std::min(max_evict_index, last_user_index);
        (void)evict_boundary;  // Unused for now

        if (evict_start >= last_user_index) {
            LOG_WARN("Cannot evict: only system messages or current user message available");
            return {-1, -1};
        }

        // PASS 1: Try to evict complete big-turns (USER → final ASSISTANT)
        LOG_DEBUG("Pass 1: Looking for complete big-turns to evict");
        for (size_t i = evict_start; i < last_user_index && tokens_freed < tokens_needed; ) {
            // Evict standalone eviction notices
            if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "context_eviction") {
                tokens_freed += messages_[i].token_count;
                if (start_index == -1) start_index = i;
                end_index = i;
                LOG_DEBUG("Pass 1: Evicting eviction notice at index " + std::to_string(i));
                i++;
                continue;
            }

            if (messages_[i].type != Message::USER) {
                LOG_WARN("Pass 1: Skipping non-USER message at index " + std::to_string(i));
                i++;
                continue;
            }

            // Found USER - scan forward to find final ASSISTANT response
            size_t turn_start = i;
            int turn_tokens = messages_[i].token_count;
            i++;

            size_t final_assistant_index = turn_start;
            bool found_final_assistant = false;

            while (i < last_user_index) {
                turn_tokens += messages_[i].token_count;

                if (messages_[i].type == Message::ASSISTANT) {
                    final_assistant_index = i;
                    found_final_assistant = true;
                    i++;

                    if (i >= last_user_index || messages_[i].type == Message::USER) {
                        break;
                    }
                } else if (messages_[i].type == Message::TOOL) {
                    i++;
                } else {
                    LOG_WARN("Pass 1: Unexpected message type at index " + std::to_string(i));
                    break;
                }
            }

            if (!found_final_assistant) {
                LOG_DEBUG("Pass 1: Skipping incomplete turn at index " + std::to_string(turn_start));
                continue;
            }

            tokens_freed += turn_tokens;
            if (start_index == -1) start_index = turn_start;
            end_index = final_assistant_index;
            LOG_DEBUG("Pass 1: Evicting big-turn [" + std::to_string(turn_start) + ", " +
                      std::to_string(final_assistant_index) + "] freeing " + std::to_string(turn_tokens) + " tokens");
        }

        if (tokens_freed >= tokens_needed) {
            LOG_INFO("Pass 1 freed " + std::to_string(tokens_freed) + " tokens (needed " +
                     std::to_string(tokens_needed) + ")");
            return {start_index, end_index};
        }

        // PASS 2: Evict mini-turns (ASSISTANT tool_call + TOOL result pairs)
        LOG_DEBUG("Pass 2: Need " + std::to_string(tokens_needed - tokens_freed) + " more tokens, evicting mini-turns");

        for (size_t i = evict_start; i < last_user_index && tokens_freed < tokens_needed; ) {
            if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "context_eviction") {
                tokens_freed += messages_[i].token_count;
                if (start_index == -1) start_index = i;
                end_index = i;
                LOG_DEBUG("Pass 2: Evicting eviction notice at index " + std::to_string(i));
                i++;
                continue;
            }

            if (messages_[i].type == Message::ASSISTANT &&
                i + 1 < last_user_index &&
                messages_[i + 1].type == Message::TOOL &&
                messages_[i + 1].tool_name != "context_eviction") {

                int pair_tokens = messages_[i].token_count + messages_[i + 1].token_count;
                tokens_freed += pair_tokens;
                if (start_index == -1) start_index = i;
                end_index = i + 1;
                LOG_DEBUG("Pass 2: Evicting mini-turn [" + std::to_string(i) + ", " +
                          std::to_string(i + 1) + "] freeing " + std::to_string(pair_tokens) + " tokens");
                i += 2;
                continue;
            }

            i++;
        }

        if (tokens_freed < tokens_needed) {
            LOG_WARN("Cannot free enough tokens: need " + std::to_string(tokens_needed) +
                     ", freed " + std::to_string(tokens_freed));
        }

        if (start_index == -1 || end_index == -1) {
            LOG_ERROR("No messages available for eviction");
            return {-1, -1};
        }

        LOG_INFO("Eviction complete: freed " + std::to_string(tokens_freed) + " tokens from range [" +
                 std::to_string(start_index) + ", " + std::to_string(end_index) + "]");
        return {start_index, end_index};
    }

    bool evict_messages_by_index(int start_msg, int end_msg) {
        if (start_msg < 0 || end_msg < 0 || start_msg > end_msg) {
            LOG_ERROR("Invalid message indices for eviction");
            return false;
        }

        if (static_cast<size_t>(end_msg) >= messages_.size()) {
            LOG_ERROR("End message index out of range");
            return false;
        }

        LOG_INFO("Evicting messages [" + std::to_string(start_msg) + ", " + std::to_string(end_msg) + "]");

        // Archive to RAG if it's a complete turn
        std::string user_question;
        std::string final_answer;

        bool contains_search_memory = false;
        for (int i = start_msg; i <= end_msg; i++) {
            if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "search_memory") {
                contains_search_memory = true;
                break;
            }
        }

        if (messages_[start_msg].type == Message::USER) {
            user_question = messages_[start_msg].content;
        }

        if (messages_[end_msg].type == Message::ASSISTANT) {
            final_answer = messages_[end_msg].content;
        }

        if (!contains_search_memory && !user_question.empty() && !final_answer.empty()) {
            ConversationTurn turn(user_question, final_answer);
            RAGManager::archive_turn(turn);
            LOG_DEBUG("Archived USER question + final ASSISTANT answer to RAG");
        }

        // Remove messages
        for (int i = start_msg; i <= end_msg; i++) {
            current_token_count_ -= messages_[i].token_count;
        }

        messages_.erase(messages_.begin() + start_msg, messages_.begin() + end_msg + 1);

        int num_messages = end_msg - start_msg + 1;
        LOG_INFO("Successfully evicted " + std::to_string(num_messages) + " messages");

        return true;
    }

private:
    void evict_oldest_messages() {
        if (!needs_eviction(0)) return;

        LOG_DEBUG("Context full, evicting old messages...");

        while (needs_eviction(0) && messages_.size() > system_message_count_) {
            int tokens_over = get_total_tokens() - static_cast<int>(max_context_tokens_);
            auto [start_msg, end_msg] = calculate_messages_to_evict(tokens_over);

            if (start_msg == -1 || end_msg == -1) {
                LOG_WARN("Cannot calculate messages to evict, stopping eviction");
                break;
            }

            if (!evict_messages_by_index(start_msg, end_msg)) {
                LOG_ERROR("Failed to evict messages, stopping eviction");
                break;
            }
        }
    }

    bool needs_eviction(int additional_tokens) const {
        return (get_total_tokens() + additional_tokens) > static_cast<int>(max_context_tokens_);
    }

    int calculate_json_overhead() const {
        int overhead_chars = 100;
        overhead_chars += static_cast<int>(messages_.size() * 50);
        return static_cast<int>(overhead_chars / 4.0 + 0.5);
    }

    int count_tokens(const std::string& text) const {
        return static_cast<int>(text.length() / 4.0 + 0.5);
    }

    std::deque<Message> messages_;
    size_t system_message_count_ = 0;
    size_t max_context_tokens_;
    int current_token_count_ = 0;
};

// Test functions (same as before but using SimpleContextManager)
void print_test(const std::string& name) {
    std::cout << "\n========================================\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << "========================================\n";
}

void print_context_state(SimpleContextManager& ctx) {
    std::cout << "Messages: " << ctx.get_message_count() << "\n";
    std::cout << "Tokens: " << ctx.get_total_tokens() << "/" << ctx.get_max_context_tokens() << "\n";
    std::cout << "Utilization: " << (ctx.get_context_utilization() * 100) << "%\n";

    auto& messages = ctx.get_messages();
    std::cout << "Message sequence: ";
    for (const auto& msg : messages) {
        std::cout << msg.get_role()[0] << " ";
    }
    std::cout << "\n";
}

void test_system_user_only() {
    print_test("Edge Case - System + Single User Only");

    SimpleContextManager ctx(600);  // Small context

    // Add system message that takes up a lot of space
    std::string sys_text;
    for (int i = 0; i < 100; i++) {
        sys_text += "System instruction. ";
    }
    Message sys(Message::SYSTEM, sys_text, static_cast<int>(sys_text.length() / 4.0));
    ctx.add_message(sys);

    std::cout << "After system message:\n";
    print_context_state(ctx);

    // Add user message that will fill up remaining space
    std::string user_text;
    for (int i = 0; i < 50; i++) {
        user_text += "User question text. ";
    }
    Message user(Message::USER, user_text, static_cast<int>(user_text.length() / 4.0));

    std::cout << "\nAdding large user message that should trigger eviction...\n";

    try {
        ctx.add_message(user);
        std::cout << "\n❌ ERROR: Should have thrown exception!\n";
        std::cout << "\nAfter user message:\n";
        print_context_state(ctx);
    } catch (const ContextManagerError& e) {
        std::cout << "\n✓ Correctly threw exception: " << e.what() << "\n";
        std::cout << "\nContext remains valid:\n";
        print_context_state(ctx);
    }
}

void test_normal_eviction() {
    print_test("Normal Eviction - Multiple Turns");

    SimpleContextManager ctx(800);  // Small context to force eviction

    Message sys(Message::SYSTEM, "You are helpful.", 4);
    ctx.add_message(sys);

    for (int i = 0; i < 10; i++) {
        // Make messages much larger to fill context
        std::string user_text = "User message " + std::to_string(i) + ": ";
        for (int j = 0; j < 20; j++) {
            user_text += "This is content to fill the context window. ";
        }
        Message user(Message::USER, user_text, static_cast<int>(user_text.length() / 4.0));
        ctx.add_message(user);

        std::string assistant_text = "Assistant response " + std::to_string(i) + ": ";
        for (int j = 0; j < 20; j++) {
            assistant_text += "This is the assistant response content. ";
        }
        Message assistant(Message::ASSISTANT, assistant_text, static_cast<int>(assistant_text.length() / 4.0));
        ctx.add_message(assistant);

        std::cout << "\nAfter turn " << i << ":\n";
        print_context_state(ctx);
    }

    std::cout << "\n✓ Eviction should have occurred\n";
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "Context Manager Eviction Test Suite\n";
    std::cout << "===========================================\n";

    try {
        test_system_user_only();
        test_normal_eviction();

        std::cout << "\n===========================================\n";
        std::cout << "All tests completed!\n";
        std::cout << "===========================================\n";

    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
