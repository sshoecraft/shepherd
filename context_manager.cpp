#include "shepherd.h"
#include "context_manager.h"
#include "rag.h"
#include <chrono>
#include <filesystem>

// Message implementation
int64_t Message::get_current_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ContextManager base implementation
ContextManager::ContextManager(size_t max_context_tokens)
    : max_context_tokens_(max_context_tokens) {

    if (max_context_tokens_ < 512) {
        throw ContextManagerError("Context size too small: " + std::to_string(max_context_tokens_) +
                                  " (minimum 512 tokens required)");
    }

    LOG_DEBUG("ContextManager created with " + std::to_string(max_context_tokens_) + " max tokens");
}

size_t ContextManager::get_message_count() const {
    return messages_.size();
}

size_t ContextManager::get_max_context_tokens() const {
    return max_context_tokens_;
}

void ContextManager::set_max_context_tokens(size_t max_tokens) {
    max_context_tokens_ = max_tokens;
    LOG_DEBUG("Updated max context tokens to " + std::to_string(max_tokens));
}

void ContextManager::add_message(const Message& message) {
    // In server mode, throw exception instead of evicting
    // Client is responsible for managing context window
    if (g_server_mode && needs_eviction(message.token_count)) {
        throw ContextManagerError(
            "Context limit exceeded in server mode: would need " +
            std::to_string(get_total_tokens() + message.token_count) + " tokens but limit is " +
            std::to_string(max_context_tokens_) + " tokens. Client must manage context window.");
    }

    // Check if we need to evict BEFORE adding (only if auto-eviction is enabled)
    // For backends with KV cache (llama.cpp, TensorRT), eviction is handled by KV cache callbacks
    if (auto_evict && needs_eviction(message.token_count)) {
        // Calculate how many tokens we need to free to fit this message
        int tokens_over = get_total_tokens() + message.token_count - static_cast<int>(get_available_tokens());

        // Try to calculate if eviction is possible
        auto [start_msg, end_msg] = calculate_messages_to_evict(tokens_over);

        if (start_msg == -1 || end_msg == -1) {
            // Cannot evict enough space for this message - prevent addition entirely
            throw ContextManagerError(
                "Cannot add message: would exceed context limit (" +
                std::to_string(get_total_tokens() + message.token_count) + " > " +
                std::to_string(max_context_tokens_) + " tokens) and no messages available for eviction. " +
                "Consider reducing message size or clearing context.");
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
                std::to_string(get_available_tokens() - get_total_tokens()) + " tokens available). " +
                "Cannot add message without violating context constraints.");
        }
    }

    // Add the message to our deque
    messages_.push_back(message);

    dprintf(2, "BEFORE increment: current_token_count_=%d, adding=%d",
            current_token_count_, message.token_count);
    current_token_count_ += message.token_count;
    dprintf(2, "AFTER increment: current_token_count_=%d/%zu",
            current_token_count_, max_context_tokens_);

    // Track system messages (they're preserved during eviction)
    if (message.type == Message::SYSTEM && messages_.size() <= system_message_count_ + 1) {
        system_message_count_++;
    }

    LOG_DEBUG("Added " + message.get_role() + " message with " +
              std::to_string(message.token_count) + " tokens");
    LOG_DEBUG("Total tokens: " + std::to_string(current_token_count_) + "/" +
              std::to_string(max_context_tokens_));
}

double ContextManager::get_context_utilization() const {
    if (max_context_tokens_ == 0) {
        return 0.0;
    }
    return static_cast<double>(get_total_tokens()) / static_cast<double>(max_context_tokens_);
}

void ContextManager::clear() {
    messages_.clear();
    current_token_count_ = 0;
    system_message_count_ = 0;
    LOG_DEBUG("Context cleared");
}

void ContextManager::remove_last_messages(size_t count) {
    if (count == 0 || messages_.empty()) {
        return;
    }

    size_t to_remove = std::min(count, messages_.size());

    // Remove from the end and update token count
    for (size_t i = 0; i < to_remove; i++) {
        current_token_count_ -= messages_.back().token_count;
        messages_.pop_back();
    }

    LOG_DEBUG("Removed last " + std::to_string(to_remove) + " messages from context");
    LOG_DEBUG("Total tokens: " + std::to_string(current_token_count_) + "/" + std::to_string(max_context_tokens_));
}

void ContextManager::evict_oldest_messages() {
    if (!needs_eviction(0)) {
        return;
    }

    LOG_DEBUG("Context full, evicting old messages...");

    // Evict messages in batches until we have enough space
    while (needs_eviction(0) && messages_.size() > system_message_count_) {
        // Calculate how many tokens we need to free
        int tokens_over = get_total_tokens() - static_cast<int>(get_available_tokens());

        // Try to evict enough messages to free the needed space
        auto [start_msg, end_msg] = calculate_messages_to_evict(tokens_over);

        if (start_msg == -1 || end_msg == -1) {
            LOG_WARN("Cannot calculate messages to evict, stopping eviction");
            break;
        }

        // Use the shared eviction logic
        if (!evict_messages_by_index(start_msg, end_msg)) {
            LOG_ERROR("Failed to evict messages, stopping eviction");
            break;
        }
    }
}

bool ContextManager::needs_eviction(int additional_tokens) const {
    return (get_total_tokens() + additional_tokens) > static_cast<int>(get_available_tokens());
}

size_t ContextManager::get_available_tokens() const {
    return max_context_tokens_;
}

int ContextManager::get_total_tokens() const {
    return current_token_count_ + calculate_json_overhead();
}

int ContextManager::get_cached_message_tokens() const {
    int cached_tokens = 0;
    for (const auto& msg : messages_) {
        if (msg.in_kv_cache) {
            cached_tokens += msg.token_count;
        }
    }
    return cached_tokens;
}

void ContextManager::recalculate_total_tokens() {
    int old_count = current_token_count_;
    current_token_count_ = 0;
    for (const auto& message : messages_) {
        current_token_count_ += message.token_count;
    }
    dprintf(3, "RECALCULATE: old=%d, new=%d, messages=%zu",
            old_count, current_token_count_, messages_.size());
}

void ContextManager::adjust_token_count(int delta) {
    dprintf(3, "ADJUST: current=%d, delta=%d, new=%d",
            current_token_count_, delta, current_token_count_ + delta);
    current_token_count_ += delta;
}

std::pair<int, int> ContextManager::calculate_messages_to_evict(int tokens_needed, size_t max_evict_index) const {
    // Two-pass eviction strategy:
    // PASS 1: Evict complete big-turns (USER → final ASSISTANT) - can archive to RAG
    // PASS 2: Evict mini-turns (ASSISTANT tool_call + TOOL result pairs) - just delete
    // Both passes can evict eviction notices (tool_name == "context_eviction") individually

    int tokens_freed = 0;
    int start_index = -1;
    int end_index = -1;

    // Find first non-system message
    size_t evict_start = system_message_count_;

    // Find last USER message - this is the current turn we're responding to
    // Scan ALL messages (not just cached ones) to find the actual current user message
    size_t last_user_index = evict_start;
    for (int i = static_cast<int>(messages_.size()) - 1; i >= static_cast<int>(evict_start); i--) {
        if (messages_[i].type == Message::USER) {
            last_user_index = i;
            break;
        }
    }

    // Determine eviction boundary: ONLY evict cached messages (max_evict_index)
    // Note: Pass 1 evicts before last_user_index, Pass 2 evicts after it
    size_t evict_boundary = std::min(max_evict_index, last_user_index);

    // Check if there's anything at all to evict
    if (evict_start >= max_evict_index) {
        LOG_WARN("Cannot evict: no messages available for eviction");
        return {-1, -1};
    }

    // PASS 1: Try to evict complete big-turns (USER → final ASSISTANT)
    LOG_DEBUG("Pass 1: Looking for complete big-turns to evict");
    for (size_t i = evict_start; i < last_user_index && tokens_freed < tokens_needed; ) {
        // Evict standalone eviction notices
        if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "context_eviction") {
            tokens_freed += messages_[i].token_count;
            if (start_index == -1) {
                start_index = i;
            }
            end_index = i;
            LOG_DEBUG("Pass 1: Evicting eviction notice at index " + std::to_string(i));
            i++;
            continue;
        }

        // Skip non-USER messages (shouldn't happen in healthy context)
        if (messages_[i].type != Message::USER) {
            std::string msg_type = messages_[i].get_role();
            std::string content_preview = messages_[i].content.substr(0, std::min(size_t(50), messages_[i].content.length()));
            if (messages_[i].content.length() > 50) content_preview += "...";
            LOG_WARN("Pass 1: Skipping non-USER message at index " + std::to_string(i) +
                     " (type: " + msg_type + ", content: \"" + content_preview + "\")");
            i++;
            continue;
        }

        // Found USER - scan forward to find final ASSISTANT response
        size_t turn_start = i;
        int turn_tokens = messages_[i].token_count;
        i++; // Move past USER

        // Scan for final ASSISTANT message in this turn
        size_t final_assistant_index = turn_start;
        bool found_final_assistant = false;

        while (i < last_user_index) {
            turn_tokens += messages_[i].token_count;

            if (messages_[i].type == Message::ASSISTANT) {
                final_assistant_index = i;
                found_final_assistant = true;
                i++;

                // Check if next message starts a new turn
                if (i >= last_user_index || messages_[i].type == Message::USER) {
                    break; // End of this turn
                }
            } else if (messages_[i].type == Message::TOOL) {
                // Tool result or eviction notice - keep scanning for final assistant response
                i++;
            } else {
                // Unexpected message type
                std::string msg_type = messages_[i].get_role();
                std::string content_preview = messages_[i].content.substr(0, std::min(size_t(50), messages_[i].content.length()));
                if (messages_[i].content.length() > 50) content_preview += "...";
                LOG_WARN("Pass 1: Unexpected message type at index " + std::to_string(i) +
                         " (type: " + msg_type + ", content: \"" + content_preview + "\")");
                break;
            }
        }

        if (!found_final_assistant) {
            // Incomplete turn (USER with no final response), skip it
            LOG_DEBUG("Pass 1: Skipping incomplete turn at index " + std::to_string(turn_start));
            continue;
        }

        // Evict this complete turn (from USER through final ASSISTANT, including all mini-turns)
        tokens_freed += turn_tokens;
        if (start_index == -1) {
            start_index = turn_start;
        }
        end_index = final_assistant_index;
        LOG_DEBUG("Pass 1: Evicting big-turn [" + std::to_string(turn_start) + ", " +
                  std::to_string(final_assistant_index) + "] freeing " + std::to_string(turn_tokens) + " tokens");
    }

    // If Pass 1 freed enough, we're done
    if (tokens_freed >= tokens_needed) {
        LOG_INFO("Pass 1 freed " + std::to_string(tokens_freed) + " tokens (needed " +
                 std::to_string(tokens_needed) + ")");
        return {start_index, end_index};
    }

    // PASS 2: Evict mini-turns from current turn (ASSISTANT tool_call + TOOL result pairs)
    // IMPORTANT: Start at last_user_index + 1 to PROTECT the current user message from eviction
    LOG_DEBUG("Pass 2: Need " + std::to_string(tokens_needed - tokens_freed) + " more tokens, evicting mini-turns from current turn");

    for (size_t i = last_user_index + 1; i < max_evict_index && tokens_freed < tokens_needed; ) {
        // Evict standalone eviction notices
        if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "context_eviction") {
            tokens_freed += messages_[i].token_count;
            if (start_index == -1) {
                start_index = i;
            }
            end_index = i;
            LOG_DEBUG("Pass 2: Evicting eviction notice at index " + std::to_string(i));
            i++;
            continue;
        }

        // Look for ASSISTANT + TOOL pairs (mini-turns)
        if (messages_[i].type == Message::ASSISTANT &&
            i + 1 < max_evict_index &&
            messages_[i + 1].type == Message::TOOL &&
            messages_[i + 1].tool_name != "context_eviction") {

            int pair_tokens = messages_[i].token_count + messages_[i + 1].token_count;
            tokens_freed += pair_tokens;
            if (start_index == -1) {
                start_index = i;
            }
            end_index = i + 1;
            LOG_DEBUG("Pass 2: Evicting mini-turn [" + std::to_string(i) + ", " +
                      std::to_string(i + 1) + "] freeing " + std::to_string(pair_tokens) + " tokens");
            i += 2; // Skip both messages
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

std::pair<int, int> ContextManager::evict_messages(int tokens_needed) {
    // Calculate which messages to evict
    auto [start_msg, end_msg] = calculate_messages_to_evict(tokens_needed);

    if (start_msg == -1 || end_msg == -1) {
        LOG_ERROR("Cannot calculate messages to evict");
        return {-1, -1};
    }

    // Use the shared eviction logic
    if (!evict_messages_by_index(start_msg, end_msg)) {
        return {-1, -1};
    }

    return {start_msg, end_msg};
}

bool ContextManager::evict_messages_by_index(int start_msg, int end_msg) {
    if (start_msg < 0 || end_msg < 0 || start_msg > end_msg) {
        LOG_ERROR("Invalid message indices for eviction: [" + std::to_string(start_msg) +
                  ", " + std::to_string(end_msg) + "]");
        return false;
    }

    if (static_cast<size_t>(end_msg) >= messages_.size()) {
        LOG_ERROR("End message index " + std::to_string(end_msg) +
                  " out of range (have " + std::to_string(messages_.size()) + " messages)");
        return false;
    }

    LOG_INFO("Evicting messages [" + std::to_string(start_msg) + ", " + std::to_string(end_msg) + "]");

    // Archive to RAG: extract USER question + final ASSISTANT answer (discard tool messages)
    // The range should be: USER → [ASSISTANT tools + TOOL results]* → ASSISTANT final
    std::string user_question;
    std::string final_answer;

    // Check if this turn contains search_memory tool calls - don't archive those to RAG
    // (would create infinite loop: search returns archived content, we archive that search result...)
    bool contains_search_memory = false;
    for (int i = start_msg; i <= end_msg; i++) {
        if (messages_[i].type == Message::TOOL && messages_[i].tool_name == "search_memory") {
            contains_search_memory = true;
            LOG_DEBUG("Turn contains search_memory result - skipping RAG archival");
            break;
        }
    }

    // Find USER at start
    if (messages_[start_msg].type == Message::USER) {
        user_question = messages_[start_msg].content;
    }

    // Find final ASSISTANT at end
    if (messages_[end_msg].type == Message::ASSISTANT) {
        final_answer = messages_[end_msg].content;
    }

    // Archive the clean Q&A pair to RAG (unless it contains search_memory)
    if (!contains_search_memory && !user_question.empty() && !final_answer.empty()) {
        ConversationTurn turn(user_question, final_answer);
        RAGManager::archive_turn(turn);
        LOG_DEBUG("Archived USER question + final ASSISTANT answer to RAG");
    } else if (contains_search_memory) {
        LOG_DEBUG("Skipped RAG archival for turn containing search_memory");
    } else {
        LOG_DEBUG("Could not extract clean Q&A pair for archiving");
    }

    // Calculate evicted tokens for logging before erasing
    int evicted_tokens = 0;
    for (int i = start_msg; i <= end_msg; i++) {
        evicted_tokens += messages_[i].token_count;
        dprintf(5, "EVICT msg[%d]: %d tokens (role=%s)",
                i, messages_[i].token_count, messages_[i].get_role().c_str());
    }

    // Erase evicted messages from the deque
    messages_.erase(messages_.begin() + start_msg, messages_.begin() + end_msg + 1);

    // CRITICAL: Recalculate the total token count from the remaining messages
    // This eliminates any drift that may have accumulated.
    recalculate_total_tokens();

    dprintf(2, "EVICT: removed %d tokens, current=%d/%zu",
            evicted_tokens, current_token_count_, max_context_tokens_);

    int num_messages = end_msg - start_msg + 1;
    LOG_INFO("Successfully evicted " + std::to_string(num_messages) + " messages");
    LOG_DEBUG("Remaining tokens: " + std::to_string(current_token_count_) + "/" +
              std::to_string(max_context_tokens_));

    return true;
}