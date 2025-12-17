#include "session.h"
#include "shepherd.h"
#include "backend.h"
#include "rag.h"
#include <iostream>


std::vector<std::pair<int, int>> Session::calculate_messages_to_evict(int tokens_needed) {
    // Two-pass eviction strategy:
    // PASS 1: Evict complete big-turns (USER → final ASSISTANT) - can archive to RAG
    // PASS 2: Evict mini-turns (ASSISTANT tool_call + TOOL result pairs) - just delete
    // Returns vector of non-contiguous ranges to preserve protected messages

    std::vector<std::pair<int, int>> ranges;
    int tokens_freed = 0;

    // Find last USER message - this is the current turn we're responding to
    int last_user_index = -1;
    for (int i = static_cast<int>(messages.size()) - 1; i >= 0; i--) {
        if (messages[i].role == Message::USER) {
            last_user_index = i;
            break;
        }
    }

    // PASS 1: Try to evict complete big-turns (USER → final ASSISTANT)
    dout(1) << "Pass 1: Looking for complete big-turns to evict" << std::endl;

    for (size_t i = 0; i < messages.size() && tokens_freed < tokens_needed; ) {
        // Skip if we've reached the current user message
        if (static_cast<int>(i) >= last_user_index) {
            break;
        }

        // Skip non-USER messages (e.g., system message at index 0)
        if (messages[i].role != Message::USER) {
            dout(1) << "Pass 1: Skipping non-USER message at index " + std::to_string(i) +
                      " (type: " + messages[i].get_role() + ")" << std::endl;
            i++;
            continue;
        }

        // Found USER - scan forward to find final ASSISTANT response
        size_t turn_start = i;
        int turn_tokens = messages[i].tokens;
        i++; // Move past USER

        // Scan for final ASSISTANT message in this turn
        size_t final_assistant_index = turn_start;
        bool found_final_assistant = false;

        while (i < messages.size() && static_cast<int>(i) < last_user_index) {
            turn_tokens += messages[i].tokens;

            if (messages[i].role == Message::ASSISTANT) {
                final_assistant_index = i;
                found_final_assistant = true;
                i++;

                // Check if next message starts a new turn
                if (i >= messages.size() || static_cast<int>(i) >= last_user_index ||
                    messages[i].role == Message::USER) {
                    break; // End of this turn
                }
            } else if (messages[i].role == Message::TOOL_RESPONSE) {
                // Tool result - keep scanning for final assistant response
                i++;
            } else {
                // Unexpected message type
                dout(1) << std::string("WARNING: ") +"Pass 1: Unexpected message type at index " + std::to_string(i) +
                         " (type: " + messages[i].get_role() + ")" << std::endl;

                break;
            }
        }

        if (!found_final_assistant) {
            // Incomplete turn (USER with no final response), skip it
            dout(1) << "Pass 1: Skipping incomplete turn at index " + std::to_string(turn_start) << std::endl;
            continue;
        }

        // Evict this complete turn (from USER through final ASSISTANT, including all mini-turns)
        tokens_freed += turn_tokens;
        ranges.push_back({turn_start, final_assistant_index});
        dout(1) << "Pass 1: Evicting big-turn [" + std::to_string(turn_start) + ", " +
                 std::to_string(final_assistant_index) + "] freeing " + std::to_string(turn_tokens) + " tokens" << std::endl;
    }

    // If Pass 1 freed enough, we're done
    if (tokens_freed >= tokens_needed) {
        dout(1) << "Pass 1 freed " + std::to_string(tokens_freed) + " tokens (needed " +
                 std::to_string(tokens_needed) + ")" << std::endl;

        return ranges;
    }

    // PASS 2: Evict mini-turns from current turn (ASSISTANT tool_call + TOOL result pairs)
    // IMPORTANT: Start at last_user_index + 1 to PROTECT the current user message from eviction
    // ALSO PROTECT last_assistant_message which provides critical context for tool results
    dout(1) << "Pass 2: Need " + std::to_string(tokens_needed - tokens_freed) +
              " more tokens, evicting mini-turns from current turn" << std::endl;
    dout(1) << "Protected messages: last_user=" + std::to_string(last_user_message_index) +
              ", last_assistant=" + std::to_string(last_assistant_message_index) << std::endl;

    if (last_user_index >= 0) {
        for (size_t i = last_user_index + 1; i < messages.size() && tokens_freed < tokens_needed; ) {
            // Look for ASSISTANT + TOOL pairs (mini-turns)
            if (messages[i].role == Message::ASSISTANT &&
                i + 1 < messages.size() &&
                messages[i + 1].role == Message::TOOL_RESPONSE) {

                // Skip if this is the last assistant message (provides context for current tool result)
                if (static_cast<int>(i) == last_assistant_message_index) {
                    dout(1) << "Pass 2: Skipping protected last_assistant_message at index " + std::to_string(i) << std::endl;
                    i += 2; // Skip both ASSISTANT and TOOL
                    continue;
                }

                int pair_tokens = messages[i].tokens + messages[i + 1].tokens;
                tokens_freed += pair_tokens;
                ranges.push_back({i, i + 1});
                dout(1) << "Pass 2: Evicting mini-turn [" + std::to_string(i) + ", " +
                        std::to_string(i + 1) + "] freeing " + std::to_string(pair_tokens) + " tokens" << std::endl;
                i += 2; // Skip both messages
                continue;
            }

            // Skip other messages in current turn
            i++;
        }
    }

    // Check if we freed enough
    if (tokens_freed < tokens_needed) {
        dout(1) << std::string("WARNING: ") +"Cannot free enough tokens: need " + std::to_string(tokens_needed) +
                 ", freed " + std::to_string(tokens_freed) << std::endl;
    }

    if (ranges.empty()) {
        std::cerr << "No messages available for eviction" << std::endl;
        return {};
    }

    dout(1) << "Eviction complete: freed " + std::to_string(tokens_freed) +
             " tokens from " + std::to_string(ranges.size()) + " ranges" << std::endl;
    return ranges;
}

bool Session::evict_messages(const std::vector<std::pair<int, int>>& ranges) {
    if (ranges.empty()) {
        std::cerr << "No ranges specified for eviction" << std::endl;
        return false;
    }

    // Validate all ranges first
    for (const auto& [start_idx, end_idx] : ranges) {
        if (start_idx < 0 || end_idx < 0 || start_idx > end_idx) {
            std::cerr << "Invalid message indices for eviction: [" + std::to_string(start_idx) +
                      ", " + std::to_string(end_idx) + "]" << std::endl;
            return false;
        }

        if (static_cast<size_t>(end_idx) >= messages.size()) {
            std::cerr << "End message index " + std::to_string(end_idx) +
                      " out of range (have " + std::to_string(messages.size()) + " messages)" << std::endl;
            return false;
        }
    }

    dout(1) << "Evicting " + std::to_string(ranges.size()) + " message range(s)" << std::endl;

    // Archive complete turns to RAG before evicting
    // Process all ranges to find USER → ASSISTANT pairs
    for (const auto& [start_idx, end_idx] : ranges) {
        for (int i = start_idx; i <= end_idx; ) {
            // Find the start of a turn (a USER message)
            if (messages[i].role != Message::USER) {
                i++;
                continue;
            }

            int turn_start_idx = i;
            int final_assistant_idx = -1;
            bool turn_contains_tool_call = false;

            // Scan forward to find the end of this turn (within this range)
            int j = i + 1;
            for ( ; j <= end_idx; j++) {
                if (messages[j].role == Message::TOOL_RESPONSE) {
                    turn_contains_tool_call = true;
                }
                if (messages[j].role == Message::ASSISTANT) {
                    final_assistant_idx = j; // Keep track of the latest assistant response
                }
                // The turn ends if we hit the next USER message
                if (messages[j].role == Message::USER) {
                    break;
                }
            }

            // If we found a valid USER -> ASSISTANT pair within the turn...
            if (final_assistant_idx != -1) {
                // ...and it doesn't contain a tool call, archive it.
                if (!turn_contains_tool_call) {
                    const std::string& user_question = messages[turn_start_idx].content;
                    const std::string& final_answer = messages[final_assistant_idx].content;
                    ConversationTurn turn(user_question, final_answer);
                    RAGManager::archive_turn(turn);
                    dout(1) << "Archived USER question at index " + std::to_string(turn_start_idx) +
                              " with final ASSISTANT answer at index " + std::to_string(final_assistant_idx) << std::endl;
                } else {
                    dout(1) << "Skipped RAG archival for turn starting at index " + std::to_string(turn_start_idx) +
                              " because it contains a tool call." << std::endl;
                }
            }

            // Continue scanning from where this turn ended
            i = j;
        }
    }

    // Calculate total evicted tokens and messages
    int total_evicted_tokens = 0;
    int total_evicted_messages = 0;
    for (const auto& [start_idx, end_idx] : ranges) {
        for (int i = start_idx; i <= end_idx; i++) {
            total_evicted_tokens += messages[i].tokens;
        }
        total_evicted_messages += (end_idx - start_idx + 1);
    }

    // Erase messages in reverse order to avoid index shifting
    for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
        auto [start_idx, end_idx] = *it;
        dout(1) << "Erasing range [" + std::to_string(start_idx) + ", " + std::to_string(end_idx) + "]" << std::endl;
        messages.erase(messages.begin() + start_idx, messages.begin() + end_idx + 1);
    }

    // Update tracked message indices to account for eviction
    // For each tracked index, calculate how many messages before it were evicted
    if (last_user_message_index >= 0) {
        bool was_evicted = false;
        int offset = 0;

        for (const auto& [start_idx, end_idx] : ranges) {
            if (last_user_message_index >= start_idx && last_user_message_index <= end_idx) {
                // This index was evicted
                dout(1) << "last_user_message_index " + std::to_string(last_user_message_index) +
                          " was evicted, resetting" << std::endl;
                last_user_message_index = -1;
                last_user_message_tokens = 0;
                was_evicted = true;
                break;
            } else if (last_user_message_index > end_idx) {
                // This range was before the index, so count it in offset
                offset += (end_idx - start_idx + 1);
            }
        }

        if (!was_evicted && offset > 0) {
            int old_idx = last_user_message_index;
            last_user_message_index -= offset;
            dout(1) << "Adjusted last_user_message_index from " + std::to_string(old_idx) +
                      " to " + std::to_string(last_user_message_index) << std::endl;
        }
    }

    if (last_assistant_message_index >= 0) {
        bool was_evicted = false;
        int offset = 0;

        for (const auto& [start_idx, end_idx] : ranges) {
            if (last_assistant_message_index >= start_idx && last_assistant_message_index <= end_idx) {
                // This index was evicted
                dout(1) << "last_assistant_message_index " + std::to_string(last_assistant_message_index) +
                          " was evicted, resetting" << std::endl;
                last_assistant_message_index = -1;
                last_assistant_message_tokens = 0;
                was_evicted = true;
                break;
            } else if (last_assistant_message_index > end_idx) {
                // This range was before the index, so count it in offset
                offset += (end_idx - start_idx + 1);
            }
        }

        if (!was_evicted && offset > 0) {
            int old_idx = last_assistant_message_index;
            last_assistant_message_index -= offset;
            dout(1) << "Adjusted last_assistant_message_index from " + std::to_string(old_idx) +
                      " to " + std::to_string(last_assistant_message_index) << std::endl;
        }
    }

    // Update total token count from API
    total_tokens -= total_evicted_tokens;

    // Update last_prompt_tokens baseline to reflect eviction
    // This keeps the delta calculation accurate for API backends
    last_prompt_tokens -= total_evicted_tokens;

    dout(1) << "Successfully evicted " + std::to_string(total_evicted_messages) +
             " messages from " + std::to_string(ranges.size()) + " ranges, freed " +
             std::to_string(total_evicted_tokens) + " tokens" << std::endl;

    return true;
}

void Session::add_message(Message::Role role,
                         const std::string& content,
                         const std::string& tool_name,
                         const std::string& tool_id,
                         int max_tokens) {
    dout(1) << "Session::add_message called: role=" + std::to_string(static_cast<int>(role)) +
             ", max_tokens=" + std::to_string(max_tokens) +
             ", total_tokens=" + std::to_string(total_tokens) +
             ", context_size=" + std::to_string(backend ? backend->context_size : 0) << std::endl;

    // Calculate prompt_tokens for eviction decisions
    int prompt_tokens = 0;
    if (backend) {
        // API backends: use EMA on formatted JSON
        // GPU backends: tokenize formatted message through chat template
        prompt_tokens = backend->count_message_tokens(role, content, tool_name, tool_id);
        dout(1) << "Calculated prompt_tokens: " + std::to_string(prompt_tokens) << std::endl;
    }

    // Calculate max_tokens if not provided
    if (max_tokens == 0 && backend && backend->context_size > 0) {
        // Reserve space for critical context that must be preserved:
        // - System message (always needed for model behavior)
        // - Last user message (the question that triggered this response)
        // - Last assistant message (the tool call or previous response)
        // Without these, tool results and responses would lack necessary context
        int reserved = system_message_tokens;
        if (last_user_message_index >= 0) {
            reserved += last_user_message_tokens;
        }
        if (last_assistant_message_index >= 0) {
            reserved += last_assistant_message_tokens;
        }

        // Calculate available space after reserving context and current message
        int available = backend->context_size - reserved - prompt_tokens;
        max_tokens = (available > 0) ? available : 0;

        // Cap max_tokens at desired completion size to avoid MAX_TOKENS_TOO_HIGH errors
        // This ensures we leave room for the actual prompt which may be larger than estimated
        if (max_tokens > desired_completion_tokens) {
            max_tokens = desired_completion_tokens;
        }

        dout(1) << "Calculated max_tokens: " + std::to_string(max_tokens) +
                 " (context=" + std::to_string(backend->context_size) +
                 ", reserved=" + std::to_string(reserved) +
                 " [system=" + std::to_string(system_message_tokens) +
                 ", last_user=" + std::to_string(last_user_message_tokens) +
                 ", last_asst=" + std::to_string(last_assistant_message_tokens) + "]" +
                 ", prompt=" + std::to_string(prompt_tokens) +
                 ", desired_completion=" + std::to_string(desired_completion_tokens) + ")" << std::endl;

    }

    // Check if we need to evict BEFORE sending to backend (only if auto-eviction is enabled)
    if (auto_evict && backend && backend->context_size > 0) {

        if (needs_eviction(prompt_tokens)) {
            // Calculate how many tokens we need to free (message + completion space)
            int tokens_over = (total_tokens + prompt_tokens + desired_completion_tokens) - backend->context_size;

            dout(1) << "Auto-eviction triggered: need to free " + std::to_string(tokens_over) + " tokens" << std::endl;
            dout(1) << "  current state: total=" + std::to_string(total_tokens) +
                     ", messages=" + std::to_string(messages.size()) +
                     ", prompt=" + std::to_string(prompt_tokens) +
                     ", max=" + std::to_string(max_tokens) +
                     ", desired_completion=" + std::to_string(desired_completion_tokens) << std::endl;

            // Calculate which messages to evict
            auto ranges = calculate_messages_to_evict(tokens_over);

            if (ranges.empty()) {
                // Cannot evict enough space - error will be signaled via callback
                std::string error = "Cannot add message: would exceed context limit (" +
                            std::to_string(total_tokens + prompt_tokens) + " > " +
                            std::to_string(backend->context_size) + " tokens) and no messages available for eviction. " +
                            "Consider reducing message size or clearing context.";
                std::cerr << error << std::endl;
                // Backend callback will signal ERROR - just return
                return;
            }

            // Perform eviction
            if (!evict_messages(ranges)) {
                std::string error = "Eviction failed unexpectedly when trying to free space for message";
                std::cerr << error << std::endl;
                // Backend callback will signal ERROR - just return
                return;
            }

            // Eviction succeeded - recalculate max_tokens using same logic as initial calculation
            int reserved = system_message_tokens;
            if (last_user_message_index >= 0) {
                reserved += last_user_message_tokens;
            }
            if (last_assistant_message_index >= 0) {
                reserved += last_assistant_message_tokens;
            }
            int available = backend->context_size - reserved - prompt_tokens;
            max_tokens = (available > 0) ? available : 0;

            // Cap max_tokens at desired completion size (same as initial calculation)
            if (max_tokens > desired_completion_tokens) {
                max_tokens = desired_completion_tokens;
            }

            dout(1) << "Recalculated max_tokens after eviction: " + std::to_string(max_tokens) +
                     " (reserved=" + std::to_string(reserved) +
                     ", prompt=" + std::to_string(prompt_tokens) +
                     ", desired_completion=" + std::to_string(desired_completion_tokens) + ")" << std::endl;

        }
    }

    // Delegate to backend to format and send the message
    // All output flows through the backend's callback (CONTENT, TOOL_CALL, ERROR, STOP)
    // Session is updated by backend on success
    backend->add_message(*this, role, content, tool_name, tool_id, max_tokens);

    // Note: Auto-continuation logic would need to be handled by frontend
    // based on STOP callback with finish_reason="length"
}

bool Session::needs_eviction(int additional_tokens) const {
    if (!backend || backend->context_size == 0) {
        return false;  // No limit set
    }

    // Check if new message + completion would exceed available space
    int required = total_tokens + additional_tokens + desired_completion_tokens;
    int available = get_available_tokens();
    bool needs = required >= available;

    dout(1) << "needs_eviction check: total=" + std::to_string(total_tokens) +
              ", additional=" + std::to_string(additional_tokens) +
              ", desired_completion=" + std::to_string(desired_completion_tokens) +
              ", required=" + std::to_string(required) +
              ", available=" + std::to_string(available) +
              ", needs_eviction=" + (needs ? "true" : "false") << std::endl;

    return needs;
}

int Session::get_available_tokens() const {
    if (!backend) return 0;
    return backend->context_size;
}

void Session::dump() const {
    std::cout << "========== SESSION DUMP ==========" << std::endl;
    std::cout << "System message: " << system_message  << std::endl;
    std::cout << "System message tokens: " << system_message_tokens  << std::endl;
    std::cout  << std::endl;

    std::cout << "Messages (" << messages.size() << " total):" << std::endl;
    for (size_t i = 0; i < messages.size(); i++) {
        std::cout << "messages[" << i << "]: " << messages[i]  << std::endl;
    }
    std::cout  << std::endl;

    if (!tools.empty()) {
        std::cout << "Tools (" << tools.size() << " total):" << std::endl;
        for (size_t i = 0; i < tools.size(); i++) {
            std::cout << "tools[" << i << "]: " << tools[i].name  << std::endl;
            std::cout << "  description: " << tools[i].description  << std::endl;
            std::cout << "  parameters: " << tools[i].parameters_text()  << std::endl;
        }
        std::cout  << std::endl;
    }

    std::cout << "Total tokens: " << total_tokens  << std::endl;
    std::cout << "Last prompt tokens: " << last_prompt_tokens  << std::endl;
    std::cout << "Auto-evict: " << (auto_evict ? "true" : "false")  << std::endl;
    std::cout << "Desired completion tokens: " << desired_completion_tokens  << std::endl;
    std::cout << "==================================" << std::endl;
}

void Session::switch_backend(Backend* new_backend) {
    dout(1) << "Switching backend from " + backend->backend_name + " to " + new_backend->backend_name << std::endl;

    // Switch backend pointer
    backend = new_backend;

    // Reset token counters (different backends count differently)
    total_tokens = 0;
    last_prompt_tokens = 0;

    // Messages remain intact - conversation history preserved
    // First message to new backend will establish new baseline
    dout(1) << "Backend switched. Token counters reset. Message history preserved." << std::endl;
}

void Session::clear() {
    dout(1) << "Clearing session context" << std::endl;

    // Clear all messages
    messages.clear();

    // Reset token counters
    total_tokens = 0;
    last_prompt_tokens = 0;

    // Reset message tracking indices
    last_user_message_index = -1;
    last_user_message_tokens = 0;
    last_assistant_message_index = -1;
    last_assistant_message_tokens = 0;

    dout(1) << "Session cleared. System message and tools preserved." << std::endl;
}
