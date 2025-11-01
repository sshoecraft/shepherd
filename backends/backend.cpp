
#include "shepherd.h"
#include "backends/backend.h"

//Backend::Backend(ize_t context_size) : config(config) {
Backend::Backend(size_t ctx_size) : context_size(ctx_size) {
    LOG_DEBUG("Backend base constructor with " + std::to_string(ctx_size));
    // Parse backend-specific config (called after derived class is constructed)
}

#if 0
void Backend::update_token_counts_from_api(int prompt_tokens, int completion_tokens, int estimated_prompt_tokens) {
    // Update user message with actual prompt token count if different
    if (prompt_tokens != estimated_prompt_tokens) {
        auto& messages = context_manager_->get_messages();
        if (!messages.empty() && messages.back().type == Message::USER) {
            messages.back().token_count = prompt_tokens;
            // Recalculate total token count since we changed a message
            context_manager_->recalculate_total_tokens();
        }
    }

    // Note: completion tokens are handled when creating the assistant message
    // This method just handles updating the prompt token count
}
#endif
