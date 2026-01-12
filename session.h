
#pragma once

#include <deque>
#include <vector>
#include <functional>
#include "message.h"
#include "nlohmann/json.hpp"

// Forward declarations to avoid circular dependency
class Backend;
struct Response;

/// Sampling parameters - unified across all backends
/// Values of -1 (or -999 for penalties) mean "use backend default"
struct SamplingParams {
    // Basic sampling
    float temperature = -1.0f;      // <= 0 = greedy
    float top_p = -1.0f;            // 1.0 = disabled
    int top_k = -1;                 // <= 0 = disabled
    float min_p = -1.0f;            // 0.0 = disabled (llama.cpp default: 0.05)
    float typ_p = -1.0f;            // typical_p, 1.0 = disabled
    float top_n_sigma = -1.0f;      // -1.0 = disabled

    // Dynamic temperature
    float dynatemp_range = -1.0f;   // 0.0 = disabled
    float dynatemp_exponent = -1.0f;

    // Repetition penalties
    float repetition_penalty = -1.0f;   // 1.0 = disabled (llama calls this penalty_repeat)
    float presence_penalty = -999.0f;   // 0.0 = disabled
    float frequency_penalty = -999.0f;  // 0.0 = disabled
    int penalty_last_n = -1;            // tokens to consider for penalties

    // DRY (Don't Repeat Yourself) sampler
    float dry_multiplier = -1.0f;   // 0.0 = disabled
    float dry_base = -1.0f;
    int dry_allowed_length = -1;
    int dry_penalty_last_n = -1;

    // XTC sampler
    float xtc_probability = -1.0f;  // 0.0 = disabled
    float xtc_threshold = -1.0f;

    // Mirostat
    int mirostat = -1;              // 0 = disabled, 1 = v1, 2 = v2
    float mirostat_tau = -1.0f;
    float mirostat_eta = -1.0f;

    // Other
    uint32_t seed = 0;              // 0 = random
    int min_keep = -1;              // minimum tokens to keep after sampling
    float length_penalty = -999.0f; // length penalty for beam search
    int no_repeat_ngram_size = -1;  // prevent repetition of n-grams
};

class Session {
public:
	Backend *backend;

	// System message ("You are an unhelpful idiot", etc)
	std::string system_message;
	int system_message_tokens = 0;

	// List of messages (user/tool/model)
    std::deque<Message> messages;

    // Last prompt_tokens from API (for delta calculation)
    int last_prompt_tokens = 0;

    // Total tokens from API (authoritative source of truth)
    int total_tokens = 0;

    // Track last user and assistant messages to preserve critical context
    // These must be protected from eviction and reserved in space calculations
    int last_user_message_index = -1;
    int last_user_message_tokens = 0;
    int last_assistant_message_index = -1;
    int last_assistant_message_tokens = 0;

    // Auto-eviction flag: when true, proactively evict messages before exceeding context limit
    // Set to true when user's context_size < backend's API context size
    bool auto_evict = false;

    // Desired completion tokens - calculated once during initialization
    // Scales from ~1,374 tokens (8K context) to 4,096 (32K+ context)
    // Used for max_tokens calculation and eviction decisions
    int desired_completion_tokens = 0;

    /// Tool definitions available in this session
    /// This is the single source of truth for tools in a session
    /// - CLI mode: populated from CLI's Tools instance at startup
    /// - Server mode: populated from client request
    struct Tool {
        std::string name;           // Tool name (sanitized)
        std::string description;    // Tool description
        nlohmann::json parameters;  // Full JSON schema object (not string)

        // Helper to get parameters as string for legacy code
        std::string parameters_json() const {
            return parameters.dump();
        }

        // Helper to get parameter string for simple text format
        std::string parameters_text() const {
            // Extract required parameters for simple format like "query: string"
            std::string result;
            if (parameters.contains("properties")) {
                for (auto& [key, value] : parameters["properties"].items()) {
                    if (!result.empty()) result += ", ";
                    result += key + ": ";
                    if (value.contains("type")) {
                        result += value["type"].get<std::string>();
                    } else {
                        result += "any";
                    }
                }
            }
            return result.empty() ? "none" : result;
        }
    };
    std::vector<Tool> tools;

    // Sampling parameters (per-request overrides from API)
    SamplingParams sampling;

    // Main message interface - handles eviction and delegates to backend
    /// @brief Add a message to the session with automatic eviction if needed

    /// @brief Add a message and generate response
    /// @param role Message role (USER, ASSISTANT, TOOL_RESPONSE)
    /// @param content Message content
    /// @param tool_name Tool name (for TOOL_RESPONSE messages)
    /// @param tool_id Tool call ID (for TOOL_RESPONSE messages)
    /// @param max_tokens Max tokens for assistant response (0 = auto-calculate)
    /// All output flows through backend callback (CONTENT, TOOL_CALL, ERROR, STOP)
    void add_message(Message::Role role,
                    const std::string& content,
                    const std::string& tool_name = "",
                    const std::string& tool_id = "",
                    int max_tokens = 0);

    // Eviction methods - used by both API and GPU backends
    // Two-pass strategy: Pass 1 evicts complete turns, Pass 2 evicts mini-turns

    /// @brief Calculate which messages to evict using two-pass strategy
    /// @param tokens_needed Number of tokens to free
    /// @return Vector of (start_index, end_index) ranges to evict, empty if cannot evict enough
    ///         Each range is inclusive and non-contiguous to preserve protected messages
    std::vector<std::pair<int, int>> calculate_messages_to_evict(int tokens_needed);

    /// @brief Actually remove messages from session
    /// @param ranges Vector of (start_idx, end_idx) ranges to evict (inclusive)
    /// @return true if successful
    bool evict_messages(const std::vector<std::pair<int, int>>& ranges);

    /// @brief Dump session contents to stdout for debugging
    void dump() const;

    /// @brief Clear session context (messages and token counters)
    void clear();

    /// @brief Switch to a different provider/backend
    /// @param new_backend New backend to switch to
    /// Keeps message history, resets token counters
    void switch_backend(Backend* new_backend);

private:
    // Helper methods for auto-eviction
    /// @brief Check if adding message would exceed context limit
    bool needs_eviction(int additional_tokens) const;

    /// @brief Get available tokens (may reserve space for response)
    int get_available_tokens() const;
};

/// @brief Calculate truncation scale factor based on context size
/// @param context_size Total context window size in tokens
/// @return Scale factor between 0.33 and 0.6 for truncating user input and tool results
/// @details Larger contexts get larger percentages to allow more generous limits
///          while still preserving space for responses:
///          - 8k context  → 35.7% (allows ~1,250 tokens)
///          - 16k context → 38.4% (allows ~4,416 tokens)
///          - 32k context → 43.8% (allows ~12,045 tokens)
///          - 64k context → 54.6% (allows ~32,487 tokens)
inline double calculate_truncation_scale(int context_size) {
    // Linear scaling: larger contexts get more generous limits
    double scale = 0.33 + (context_size / 80000.0) * 0.27;
    // Clamp between 33% and 60%
    return std::max(0.33, std::min(0.6, scale));
}

/// @brief Calculate desired completion tokens to reserve for assistant responses
/// @param context_size Total context window size in tokens
/// @param max_output_tokens Model's maximum output token limit (0 = no limit)
/// @return Number of tokens to reserve for completion, capped by model's max_output_tokens
/// @details Uses calculate_truncation_scale() on half the context size to determine
///          a reasonable completion size that scales with context. Result is capped
///          at the model's max_output_tokens limit.
///          - 8K context  → 1,374 tokens
///          - 16K context → 2,856 tokens
///          - 32K+ context → 4,096 tokens (capped by model max_output_tokens)
inline int calculate_desired_completion_tokens(int context_size, int max_output_tokens) {
    if (context_size == 0) return 0;

    // Use half the context size to calculate a more conservative scale
    double scale = calculate_truncation_scale(context_size / 2);
    int desired = static_cast<int>((context_size / 2) * scale);

    // Cap at model's max_output_tokens if specified
    if (max_output_tokens > 0 && desired > max_output_tokens) {
        return max_output_tokens;
    }

    return desired;
}
