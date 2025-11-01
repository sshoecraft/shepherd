
#include "shepherd.h"
#include "llamacpp.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "nlohmann/json.hpp"
#include "minja.hpp"
#include "models.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <ctime>
#include <filesystem>

#ifdef ENABLE_LLAMACPP
#include "../llama.cpp/include/llama.h"
#include "../llama.cpp/src/llama-batch.h"
#include "../llama.cpp/common/chat.h"
#endif


// LlamaCppBackend implementation
LlamaCppBackend::LlamaCppBackend(size_t max_context_tokens)
    : Backend(max_context_tokens),
      model_config(ModelConfig::create_generic()) {
    // Set public variables per RULES.md
    backend_name = "llamacpp";
    context_size = max_context_tokens;
    is_local = true;  // Local GPU backend
    // model_name will be set in initialize_old()

    LOG_DEBUG("LlamaCppBackend created with context_size: " + std::to_string(context_size));

    // Parse backend-specific config
    std::string backend_cfg = config->backend_config(backend_name);
    parse_backend_config(backend_cfg);
}

void LlamaCppBackend::parse_backend_config(const std::string& json) {
    if (json.empty() || json == "{}") {
        return;  // No config, use defaults
    }

    try {
        auto j = nlohmann::json::parse(json);

        if (j.contains("temperature")) temperature = j["temperature"].get<float>();
        if (j.contains("top_p")) top_p = j["top_p"].get<float>();
        if (j.contains("top_k")) top_k = j["top_k"].get<int>();
        if (j.contains("min_keep")) min_keep = j["min_keep"].get<int>();
        if (j.contains("penalty_repeat")) penalty_repeat = j["penalty_repeat"].get<float>();
        if (j.contains("penalty_freq")) penalty_freq = j["penalty_freq"].get<float>();
        if (j.contains("penalty_present")) penalty_present = j["penalty_present"].get<float>();
        if (j.contains("penalty_last_n")) penalty_last_n = j["penalty_last_n"].get<int>();
        if (j.contains("gpu_layers")) gpu_layers = j["gpu_layers"].get<int>();
        if (j.contains("tensor_parallel")) tensor_parallel = j["tensor_parallel"].get<int>();
        if (j.contains("pipeline_parallel")) pipeline_parallel = j["pipeline_parallel"].get<int>();

        LOG_DEBUG("Loaded llamacpp backend config: temperature=" + std::to_string(temperature) +
                  ", gpu_layers=" + std::to_string(gpu_layers) +
                  ", tensor_parallel=" + std::to_string(tensor_parallel));

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse llamacpp backend config: " + std::string(e.what()));
    }
}

void LlamaCppBackend::log_token_state(const std::string& context) const {
    // Will be updated in later step - needs Session reference
}

int LlamaCppBackend::count_tokens_in_text(const std::string& text) const {
#ifdef ENABLE_LLAMACPP
    if (model) {
        // Use actual llama.cpp tokenization with vocab
        const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model));
        // Negative return value gives us the token count
        // Note: add_special=false because minja template handles special tokens
        int n_tokens = -llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, false, true);
        return n_tokens;
    }
#endif
    // Fallback approximation when model not loaded
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

int LlamaCppBackend::count_message_tokens(Message::Type type,
                                         const std::string& content,
                                         const std::string& tool_name,
                                         const std::string& tool_id) {
#ifdef ENABLE_LLAMACPP
    if (!model || !template_node) {
        // Fallback to simple text tokenization if model not initialized
        return count_tokens_in_text(content);
    }

    // Format message through minja template to get exact token count
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);
    auto context = minja::Context::builtins();

    // Build messages array with just this message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();

    // Convert Message::Type to role string
    std::string role;
    switch (type) {
        case Message::SYSTEM:
            role = "system";
            break;
        case Message::USER:
            role = "user";
            break;
        case Message::ASSISTANT:
            role = "assistant";
            break;
        case Message::TOOL:
            role = "tool";
            break;
        default:
            role = "user";
            break;
    }

    msg_obj.set("role", minja::Value(role));
    msg_obj.set("content", minja::Value(content));

    // Add tool metadata if present
    if (!tool_name.empty()) {
        msg_obj.set("name", minja::Value(tool_name));
    }
    if (!tool_id.empty()) {
        msg_obj.set("tool_call_id", minja::Value(tool_id));
    }

    messages.push_back(msg_obj);

    // Set template context
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(false));

    // Render through template
    std::string rendered;
    try {
        rendered = (*template_ptr)->render(context);
    } catch (const std::exception& e) {
        LOG_WARN("Exception rendering message through template: " + std::string(e.what()));
        // Fallback to simple tokenization
        return count_tokens_in_text(content);
    }

    // Tokenize the rendered message to get exact count
    const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model));
    std::vector<llama_token> tokens(rendered.length() + 256);
    int n_tokens = llama_tokenize(vocab, rendered.c_str(), rendered.length(),
                                   tokens.data(), tokens.size(), false, true);

    if (n_tokens < 0) {
        LOG_WARN("Failed to tokenize rendered message");
        return count_tokens_in_text(content);
    }

    LOG_DEBUG("count_message_tokens: role=" + role +
              ", content_len=" + std::to_string(content.length()) +
              ", rendered_len=" + std::to_string(rendered.length()) +
              ", tokens=" + std::to_string(n_tokens));

    return n_tokens;
#else
    return count_tokens_in_text(content);
#endif
}

LlamaCppBackend::~LlamaCppBackend() {
    shutdown();
}

ModelConfig LlamaCppBackend::get_model_config() const {
#ifdef ENABLE_LLAMACPP
    return model_config;
#else
    return ModelConfig::create_generic();
#endif
}

std::vector<std::string> LlamaCppBackend::get_tool_call_markers() const {
#ifdef ENABLE_LLAMACPP
    return tool_call_markers;
#else
    return {};
#endif
}

bool LlamaCppBackend::is_ready() const {
#ifdef ENABLE_LLAMACPP
    return initialized; // && model && model_ctx;
#else
    return false;
#endif
}

uint32_t LlamaCppBackend::evict_to_free_space(uint32_t tokens_needed) {
#ifdef ENABLE_LLAMACPP
    LOG_INFO("KV cache full - need to free " + std::to_string(tokens_needed) + " tokens");

    if (!current_session) {
        LOG_ERROR("No current session - cannot evict");
        return UINT32_MAX;
    }

    // Debug: Check KV cache actual usage
    int kv_used = get_context_token_count();
    int cached_tokens = 0;
    for (const auto& msg : current_session->messages) {
        cached_tokens += msg.tokens;
    }

    LOG_DEBUG("KV cache state: used_tokens=" + std::to_string(kv_used) +
              ", max_ctx=" + std::to_string(context_size) +
              ", cached_tokens=" + std::to_string(cached_tokens) +
              ", session_messages=" + std::to_string(current_session->messages.size()));

    // In server mode, throw exception instead of evicting
    // Client is responsible for managing context window
    if (g_server_mode) {
        LOG_ERROR("KV cache full in server mode - throwing ContextFullException");
        // Format error in OpenAI-compatible format
        // Use actual KV cache size, not cached_tokens which doesn't include formatting overhead
        int total_needed = kv_used + tokens_needed;
        throw ContextFullException("This model's maximum context length is " +
            std::to_string(context_size) + " tokens. However, your messages resulted in " +
            std::to_string(total_needed) + " tokens.");
    }

    // Get KV cache memory handle for eviction operations
    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_memory_t mem = llama_get_memory(ctx);

    LOG_DEBUG("Found " + std::to_string(current_session->messages.size()) + " messages in KV cache");

    // Use current_session for eviction calculation
    // Cast away const for eviction operation
    Session* mutable_session = const_cast<Session*>(current_session);
    auto [start_msg, end_msg] = mutable_session->calculate_messages_to_evict(tokens_needed);

    if (start_msg == -1 || end_msg == -1) {
        LOG_ERROR("Cannot calculate messages to evict - no space can be freed");
        return 0;  // Signal failure: eviction cannot proceed
    }

    // Calculate token positions for these messages
    int start_pos = 0;
    for (int i = 0; i < start_msg; i++) {
        start_pos += current_session->messages[i].tokens;
    }

    int end_pos = start_pos;
    for (int i = start_msg; i <= end_msg; i++) {
        end_pos += current_session->messages[i].tokens;
    }
    end_pos--; // end_pos is inclusive

    dprintf(3, "EVICT range: messages[%d,%d] = KV positions[%d,%d]\n",
            start_msg, end_msg, start_pos, end_pos);

    LOG_INFO("Evicting messages [" + std::to_string(start_msg) + ", " + std::to_string(end_msg) +
             "] = tokens [" + std::to_string(start_pos) + ", " + std::to_string(end_pos) + "]");

    // Call llama.cpp API to evict from KV cache
    llama_memory_seq_rm(mem, 0, start_pos, end_pos + 1); // +1 because llama uses exclusive end

    int tokens_removed = end_pos - start_pos + 1;

    // CRITICAL: Shift remaining tokens down to keep positions contiguous
    llama_memory_seq_add(mem, 0, end_pos + 1, -1, -tokens_removed);
    LOG_DEBUG("Shifted KV cache positions >= " + std::to_string(end_pos + 1) + " down by " + std::to_string(tokens_removed));

    // Archive to RAG and remove messages from session
    if (!mutable_session->evict_messages(start_msg, end_msg)) {
        LOG_ERROR("Failed to evict messages from session");
        return UINT32_MAX;
    }

    int num_messages = end_msg - start_msg + 1;
    LOG_DEBUG("Evicted " + std::to_string(num_messages) + " messages from cache");
    LOG_INFO("Successfully evicted " + std::to_string(num_messages) + " messages (" +
             std::to_string(tokens_removed) + " tokens) from KV cache");

    // KV cache is the source of truth - query actual state
    dprintf(2, "KV cache after eviction: %d tokens\n", get_context_token_count());
    log_token_state("After eviction");

    // Return the new head position (where freed space begins) as required by callback API
    LOG_DEBUG("Eviction complete - returning new head position: " + std::to_string(start_pos));
    return static_cast<uint32_t>(start_pos);
#else
    LOG_ERROR("llama.cpp not enabled");
    return UINT32_MAX;
#endif
}

void LlamaCppBackend::shutdown() {
    if (!initialized) {
        return;
    }

#ifdef ENABLE_LLAMACPP
    // Cleanup llama.cpp resources properly
    if (model_ctx) {
        llama_free(static_cast<llama_context*>(model_ctx));
        model_ctx = nullptr;
    }
    if (model) {
        llama_model_free(static_cast<llama_model*>(model));
        model = nullptr;
    }

    // Cleanup chat templates
    if (chat_templates) {
        common_chat_templates_free(static_cast<common_chat_templates*>(chat_templates));
        chat_templates = nullptr;
    }
#endif

    initialized = false;
    LOG_DEBUG("LlamaCppBackend shutdown complete");
}

int LlamaCppBackend::get_context_token_count() const {
#ifdef ENABLE_LLAMACPP
    if (!model_ctx) {
        return 0;
    }

    // Query actual KV cache state - this is the source of truth
    llama_memory_t mem = llama_get_memory(static_cast<llama_context*>(model_ctx));
    llama_pos actual_max_pos = llama_memory_seq_pos_max(mem, 0);  // sequence 0

    // max_pos is the highest position (0-based), so +1 for token count
    // Return -1 (empty cache) as 0
    return (actual_max_pos >= 0) ? (actual_max_pos + 1) : 0;
#else
    return 0;
#endif
}

std::string LlamaCppBackend::generate(int max_tokens) {
    LOG_DEBUG("=== GENERATE START ===");
    if (!is_ready()) {
        throw std::runtime_error("LlamaCpp backend not initialized");
    }

    LOG_DEBUG("Getting tools from registry");
    auto& registry = ToolRegistry::instance();
    auto available_tools = registry.list_tools();

    // Extract tool call markers from chat template on first call
    if (!have_tool_markers && chat_templates && current_session) {
        try {
            common_chat_templates_inputs inputs{};

            // Convert session messages (all messages in current_session are in KV cache)
            for (int i = 0; i < static_cast<int>(current_session->messages.size()); i++) {
                const auto& msg = current_session->messages[i];
                common_chat_msg tmpl_msg;
                tmpl_msg.role = msg.get_role();
                tmpl_msg.content = msg.content;
                inputs.messages.push_back(tmpl_msg);
            }

            // Add "python" builtin tool to trigger preserved_tokens
            common_chat_tool tmpl_tool;
            tmpl_tool.name = "python";
            tmpl_tool.description = "Execute python code";
            tmpl_tool.parameters = R"({"type":"object","properties":{"code":{"type":"string"}},"required":["code"]})";
            inputs.tools.push_back(tmpl_tool);

            inputs.add_generation_prompt = true;

            // Apply template to get preserved_tokens
            auto params = common_chat_templates_apply(
                static_cast<common_chat_templates*>(chat_templates),
                inputs
            );

            if (!params.preserved_tokens.empty()) {
                tool_call_markers = params.preserved_tokens;
                LOG_INFO("Extracted " + std::to_string(tool_call_markers.size()) + " tool call markers from template:");
                for (const auto& marker : tool_call_markers) {
                    LOG_INFO("  - " + marker);
                }
            } else {
                LOG_INFO("No preserved_tokens from template - checking vocabulary");

                // Check if model has <|python_tag|> in its vocabulary
                std::vector<llama_token> tokens;
                tokens.resize(8);  // Should only need 1-2 tokens max
                std::string test_marker = "<|python_tag|>";
                const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model));
                int n_tokens = llama_tokenize(
                    vocab,
                    test_marker.c_str(),
                    test_marker.length(),
                    tokens.data(),
                    tokens.size(),
                    false,  // add_special
                    true    // parse_special
                );

                // If it tokenizes to exactly 1 token, it's a special token in vocabulary
                if (n_tokens == 1) {
                    tool_call_markers.push_back(test_marker);
                    LOG_INFO("Found <|python_tag|> in model vocabulary (token " + std::to_string(tokens[0]) + ")");
                } else {
                    LOG_INFO("Model does not have <|python_tag|> special token (tokenized to " + std::to_string(n_tokens) + " tokens)");
                }
            }

            have_tool_markers = true;
        } catch (const std::exception& e) {
            LOG_INFO("Exception during tool marker extraction: " + std::string(e.what()));
            have_tool_markers = true;
        }
    }

    // All messages are already decoded in KV cache from add_*_message() calls
    // We just need to run the generation loop now

    // In debug mode, show what's in the KV cache
    if (g_debug_level && current_session) {
        LOG_DEBUG("=== MESSAGES IN KV CACHE ===");
        char line[128];
        for (int i = 0; i < static_cast<int>(current_session->messages.size()); i++) {
            const auto& msg = current_session->messages[i];
            line[0] = 0;
            // Format: "[role] content..." with max 128 chars total, newlines replaced with spaces

            // Replace newlines with spaces in content
            std::string content_clean = msg.content;
            for (size_t j = 0; j < content_clean.length(); j++) {
                if (content_clean[j] == '\n' || content_clean[j] == '\r') {
                    content_clean[j] = ' ';
                }
            }

            const char* role = msg.get_role().c_str();
            const char* content = content_clean.c_str();

            // Truncate content to fit in remaining space
            int prefix_len = snprintf(line, sizeof(line), "[%s] ", role);
            if (prefix_len > 0 && prefix_len < (int)sizeof(line) - 4) {
                int remaining = sizeof(line) - prefix_len - 4; // -4 for "..." + null
                if (content_clean.length() > (size_t)remaining) {
                    strncat(line, content, remaining);
                    strcat(line, "...");
                } else {
                    strcat(line, content);
                }
            }
            LOG_DEBUG(line);
        }
        LOG_DEBUG("=== END KV CACHE ===");
    }

    LOG_DEBUG("Running generation (messages already cached)");

    // Before generation, we need to add the generation prompt to KV cache
    // Use model-specific tag from config (no hardcoded model family checks!)
    std::string generation_prompt = model_config.assistant_start_tag;

    // Declare these outside the if block so they're accessible later
    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_model* mdl = static_cast<llama_model*>(this->model);
    int n_tokens = 0;

    // Tokenize and decode the generation prompt into KV cache (if not empty)
    if (!generation_prompt.empty()) {
        const llama_vocab* vocab = llama_model_get_vocab(mdl);

        std::vector<llama_token> prompt_tokens(generation_prompt.length() + 256);
        n_tokens = llama_tokenize(vocab, generation_prompt.c_str(), generation_prompt.length(),
                                   prompt_tokens.data(), prompt_tokens.size(), false, true);

        if (n_tokens > 0) {
            prompt_tokens.resize(n_tokens);
            LOG_DEBUG("Decoding generation prompt (" + generation_prompt + "): " + std::to_string(n_tokens) + " tokens");

            for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
                int batch_size = std::min(n_batch, static_cast<int>(prompt_tokens.size() - i));
                llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERROR("Failed to decode generation prompt at position " + std::to_string(i));
                }
            }
        }
    }

    // Suppress streaming when tools are available OR in server mode
    // Server mode should NEVER stream output (it goes to server.log)
    bool suppress_stream = g_server_mode || !available_tools.empty();
    std::string raw_response = run_inference("", max_tokens, suppress_stream);  // Empty prompt since everything is cached
    LOG_DEBUG("Got raw response length: " + std::to_string(raw_response.length()));
    LOG_DEBUG("Raw response: " + raw_response);

    // CRITICAL FIX: After generation, add the closing tag to KV cache
    // The generation prompt added assistant_start_tag, and run_inference() generated tokens,
    // but we never added the assistant_end_tag closing tag! This causes malformed context on next generate()
    int n_closing = 0;
    if (!model_config.assistant_end_tag.empty()) {
        // Tokenize and add the closing tag to KV cache
        const llama_vocab* vocab = llama_model_get_vocab(mdl);
        std::vector<llama_token> closing_tokens(16);
        n_closing = llama_tokenize(vocab, model_config.assistant_end_tag.c_str(),
                                    model_config.assistant_end_tag.length(),
                                    closing_tokens.data(), closing_tokens.size(), false, true);

        if (n_closing > 0) {
            closing_tokens.resize(n_closing);
            llama_batch closing_batch = llama_batch_get_one(closing_tokens.data(), n_closing);
            if (llama_decode(ctx, closing_batch) != 0) {
                LOG_WARN("Failed to decode closing tag into KV cache");
            } else {
                LOG_DEBUG("Added closing tag to KV cache: " + model_config.assistant_end_tag);
            }
        }
    }

    // Store prompt token count for server to return (like API backends do)
    // This is the total number of tokens in the KV cache before generation
    // Calculate total tokens from current session messages
    int total_tokens = 0;
    if (current_session) {
        for (const auto& msg : current_session->messages) {
            total_tokens += msg.tokens;
        }
    }
    last_prompt_tokens = total_tokens;

    // Store the actual tokens added to KV cache for this assistant message
    // This includes: generation_prompt + generated_tokens + closing_tag
    // Note: last_completion_tokens is set by run_inference() to n_generated
    last_assistant_kv_tokens = n_tokens + last_completion_tokens + n_closing;

    // Return response directly - main will handle tool parsing and cleanup
    return raw_response;
}

std::string LlamaCppBackend::run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming) {
#ifdef ENABLE_LLAMACPP
    // Reset cancellation flag at start of generation
    g_generation_cancelled = false;

    if (!model || !model_ctx) {
        LOG_ERROR("llama.cpp model or context not initialized");
        return "Error: Model not initialized";
    }

    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_model* mdl = static_cast<llama_model*>(this->model);
    const llama_vocab* vocab = llama_model_get_vocab(mdl);

    // Initialize sampler chain with configured sampling parameters
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add samplers in the recommended order (from llama.cpp examples)
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, min_keep));

    // Add repetition penalties BEFORE temperature to discourage repetitive patterns
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        penalty_last_n,      // last n tokens to penalize
        penalty_repeat,      // repetition penalty (1.0 = disabled)
        penalty_freq,        // frequency penalty (0.0 = disabled)
        penalty_present));   // presence penalty (0.0 = disabled)

    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));           // greedy sampling from dist

    LOG_DEBUG("Sampling params: temperature=" + std::to_string(temperature) +
              ", top_p=" + std::to_string(top_p) +
              ", top_k=" + std::to_string(top_k) +
              ", min_keep=" + std::to_string(min_keep) +
              ", penalty_repeat=" + std::to_string(penalty_repeat) +
              ", penalty_freq=" + std::to_string(penalty_freq) +
              ", penalty_present=" + std::to_string(penalty_present) +
              ", penalty_last_n=" + std::to_string(penalty_last_n));

    // If prompt_text is empty, everything is already in KV cache
    // Just skip straight to generation
    if (!prompt_text.empty()) {
        // Legacy path: tokenize and decode prompt
        // This is kept for backward compatibility but shouldn't be used with stateful KV cache
        LOG_WARN("run_inference() called with non-empty prompt - this is wasteful with stateful KV cache");

        std::vector<llama_token> prompt_tokens(prompt_text.length() + 256);
        int n_prompt_tokens = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.length(),
                                             prompt_tokens.data(), prompt_tokens.size(), false, true);

        if (n_prompt_tokens < 0) {
            LOG_ERROR("Failed to tokenize input text");
            llama_sampler_free(sampler);
            return "Error: Tokenization failed";
        }

        prompt_tokens.resize(n_prompt_tokens);
        LOG_DEBUG("Evaluating " + std::to_string(n_prompt_tokens) + " prompt tokens");

        // Check if prompt alone is larger than entire context window
        if (n_prompt_tokens > static_cast<int>(context_size)) {
            LOG_ERROR("Prompt too large for context: " + std::to_string(n_prompt_tokens) + " tokens exceeds max context size " +
                      std::to_string(context_size));
            llama_sampler_free(sampler);
            if (g_server_mode) {
                throw ContextFullException("This model's maximum context length is " +
                    std::to_string(context_size) + " tokens. However, your messages resulted in " +
                    std::to_string(n_prompt_tokens) + " tokens.");
            }
            return "Error: Prompt too large for context window";
        }

        // Evaluate prompt tokens in batches using configured batch size
        for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
            // Check for cancellation between batches
            if (g_generation_cancelled) {
                LOG_INFO("Generation cancelled during prompt processing");
                llama_sampler_free(sampler);
                return "";
            }

            int batch_size = std::min(n_batch, static_cast<int>(prompt_tokens.size() - i));

            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERROR("Failed to evaluate prompt tokens at position " + std::to_string(i));
                llama_sampler_free(sampler);
                return "Error: Evaluation failed";
            }
        }
    } else {
        LOG_DEBUG("Prompt already cached, skipping tokenization/decoding");
    }

    // Generate tokens
    std::string response;
    int n_generated = 0;

    // Calculate max generation tokens: context_size - system - current_user
    // These token counts include ALL template overhead (saved when messages were decoded)
    int available_for_generation = static_cast<int>(context_size) - system_formatted_tokens - current_user_formatted_tokens;

    // Use explicit max_tokens if provided, otherwise use all available space
    int max_gen_tokens = max_tokens > 0 ? max_tokens : available_for_generation;

    LOG_DEBUG("Generation limits: available=" + std::to_string(available_for_generation) +
              " (context=" + std::to_string(context_size) +
              " - system=" + std::to_string(system_formatted_tokens) +
              " - user=" + std::to_string(current_user_formatted_tokens) +
              "), max_gen_tokens=" + std::to_string(max_gen_tokens));

    // Start timing for t/s measurement
    auto gen_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_gen_tokens; i++) {
        // Check for cancellation (from SIGUSR1 signal)
        if (g_generation_cancelled) {
            LOG_INFO("Generation cancelled by signal");
            break;
        }

        // Sample next token
        llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

        // Check for end of generation using llama.cpp's native EOG detection
        // This handles all model-specific end tokens automatically
        if (llama_vocab_is_eog(vocab, next_token)) {
            LOG_DEBUG("End of generation token detected");
            break;
        }

        // Accept the token
        llama_sampler_accept(sampler, next_token);

        // Convert token to text (filter special tokens with false parameter)
        char token_str[256];
        int token_len = llama_token_to_piece(vocab, next_token, token_str, sizeof(token_str), 0, false);

        if (token_len > 0) {
            // Accumulate for final response
            response.append(token_str, token_len);

            // Stream output only if not suppressed (suppressed for tool calls)
            if (!suppress_streaming) {
                std::cout << std::string(token_str, token_len) << std::flush;
            }
        }

        // Evaluate the generated token - retry once if eviction happens
        llama_batch single_batch = llama_batch_get_one(&next_token, 1);

        bool decode_ok = false;
        for (int retry = 0; retry < 2; retry++) {
            if (llama_decode(ctx, single_batch) == 0) {
                decode_ok = true;
                break;
            }
            if (retry == 0) {
                LOG_DEBUG("Token decode failed (likely eviction), retrying");
            }
        }

        if (!decode_ok) {
            LOG_ERROR("Failed to evaluate generated token after retries");
            break;
        }

        n_generated++;
    }

    // End timing and calculate t/s
    auto gen_end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end_time - gen_start_time);
    double seconds = duration.count() / 1000.0;
    double tokens_per_second = (n_generated > 0 && seconds > 0) ? (n_generated / seconds) : 0.0;

    llama_sampler_free(sampler);

    // Add newline after streaming is complete
    if (n_generated > 0) {
        std::cout << std::endl;
    }

    // Always show performance metrics to stderr (visible even without debug mode)
    if (n_generated > 0) {
        int context_used = get_context_token_count();
        int context_max = context_size;
        std::cerr << "\033[90m[Decode: " << n_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tokens_per_second << " t/s, "
                  << "context: " << context_used << "/" << context_max << "]\033[0m" << std::endl;
    }

    LOG_INFO("Generation (decode): " + std::to_string(n_generated) + " tokens in " +
             std::to_string(seconds) + "s (" + std::to_string(tokens_per_second) + " t/s)");

    // Store completion token count for server to return (like API backends do)
    last_completion_tokens = n_generated;

    return response;
#else
    // Fallback when llama.cpp not available
    LOG_ERROR("llama.cpp not compiled in");
    return "Error: llama.cpp not available";
#endif
}





std::map<std::string, std::any> LlamaCppBackend::parse_json_to_args(const std::string& args_str) {
    std::map<std::string, std::any> args;

    LOG_DEBUG("Parsing tool arguments: " + args_str);

    // Try JSON parsing first
    if (args_str.find('{') != std::string::npos && args_str.find('}') != std::string::npos) {
        try {
            // Extract "key": "value" patterns from JSON-like format
            std::regex json_regex("\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\"");
            std::sregex_iterator iter(args_str.begin(), args_str.end(), json_regex);
            std::sregex_iterator end;

            for (; iter != end; ++iter) {
                std::smatch match = *iter;
                args[match[1].str()] = match[2].str();
            }

            if (!args.empty()) {
                LOG_DEBUG("Successfully parsed JSON-style arguments");
                return args;
            }
        } catch (...) {
            LOG_DEBUG("JSON parsing failed, trying other formats");
        }
    }

    // Try simple key=value format
    std::regex kv_regex("(\\w+)\\s*=\\s*\"([^\"]*)\"");
    std::sregex_iterator iter(args_str.begin(), args_str.end(), kv_regex);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        args[match[1].str()] = match[2].str();
    }

    if (!args.empty()) {
        LOG_DEBUG("Successfully parsed key=value arguments");
        return args;
    }

    // If no structured format found, treat the entire string as a single argument
    // This handles cases where the model outputs just the value without structure
    if (!args_str.empty()) {
        // Try to infer the parameter based on common patterns
        if (args_str.find('/') != std::string::npos || args_str.find('.') != std::string::npos) {
            args["path"] = args_str;
        } else {
            args["command"] = args_str;
        }
        LOG_DEBUG("Using fallback argument parsing");
    }

    return args;
}

std::string LlamaCppBackend::render_message(const Message& msg, bool add_generation_prompt) {
#ifdef ENABLE_LLAMACPP
    if (!template_node) {
        LOG_ERROR("No template node available, falling back to simple format");
        return msg.get_role() + ": " + msg.content + "\n\n";
    }

    // Get the actual template node from the void pointer
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);

    // Create minja context
    auto context = minja::Context::builtins();

    // Provide strftime_now function for current date
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        if (args.args.empty()) return minja::Value("");
        std::string format = args.args[0].get<std::string>();

        time_t now = time(nullptr);
        struct tm* tm_info = localtime(&now);
        char buffer[128];
        strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
        return minja::Value(std::string(buffer));
    });
    context->set("strftime_now", strftime_now);

    // Set date_string directly (like llama.cpp does)
    time_t now = time(nullptr);
    struct tm* tm_info = localtime(&now);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%d %b %Y", tm_info);
    context->set("date_string", minja::Value(std::string(date_buffer)));

    // Convert single message to minja array with just one element
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();
    msg_obj.set("role", minja::Value(msg.get_role()));
    msg_obj.set("content", minja::Value(msg.content));

    // Add tool metadata if present
    if (!msg.tool_name.empty()) {
        msg_obj.set("name", minja::Value(msg.tool_name));
    }
    if (!msg.tool_call_id.empty()) {
        msg_obj.set("tool_call_id", minja::Value(msg.tool_call_id));
    }

    messages.push_back(msg_obj);

    context->set("bos_token", minja::Value("<|begin_of_text|>"));
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(add_generation_prompt));

    std::string rendered = (*template_ptr)->render(context);
    LOG_DEBUG("Rendered message (" + std::to_string(rendered.length()) + " chars, role=" + msg.get_role() + ")");
    return rendered;
#else
    // Fallback when llama.cpp not available
    return msg.get_role() + ": " + msg.content + "\n\n";
#endif
}

bool LlamaCppBackend::format_and_decode_message(Message& msg) {
#ifdef ENABLE_LLAMACPP
    if (!model || !model_ctx) {
        LOG_ERROR("Model or context not initialized");
        return false;
    }

    // Render just this single message through the template
    std::string rendered_msg = render_message(msg, false);

    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_model* mdl = static_cast<llama_model*>(model);
    const llama_vocab* vocab = llama_model_get_vocab(mdl);

    // Tokenize the rendered message
    std::vector<llama_token> msg_tokens(rendered_msg.length() + 256);
    int n_tokens = llama_tokenize(vocab, rendered_msg.c_str(), rendered_msg.length(),
                                   msg_tokens.data(), msg_tokens.size(), false, true);

    if (n_tokens < 0) {
        LOG_ERROR("Failed to tokenize message");
        return false;
    }
    msg_tokens.resize(n_tokens);

    // CRITICAL: Update message token count to the FORMATTED count (includes all template overhead)
    // This is the actual number of tokens in KV cache, not just the raw content
    msg.tokens = n_tokens;

    LOG_DEBUG("Decoding " + std::to_string(n_tokens) + " tokens for new message (updated msg.tokens)");

    // Check if message alone is larger than entire context window
    if (n_tokens > static_cast<int>(context_size)) {
        LOG_ERROR("Message too large for context: " + std::to_string(n_tokens) + " tokens exceeds max context size " +
                  std::to_string(context_size));
        if (g_server_mode) {
            throw ContextFullException("This model's maximum context length is " +
                std::to_string(context_size) + " tokens. However, your messages resulted in " +
                std::to_string(n_tokens) + " tokens.");
        }
        return false;
    }

    // Start timing for prompt processing (prefill) speed
    auto prefill_start_time = std::chrono::high_resolution_clock::now();

    // Decode the message tokens in batches
    // Retry once if decode fails (likely due to KV cache eviction mid-operation)
    const int MAX_DECODE_RETRIES = 1;
    int retry_count = 0;

    while (retry_count <= MAX_DECODE_RETRIES) {
        bool decode_failed = false;

        for (size_t i = 0; i < msg_tokens.size(); i += n_batch) {
            int batch_size = std::min(n_batch, static_cast<int>(msg_tokens.size() - i));

            llama_batch batch = llama_batch_get_one(msg_tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch) != 0) {
                if (retry_count < MAX_DECODE_RETRIES) {
                    LOG_WARN("Decode failed at position " + std::to_string(i) + ", retrying (attempt " +
                            std::to_string(retry_count + 1) + "/" + std::to_string(MAX_DECODE_RETRIES + 1) + ")");
                    decode_failed = true;
                    break;
                } else {
                    LOG_ERROR("Failed to decode message tokens after " + std::to_string(MAX_DECODE_RETRIES + 1) + " attempts");
                    return false;
                }
            }
        }

        if (!decode_failed) {
            break; // Success!
        }

        retry_count++;
    }

    // End timing and calculate prompt processing (prefill) speed
    auto prefill_end_time = std::chrono::high_resolution_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time - prefill_start_time);
    double prefill_seconds = prefill_duration.count() / 1000.0;
    double prefill_tokens_per_second = (n_tokens > 0 && prefill_seconds > 0) ? (n_tokens / prefill_seconds) : 0.0;

    // KV cache is the source of truth - use get_context_token_count() to query actual state
    dprintf(2, "KV cache after decode: %d tokens\n", get_context_token_count());
    log_token_state("After decode to KV cache");

    // Always show performance metrics to stderr (visible even without debug mode)
    // Include context window utilization
    int context_used = get_context_token_count();
    int context_max = context_size;
    std::cerr << "\033[90m[Prefill: " << n_tokens << " tokens, "
              << std::fixed << std::setprecision(1) << prefill_tokens_per_second << " t/s, "
              << "context: " << context_used << "/" << context_max << "]\033[0m" << std::endl;

    LOG_INFO("Prompt processing: " + std::to_string(n_tokens) + " tokens in " +
             std::to_string(prefill_seconds) + "s (" + std::to_string(prefill_tokens_per_second) + " t/s)");

    return true;
#else
    return false;
#endif
}




std::string LlamaCppBackend::format_system_message_with_tools(const std::string& system_content,
                                                               const std::vector<Session::Tool>& tools) {
    // For models that don't need special tool formatting (most models)
    if (model_config.family != ModelFamily::QWEN_2_X &&
        model_config.family != ModelFamily::LLAMA_3_X) {
        // Simple format: append tool descriptions to system message
        if (tools.empty()) {
            return system_content;
        }

        std::string result = system_content;
        if (!result.empty() && result.back() != '\n') {
            result += "\n\n";
        }

        result += "Here are the available tools:\n\n";
        for (const auto& tool : tools) {
            result += "- " + tool.name + ": " + tool.description +
                      " (parameters: " + tool.parameters_text() + ")\n";
        }

        // Add tool call format instructions
        result += "\n**IMPORTANT - Tool Call Format:**\n";
        result += "Tool calls must be raw JSON as the ENTIRE message (no markdown, no text, no quotes).\n";
        result += "Format: {\"name\": \"tool_name\", \"parameters\": {\"key\": \"value\"}}\n";
        result += "Use \"name\" (not \"tool_name\") and \"parameters\" (not \"params\").\n";
        result += "If a tool fails, continue gracefully without retrying indefinitely.\n";

        return result;
    }

    // For Qwen 2.x: Use minja template to format tools
    if (model_config.family == ModelFamily::QWEN_2_X && template_node) {
        // Build tools array for minja
        auto tools_array = minja::Value::array();
        for (const auto& tool : tools) {
            auto tool_obj = minja::Value::object();
            tool_obj.set("name", minja::Value(tool.name));
            tool_obj.set("description", minja::Value(tool.description));

            // Set parameters - pass the JSON object directly (not as string)
            // The template expects a proper object structure, not a JSON string
            if (!tool.parameters.empty()) {
                tool_obj.set("parameters", minja::Value(tool.parameters));
            }

            tools_array.push_back(tool_obj);
        }

        // Use Qwen's special tool system format
        auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);
        auto context = minja::Context::builtins();

        // Set tools in context - try both variable names
        context->set("tools", tools_array);
        context->set("builtin_tools", tools_array);

        // Render the system message with tools
        std::string qwen_system = system_content.empty() ? "You are a helpful assistant." : system_content;

        auto messages = minja::Value::array();
        auto system_msg = minja::Value::object();
        system_msg.set("role", minja::Value("system"));
        system_msg.set("content", minja::Value(qwen_system));
        messages.push_back(system_msg);

        context->set("messages", messages);
        context->set("add_generation_prompt", minja::Value(false));

        try {
            auto rendered = (*template_ptr)->render(context);
            if (rendered.empty()) {
                LOG_ERROR("Failed to render Qwen system message with tools");
                return system_content;  // Fallback
            }
            return rendered;
        } catch (const std::exception& e) {
            LOG_ERROR("Exception rendering Qwen system message: " + std::string(e.what()));
            return system_content;  // Fallback
        }
    }

    // For Llama 3.x: Format tools as JSON schemas in first user message
    if (model_config.family == ModelFamily::LLAMA_3_X) {
        // Llama 3.x puts tools in the first user message, not system
        // Return unmodified system message
        return system_content;
    }

    // Default fallback
    return system_content;
}



Response LlamaCppBackend::generate_from_session(const Session& session, int max_tokens) {
#ifdef ENABLE_LLAMACPP
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "LlamaCpp backend not initialized";
        return err_resp;
    }

    LOG_DEBUG("LlamaCpp generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // PREFIX CACHING: Compare incoming session with backend_session (what's in KV cache)
    size_t cached_count = backend_session.messages.size();

    LOG_DEBUG("Backend has " + std::to_string(cached_count) + " cached messages, " +
              "incoming session has " + std::to_string(session.messages.size()) + " messages");

    // Account for system message offset: backend_session may have SYSTEM at [0], but session doesn't
    size_t backend_offset = 0;
    if (!backend_session.messages.empty() && backend_session.messages[0].type == Message::SYSTEM) {
        backend_offset = 1;
        LOG_DEBUG("Backend has system message at index 0, offsetting comparison by 1");
    }

    // Find how many messages match (prefix caching)
    // Compare session.messages[i] with backend_session.messages[i + backend_offset]
    size_t matching_prefix = 0;
    for (size_t i = 0; i < session.messages.size(); i++) {
        size_t backend_idx = i + backend_offset;

        if (backend_idx >= cached_count) {
            // No more cached messages to compare
            break;
        }

        const auto& cached_msg = backend_session.messages[backend_idx];
        const auto& session_msg = session.messages[i];

        // Compare role and content to see if they match
        if (cached_msg.get_role() == session_msg.get_role() &&
            cached_msg.content == session_msg.content) {
            matching_prefix++;
        } else {
            // Messages diverged
            LOG_WARN("DIVERGENCE at session message " + std::to_string(i) + " (backend index " + std::to_string(backend_idx) + "):");
            LOG_WARN("  Cached: [" + cached_msg.get_role() + "] " + cached_msg.content.substr(0, 50) + "...");
            LOG_WARN("  Session: [" + session_msg.get_role() + "] " + session_msg.content.substr(0, 50) + "...");
            break;
        }
    }

    LOG_DEBUG("Prefix match: " + std::to_string(matching_prefix) + " messages (out of " + std::to_string(session.messages.size()) + " in session)");

    // Check if backend has more messages than what matched
    // Keep: system message (if present) + matching_prefix messages
    size_t expected_backend_count = backend_offset + matching_prefix;

    if (cached_count > expected_backend_count) {
        LOG_WARN("Conversation diverged - clearing " + std::to_string(cached_count - expected_backend_count) + " cached messages");

        // Clear KV cache from divergence point onward
        llama_context* ctx = static_cast<llama_context*>(model_ctx);
        llama_memory_t mem = llama_get_memory(ctx);

        // Calculate token position to clear from
        int clear_from_pos = 0;
        for (size_t i = 0; i < expected_backend_count; i++) {
            clear_from_pos += backend_session.messages[i].tokens;
        }

        llama_memory_seq_rm(mem, 0, clear_from_pos, -1);
        LOG_DEBUG("Cleared KV cache from token position " + std::to_string(clear_from_pos));

        // Remove diverged messages from backend_session
        while (backend_session.messages.size() > expected_backend_count) {
            backend_session.messages.pop_back();
        }
    }

    // Set current_session for eviction callbacks
    current_session = &backend_session;

    // Handle system message if present (server mode sends it separately)
    if (!session.system_message.empty() &&
        (backend_session.messages.empty() || backend_session.messages[0].type != Message::SYSTEM)) {

        LOG_DEBUG("Adding system message to KV cache");

        // Format system message with tools
        std::string formatted_system = format_system_message_with_tools(
            session.system_message,
            session.tools
        );

        // Create system message
        Message sys_msg(Message::SYSTEM, formatted_system, 0);

        // Count tokens
        sys_msg.tokens = count_message_tokens(Message::SYSTEM, formatted_system, "", "");

        // Decode to KV cache
        if (!format_and_decode_message(sys_msg)) {
            LOG_ERROR("Failed to decode system message to KV cache");
            Response err_resp;
            err_resp.success = false;
            err_resp.code = Response::ERROR;
            err_resp.finish_reason = "error";
            err_resp.error = "Failed to decode system message to KV cache";
            return err_resp;
        }

        // Add to backend_session
        backend_session.messages.push_back(sys_msg);
        backend_session.system_message = session.system_message;
        LOG_DEBUG("System message added to KV cache (" + std::to_string(sys_msg.tokens) + " tokens)");
    }

    // Add NEW messages (from matching_prefix onward)
    size_t new_messages = session.messages.size() - matching_prefix;
    if (new_messages > 0) {
        LOG_DEBUG("Adding " + std::to_string(new_messages) + " new messages to KV cache");

        for (size_t i = matching_prefix; i < session.messages.size(); i++) {
            const auto& msg = session.messages[i];
            Message msg_copy = msg;

            // Set token count if not already set
            if (msg_copy.tokens == 0) {
                msg_copy.tokens = count_message_tokens(msg_copy.type, msg_copy.content,
                                                        msg_copy.tool_name, msg_copy.tool_call_id);
            }

            // Decode to KV cache
            if (!format_and_decode_message(msg_copy)) {
                LOG_ERROR("Failed to decode message to KV cache");
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::ERROR;
                err_resp.finish_reason = "error";
                err_resp.error = "Failed to decode message to KV cache";
                return err_resp;
            }

            // Successfully decoded - add to backend_session
            backend_session.messages.push_back(msg_copy);

            // Update last_user/last_assistant for eviction protection
            if (msg.type == Message::USER) {
                backend_session.last_user_message_index = backend_session.messages.size() - 1;
                backend_session.last_user_message_tokens = msg_copy.tokens;
            } else if (msg.type == Message::ASSISTANT) {
                backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
                backend_session.last_assistant_message_tokens = msg_copy.tokens;
            }

            LOG_DEBUG("Decoded message " + std::to_string(i) + " to KV cache");
        }

        LOG_DEBUG("Prefix caching: " + std::to_string(matching_prefix) + " cached, " +
                  std::to_string(new_messages) + " new, total " + std::to_string(session.messages.size()));
    } else {
        LOG_DEBUG("All messages already in KV cache (100% prefix cache hit)");
    }

    // Copy tools to backend_session
    backend_session.tools = session.tools;
    backend_session.system_message = session.system_message;

    // Update member variables used by generate() for token calculations
    // System message tokens
    if (!backend_session.messages.empty() && backend_session.messages[0].type == Message::SYSTEM) {
        system_formatted_tokens = backend_session.messages[0].tokens;
    } else {
        system_formatted_tokens = 0;
    }

    // Last user message tokens
    if (backend_session.last_user_message_index >= 0) {
        current_user_formatted_tokens = backend_session.last_user_message_tokens;
    } else {
        current_user_formatted_tokens = 0;
    }

    LOG_DEBUG("Server mode generation - system_tokens=" + std::to_string(system_formatted_tokens) +
              ", user_tokens=" + std::to_string(current_user_formatted_tokens));

    // Generate response
    std::string result = generate(max_tokens);

    // Wrap in Response
    Response success_resp;
    success_resp.success = true;
    success_resp.code = Response::SUCCESS;
    success_resp.content = result;
    success_resp.finish_reason = "stop";
    success_resp.prompt_tokens = last_prompt_tokens;
    success_resp.completion_tokens = last_completion_tokens;
    return success_resp;
#else
    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.finish_reason = "error";
    err_resp.error = "LlamaCpp backend not compiled in";
    return err_resp;
#endif
}

// Backend interface - initialize from Session
void LlamaCppBackend::initialize(Session& session) {
#ifdef ENABLE_LLAMACPP
    if (initialized) {
        LOG_WARN("LlamaCppBackend already initialized");
        return;
    }

    // Construct full model path from config
    std::string model_filename = config->model;
    std::string model_dir = config->model_path;

    if (model_filename.empty()) {
        throw BackendError("Model name is required for LlamaCpp backend");
    }

    std::string full_model_path;

    // Check if model_filename is already a full path (absolute path)
    if (model_filename[0] == '/' || model_filename[0] == '~') {
        // Model is already a full path - use it directly
        full_model_path = model_filename;
    } else {
        // Combine model_path directory with model filename
        if (model_dir.empty()) {
            throw BackendError("Model path directory is required when model is not an absolute path");
        }
        full_model_path = (std::filesystem::path(model_dir) / model_filename).string();
    }

    // Expand ~ if present
    if (!full_model_path.empty() && full_model_path[0] == '~') {
        full_model_path = Config::get_home_directory() + full_model_path.substr(1);
    }

    LOG_INFO("Using model path: " + full_model_path);

    this->model_path = full_model_path;
    model_name = full_model_path;  // Set public variable

    // Suppress llama.cpp logging unless in debug mode (but always show errors)
    if (!g_debug_level) {
        llama_log_set([](enum ggml_log_level level, const char * text, void * user_data) {
            // Only show ERROR messages (suppress INFO and WARN unless debug enabled)
            if (level == GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        }, nullptr);
    }

    // Detect GPU support (can be disabled with GGML_METAL=0 or GGML_NO_METAL=1)
    bool has_gpu = llama_supports_gpu_offload();
    const char* disable_metal = getenv("GGML_METAL");
    const char* no_metal = getenv("GGML_NO_METAL");
    if ((disable_metal && strcmp(disable_metal, "0") == 0) ||
        (no_metal && strcmp(no_metal, "1") == 0)) {
        has_gpu = false;
        LOG_INFO("Metal GPU disabled by environment variable");
    }

    // Load actual llama.cpp model
    llama_model_params model_params = llama_model_default_params();

    if (has_gpu) {
        // Priority: environment variable > config setting > auto
        const char* gpu_layers_env = getenv("GGML_N_GPU_LAYERS");
        if (gpu_layers_env) {
            model_params.n_gpu_layers = atoi(gpu_layers_env);
            LOG_INFO("Using GPU layer count from GGML_N_GPU_LAYERS env: " + std::to_string(model_params.n_gpu_layers));
        } else if (gpu_layers >= 0) {
            model_params.n_gpu_layers = gpu_layers;
            LOG_INFO("Using GPU layer count from config: " + std::to_string(model_params.n_gpu_layers));
        } else {
            // -1 = auto: set to extremely high number - llama.cpp will automatically cap at actual layer count
            model_params.n_gpu_layers = INT32_MAX;
            LOG_INFO("Auto GPU layers (loading all layers to GPU)");
        }

        // Multi-GPU support: Use tensor_parallel config to control GPU usage
        if (tensor_parallel == 0 || tensor_parallel > 1) {
            // TP=0 (auto) or TP>1: Use multiple GPUs with LAYER split mode
            model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            model_params.main_gpu = 0;

            // Configure tensor_split array to distribute model across GPUs
            int num_gpus = tensor_parallel > 0 ? tensor_parallel : 16;  // Max 16 GPUs if auto
            tensor_split.resize(128, 0.0f);  // Initialize all to 0 (unused)
            for (int i = 0; i < num_gpus && i < 128; i++) {
                tensor_split[i] = 1.0f;  // Equal proportion for each GPU
            }
            model_params.tensor_split = tensor_split.data();

            // Configure devices array
            size_t dev_count = ggml_backend_dev_count();
            gpu_devices.clear();
            for (size_t i = 0; i < dev_count && (int)gpu_devices.size() < num_gpus; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev) {
                    const char* dev_name = ggml_backend_dev_name(dev);
                    // Only include CUDA devices
                    if (dev_name && strstr(dev_name, "CUDA") != nullptr) {
                        gpu_devices.push_back(static_cast<void*>(dev));
                        LOG_DEBUG("Added device " + std::to_string(i) + ": " + std::string(dev_name));
                    }
                }
            }
            gpu_devices.push_back(nullptr);  // NULL terminator
            model_params.devices = reinterpret_cast<ggml_backend_dev_t*>(gpu_devices.data());

            LOG_INFO("Configured " + std::to_string(gpu_devices.size() - 1) + " devices for multi-GPU");

            if (tensor_parallel == 0) {
                LOG_INFO("Tensor parallelism: AUTO (using all available GPUs with LAYER splitting)");
            } else {
                LOG_INFO("Tensor parallelism: TP=" + std::to_string(tensor_parallel) + " GPUs with LAYER split mode");
            }
        } else {
            // TP=1: Single GPU only, no splitting
            model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
            model_params.main_gpu = 0;
            model_params.tensor_split = nullptr;
            LOG_INFO("Tensor parallelism: TP=1 (single GPU, no splitting)");
        }

        LOG_INFO("GPU detected, offloading layers to GPU (n_gpu_layers=" + std::to_string(model_params.n_gpu_layers) + ")");
    } else {
        model_params.n_gpu_layers = 0;
        LOG_INFO("GPU not available, using CPU only");
    }

    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw BackendError("Failed to load model: " + model_path);
    }

    // Log actual layer offload count
    if (has_gpu) {
        int32_t n_layers = llama_model_n_layer(static_cast<llama_model*>(model));
        LOG_INFO("Model has " + std::to_string(n_layers) + " layers, all offloaded to GPU");
    }

    LOG_INFO("LlamaCpp model loaded successfully");

    // If user didn't specify context size (0), get it from the model
    if (context_size == 0) {
        int32_t model_context_size = llama_model_n_ctx_train(static_cast<llama_model*>(model));
        if (model_context_size <= 0) {
            LOG_ERROR("No context size specified and model has none defined");
            throw BackendError("Cannot determine context size: model provides no context size and none was specified");
        }
        context_size = static_cast<size_t>(model_context_size);
        LOG_INFO("Using model's full context size: " + std::to_string(context_size));
    }
    // else: use user's specified context_size as-is

    LOG_INFO("Using context size: " + std::to_string(context_size) + " tokens");

    // Determine optimal batch size
    if (has_gpu) {
        if (context_size >= 32768) {
            n_batch = 4096;
        } else if (context_size >= 8192) {
            n_batch = 2048;
        } else {
            n_batch = std::min(static_cast<int>(context_size), 2048);
        }
    } else {
        n_batch = std::min(static_cast<int>(context_size) / 4, 1024);
    }
    LOG_INFO("Using n_batch = " + std::to_string(n_batch) + " (GPU: " + (has_gpu ? "yes" : "no") + ")");

    // Create llama context with KV space callback
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(context_size);
    ctx_params.n_batch = static_cast<uint32_t>(n_batch);
    ctx_params.offload_kqv = has_gpu;

    // Set KV cache space callback for interrupt-driven eviction
    ctx_params.kv_need_space_callback = [](uint32_t tokens_needed, void* user_data) -> uint32_t {
        auto* backend = static_cast<LlamaCppBackend*>(user_data);
        uint32_t new_head = backend->evict_to_free_space(tokens_needed);
        if (new_head == UINT32_MAX) {
            LOG_DEBUG("Eviction succeeded, retrying operation with freed space");
        }
        return new_head;
    };
    ctx_params.kv_need_space_callback_data = this;

    model_ctx = llama_init_from_model(static_cast<llama_model*>(model), ctx_params);
    if (!model_ctx) {
        llama_model_free(static_cast<llama_model*>(model));
        model = nullptr;
        throw BackendError("Failed to create llama context");
    }

    // Initialize chat templates
    auto templates = common_chat_templates_init(
        static_cast<llama_model*>(model),
        "",  // Use model's default chat template
        "",  // Default BOS token
        ""   // Default EOS token
    );
    chat_templates = templates.release();

    if (!chat_templates) {
        LOG_WARN("Failed to initialize chat templates - tool support may be limited");
    } else {
        LOG_DEBUG("Chat templates initialized successfully");
    }

    // Get chat template from model (no custom template support for now)
    const char* template_ptr = llama_model_chat_template(static_cast<llama_model*>(model), nullptr);
    const char* tool_use_template_ptr = llama_model_chat_template(static_cast<llama_model*>(model), "tool_use");

    if (template_ptr) {
        chat_template_text = std::string(template_ptr);
        LOG_INFO("Retrieved default chat template from model (" + std::to_string(chat_template_text.length()) + " characters)");

        // Dump template to file for inspection
        std::ofstream template_file("/tmp/shepherd_chat_template.jinja");
        template_file << chat_template_text;
        template_file.close();
        LOG_INFO("Chat template saved to /tmp/shepherd_chat_template.jinja for inspection");
    }

    if (tool_use_template_ptr) {
        std::string tool_use_template = std::string(tool_use_template_ptr);
        LOG_INFO("Retrieved tool_use chat template from model (" + std::to_string(tool_use_template.length()) + " characters)");

        // Use tool_use template if it has python_tag support
        if (tool_use_template.find("<|python_tag|>") != std::string::npos) {
            chat_template_text = tool_use_template;
            LOG_INFO("Using tool_use template variant for tool calling support");
        }
    }

    if (!template_ptr && !tool_use_template_ptr) {
        throw BackendError("No chat template found in model");
    }

    // Parse the chat template with minja
    try {
        minja::Options options{};
        auto parsed_template = minja::Parser::parse(chat_template_text, options);
        template_node = new std::shared_ptr<minja::TemplateNode>(parsed_template);
        LOG_INFO("Successfully parsed chat template with minja");
    } catch (const std::exception& e) {
        throw BackendError("Failed to parse chat template with minja: " + std::string(e.what()));
    }

    // Detect model family
    model_config = Models::detect_from_chat_template(chat_template_text, model_path);
    LOG_INFO("Model configuration: family=" + std::to_string(static_cast<int>(model_config.family)) +
             ", version=" + model_config.version +
             ", tool_result_role=" + model_config.tool_result_role +
             ", uses_eom_token=" + (model_config.uses_eom_token ? "true" : "false") +
             ", uses_python_tag=" + (model_config.uses_python_tag ? "true" : "false"));

    LOG_INFO("LlamaCppBackend initialized with model: " + model_path);
    initialized = true;

    // Add system message in cli mode
	if (!g_server_mode) {
        // Format system message with tools
        std::string formatted_system = format_system_message_with_tools(
            session.system_message,
            session.tools
        );

        // Use add_message to properly decode and add system message
        // This ensures it's in KV cache before being added to session.messages
        Response sys_resp = add_message(session, Message::SYSTEM, formatted_system, "", "", 0, 0);

        if (!sys_resp.success) {
            LOG_ERROR("Failed to add system message: " + sys_resp.error);
            throw BackendError("Failed to initialize system message: " + sys_resp.error);
        }

        // Update session tracking (add_message already added to session.messages)
        session.system_message_tokens = sys_resp.prompt_tokens;

        LOG_INFO("Added system message to session (tokens=" + std::to_string(sys_resp.prompt_tokens) + ")");
    }
#else
    throw BackendError("LlamaCpp backend not compiled in");
#endif
}

Response LlamaCppBackend::add_message(Session& session, Message::Type type, const std::string& content, const std::string& tool_name, const std::string& tool_id, int prompt_tokens, int max_tokens) {
#ifdef ENABLE_LLAMACPP
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "LlamaCpp backend not initialized";
        return err_resp;
    }

    // Set current session for eviction callbacks
    current_session = &session;

    LOG_DEBUG("LlamaCpp add_message called: type=" + std::to_string(type) +
              ", content_len=" + std::to_string(content.length()));

    // Count tokens for this message if not provided
    int message_tokens = prompt_tokens;
    if (message_tokens == 0) {
        if (type == Message::SYSTEM && !session.tools.empty()) {
            std::string formatted_content = format_system_message_with_tools(content, session.tools);
            message_tokens = count_message_tokens(type, formatted_content, tool_name, tool_id);
        } else {
            message_tokens = count_message_tokens(type, content, tool_name, tool_id);
        }
    }

    LOG_DEBUG("Message token count: " + std::to_string(message_tokens));

    // Create the message object (NOT in session yet)
    Message msg(type, content, message_tokens);
    msg.tool_name = tool_name;
    msg.tool_call_id = tool_id;

    // TRY to decode to KV cache FIRST
    if (!format_and_decode_message(msg)) {
        LOG_ERROR("Failed to decode message to KV cache");
        // Session is unchanged - no rollback needed
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "Failed to decode message to KV cache";
        return err_resp;
    }

    // SUCCESS - now add message to session
    session.messages.push_back(msg);
    int new_message_index = static_cast<int>(session.messages.size()) - 1;

    // Update session total tokens
    session.total_tokens += message_tokens;

    // Track for context preservation
    if (type == Message::USER) {
        session.last_user_message_index = new_message_index;
        session.last_user_message_tokens = message_tokens;
    }

    // Generate response (unless this is a system message)
    Response resp;
    resp.success = true;
    resp.code = Response::SUCCESS;

    if (type != Message::SYSTEM) {
        // Update member variables used by generate() for token calculations
        // System message tokens (first message if it's a SYSTEM type)
        if (!session.messages.empty() && session.messages[0].type == Message::SYSTEM) {
            system_formatted_tokens = session.messages[0].tokens;
        } else {
            system_formatted_tokens = 0;
        }

        // Last user message tokens
        if (type == Message::USER) {
            // Current message being added
            current_user_formatted_tokens = message_tokens;
        } else if (session.last_user_message_index >= 0) {
            // Previous user message
            current_user_formatted_tokens = session.last_user_message_tokens;
        } else {
            current_user_formatted_tokens = 0;
        }

        std::string response_text = generate(max_tokens);

        // Add assistant message to session (generation was successful)
        Message assistant_msg(Message::ASSISTANT, response_text, last_assistant_kv_tokens);
        session.messages.push_back(assistant_msg);

        // Update session total tokens with assistant response
        session.total_tokens += last_assistant_kv_tokens;

        // Update tracking
        session.last_assistant_message_index = static_cast<int>(session.messages.size()) - 1;
        session.last_assistant_message_tokens = last_assistant_kv_tokens;

        // Parse tool calls from response if present
        std::vector<ToolParser::ToolCall> tool_calls;
        if (!response_text.empty()) {
            // Check for tool call markers
            bool has_marker = false;
            for (const auto& marker : tool_call_markers) {
                if (response_text.find(marker) != std::string::npos) {
                    has_marker = true;
                    break;
                }
            }

            // Parse tool calls if markers present or response looks like JSON
            if (has_marker || (response_text[0] == '{' && response_text.find("\"name\"") != std::string::npos)) {
                auto tool_call = ToolParser::parse_tool_call(response_text, tool_call_markers);
                if (tool_call.has_value()) {
                    tool_calls.push_back(tool_call.value());
                }
            }
        }

        // Determine finish reason
        std::string finish_reason = "stop";
        if (!tool_calls.empty()) {
            finish_reason = "tool_calls";
        }

        // Build Response
        resp.content = response_text;
        resp.tool_calls = tool_calls;
        resp.prompt_tokens = last_prompt_tokens;
        resp.completion_tokens = last_completion_tokens;
        resp.finish_reason = finish_reason;
    } else {
        // System message - no generation, just success
        resp.finish_reason = "system";
        resp.prompt_tokens = message_tokens;
        resp.completion_tokens = 0;
    }

    LOG_DEBUG("add_message complete: prompt_tokens=" + std::to_string(resp.prompt_tokens) +
              ", completion_tokens=" + std::to_string(resp.completion_tokens) +
              ", finish_reason=" + resp.finish_reason);

    return resp;
#else
    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.finish_reason = "error";
    err_resp.error = "LlamaCpp backend not compiled in";
    return err_resp;
#endif
}
