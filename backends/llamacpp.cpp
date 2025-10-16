#include "llamacpp.h"
#include "../logger.h"
#include "../tools/tool.h"
#include "../tools/tool_parser.h"
#include "../nlohmann/json.hpp"
#include "../minja.hpp"
#include "../rag.h"
#include "../model_manager.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <regex>
#include <ctime>

#ifdef ENABLE_LLAMACPP
#include "../llama.cpp/include/llama.h"
#include "../llama.cpp/src/llama-batch.h"
#include "../llama.cpp/common/chat.h"
#endif

// LlamaCppTokenizer implementation
LlamaCppTokenizer::LlamaCppTokenizer(void* model)
    : model_(model) {
    LOG_DEBUG("LlamaCpp tokenizer initialized");
}

int LlamaCppTokenizer::count_tokens(const std::string& text) {
#ifdef ENABLE_LLAMACPP
    if (model_) {
        // Use actual llama.cpp tokenization with vocab
        const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model_));
        // Negative return value gives us the token count
        // Note: add_special=false because minja template handles special tokens
        int n_tokens = -llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, false, true);
        return n_tokens;
    }
#endif
    // Fallback approximation when model not loaded
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

std::vector<int> LlamaCppTokenizer::encode(const std::string& text) {
#ifdef ENABLE_LLAMACPP
    if (model_) {
        // TODO: Use actual llama.cpp tokenization
        // std::vector<int> tokens(text.length() + 100); // oversized buffer
        // int n_tokens = llama_tokenize(model_, text.c_str(), text.length(), tokens.data(), tokens.size(), true, false);
        // tokens.resize(n_tokens);
        // return tokens;
    }
#endif
    // Fallback implementation
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {
        tokens.push_back(static_cast<int>(text.substr(i, 4).length()));
    }
    return tokens;
}

std::string LlamaCppTokenizer::decode(const std::vector<int>& tokens) {
#ifdef ENABLE_LLAMACPP
    if (model_) {
        // TODO: Use actual llama.cpp detokenization
        // return llama_detokenize(model_, tokens.data(), tokens.size());
    }
#endif
    return "TODO: Implement llama.cpp decode";
}

std::string LlamaCppTokenizer::get_tokenizer_name() const {
    return "llamacpp";
}

// LlamaCppContextManager implementation
LlamaCppContextManager::LlamaCppContextManager(size_t max_context_tokens)
    : ContextManager(max_context_tokens) {
    // Disable auto-eviction - llama.cpp manages eviction via KV cache callbacks
    auto_evict_on_add_ = false;
    LOG_DEBUG("LlamaCpp context manager initialized (auto_evict_on_add=false, KV cache manages eviction)");
}

std::string LlamaCppContextManager::get_context_for_inference() {
    return get_context_for_inference(true);  // Default: add generation prompt
}

std::string LlamaCppContextManager::get_context_for_inference(bool add_generation_prompt) {
#ifdef ENABLE_LLAMACPP
    // Use minja to render messages with the chat template
    if (!template_node_) {
        LOG_ERROR("No template node available, falling back to simple format");
        // Fallback to simple format
        std::ostringstream text_builder;
        for (const auto& msg : messages_) {
            text_builder << msg.get_role() << ": " << msg.content << "\n\n";
        }
        if (add_generation_prompt) {
            text_builder << "assistant: ";
        }
        return text_builder.str();
    }

    // Get the actual template node from the void pointer
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);

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

    // Convert messages to minja array
    auto messages = minja::Value::array();

    // Render all messages using full template
    for (const auto& msg : messages_) {
        auto msg_obj = minja::Value::object();
        msg_obj.set("role", minja::Value(msg.get_role()));
        msg_obj.set("content", minja::Value(msg.content));
        messages.push_back(msg_obj);
    }

    context->set("bos_token", minja::Value("<|begin_of_text|>"));
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(add_generation_prompt));

    std::string rendered = (*template_ptr)->render(context);
    LOG_DEBUG("Template render (" + std::to_string(rendered.length()) + " chars, add_generation_prompt=" +
              (add_generation_prompt ? "true" : "false") + ")");
    return rendered;
#else
    // Fallback when llama.cpp not available
    std::ostringstream text_builder;
    for (const auto& msg : messages_) {
        text_builder << msg.get_role() << ": " << msg.content << "\n\n";
    }
    if (add_generation_prompt) {
        text_builder << "assistant: ";
    }
    return text_builder.str();
#endif
}

std::string LlamaCppContextManager::render_single_message(const Message& msg, bool add_generation_prompt) {
#ifdef ENABLE_LLAMACPP
    // Use minja to render just this one message with the chat template
    if (!template_node_) {
        LOG_ERROR("No template node available, falling back to simple format");
        // Fallback to simple format
        return msg.get_role() + ": " + msg.content + "\n\n";
    }

    // Get the actual template node from the void pointer
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);

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
    messages.push_back(msg_obj);

    context->set("bos_token", minja::Value("<|begin_of_text|>"));
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(add_generation_prompt));

    std::string rendered = (*template_ptr)->render(context);
    LOG_DEBUG("Rendered single message (" + std::to_string(rendered.length()) + " chars, role=" + msg.get_role() + ")");
    return rendered;
#else
    // Fallback when llama.cpp not available
    return msg.get_role() + ": " + msg.content + "\n\n";
#endif
}

std::pair<int, int> LlamaCppContextManager::get_token_range_for_messages(int start_msg_index, int end_msg_index) const {
    // Calculate token positions for message range
    // Assumes messages are contiguous in KV cache starting from position 0

    if (start_msg_index < 0 || end_msg_index >= static_cast<int>(messages_.size()) || start_msg_index > end_msg_index) {
        LOG_ERROR("Invalid message index range: " + std::to_string(start_msg_index) + " to " + std::to_string(end_msg_index));
        return {-1, -1};
    }

    int start_pos = 0;
    int end_pos = 0;

    // Calculate start position (sum of all tokens before start_msg_index)
    for (int i = 0; i < start_msg_index; i++) {
        start_pos += messages_[i].token_count;
    }

    // Calculate end position (sum of all tokens up to and including end_msg_index)
    end_pos = start_pos;
    for (int i = start_msg_index; i <= end_msg_index; i++) {
        end_pos += messages_[i].token_count;
    }

    LOG_DEBUG("Message range [" + std::to_string(start_msg_index) + ", " + std::to_string(end_msg_index) +
              "] = token positions [" + std::to_string(start_pos) + ", " + std::to_string(end_pos - 1) + "]");

    return {start_pos, end_pos - 1}; // Return inclusive range
}

void LlamaCppContextManager::mark_messages_evicted(int start_msg_index, int end_msg_index) {
    // After eviction, messages removed from both messages_ array and KV cache
    // No tracking needed - if it's in messages_, it's in KV cache
    int num_evicted = end_msg_index - start_msg_index + 1;
    LOG_DEBUG("Evicted " + std::to_string(num_evicted) + " messages [" +
              std::to_string(start_msg_index) + ", " + std::to_string(end_msg_index) + "]");
}

void LlamaCppContextManager::extract_message_patterns() {
#ifdef ENABLE_LLAMACPP
    if (!template_node_ || patterns_extracted_) {
        return;
    }

    LOG_INFO("Extracting message format patterns from chat template");

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);
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

    // Helper to render messages and extract just the last one's pattern
    // We need to render 2+ messages so the template doesn't add system headers
    auto extract_pattern = [&](const std::string& role, const std::string& placeholder) -> std::string {
        auto messages = minja::Value::array();

        // Add a dummy user message first (so template doesn't add system header)
        auto dummy_msg = minja::Value::object();
        dummy_msg.set("role", minja::Value("user"));
        dummy_msg.set("content", minja::Value("__DUMMY__"));
        messages.push_back(dummy_msg);

        // Now add the message we want to extract
        auto msg_obj = minja::Value::object();
        msg_obj.set("role", minja::Value(role));
        msg_obj.set("content", minja::Value(placeholder));
        messages.push_back(msg_obj);

        context->set("messages", messages);
        context->set("add_generation_prompt", minja::Value(false));
        context->set("bos_token", minja::Value("")); // No BOS

        try {
            auto result = (*template_ptr)->render(context);

            LOG_DEBUG("Rendered template for role '" + role + "': [" + result + "]");

            // Extract just the last message's formatting
            // Find where our placeholder appears
            size_t placeholder_pos = result.find(placeholder);
            if (placeholder_pos == std::string::npos) {
                LOG_ERROR("Could not find placeholder in rendered template for role: " + role);
                LOG_ERROR("Full rendered result: " + result);
                return "";
            }

            // Find the start of THIS SPECIFIC message's header by looking for the role name
            // Pattern: <|start_header_id|>ROLE<|end_header_id|>
            // Search BACKWARDS from placeholder to find the LAST occurrence (our target message)
            std::string role_header = "<|start_header_id|>" + role + "<|end_header_id|>";
            size_t msg_start = result.rfind(role_header, placeholder_pos);
            if (msg_start == std::string::npos) {
                LOG_ERROR("Could not find role header '" + role_header + "' before placeholder in rendered template");
                LOG_ERROR("Full rendered result: " + result);
                return "";
            }

            // Extract from THIS message's start to end
            return result.substr(msg_start);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to extract pattern for role '" + role + "': " + std::string(e.what()));
            return "";
        }
    };

    // Extract patterns for each role
    system_pattern_ = extract_pattern("system", "__SYSTEM_CONTENT__");
    user_pattern_ = extract_pattern("user", "__USER_CONTENT__");
    assistant_pattern_ = extract_pattern("assistant", "__ASSISTANT_CONTENT__");
    tool_pattern_ = extract_pattern("ipython", "__TOOL_CONTENT__"); // llama3 uses "ipython" for Python REPL-style tool results

    patterns_extracted_ = true;

    LOG_INFO("Extracted message patterns successfully");
    LOG_INFO("System pattern: " + system_pattern_.substr(0, 100) + "...");
    LOG_INFO("User pattern: " + user_pattern_.substr(0, 100) + "...");
    LOG_INFO("Assistant pattern: " + assistant_pattern_.substr(0, 100) + "...");
    LOG_INFO("Tool pattern: " + tool_pattern_);

    // Save patterns to file for inspection
    std::ofstream pattern_file("/tmp/shepherd_patterns.txt");
    pattern_file << "SYSTEM:\n" << system_pattern_ << "\n\n";
    pattern_file << "USER:\n" << user_pattern_ << "\n\n";
    pattern_file << "ASSISTANT:\n" << assistant_pattern_ << "\n\n";
    pattern_file << "TOOL:\n" << tool_pattern_ << "\n\n";
    pattern_file.close();
#endif
}

int LlamaCppContextManager::count_tokens(const std::string& text) {
#ifdef ENABLE_LLAMACPP
    if (model_) {
        // Use actual llama.cpp tokenization with vocab
        const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model_));
        // Negative return value gives us the token count
        // Note: add_special=false because minja template handles special tokens
        int n_tokens = -llama_tokenize(vocab, text.c_str(), text.length(), nullptr, 0, false, true);
        return n_tokens;
    }
#endif
    // Fallback approximation when model not loaded
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

void LlamaCppContextManager::clear() {
    // Call base class clear() first
    ContextManager::clear();

    // Reset llamacpp-specific state
    patterns_extracted_ = false;
    LOG_DEBUG("LlamaCppContextManager cleared");
}

int LlamaCppContextManager::calculate_json_overhead() const {
#ifdef ENABLE_LLAMACPP
    if (!model_ || messages_.empty()) {
        // Fallback estimation
        return static_cast<int>(messages_.size() * 4 + 4); // ~4 tokens per message + final prompt
    }

    // Create a minimal message set to test overhead
    std::vector<llama_chat_message> minimal_chat;
    std::vector<std::string> role_storage;

    // Create one message with minimal content to calculate template overhead
    role_storage.push_back("user");
    minimal_chat.push_back({"user", "test"});

    const char* chat_template = nullptr;

    // Get template size with minimal content
    int32_t template_size = llama_chat_apply_template(
        chat_template,
        minimal_chat.data(),
        minimal_chat.size(),
        true,
        nullptr,
        0
    );

    if (template_size <= 0) {
        // Fallback estimation for unknown templates
        return static_cast<int>(messages_.size() * 3 + 5); // Conservative estimate
    }

    // Template overhead = total template size - content size
    int content_size = 4; // "test" content
    int overhead_per_message = template_size - content_size;

    // Estimate total overhead for all messages
    int total_overhead = overhead_per_message * static_cast<int>(messages_.size());

    // Convert chars to tokens (roughly 4 chars per token)
    return static_cast<int>(total_overhead / 4.0 + 0.5);
#else
    // Fallback when llama.cpp not available
    return static_cast<int>(messages_.size() * 3 + 4);
#endif
}

// LlamaCppBackend implementation
LlamaCppBackend::LlamaCppBackend(size_t max_context_tokens)
    : BackendManager(max_context_tokens), max_context_size_(max_context_tokens),
      model_config_(ModelConfig::create_generic()) {
    // Don't create context manager yet - wait until model is loaded to get actual context size
    tokenizer_ = std::make_unique<LlamaCppTokenizer>(nullptr); // Model set later
    LOG_DEBUG("LlamaCppBackend created with max_context_size: " + std::to_string(max_context_size_));
}

LlamaCppBackend::~LlamaCppBackend() {
    shutdown();
}

bool LlamaCppBackend::initialize(const std::string& model_path, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_LLAMACPP
    if (initialized_) {
        LOG_WARN("LlamaCppBackend already initialized");
        return true;
    }

    if (model_path.empty()) {
        LOG_ERROR("Model path is required for LlamaCpp backend");
        return false;
    }

    model_path_ = model_path;

    // Suppress llama.cpp logging unless in debug mode
    extern bool g_debug_mode;
    if (!g_debug_mode) {
        llama_log_set([](enum ggml_log_level level, const char * text, void * user_data) {
            // Suppress all llama.cpp logs in non-debug mode
        }, nullptr);
    }

    // Detect GPU support
    bool has_gpu = llama_supports_gpu_offload();

    // Load actual llama.cpp model
    llama_model_params model_params = llama_model_default_params();

    if (has_gpu) {
        // Set to extremely high number - llama.cpp will automatically cap at actual layer count
        model_params.n_gpu_layers = INT32_MAX;

        // Multi-GPU support: split layers across GPUs
        // LAYER mode distributes layers evenly across available GPUs
        model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        model_params.main_gpu = 0;  // Primary GPU for single-GPU ops

        LOG_INFO("GPU detected, offloading all layers to GPU with multi-GPU layer splitting enabled");
    } else {
        model_params.n_gpu_layers = 0;
        LOG_INFO("GPU not available, using CPU only");
    }

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        LOG_ERROR("Failed to load model: " + model_path);
        return false;
    }

    // Log actual layer offload count
    if (has_gpu) {
        int32_t n_layers = llama_model_n_layer(static_cast<llama_model*>(model_));
        LOG_INFO("Model has " + std::to_string(n_layers) + " layers, all offloaded to GPU");
    }

    LOG_INFO("LlamaCpp model loaded successfully");

    // Get the model's actual context size
    int32_t model_context_size = llama_model_n_ctx_train(static_cast<llama_model*>(model_));
    if (model_context_size <= 0) {
        LOG_WARN("Model doesn't specify context size, using default 4096");
        model_context_size = 4096;
    }

    // Determine actual context size to use
    // If max_context_size_ is 0, it means "use model's full context"
    size_t actual_context_size;
    if (max_context_size_ == 0) {
        actual_context_size = static_cast<size_t>(model_context_size);
        LOG_INFO("Using model's full context size: " + std::to_string(actual_context_size));
    } else {
        // Use the smaller of requested size or model's maximum
        actual_context_size = std::min(static_cast<size_t>(model_context_size), max_context_size_);
    }
    max_context_size_ = actual_context_size;

    LOG_INFO("Using context size: " + std::to_string(actual_context_size) + " tokens");

    // Determine optimal batch size based on GPU availability and context size
    if (has_gpu) {
        // GPU available - use larger batches for better throughput
        if (actual_context_size >= 32768) {
            n_batch_ = 4096;  // Large context, process in bigger chunks
        } else if (actual_context_size >= 8192) {
            n_batch_ = 2048;  // Medium context
        } else {
            n_batch_ = std::min(static_cast<int>(actual_context_size), 2048);  // Small context
        }
    } else {
        // CPU only - use smaller batches to avoid memory issues
        n_batch_ = std::min(static_cast<int>(actual_context_size) / 4, 1024);
    }
    LOG_INFO("Using n_batch = " + std::to_string(n_batch_) + " (GPU: " + (has_gpu ? "yes" : "no") + ")");

    // Create llama context with actual size and KV space callback
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(actual_context_size);
    ctx_params.n_batch = static_cast<uint32_t>(n_batch_);

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

    model_ctx_ = llama_init_from_model(static_cast<llama_model*>(model_), ctx_params);
    if (!model_ctx_) {
        LOG_ERROR("Failed to create llama context");
        llama_model_free(static_cast<llama_model*>(model_));
        model_ = nullptr;
        return false;
    }

    // Now create the context manager with the correct size
    context_manager_ = std::make_unique<LlamaCppContextManager>(actual_context_size);

    // Update tokenizer and context manager with model
    static_cast<LlamaCppTokenizer*>(tokenizer_.get())->set_model(model_);
    static_cast<LlamaCppContextManager*>(context_manager_.get())->set_model(model_);

    // Initialize chat templates for tool support
    auto templates = common_chat_templates_init(
        static_cast<llama_model*>(model_),
        "",  // Use model's default chat template
        "",  // Default BOS token
        ""   // Default EOS token
    );
    chat_templates_ = templates.release();

    if (!chat_templates_) {
        LOG_WARN("Failed to initialize chat templates - tool support may be limited");
    } else {
        LOG_DEBUG("Chat templates initialized successfully");
        // Tool call markers will be extracted on first generation with real messages/tools
    }

#if 0
    // Extract and save chat template for inspection
    const char* chat_template = nullptr;
    if (chat_template) {
        std::ofstream template_file("/tmp/chat_template.jinja");
        template_file << chat_template;
        template_file.close();
        LOG_INFO("Chat template saved to /tmp/chat_template.jinja");
        std::cout << "\n=== CHAT TEMPLATE ===\n" << chat_template << "\n=== END TEMPLATE ===\n" << std::endl;
        exit(0); // Exit so we can examine the template
    } else {
        LOG_ERROR("No chat template found in model");
    }
#endif

    // Check if custom template path was provided
    if (!template_path.empty()) {
        LOG_INFO("Loading custom chat template from: " + template_path);
        std::ifstream template_file(template_path);
        if (!template_file.is_open()) {
            LOG_ERROR("Failed to open template file: " + template_path);
            return false;
        }
        std::stringstream buffer;
        buffer << template_file.rdbuf();
        chat_template_text_ = buffer.str();
        LOG_INFO("Loaded custom template (" + std::to_string(chat_template_text_.length()) + " characters)");
        template_file.close();
    } else {
        // Get chat template from model
        const char* template_ptr = llama_model_chat_template(static_cast<llama_model*>(model_), nullptr);
        const char* tool_use_template_ptr = llama_model_chat_template(static_cast<llama_model*>(model_), "tool_use");

        if (template_ptr) {
        chat_template_text_ = std::string(template_ptr);
        LOG_INFO("Retrieved default chat template from model (" + std::to_string(chat_template_text_.length()) + " characters)");
        LOG_INFO("Default template contains <|python_tag|>: " + std::string(chat_template_text_.find("<|python_tag|>") != std::string::npos ? "YES" : "NO"));

        // Dump template to file for inspection
        std::ofstream template_file("/tmp/shepherd_chat_template.jinja");
        template_file << chat_template_text_;
        template_file.close();
        LOG_INFO("Chat template saved to /tmp/shepherd_chat_template.jinja for inspection");
    }

    if (tool_use_template_ptr) {
        std::string tool_use_template = std::string(tool_use_template_ptr);
        LOG_INFO("Retrieved tool_use chat template from model (" + std::to_string(tool_use_template.length()) + " characters)");
        LOG_INFO("Tool_use template contains <|python_tag|>: " + std::string(tool_use_template.find("<|python_tag|>") != std::string::npos ? "YES" : "NO"));

        // Use tool_use template if it has python_tag support
        if (tool_use_template.find("<|python_tag|>") != std::string::npos) {
            chat_template_text_ = tool_use_template;
            LOG_INFO("Using tool_use template variant for tool calling support");
        }
    }

        if (!template_ptr && !tool_use_template_ptr) {
            LOG_ERROR("No chat template found in model and no custom template provided");
            LOG_ERROR("Use --template to specify a custom chat template file");
            return false;
        }
    }  // end custom template check

    // Parse the chat template with minja and store it
    try {
        minja::Options options{};
        auto parsed_template = minja::Parser::parse(chat_template_text_, options);

        // Store in a new allocated shared_ptr that won't be destroyed
        template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
        LOG_INFO("Successfully parsed chat template with minja");
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to parse chat template with minja: " + std::string(e.what()));
        LOG_ERROR("Cannot proceed without valid chat template");
        return false;
    }

    // Pass template node to context manager
    static_cast<LlamaCppContextManager*>(context_manager_.get())->set_template_node(template_node_);

    // Pattern extraction disabled - we'll render single messages directly with Jinja
    // static_cast<LlamaCppContextManager*>(context_manager_.get())->extract_message_patterns();

    // Detect model family and configure prompt behavior using centralized ModelManager
    model_config_ = ModelManager::detect_from_chat_template(chat_template_text_, model_path_);
    LOG_INFO("Model configuration: family=" + std::to_string(static_cast<int>(model_config_.family)) +
             ", version=" + model_config_.version +
             ", tool_result_role=" + model_config_.tool_result_role +
             ", uses_eom_token=" + (model_config_.uses_eom_token ? "true" : "false") +
             ", uses_python_tag=" + (model_config_.uses_python_tag ? "true" : "false"));

    LOG_INFO("LlamaCppBackend initialized with model: " + model_path);
    initialized_ = true;
    return true;
#else
    LOG_ERROR("LlamaCpp backend not compiled in");
    return false;
#endif
}


void LlamaCppBackend::add_system_message(const std::string& custom_prompt) {
    // Format system message with tools using ModelManager
    auto& registry = ToolRegistry::instance();
    std::string formatted_content = ModelManager::format_system_message(
        model_config_,
        custom_prompt,
        registry,
        template_node_
    );

    int token_count = context_manager_->count_tokens(formatted_content);
    Message system_msg(Message::SYSTEM, formatted_content, token_count);
    system_msg.in_kv_cache = false;  // Not cached yet
    context_manager_->add_message(system_msg);

    // Immediately format, tokenize, and decode the message into KV cache
    auto& messages = context_manager_->get_messages();
    if (!messages.empty()) {
        format_and_decode_message(messages.back());
    }

    LOG_DEBUG("Added formatted system message with tools to LlamaCpp backend and decoded to KV cache");
}

void LlamaCppBackend::add_user_message(const std::string& content) {
    int token_count = context_manager_->count_tokens(content);
    Message user_msg(Message::USER, content, token_count);
    user_msg.in_kv_cache = false;  // Not cached yet
    context_manager_->add_message(user_msg);

    // Immediately format, tokenize, and decode the message into KV cache
    auto& messages = context_manager_->get_messages();
    if (!messages.empty()) {
        format_and_decode_message(messages.back());
    }

    LOG_DEBUG("Added user message to LlamaCpp backend and decoded to KV cache");
}

void LlamaCppBackend::add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id) {
    int token_count = context_manager_->count_tokens(content);
    Message tool_msg(Message::TOOL, content, token_count);
    tool_msg.tool_name = tool_name;
    tool_msg.tool_call_id = tool_call_id;
    tool_msg.in_kv_cache = false;  // Not cached yet
    context_manager_->add_message(tool_msg);

    // Immediately format, tokenize, and decode the message into KV cache
    auto& messages = context_manager_->get_messages();
    if (!messages.empty()) {
        format_and_decode_message(messages.back());
    }

    LOG_DEBUG("Added tool result to LlamaCpp backend and decoded to KV cache: " + tool_name);
}

void LlamaCppBackend::add_assistant_message(const std::string& content) {
    int token_count = context_manager_->count_tokens(content);
    Message assistant_msg(Message::ASSISTANT, content, token_count);
    // Assistant message is already in KV cache because it was decoded during generation
    assistant_msg.in_kv_cache = true;
    context_manager_->add_message(assistant_msg);
    LOG_DEBUG("Added assistant message to LlamaCpp backend (already in KV cache from generation)");
}

std::string LlamaCppBackend::get_backend_name() const {
    return "llamacpp";
}

std::string LlamaCppBackend::get_model_name() const {
    return model_path_;
}

size_t LlamaCppBackend::get_max_context_size() const {
    return max_context_size_;
}

ModelConfig LlamaCppBackend::get_model_config() const {
#ifdef ENABLE_LLAMACPP
    return model_config_;
#else
    return ModelConfig::create_generic();
#endif
}

std::vector<std::string> LlamaCppBackend::get_tool_call_markers() const {
#ifdef ENABLE_LLAMACPP
    return tool_call_markers_;
#else
    return {};
#endif
}

bool LlamaCppBackend::is_ready() const {
#ifdef ENABLE_LLAMACPP
    return initialized_; // && model_ && model_ctx_;
#else
    return false;
#endif
}

uint32_t LlamaCppBackend::evict_to_free_space(uint32_t tokens_needed) {
#ifdef ENABLE_LLAMACPP
    LOG_INFO("KV cache full - need to free " + std::to_string(tokens_needed) + " tokens");

    auto* llama_ctx_mgr = dynamic_cast<LlamaCppContextManager*>(context_manager_.get());
    if (!llama_ctx_mgr) {
        LOG_ERROR("Context manager is not LlamaCppContextManager");
        return UINT32_MAX;
    }

    // Get KV cache memory handle for eviction operations
    llama_context* ctx = static_cast<llama_context*>(model_ctx_);
    llama_memory_t mem = llama_get_memory(ctx);

    // Count how many messages are confirmed in KV cache
    // Trust the in_kv_cache flags we set explicitly during add/evict operations
    size_t confirmed_count = 0;
    for (const auto& msg : context_manager_->get_messages()) {
        if (msg.in_kv_cache) confirmed_count++;
    }

    LOG_DEBUG("Found " + std::to_string(confirmed_count) + " messages in KV cache (out of " +
              std::to_string(context_manager_->get_messages().size()) + " total)");

    // Calculate which messages to evict (only from confirmed messages)
    auto [start_msg, end_msg] = context_manager_->calculate_messages_to_evict(tokens_needed, confirmed_count);

    if (start_msg == -1 || end_msg == -1) {
        LOG_ERROR("Cannot calculate messages to evict - no space can be freed");
        return 0;  // Signal failure: eviction cannot proceed
    }

    // Get token positions for these messages
    auto [start_pos, end_pos] = llama_ctx_mgr->get_token_range_for_messages(start_msg, end_msg);

    if (start_pos == -1 || end_pos == -1) {
        LOG_ERROR("Cannot calculate token positions for eviction");
        return 0;  // Signal failure: eviction cannot proceed
    }

    LOG_INFO("Evicting messages [" + std::to_string(start_msg) + ", " + std::to_string(end_msg) +
             "] = tokens [" + std::to_string(start_pos) + ", " + std::to_string(end_pos) + "]");

    // Call llama.cpp API to evict from KV cache (reuse ctx and mem from earlier)
    llama_memory_seq_rm(mem, 0, start_pos, end_pos + 1); // +1 because llama uses exclusive end

    int tokens_removed = end_pos - start_pos + 1;

    // CRITICAL: Shift remaining tokens down to keep positions contiguous
    // Without this, we create gaps that break attention/RoPE and cause infinite eviction loops
    llama_memory_seq_add(mem, 0, end_pos + 1, -1, -tokens_removed);
    LOG_DEBUG("Shifted KV cache positions >= " + std::to_string(end_pos + 1) + " down by " + std::to_string(tokens_removed));

    // Use shared context manager method to archive to RAG and remove messages
    if (!context_manager_->evict_messages_by_index(start_msg, end_msg)) {
        LOG_ERROR("Failed to evict messages from context manager");
        return UINT32_MAX;
    }

    // Mark as evicted and reset cached count
    llama_ctx_mgr->mark_messages_evicted(start_msg, end_msg);

    int num_messages = end_msg - start_msg + 1;
    LOG_INFO("Successfully evicted " + std::to_string(num_messages) + " messages (" +
             std::to_string(tokens_removed) + " tokens) from KV cache");

    // Return the new head position (where freed space begins) as required by callback API
    // After removing [start_pos, end_pos] and shifting down, the new head is start_pos
    LOG_DEBUG("Eviction complete - returning new head position: " + std::to_string(start_pos));
    return static_cast<uint32_t>(start_pos);
#else
    LOG_ERROR("llama.cpp not enabled");
    return UINT32_MAX;
#endif
}

void LlamaCppBackend::shutdown() {
    if (!initialized_) {
        return;
    }

#ifdef ENABLE_LLAMACPP
    // Cleanup llama.cpp resources properly
    if (model_ctx_) {
        llama_free(static_cast<llama_context*>(model_ctx_));
        model_ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(static_cast<llama_model*>(model_));
        model_ = nullptr;
    }

    // Cleanup chat templates
    if (chat_templates_) {
        common_chat_templates_free(static_cast<common_chat_templates*>(chat_templates_));
        chat_templates_ = nullptr;
    }
#endif

    initialized_ = false;
    LOG_DEBUG("LlamaCppBackend shutdown complete");
}

std::string LlamaCppBackend::generate(int max_tokens) {
    LOG_DEBUG("=== GENERATE START ===");
    if (!is_ready()) {
        throw BackendManagerError("LlamaCpp backend not initialized");
    }

    LOG_DEBUG("Getting tools from registry");
    auto& registry = ToolRegistry::instance();
    auto available_tools = registry.list_tools();

    // Extract tool call markers from chat template on first call
    if (!have_tool_markers_ && chat_templates_) {
        try {
            common_chat_templates_inputs inputs{};

            // Convert our messages
            for (const auto& msg : context_manager_->get_messages()) {
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
                static_cast<common_chat_templates*>(chat_templates_),
                inputs
            );

            if (!params.preserved_tokens.empty()) {
                tool_call_markers_ = params.preserved_tokens;
                LOG_INFO("Extracted " + std::to_string(tool_call_markers_.size()) + " tool call markers from template:");
                for (const auto& marker : tool_call_markers_) {
                    LOG_INFO("  - " + marker);
                }
            } else {
                LOG_INFO("No preserved_tokens from template - checking vocabulary");

                // Check if model has <|python_tag|> in its vocabulary
                std::vector<llama_token> tokens;
                tokens.resize(8);  // Should only need 1-2 tokens max
                std::string test_marker = "<|python_tag|>";
                const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model_));
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
                    tool_call_markers_.push_back(test_marker);
                    LOG_INFO("Found <|python_tag|> in model vocabulary (token " + std::to_string(tokens[0]) + ")");
                } else {
                    LOG_INFO("Model does not have <|python_tag|> special token (tokenized to " + std::to_string(n_tokens) + " tokens)");
                }
            }

            have_tool_markers_ = true;
        } catch (const std::exception& e) {
            LOG_INFO("Exception during tool marker extraction: " + std::string(e.what()));
            have_tool_markers_ = true;
        }
    }

    // All messages are already decoded in KV cache from add_*_message() calls
    // We just need to run the generation loop now

    // In debug mode, show what's in the KV cache
    extern bool g_debug_mode;
    if (g_debug_mode) {
        LOG_DEBUG("=== MESSAGES IN KV CACHE ===");
        char line[128];
        for (const auto& msg : context_manager_->get_messages()) {
            line[0] = 0;
            // Format: "[role] content..." with max 128 chars total, newlines replaced with spaces

            // Replace newlines with spaces in content
            std::string content_clean = msg.content;
            for (size_t i = 0; i < content_clean.length(); i++) {
                if (content_clean[i] == '\n' || content_clean[i] == '\r') {
                    content_clean[i] = ' ';
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
    // For Qwen models: "<|im_start|>assistant\n"
    std::string generation_prompt;
    if (model_config_.family == ModelFamily::QWEN_2_X) {
        generation_prompt = "<|im_start|>assistant\n";
    } else if (model_config_.family == ModelFamily::LLAMA_3_X) {
        generation_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else {
        // Generic fallback
        generation_prompt = "assistant: ";
    }

    // Tokenize and decode the generation prompt into KV cache
    llama_context* ctx = static_cast<llama_context*>(model_ctx_);
    llama_model* model = static_cast<llama_model*>(model_);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> prompt_tokens(generation_prompt.length() + 256);
    int n_tokens = llama_tokenize(vocab, generation_prompt.c_str(), generation_prompt.length(),
                                   prompt_tokens.data(), prompt_tokens.size(), false, true);

    if (n_tokens > 0) {
        prompt_tokens.resize(n_tokens);
        LOG_DEBUG("Decoding generation prompt (" + generation_prompt + "): " + std::to_string(n_tokens) + " tokens");

        for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_) {
            int batch_size = std::min(n_batch_, static_cast<int>(prompt_tokens.size() - i));
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch) != 0) {
                LOG_ERROR("Failed to decode generation prompt at position " + std::to_string(i));
            }
        }
    }

    // Suppress streaming when tools are available to avoid showing JSON tool calls
    bool suppress_stream = !available_tools.empty();
    std::string raw_response = run_inference("", max_tokens, suppress_stream);  // Empty prompt since everything is cached
    LOG_DEBUG("Got raw response length: " + std::to_string(raw_response.length()));
    LOG_DEBUG("Raw response: " + raw_response);

    // Return response directly - main will handle tool parsing and cleanup
    return raw_response;
}

std::string LlamaCppBackend::get_context_with_tools() {
#ifdef ENABLE_LLAMACPP
    // Use our instruction executor instead of llama.cpp chat templates
    return context_manager_->get_context_for_inference();
#else
    return context_manager_->get_context_for_inference();
#endif
}

std::string LlamaCppBackend::run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming) {
#ifdef ENABLE_LLAMACPP
    if (!model_ || !model_ctx_) {
        LOG_ERROR("llama.cpp model or context not initialized");
        return "Error: Model not initialized";
    }

    llama_context* ctx = static_cast<llama_context*>(model_ctx_);
    llama_model* model = static_cast<llama_model*>(model_);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Initialize sampler chain with configured sampling parameters
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add samplers in the recommended order (from llama.cpp examples)
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k_));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p_, min_keep_));

    // Add repetition penalties BEFORE temperature to discourage repetitive patterns
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        penalty_last_n_,      // last n tokens to penalize
        penalty_repeat_,      // repetition penalty (1.0 = disabled)
        penalty_freq_,        // frequency penalty (0.0 = disabled)
        penalty_present_));   // presence penalty (0.0 = disabled)

    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));           // greedy sampling from dist

    LOG_DEBUG("Sampling params: temperature=" + std::to_string(temperature_) +
              ", top_p=" + std::to_string(top_p_) +
              ", top_k=" + std::to_string(top_k_) +
              ", min_keep=" + std::to_string(min_keep_) +
              ", penalty_repeat=" + std::to_string(penalty_repeat_) +
              ", penalty_freq=" + std::to_string(penalty_freq_) +
              ", penalty_present=" + std::to_string(penalty_present_) +
              ", penalty_last_n=" + std::to_string(penalty_last_n_));

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

        // Evaluate prompt tokens in batches using configured batch size
        for (size_t i = 0; i < prompt_tokens.size(); i += n_batch_) {
            int batch_size = std::min(n_batch_, static_cast<int>(prompt_tokens.size() - i));

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

    // Calculate max generation tokens based on what MUST stay in context:
    // - System prompt (message 0)
    // - Current user message (last user message)
    // Everything else is evictable to make room for the response
    auto& messages = context_manager_->get_messages();
    int system_tokens = 0;
    int current_user_tokens = 0;

    if (!messages.empty()) {
        // System message is always first
        system_tokens = messages[0].token_count;

        // Find last user message
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (it->get_role() == "user") {
                current_user_tokens = it->token_count;
                break;
            }
        }

        // Safety check: verify user message exists
        if (current_user_tokens == 0) {
            LOG_ERROR("CRITICAL: No user message found in context - cannot generate without user input");
            llama_sampler_free(sampler);
            return "Error: No user message in context. This indicates a critical eviction bug.";
        }
    }

    int protected_tokens = system_tokens + current_user_tokens + 200; // 200 token buffer for safety
    int available_for_generation = static_cast<int>(max_context_size_) - protected_tokens;

    // Trust eviction system - it will free old messages as needed
    int max_gen_tokens = max_tokens > 0 ? max_tokens : available_for_generation;

    LOG_DEBUG("Generation limits: " + std::to_string(max_gen_tokens) + " max tokens (protected: system=" +
              std::to_string(system_tokens) + " + user=" + std::to_string(current_user_tokens) +
              " + buffer=200, evictable_for_response: " + std::to_string(available_for_generation) + ")");

    // Start timing for t/s measurement
    auto gen_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_gen_tokens; i++) {
        // Sample next token
        llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

        // Check for end of generation using llama.cpp's native EOG detection
        // This handles all model-specific end tokens automatically
        const struct llama_vocab* vocab = llama_model_get_vocab(model);
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
        std::cerr << "\033[90m[Decode: " << n_generated << " tokens, "
                  << std::fixed << std::setprecision(1) << tokens_per_second << " t/s]\033[0m" << std::endl;
    }

    LOG_INFO("Generation (decode): " + std::to_string(n_generated) + " tokens in " +
             std::to_string(seconds) + "s (" + std::to_string(tokens_per_second) + " t/s)");
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

bool LlamaCppBackend::format_and_decode_message(Message& msg) {
#ifdef ENABLE_LLAMACPP
    if (!model_ || !model_ctx_) {
        LOG_ERROR("Model or context not initialized");
        return false;
    }

    // Render ONLY this single message (not the entire context!)
    auto* llama_ctx_mgr = static_cast<LlamaCppContextManager*>(context_manager_.get());
    std::string rendered_msg = llama_ctx_mgr->render_single_message(msg, false);

    // In debug mode, show the rendered message being sent to the model
    extern bool g_debug_mode;
    if (g_debug_mode) {
        LOG_DEBUG("=== MESSAGE TO DECODE ===");
        LOG_DEBUG(rendered_msg);
        LOG_DEBUG("=== END MESSAGE ===");
    }

    llama_context* ctx = static_cast<llama_context*>(model_ctx_);
    llama_model* model = static_cast<llama_model*>(model_);
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize just this message
    std::vector<llama_token> msg_tokens(rendered_msg.length() + 256);
    int n_tokens = llama_tokenize(vocab, rendered_msg.c_str(), rendered_msg.length(),
                                   msg_tokens.data(), msg_tokens.size(), false, true);

    if (n_tokens < 0) {
        LOG_ERROR("Failed to tokenize message");
        return false;
    }
    msg_tokens.resize(n_tokens);

    LOG_DEBUG("Decoding " + std::to_string(n_tokens) + " tokens for new message");

    // Start timing for prompt processing (prefill) speed
    auto prefill_start_time = std::chrono::high_resolution_clock::now();

    // Decode the message tokens in batches
    // Retry once if decode fails (likely due to KV cache eviction mid-operation)
    const int MAX_DECODE_RETRIES = 1;
    int retry_count = 0;

    while (retry_count <= MAX_DECODE_RETRIES) {
        bool decode_failed = false;

        for (size_t i = 0; i < msg_tokens.size(); i += n_batch_) {
            int batch_size = std::min(n_batch_, static_cast<int>(msg_tokens.size() - i));

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

    // Mark message as successfully cached
    msg.in_kv_cache = true;

    // Always show performance metrics to stderr (visible even without debug mode)
    std::cerr << "\033[90m[Prefill: " << n_tokens << " tokens, "
              << std::fixed << std::setprecision(1) << prefill_tokens_per_second << " t/s]\033[0m" << std::endl;

    LOG_INFO("Prompt processing: " + std::to_string(n_tokens) + " tokens in " +
             std::to_string(prefill_seconds) + "s (" + std::to_string(prefill_tokens_per_second) + " t/s)");

    return true;
#else
    return false;
#endif
}






