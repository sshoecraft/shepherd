#include "tensorrt.h"
#include "../logger.h"
#include "../tools/tool.h"
#include "../minja.hpp"
#include "../nlohmann/json.hpp"
#include "../global_args.h"
#include "../model_config.h"
#include "tokenizers_cpp.h"

#ifdef ENABLE_TENSORRT
#include "tensorrt_llm/executor/executor.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <variant>
#include <dlfcn.h>
#include <algorithm>
#include <unistd.h>
#include <linux/limits.h>
#include <cstring>

// Force plugin library symbols to be loaded with RTLD_GLOBAL
// This ensures all plugins are visible to TensorRT
namespace {
    struct PluginForcer {
        PluginForcer() {
            LOG_DEBUG("PluginForcer: Loading TensorRT base plugin library...");
            // Load base TensorRT plugin library first (contains plugin registry)
            void* base_handle = dlopen("libnvinfer_plugin.so.10", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
            if (!base_handle) {
                LOG_WARN("Could not load base TensorRT plugin library: " + std::string(dlerror()));
            } else {
                LOG_DEBUG("Base TensorRT plugin library loaded");
            }

            LOG_DEBUG("PluginForcer: Loading TensorRT-LLM plugin library...");
            // Now load TensorRT-LLM plugins
            void* handle = dlopen("libnvinfer_plugin_tensorrt_llm.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
            if (!handle) {
                const char* err = dlerror();
                if (err) {
                    LOG_ERROR("Could not preload TensorRT-LLM plugin library: " + std::string(err));
                }
            } else {
                LOG_DEBUG("TensorRT-LLM plugin library loaded successfully!");

                // THIS IS THE CRITICAL STEP: Call initTrtLlmPlugins just like Python does
                // From Python: handle.initTrtLlmPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))
                typedef bool (*InitPluginsFunc)(void*, const char*);
                InitPluginsFunc initTrtLlmPlugins = (InitPluginsFunc)dlsym(handle, "initTrtLlmPlugins");
                if (initTrtLlmPlugins) {
                    LOG_DEBUG("Found initTrtLlmPlugins, calling it with namespace 'tensorrt_llm'...");
                    bool success = initTrtLlmPlugins(nullptr, "tensorrt_llm");
                    if (success) {
                        LOG_DEBUG("initTrtLlmPlugins succeeded! Plugins should now be registered.");
                    } else {
                        LOG_ERROR("initTrtLlmPlugins returned false!");
                    }
                } else {
                    LOG_ERROR("initTrtLlmPlugins not found: " + std::string(dlerror()));
                }
            }
        }
    };
    // Static instance to run before main
    static PluginForcer plugin_forcer;
}
#endif

// TensorRTTokenizer implementation
TensorRTTokenizer::TensorRTTokenizer(const std::string& tokenizer_path)
    : tokenizer_path_(tokenizer_path) {
    LOG_DEBUG("TensorRT tokenizer initialized");
    if (!tokenizer_path.empty()) {
        load_tokenizer(tokenizer_path);
    }
}

TensorRTTokenizer::~TensorRTTokenizer() {
    if (tokenizer_) {
        delete static_cast<tokenizers::Tokenizer*>(tokenizer_);
        tokenizer_ = nullptr;
    }
}

bool TensorRTTokenizer::load_tokenizer(const std::string& tokenizer_path) {
    tokenizer_path_ = tokenizer_path;

    // Load tokenizer.json file
    std::string json_file = tokenizer_path + "/tokenizer.json";
    std::ifstream file(json_file);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open tokenizer file: " + json_file);
        return UINT32_MAX;
    }

    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    try {
        auto tok = tokenizers::Tokenizer::FromBlobJSON(json_blob);
        tokenizer_ = tok.release();
        LOG_INFO("Loaded tokenizer from: " + json_file);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load tokenizer: " + std::string(e.what()));
        return UINT32_MAX;
    }
}

int TensorRTTokenizer::count_tokens(const std::string& text) {
    if (!tokenizer_) {
        // Fallback approximation
        return static_cast<int>(text.length() / 4.0 + 0.5);
    }
    auto* tok = static_cast<tokenizers::Tokenizer*>(tokenizer_);
    return static_cast<int>(tok->Encode(text).size());
}

std::vector<int> TensorRTTokenizer::encode(const std::string& text) {
    if (!tokenizer_) {
        LOG_ERROR("Tokenizer not loaded");
        return {};
    }
    auto* tok = static_cast<tokenizers::Tokenizer*>(tokenizer_);
    auto tokens32 = tok->Encode(text);
    // Convert int32_t to int
    std::vector<int> result(tokens32.begin(), tokens32.end());

    // CRITICAL FIX: Llama 3.1 requires BOS token (128000) at the start
    // Without this, the model generates garbage (token 247 repeatedly)
    if (!result.empty() && result[0] != 128000) {
        result.insert(result.begin(), 128000);  // Add <|begin_of_text|>
    }

    return result;
}

std::string TensorRTTokenizer::decode(const std::vector<int>& tokens) {
    if (!tokenizer_) {
        LOG_ERROR("Tokenizer not loaded");
        return "";
    }
    auto* tok = static_cast<tokenizers::Tokenizer*>(tokenizer_);
    // Convert int to int32_t
    std::vector<int32_t> tokens32(tokens.begin(), tokens.end());
    return tok->Decode(tokens32);
}

std::string TensorRTTokenizer::get_tokenizer_name() const {
    return "tensorrt";
}

// TensorRTContextManager implementation
TensorRTContextManager::TensorRTContextManager(size_t max_context_tokens)
    : ContextManager(max_context_tokens),
      model_config_(ModelConfig::create_generic()) {
    LOG_DEBUG("TensorRT context manager initialized");
}

void TensorRTContextManager::add_message(const Message& message) {
    // Track cumulative token position before this message
    size_t tokens_before = 0;
    if (!messages_.empty()) {
        tokens_before = message_token_positions_.back() + messages_.back().token_count;
    }
    message_token_positions_.push_back(tokens_before);

    // Call base class to add message
    ContextManager::add_message(message);
}

std::vector<int> TensorRTContextManager::get_messages_in_token_range(size_t start_token, size_t end_token) const {
    std::vector<int> affected_messages;

    for (size_t i = 0; i < messages_.size(); ++i) {
        size_t msg_start = message_token_positions_[i];
        size_t msg_end = msg_start + messages_[i].token_count;

        // Check if message overlaps with token range
        if (msg_start < end_token && msg_end > start_token) {
            affected_messages.push_back(static_cast<int>(i));
        }
    }

    return affected_messages;
}

size_t TensorRTContextManager::get_tokens_before_message(int msg_index) const {
    if (msg_index < 0 || static_cast<size_t>(msg_index) >= message_token_positions_.size()) {
        return 0;
    }
    return message_token_positions_[msg_index];
}

std::string TensorRTContextManager::get_context_for_inference() {
    // Use minja to render messages with the chat template (if available)
    if (!template_node_) {
        LOG_ERROR("No template node available, falling back to simple format");
        // Fallback to simple format
        std::ostringstream text_builder;
        for (const auto& msg : messages_) {
            text_builder << msg.get_role() << ": " << msg.content << "\n\n";
        }
        text_builder << "assistant: ";
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
    for (const auto& msg : messages_) {
        auto msg_obj = minja::Value::object();
        // Use model-specific role mapping
        msg_obj.set("role", minja::Value(msg.get_role_for_model(model_config_)));
        msg_obj.set("content", minja::Value(msg.content));
        messages.push_back(msg_obj);
    }

    context->set("bos_token", minja::Value("<|begin_of_text|>"));
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(true));

    // Set builtin_tools to enable <|eom_id|> generation for tool calls
    auto builtin_tools_array = minja::Value::array();
    builtin_tools_array.push_back(minja::Value("ipython"));
    context->set("builtin_tools", builtin_tools_array);

    std::string rendered = (*template_ptr)->render(context);
    LOG_DEBUG("Rendered template (" + std::to_string(rendered.length()) + " chars)");
    return rendered;
}

int TensorRTContextManager::count_tokens(const std::string& text) {
    return static_cast<int>(text.length() / 4.0 + 0.5);
}

int TensorRTContextManager::calculate_json_overhead() const {
    // TensorRT uses simple text formatting, minimal overhead
    return 10;  // Small overhead for role labels and formatting
}

void TensorRTContextManager::evict_old_messages() {
    // This is called internally for basic eviction
    // For KV cache event-driven eviction, we use evict_messages_by_index()
    while (needs_eviction(0) && !messages_.empty()) {
        // Remove oldest message
        current_token_count_ -= messages_[0].token_count;
        messages_.pop_front();
        message_token_positions_.erase(message_token_positions_.begin());
    }
}

// TensorRTBackend implementation
TensorRTBackend::TensorRTBackend(size_t max_context_tokens)
    : BackendManager(max_context_tokens) {
    // Don't create context_manager here - will be created in initialize() after reading model config
    tokenizer_ = std::make_unique<TensorRTTokenizer>("");
    LOG_DEBUG("TensorRTBackend created");
}

TensorRTBackend::~TensorRTBackend() {
    shutdown();
}

#ifdef ENABLE_TENSORRT
ModelConfig TensorRTBackend::detect_model_family() {
    // Primary detection: Analyze chat template content (most reliable)
    if (!chat_template_text_.empty()) {
        LOG_DEBUG("Detecting model family from chat template...");

        // Llama 3.x: Has "Environment: ipython" and <|eom_id|>
        if (chat_template_text_.find("Environment: ipython") != std::string::npos &&
            chat_template_text_.find("<|eom_id|>") != std::string::npos) {
            LOG_INFO("Detected Llama 3.x model family from chat template");

            // Try to extract version from model path
            std::string version = "3.1";  // Default
            if (model_path_.find("3.3") != std::string::npos) version = "3.3";
            else if (model_path_.find("3.2") != std::string::npos) version = "3.2";
            else if (model_path_.find("3.1") != std::string::npos) version = "3.1";
            else if (model_path_.find("3.0") != std::string::npos) version = "3.0";

            return ModelConfig::create_llama_3x(version);
        }

        // GLM-4.x: Has <|observation|>, <think>, [gMASK]
        if (chat_template_text_.find("<|observation|>") != std::string::npos ||
            (chat_template_text_.find("<think>") != std::string::npos &&
             chat_template_text_.find("[gMASK]") != std::string::npos)) {
            LOG_INFO("Detected GLM-4.x model family from chat template");

            // Detect version
            std::string version = "4";
            if (model_path_.find("4.6") != std::string::npos) version = "4.6";
            else if (model_path_.find("4.5") != std::string::npos) version = "4.5";

            return ModelConfig::create_glm_4(version);
        }

        // Qwen 2.x: Has <|im_start|>
        if (chat_template_text_.find("<|im_start|>") != std::string::npos) {
            LOG_INFO("Detected Qwen 2.x model family from chat template");

            std::string version = "2.5";
            if (model_path_.find("2.5") != std::string::npos) version = "2.5";
            else if (model_path_.find("2.0") != std::string::npos) version = "2.0";

            return ModelConfig::create_qwen_2x(version);
        }

        // Add more template-based detection patterns here as needed
        // - Mistral: [INST]
        // - Gemma: <start_of_turn>
        // - etc.
    }

    // Fallback: Parse config.json for model_type
    std::string config_file = model_path_ + "/config.json";
    std::ifstream file(config_file);
    if (file.is_open()) {
        std::string config_json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        LOG_DEBUG("Detecting model family from config.json...");

        // Look for model_type field
        size_t model_type_pos = config_json.find("\"model_type\"");
        if (model_type_pos != std::string::npos) {
            size_t colon_pos = config_json.find(":", model_type_pos);
            size_t quote_start = config_json.find("\"", colon_pos);
            size_t quote_end = config_json.find("\"", quote_start + 1);

            if (quote_start != std::string::npos && quote_end != std::string::npos) {
                std::string model_type = config_json.substr(quote_start + 1, quote_end - quote_start - 1);
                LOG_DEBUG("Found model_type: " + model_type);

                if (model_type == "llama") {
                    // Check for Llama 3.x specific tokens in tokenizer
                    if (config_json.find("<|begin_of_text|>") != std::string::npos ||
                        config_json.find("<|eom_id|>") != std::string::npos) {
                        LOG_INFO("Detected Llama 3.x from config.json");
                        return ModelConfig::create_llama_3x();
                    }
                }
                else if (model_type == "chatglm" || model_type == "glm") {
                    LOG_INFO("Detected GLM-4.x from config.json");
                    return ModelConfig::create_glm_4();
                }
                else if (model_type == "qwen2") {
                    LOG_INFO("Detected Qwen 2.x from config.json");
                    return ModelConfig::create_qwen_2x();
                }
            }
        }
    }

    // Final fallback: Generic model
    LOG_WARN("Could not detect specific model family, using generic configuration");
    return ModelConfig::create_generic();
}
#endif

bool TensorRTBackend::initialize(const std::string& model_path, const std::string& api_key, const std::string& template_path) {
#ifdef ENABLE_TENSORRT
    if (model_path.empty()) {
        LOG_ERROR("Model path is required for TensorRT backend");
        return UINT32_MAX;
    }

    model_path_ = model_path;
    model_name_ = model_path;

    try {
        LOG_INFO("Initializing TensorRT-LLM Executor with model: " + model_path);

        // Parse config.json to check parallelism settings
        std::string config_file = model_path + "/config.json";
        std::ifstream file(config_file);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open config file: " + config_file);
            return UINT32_MAX;
        }

        std::string config_json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // Simple JSON parsing to extract world_size and max_seq_len
        int required_world_size = 1;
        int max_seq_len = 2048; // Default fallback

        // Extract world_size from mapping section
        size_t mapping_pos = config_json.find("\"mapping\"");
        if (mapping_pos != std::string::npos) {
            size_t world_size_pos = config_json.find("\"world_size\"", mapping_pos);
            if (world_size_pos != std::string::npos) {
                size_t colon_pos = config_json.find(":", world_size_pos);
                if (colon_pos != std::string::npos) {
                    size_t comma_pos = config_json.find_first_of(",}", colon_pos);
                    if (comma_pos != std::string::npos) {
                        std::string value = config_json.substr(colon_pos + 1, comma_pos - colon_pos - 1);
                        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
                        required_world_size = std::stoi(value);
                    }
                }
            }
        }

        // Extract max_seq_len from build_config section
        size_t build_config_pos = config_json.find("\"build_config\"");
        if (build_config_pos != std::string::npos) {
            size_t max_seq_pos = config_json.find("\"max_seq_len\"", build_config_pos);
            if (max_seq_pos != std::string::npos) {
                size_t colon_pos = config_json.find(":", max_seq_pos);
                if (colon_pos != std::string::npos) {
                    size_t comma_pos = config_json.find_first_of(",}", colon_pos);
                    if (comma_pos != std::string::npos) {
                        std::string value = config_json.substr(colon_pos + 1, comma_pos - colon_pos - 1);
                        value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
                        max_seq_len = std::stoi(value);
                    }
                }
            }
        }

        // Only auto-detect context size if not explicitly set (max_context_size_ == 0 means auto)
        if (max_context_size_ == 0) {
            max_context_size_ = max_seq_len;
            LOG_INFO("Model max sequence length: " + std::to_string(max_seq_len));
        } else {
            LOG_INFO("Using explicitly configured context size: " + std::to_string(max_context_size_));
        }

        // Create context manager with the determined context size
        context_manager_ = std::make_unique<TensorRTContextManager>(max_context_size_);
        LOG_DEBUG("Created TensorRTContextManager with " + std::to_string(max_context_size_) + " tokens");

        LOG_INFO("Model requires world_size: " + std::to_string(required_world_size));

        // Check if we're already running under MPI
        int current_world_size = 1;
        const char* mpi_world_env = getenv("OMPI_COMM_WORLD_SIZE");
        if (mpi_world_env) {
            current_world_size = std::stoi(mpi_world_env);
        }

        // If model requires multi-GPU but we're not under MPI, re-exec with mpirun
        if (required_world_size > 1 && current_world_size == 1) {
            LOG_INFO("Model requires " + std::to_string(required_world_size) + " GPUs, re-launching with mpirun...");

            // Build mpirun command with original arguments
            std::vector<std::string> args;
            args.push_back("mpirun");
            args.push_back("-n");
            args.push_back(std::to_string(required_world_size));
            args.push_back("--stdin");
            args.push_back("0");  // Forward stdin only to rank 0
            args.push_back("--bind-to");
            args.push_back("none");  // Don't bind processes to cores (helps with I/O)
            // Merge stdout/stderr from all ranks
            args.push_back("-x");
            args.push_back("OMPI_MCA_orte_timestamp_output=0");

            // Get original command-line arguments
            int orig_argc = 0;
            char** orig_argv = nullptr;
            get_global_args(orig_argc, orig_argv);

            // Add all original arguments (starting from argv[0] which is the executable)
            if (orig_argc > 0 && orig_argv) {
                for (int i = 0; i < orig_argc; ++i) {
                    args.push_back(orig_argv[i]);
                }
            } else {
                // Fallback if global args not available
                char exe_path[PATH_MAX];
                ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
                if (len != -1) {
                    exe_path[len] = '\0';
                    args.push_back(exe_path);
                } else {
                    LOG_ERROR("Failed to get executable path for mpirun re-exec");
                    return UINT32_MAX;
                }
            }

            // Convert to char* array for execvp
            std::vector<char*> c_args;
            for (auto& arg : args) {
                c_args.push_back(const_cast<char*>(arg.c_str()));
            }
            c_args.push_back(nullptr);

            // Re-exec with mpirun
            LOG_INFO("Executing: " + [&]() {
                std::string cmd;
                for (const auto& arg : args) cmd += arg + " ";
                return cmd;
            }());

            execvp("mpirun", c_args.data());

            // If we get here, exec failed
            LOG_ERROR("Failed to exec mpirun: " + std::string(strerror(errno)));
            return UINT32_MAX;
        }

        // Verify we have the correct world size if running under MPI
        if (required_world_size > 1 && current_world_size != required_world_size) {
            LOG_ERROR("MPI world size (" + std::to_string(current_world_size) + ") doesn't match model requirement (" + std::to_string(required_world_size) + ")");
            return UINT32_MAX;
        }

        LOG_INFO("World size verified, creating ExecutorConfig...");

        // Create ExecutorConfig with reasonable defaults
        namespace tle = tensorrt_llm::executor;

        tle::ExecutorConfig config;
        config.setMaxBeamWidth(1);  // Greedy decoding

        // Set KV cache configuration with event monitoring enabled
        tle::KvCacheConfig kvCacheConfig;
        kvCacheConfig.setEnableBlockReuse(true);
        kvCacheConfig.setEventBufferMaxSize(1000);  // Enable event buffering
        config.setKvCacheConfig(kvCacheConfig);

        // Create Executor (loads the TensorRT engine)
        LOG_INFO("About to create TensorRT-LLM Executor...");
        auto* executor = new tle::Executor(
            model_path,
            tle::ModelType::kDECODER_ONLY,
            config
        );

        executor_ = static_cast<void*>(executor);

        LOG_INFO("Executor created successfully, getting KV cache event manager...");

        // Get KV cache event manager
        auto event_mgr_opt = executor->getKVCacheEventManager();
        LOG_INFO("getKVCacheEventManager() returned");

        if (event_mgr_opt.has_value()) {
            // Store the shared_ptr in a new allocation
            event_manager_ = new std::shared_ptr<tle::KVCacheEventManager>(*event_mgr_opt);
            LOG_INFO("KV cache event monitoring enabled");

            // Start event monitoring thread
            monitoring_events_ = true;
            kv_event_monitor_thread_ = std::thread(&TensorRTBackend::monitor_kv_events, this);
        } else {
            LOG_WARN("KV cache event manager not available");
        }

        LOG_INFO("Starting tokenizer load...");

        // Load tokenizer from model_path (engine directory)
        LOG_INFO("Loading tokenizer from: " + model_path);
        if (!static_cast<TensorRTTokenizer*>(tokenizer_.get())->load_tokenizer(model_path)) {
            LOG_ERROR("Failed to load tokenizer from: " + model_path);
            LOG_ERROR("Please ensure tokenizer.json exists in the engine directory");
            return UINT32_MAX;
        }

        LOG_INFO("Tokenizer initialization complete");

        // Load chat template from tokenizer_config.json
        LOG_INFO("Loading chat template from tokenizer_config.json...");
        std::string tokenizer_config_file = model_path + "/tokenizer_config.json";
        std::ifstream config_stream(tokenizer_config_file);
        if (config_stream.is_open()) {
            try {
                nlohmann::json tokenizer_config;
                config_stream >> tokenizer_config;
                config_stream.close();

                if (tokenizer_config.contains("chat_template")) {
                    chat_template_text_ = tokenizer_config["chat_template"].get<std::string>();
                    LOG_INFO("Loaded chat template (" + std::to_string(chat_template_text_.length()) + " chars)");

                    // Parse the chat template with minja
                    try {
                        minja::Options options{};
                        auto parsed_template = minja::Parser::parse(chat_template_text_, options);
                        template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
                        LOG_INFO("Successfully parsed chat template with minja");

                        // Pass template to context manager
                        auto* trt_ctx_mgr = dynamic_cast<TensorRTContextManager*>(context_manager_.get());
                        if (trt_ctx_mgr) {
                            trt_ctx_mgr->set_template_node(template_node_);
                            LOG_DEBUG("Set template node in context manager");
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("Failed to parse chat template: " + std::string(e.what()));
                        LOG_WARN("Will fall back to simple format");
                    }
                } else {
                    LOG_WARN("No chat_template found in tokenizer_config.json");
                    LOG_WARN("Will use simple format");
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to parse tokenizer_config.json: " + std::string(e.what()));
                LOG_WARN("Will use simple format");
            }
        } else {
            LOG_WARN("tokenizer_config.json not found at: " + tokenizer_config_file);
            LOG_WARN("Will use simple format");
        }

        // Detect model family and configure prompt behavior
        model_config_ = detect_model_family();
        LOG_INFO("Model configuration: family=" + std::to_string(static_cast<int>(model_config_.family)) +
                 ", version=" + model_config_.version +
                 ", tool_result_role=" + model_config_.tool_result_role +
                 ", uses_eom_token=" + (model_config_.uses_eom_token ? "true" : "false"));

        // Pass model config to context manager
        auto* trt_ctx_mgr = dynamic_cast<TensorRTContextManager*>(context_manager_.get());
        if (trt_ctx_mgr) {
            trt_ctx_mgr->set_model_config(model_config_);
            LOG_DEBUG("Set model config in context manager");
        }

        initialized_ = true;
        LOG_INFO("TensorRT-LLM Executor initialized successfully");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize TensorRT executor: " + std::string(e.what()));
        return UINT32_MAX;
    }
#else
    LOG_ERROR("TensorRT backend not compiled in");
    return UINT32_MAX;
#endif
}

std::string TensorRTBackend::generate(int max_tokens) {
#ifdef ENABLE_TENSORRT
    if (!is_ready()) {
        throw BackendManagerError("TensorRT backend not initialized");
    }

    namespace tle = tensorrt_llm::executor;
    auto* executor = static_cast<tle::Executor*>(executor_);

    try {
        // Get context as text (chat template is already applied by context manager)
        std::string prompt_text = context_manager_->get_context_for_inference();

        // Always show full prompt in debug mode - user needs to see exactly what's being sent
        extern bool g_debug_mode;
        if (g_debug_mode) {
            LOG_DEBUG("========== PROMPT BEING SENT TO MODEL ==========");
            LOG_DEBUG(prompt_text);
            LOG_DEBUG("================================================");
        }

        // Tokenize the prompt (no additional formatting needed - chat template already applied)
        if (!tokenizer_) {
            LOG_ERROR("Tokenizer is NULL! Cannot encode prompt");
            throw BackendManagerError("Tokenizer not initialized");
        }
        std::vector<int> tokens = tokenizer_->encode(prompt_text);
        if (tokens.empty()) {
            throw BackendManagerError("Failed to tokenize prompt");
        }

        // Convert to int32_t
        std::vector<int32_t> input_tokens(tokens.begin(), tokens.end());
        LOG_DEBUG("Tokenized prompt to " + std::to_string(input_tokens.size()) + " tokens");

        // Create sampling config
        tle::SamplingConfig samplingConfig;
        samplingConfig.setBeamWidth(1);
        samplingConfig.setTemperature(0.7f);

        // Create output config
        tle::OutputConfig outputConfig;
        outputConfig.returnLogProbs = false;
        outputConfig.returnContextLogits = false;
        outputConfig.returnGenerationLogits = false;

        // Create request
        int actual_max_tokens = (max_tokens > 0) ? max_tokens : 512;
        tle::Request request(
            input_tokens,
            actual_max_tokens,
            true,  // streaming
            samplingConfig,
            outputConfig
        );

        // Enqueue request
        uint64_t request_id = executor->enqueueRequest(request);
        current_request_id_ = request_id;
        request_active_ = true;

        LOG_DEBUG("Request enqueued with ID: " + std::to_string(request_id));

        // Collect responses
        std::string response_text;
        std::vector<int32_t> output_tokens;

        while (request_active_) {
            // Wait for responses with timeout
            auto responses = executor->awaitResponses(
                request_id,
                std::chrono::milliseconds(100)
            );

            for (const auto& response : responses) {
                if (response.hasError()) {
                    LOG_ERROR("Response error: " + response.getErrorMsg());
                    request_active_ = false;
                    throw BackendManagerError("TensorRT generation failed: " + response.getErrorMsg());
                }

                const auto& result = response.getResult();

                // Get output tokens from first beam
                if (!result.outputTokenIds.empty() && !result.outputTokenIds[0].empty()) {
                    const auto& beam_tokens = result.outputTokenIds[0];

                    // Accumulate new tokens
                    size_t prev_size = output_tokens.size();
                    output_tokens.insert(output_tokens.end(), beam_tokens.begin(), beam_tokens.end());

                    // Detokenize new tokens
                    std::vector<int> new_tokens_int(beam_tokens.begin(), beam_tokens.end());
                    std::string new_text = tokenizer_->decode(new_tokens_int);
                    response_text += new_text;

                    // Stream output
                    std::cout << new_text << std::flush;
                }

                // Check if generation is complete
                if (result.isFinal) {
                    LOG_DEBUG("Generation complete");
                    request_active_ = false;
                    break;
                }
            }
        }

        std::cout << std::endl;
        LOG_DEBUG("Generated " + std::to_string(output_tokens.size()) + " tokens");

        // Always show full response in debug mode - user needs to see exactly what model returned
        if (g_debug_mode) {
            LOG_DEBUG("========== MODEL RESPONSE ==========");
            LOG_DEBUG(response_text);
            LOG_DEBUG("====================================");
        }

        // Handle <|eom_id|> token for continued tool execution (Llama 3.x)
        if (model_config_.uses_eom_token) {
            // Check if response contains <|eom_id|> (end of message, not end of turn)
            size_t eom_pos = response_text.find("<|eom_id|>");
            if (eom_pos != std::string::npos) {
                // Strip the <|eom_id|> token from response
                response_text = response_text.substr(0, eom_pos);
                LOG_DEBUG("Found <|eom_id|> token - model expects tool execution continuation");
                // The main loop in main.cpp will continue to execute the tool
            }
            // Note: If we find <|eot_id|> instead, that means final response (no stripping needed)
        }

        return response_text;

    } catch (const std::exception& e) {
        request_active_ = false;
        LOG_ERROR("TensorRT generation error: " + std::string(e.what()));
        throw BackendManagerError("TensorRT generation failed: " + std::string(e.what()));
    }
#else
    LOG_ERROR("TensorRT backend not compiled in");
    return "Error: TensorRT not available";
#endif
}

void TensorRTBackend::add_system_message(const std::string& content) {
    // System message already includes tool list from main.cpp - no need to append again
    std::string full_content = content;

    // LOG_DEBUG("========== SYSTEM MESSAGE BEING ADDED ==========");
    // LOG_DEBUG(full_content);
    // LOG_DEBUG("================================================");

    int token_count = context_manager_->count_tokens(full_content);
    Message system_msg(Message::SYSTEM, full_content, token_count);
    context_manager_->add_message(system_msg);
    LOG_DEBUG("Added system message to TensorRT backend");
}

void TensorRTBackend::add_user_message(const std::string& content) {
    int token_count = context_manager_->count_tokens(content);
    Message user_msg(Message::USER, content, token_count);
    context_manager_->add_message(user_msg);
    LOG_DEBUG("Added user message to TensorRT backend");
}

void TensorRTBackend::add_tool_result(const std::string& tool_name, const std::string& content, const std::string& tool_call_id) {
    int token_count = context_manager_->count_tokens(content);
    Message tool_msg(Message::TOOL, content, token_count);
    tool_msg.tool_name = tool_name;
    tool_msg.tool_call_id = tool_call_id;
    context_manager_->add_message(tool_msg);
    LOG_DEBUG("Added tool result to TensorRT backend: " + tool_name);
}

void TensorRTBackend::add_assistant_message(const std::string& content) {
    int token_count = context_manager_->count_tokens(content);
    Message assistant_msg(Message::ASSISTANT, content, token_count);
    context_manager_->add_message(assistant_msg);
    LOG_DEBUG("Added assistant message to TensorRT backend");
}

std::string TensorRTBackend::get_backend_name() const {
    return "tensorrt";
}

std::string TensorRTBackend::get_model_name() const {
    return model_name_;
}

size_t TensorRTBackend::get_max_context_size() const {
#ifdef ENABLE_TENSORRT
    return max_context_size_;
#else
    return 4096;
#endif
}

bool TensorRTBackend::is_ready() const {
    return initialized_;
}

ModelConfig TensorRTBackend::get_model_config() const {
#ifdef ENABLE_TENSORRT
    return model_config_;
#else
    return ModelConfig::create_generic();
#endif
}

uint32_t TensorRTBackend::evict_to_free_space(uint32_t tokens_needed) {
    LOG_INFO("TensorRT KV cache eviction not yet implemented");
    // TensorRT-LLM manages KV cache internally
    // This would need to be coordinated with the executor
    return UINT32_MAX;
}

void TensorRTBackend::shutdown() {
#ifdef ENABLE_TENSORRT
    // Stop event monitoring thread first
    if (monitoring_events_) {
        monitoring_events_ = false;
        if (kv_event_monitor_thread_.joinable()) {
            kv_event_monitor_thread_.join();
        }
    }

    // Clean up chat template
    if (template_node_) {
        delete static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);
        template_node_ = nullptr;
    }

    // Clean up event manager
    if (event_manager_) {
        namespace tle = tensorrt_llm::executor;
        delete static_cast<std::shared_ptr<tle::KVCacheEventManager>*>(event_manager_);
        event_manager_ = nullptr;
    }

    // Shutdown executor
    if (executor_) {
        namespace tle = tensorrt_llm::executor;
        auto* executor = static_cast<tle::Executor*>(executor_);

        try {
            executor->shutdown();
            delete executor;
        } catch (const std::exception& e) {
            LOG_ERROR("Error during TensorRT executor shutdown: " + std::string(e.what()));
        }

        executor_ = nullptr;
    }
#endif
    initialized_ = false;
    LOG_DEBUG("TensorRTBackend shutdown");
}

#ifdef ENABLE_TENSORRT
void TensorRTBackend::monitor_kv_events() {
    namespace tle = tensorrt_llm::executor;

    if (!event_manager_) {
        LOG_ERROR("Event manager not initialized");
        return;
    }

    auto* event_mgr = static_cast<std::shared_ptr<tle::KVCacheEventManager>*>(event_manager_);
    LOG_INFO("KV cache event monitoring thread started");

    while (monitoring_events_) {
        try {
            // Poll for events with 100ms timeout
            auto events = (*event_mgr)->getLatestEvents(std::chrono::milliseconds(100));

            for (const auto& event : events) {
                // Check if this is a removal event
                if (std::holds_alternative<tle::KVCacheRemovedData>(event.data)) {
                    const auto& removed = std::get<tle::KVCacheRemovedData>(event.data);

                    LOG_INFO("KV cache eviction detected: " +
                             std::to_string(removed.blockHashes.size()) + " blocks removed");

                    // Convert IdType vector to uint64_t vector
                    std::vector<uint64_t> block_hashes;
                    for (const auto& hash : removed.blockHashes) {
                        block_hashes.push_back(static_cast<uint64_t>(hash));
                    }

                    // Handle the eviction
                    handle_kv_cache_removed(block_hashes);
                }
                // Could also handle KVCacheStoredData to track block mappings
                else if (std::holds_alternative<tle::KVCacheStoredData>(event.data)) {
                    const auto& stored = std::get<tle::KVCacheStoredData>(event.data);

                    // Track which blocks were stored (for future mapping)
                    std::lock_guard<std::mutex> lock(block_map_mutex_);
                    for (const auto& block : stored.blocks) {
                        // TODO: Map block hash to token positions
                        // This requires knowing token positions when blocks are created
                        LOG_DEBUG("Block stored: hash=" + std::to_string(block.blockHash) +
                                 " tokens=" + std::to_string(block.tokens.size()));
                    }
                }
            }

        } catch (const std::exception& e) {
            LOG_ERROR("Error in KV cache event monitoring: " + std::string(e.what()));
        }
    }

    LOG_INFO("KV cache event monitoring thread stopped");
}

void TensorRTBackend::handle_kv_cache_removed(const std::vector<uint64_t>& block_hashes) {
    // Get context manager
    auto* trt_ctx_mgr = dynamic_cast<TensorRTContextManager*>(context_manager_.get());
    if (!trt_ctx_mgr) {
        LOG_ERROR("Context manager is not TensorRTContextManager");
        return;
    }

    // Try to determine which messages were affected
    // This is a simplified approach - we don't have perfect block->token mapping yet
    // So we'll estimate based on block count and evict oldest messages

    std::lock_guard<std::mutex> lock(block_map_mutex_);

    // Estimate tokens evicted (rough approximation)
    // TensorRT typically uses 64-128 tokens per block
    size_t estimated_tokens_evicted = block_hashes.size() * 64;

    LOG_INFO("Estimated " + std::to_string(estimated_tokens_evicted) +
             " tokens evicted based on " + std::to_string(block_hashes.size()) + " blocks");

    // Calculate which messages to evict from our tracking
    auto [start_msg, end_msg] = context_manager_->calculate_messages_to_evict(
        static_cast<uint32_t>(estimated_tokens_evicted)
    );

    if (start_msg == -1 || end_msg == -1) {
        LOG_WARN("Could not determine messages to evict");
        return;
    }

    // Use the shared eviction logic - this will:
    // 1. Archive to RAG
    // 2. Remove messages from deque
    // 3. Insert NOTICE message
    if (!context_manager_->evict_messages_by_index(start_msg, end_msg)) {
        LOG_ERROR("Failed to evict messages from context manager");
        return;
    }

    // Clear block mappings for evicted blocks
    for (uint64_t hash : block_hashes) {
        block_to_tokens_.erase(hash);
    }

    LOG_INFO("Successfully handled KV cache eviction, added notice to context");
}
#endif
