#include "shepherd.h"
#include "tensorrt.h"
#include "session.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "models.h"
#include "llama.cpp/vendor/minja/minja.hpp"
#include "nlohmann/json.hpp"
#include "config.h"
#include "rag.h"
#include "terminal_io.h"

#ifdef ENABLE_TENSORRT
#include "tokenizers_c.h"  // Use C API instead of broken C++ wrapper
#include "tensorrt_llm/executor/executor.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
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
#include <filesystem>

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

#ifdef ENABLE_TENSORRT
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
        tokenizers_free(static_cast<TokenizerHandle>(tokenizer_));
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
        return false;
    }

    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    LOG_INFO("Creating tokenizer from JSON (" + std::to_string(json_blob.length()) + " bytes)...");

    // Use C API (which works) instead of broken C++ wrapper
    TokenizerHandle handle = tokenizers_new_from_str(json_blob.data(), json_blob.length());
    if (!handle) {
        LOG_ERROR("Failed to create tokenizer from JSON");
        return false;
    }

    LOG_INFO("Tokenizer handle created successfully");
    tokenizer_ = handle;
    LOG_INFO("Loaded tokenizer from: " + json_file);
    return true;
}

int TensorRTTokenizer::count_tokens(const std::string& text) {
    if (!tokenizer_) {
        // Fallback approximation
        return static_cast<int>(text.length() / 4.0 + 0.5);
    }

    // Use C API
    TokenizerHandle handle = static_cast<TokenizerHandle>(tokenizer_);
    TokenizerEncodeResult result;
    tokenizers_encode(handle, text.data(), text.length(), 0, &result);
    int count = static_cast<int>(result.len);
    tokenizers_free_encode_results(&result, 1);
    return count;
}

std::vector<int> TensorRTTokenizer::encode(const std::string& text, bool add_special_tokens) {
    if (!tokenizer_) {
        LOG_ERROR("Tokenizer not loaded");
        return {};
    }

    // Use C API
    TokenizerHandle handle = static_cast<TokenizerHandle>(tokenizer_);
    TokenizerEncodeResult result;
    tokenizers_encode(handle, text.data(), text.length(), static_cast<int>(add_special_tokens), &result);

    // Convert to std::vector<int>
    std::vector<int> tokens;
    tokens.reserve(result.len);
    for (size_t i = 0; i < result.len; i++) {
        tokens.push_back(static_cast<int>(result.token_ids[i]));
    }

    tokenizers_free_encode_results(&result, 1);
    return tokens;
}

std::string TensorRTTokenizer::decode(const std::vector<int>& tokens) {
    if (!tokenizer_) {
        LOG_ERROR("Tokenizer not loaded");
        return "";
    }
    if (tokens.empty()) return "";

    // Use C API (which works!)
    TokenizerHandle handle = static_cast<TokenizerHandle>(tokenizer_);

    // Convert vector<int> to vector<uint32_t> for C API
    std::vector<uint32_t> tokens32(tokens.begin(), tokens.end());

    tokenizers_decode(handle, tokens32.data(), tokens32.size(), 0);

    const char* decoded_data;
    size_t decoded_len;
    tokenizers_get_decode_str(handle, &decoded_data, &decoded_len);

    return std::string(decoded_data, decoded_len);
}

std::string TensorRTTokenizer::get_tokenizer_name() const {
    return "tensorrt";
}
#else
// Stub implementations when TensorRT is not enabled
TensorRTTokenizer::TensorRTTokenizer(const std::string&) {}
TensorRTTokenizer::~TensorRTTokenizer() {}
bool TensorRTTokenizer::load_tokenizer(const std::string&) { return false; }
int TensorRTTokenizer::count_tokens(const std::string& text) {
    return static_cast<int>(text.length() / 4.0 + 0.5);
}
std::vector<int> TensorRTTokenizer::encode(const std::string&) { return {}; }
std::string TensorRTTokenizer::decode(const std::vector<int>&) { return ""; }
std::string TensorRTTokenizer::get_tokenizer_name() const { return "tensorrt"; }
#endif

// TensorRTBackend implementation
TensorRTBackend::TensorRTBackend(size_t context_size)
    : Backend(context_size)
#ifdef ENABLE_TENSORRT
      ,backend_session(),
      current_session(nullptr),
      model_config_(ModelConfig::create_generic())
#endif
{
    // Set public Backend variables
    backend_name = "tensorrt";
    this->context_size = context_size;
    is_local = true;  // Local GPU backend

#ifdef ENABLE_TENSORRT
    tokenizer_ = std::make_unique<TensorRTTokenizer>("");
#endif
    LOG_DEBUG("TensorRTBackend created with context_size: " + std::to_string(context_size));
}

TensorRTBackend::~TensorRTBackend() {
#ifdef ENABLE_TENSORRT
    shutdown();
#endif
}

#ifdef ENABLE_TENSORRT
ModelConfig TensorRTBackend::detect_model_family() {
    // Use centralized model detection from Models class
    ModelConfig config = Models::detect_from_chat_template(chat_template_text_, model_path_);

    // If template detection failed, try config.json as fallback
    if (config.family == ModelFamily::GENERIC) {
        std::string config_file = model_path_ + "/config.json";
        std::ifstream file(config_file);
        if (file.is_open()) {
            std::string config_json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            file.close();

            LOG_DEBUG("Detecting model family from config.json...");

            // TensorRT engines might have a different config structure
            // Check for architecture field in pretrained_config first
            size_t arch_pos = config_json.find("\"architecture\"");
            if (arch_pos != std::string::npos) {
                size_t colon_pos = config_json.find(":", arch_pos);
                size_t quote_start = config_json.find("\"", colon_pos);
                size_t quote_end = config_json.find("\"", quote_start + 1);

                if (quote_start != std::string::npos && quote_end != std::string::npos) {
                    std::string architecture = config_json.substr(quote_start + 1, quote_end - quote_start - 1);
                    LOG_DEBUG("Found architecture: " + architecture);

                    // Check for Qwen3ForCausalLM
                    if (architecture == "Qwen3ForCausalLM") {
                        LOG_INFO("Detected Qwen 3.x (MindLink) from config.json architecture");
                        return ModelConfig::create_qwen_3x();
                    }
                }
            }

            // Also check for qwen_type field
            size_t qwen_type_pos = config_json.find("\"qwen_type\"");
            if (qwen_type_pos != std::string::npos) {
                size_t colon_pos = config_json.find(":", qwen_type_pos);
                size_t quote_start = config_json.find("\"", colon_pos);
                size_t quote_end = config_json.find("\"", quote_start + 1);

                if (quote_start != std::string::npos && quote_end != std::string::npos) {
                    std::string qwen_type = config_json.substr(quote_start + 1, quote_end - quote_start - 1);
                    LOG_DEBUG("Found qwen_type: " + qwen_type);

                    if (qwen_type == "qwen3") {
                        LOG_INFO("Detected Qwen 3.x from config.json qwen_type");
                        return ModelConfig::create_qwen_3x();
                    } else if (qwen_type == "qwen2") {
                        LOG_INFO("Detected Qwen 2.x from config.json qwen_type");
                        return ModelConfig::create_qwen_2x();
                    }
                }
            }

            // Look for model_type field (standard HF format)
            size_t model_type_pos = config_json.find("\"model_type\"");
            if (model_type_pos != std::string::npos) {
                size_t colon_pos = config_json.find(":", model_type_pos);
                size_t quote_start = config_json.find("\"", colon_pos);
                size_t quote_end = config_json.find("\"", quote_start + 1);

                if (quote_start != std::string::npos && quote_end != std::string::npos) {
                    std::string model_type = config_json.substr(quote_start + 1, quote_end - quote_start - 1);
                    LOG_DEBUG("Found model_type: " + model_type);

                    // Convert to lowercase for case-insensitive comparison
                    std::transform(model_type.begin(), model_type.end(), model_type.begin(), ::tolower);

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
                    else if (model_type == "qwen" || model_type == "qwen3") {
                        // Check path for MindLink or Qwen3 indicators
                        std::string path_lower = model_path_;
                        std::transform(path_lower.begin(), path_lower.end(), path_lower.begin(), ::tolower);

                        if (path_lower.find("mindlink") != std::string::npos ||
                            path_lower.find("qwen3") != std::string::npos ||
                            path_lower.find("qwen-3") != std::string::npos) {
                            LOG_INFO("Detected Qwen 3.x (or MindLink) from config.json");
                            return ModelConfig::create_qwen_3x();
                        } else {
                            LOG_INFO("Detected Qwen 2.x from config.json");
                            return ModelConfig::create_qwen_2x();
                        }
                    }
                }
            }
        }
    }

    return config;
}

void TensorRTBackend::initialize(Session& session) {
    if (initialized) {
        LOG_WARN("TensorRTBackend already initialized");
        return;
    }

    // Get model path from config
    std::string model_filename = config->model;
    std::string model_dir = config->model_path;

    if (model_filename.empty()) {
        throw BackendError("Model name is required for TensorRT backend");
    }

    // Construct full path
    std::string full_model_path;
    if (model_filename[0] == '/' || model_filename[0] == '~') {
        full_model_path = model_filename;
    } else {
        full_model_path = (std::filesystem::path(model_dir) / model_filename).string();
    }

    if (!full_model_path.empty() && full_model_path[0] == '~') {
        full_model_path = Config::get_home_directory() + full_model_path.substr(1);
    }

    model_path_ = full_model_path;
    model_name = full_model_path;  // Set public variable

    try {
        LOG_INFO("Initializing TensorRT-LLM Executor with model: " + model_path_);

        // Parse config.json to check parallelism settings
        std::string config_file = model_path_ + "/config.json";
        LOG_DEBUG("About to open config file: " + config_file);
        std::ifstream file(config_file);
        if (!file.is_open()) {
            throw BackendError("Failed to open config file: " + config_file);
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

        // Update context_size if auto-detect
        if (context_size == 0) {
            context_size = max_seq_len;
            LOG_INFO("Model max sequence length: " + std::to_string(max_seq_len));
        } else {
            LOG_INFO("Using explicitly configured context size: " + std::to_string(context_size));
        }

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
            args.push_back("none");  // Don't bind processes to cores
            args.push_back("-x");
            args.push_back("OMPI_MCA_orte_timestamp_output=0");
            args.push_back("-x");
            args.push_back("SHEPHERD_INTERACTIVE");  // Preserve interactivity flag across MPI

            // Get original command-line arguments
            int orig_argc = 0;
            char** orig_argv = nullptr;
            get_global_args(orig_argc, orig_argv);

            if (orig_argc > 0 && orig_argv) {
                for (int i = 0; i < orig_argc; ++i) {
                    args.push_back(orig_argv[i]);
                }
            } else {
                char exe_path[PATH_MAX];
                ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
                if (len != -1) {
                    exe_path[len] = '\0';
                    args.push_back(exe_path);
                } else {
                    throw BackendError("Failed to get executable path for mpirun re-exec");
                }
            }

            // Convert to char* array for execvp
            std::vector<char*> c_args;
            for (auto& arg : args) {
                c_args.push_back(const_cast<char*>(arg.c_str()));
            }
            c_args.push_back(nullptr);

            LOG_INFO("Executing: " + [&]() {
                std::string cmd;
                for (const auto& arg : args) cmd += arg + " ";
                return cmd;
            }());

            execvp("mpirun", c_args.data());

            throw BackendError("Failed to exec mpirun: " + std::string(strerror(errno)));
        }

        // Verify we have the correct world size if running under MPI
        if (required_world_size > 1 && current_world_size != required_world_size) {
            throw BackendError("MPI world size (" + std::to_string(current_world_size) +
                             ") doesn't match model requirement (" + std::to_string(required_world_size) + ")");
        }

        LOG_INFO("World size verified, creating ExecutorConfig...");

        // Create ExecutorConfig
        namespace tle = tensorrt_llm::executor;

        tle::ExecutorConfig config;
        config.setMaxBeamWidth(1);  // Greedy decoding

        // Set KV cache configuration with event monitoring enabled
        tle::KvCacheConfig kvCacheConfig;
        kvCacheConfig.setEnableBlockReuse(true);
        kvCacheConfig.setEventBufferMaxSize(1000);
        config.setKvCacheConfig(kvCacheConfig);

        // Set scheduler policy to GUARANTEED_NO_EVICT
        // This prevents TensorRT from auto-evicting - it will fail requests instead
        // allowing us to handle eviction based on g_server_mode
        tle::SchedulerConfig schedulerConfig(tle::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT);
        config.setSchedulerConfig(schedulerConfig);
        LOG_INFO("TensorRT scheduler configured with GUARANTEED_NO_EVICT policy");

        // Create Executor (loads the TensorRT engine)
        LOG_INFO("About to create TensorRT-LLM Executor...");
        LOG_DEBUG("Model path: " + model_path_);
        LOG_DEBUG("Model path length: " + std::to_string(model_path_.length()));
        auto* executor = new tle::Executor(
            model_path_,
            tle::ModelType::kDECODER_ONLY,
            config
        );

        executor_ = static_cast<void*>(executor);
        LOG_INFO("Executor created successfully");

        // Get KV cache event manager
        auto event_mgr_opt = executor->getKVCacheEventManager();
        if (event_mgr_opt.has_value()) {
            event_manager_ = new std::shared_ptr<tle::KVCacheEventManager>(*event_mgr_opt);
            LOG_INFO("KV cache event monitoring enabled");

            // Start event monitoring thread
            monitoring_events_ = true;
            kv_event_monitor_thread_ = std::thread(&TensorRTBackend::monitor_kv_events, this);
        } else {
            LOG_WARN("KV cache event manager not available");
        }

        // Load tokenizer
        LOG_INFO("Loading tokenizer from: " + model_path_);
        if (!tokenizer_->load_tokenizer(model_path_)) {
            throw BackendError("Failed to load tokenizer from: " + model_path_);
        }
        LOG_INFO("Tokenizer loaded");

        // Load chat template from tokenizer_config.json
        LOG_INFO("Loading chat template...");
        std::string tokenizer_config_file = model_path_ + "/tokenizer_config.json";
        std::ifstream config_stream(tokenizer_config_file);
        if (config_stream.is_open()) {
            try {
                nlohmann::json tokenizer_config;
                config_stream >> tokenizer_config;
                config_stream.close();

                if (tokenizer_config.contains("chat_template")) {
                    chat_template_text_ = tokenizer_config["chat_template"].get<std::string>();
                    LOG_INFO("Loaded chat template (" + std::to_string(chat_template_text_.length()) + " chars)");

                    // Parse with minja
                    try {
                        minja::Options options{};
                        auto parsed_template = minja::Parser::parse(chat_template_text_, options);
                        template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
                        LOG_INFO("Chat template parsed successfully");
                    } catch (const std::exception& e) {
                        LOG_ERROR("Failed to parse chat template: " + std::string(e.what()));
                        LOG_WARN("Will fall back to simple format");
                    }
                }

                // Load stop tokens (eos_token) from tokenizer config
                if (tokenizer_config.contains("eos_token")) {
                    std::string eos_token;
                    if (tokenizer_config["eos_token"].is_string()) {
                        eos_token = tokenizer_config["eos_token"].get<std::string>();
                    } else if (tokenizer_config["eos_token"].is_object() &&
                               tokenizer_config["eos_token"].contains("content")) {
                        eos_token = tokenizer_config["eos_token"]["content"].get<std::string>();
                    }
                    if (!eos_token.empty()) {
                        stop_tokens_.push_back(eos_token);
                        LOG_INFO("Loaded stop token from tokenizer config: " + eos_token);
                    }
                }

                // Load BOS token configuration for filtering
                add_bos_token_ = tokenizer_config.value("add_bos_token", false);
                if (tokenizer_config.contains("bos_token")) {
                    if (tokenizer_config["bos_token"].is_string()) {
                        std::string bos_token_str = tokenizer_config["bos_token"].get<std::string>();
                        if (!bos_token_str.empty()) {
                            std::vector<int> bos_encoded = tokenizer_->encode(bos_token_str);
                            if (!bos_encoded.empty()) {
                                bos_token_id_ = bos_encoded[0];
                                LOG_INFO("Loaded BOS token: " + bos_token_str + " (ID: " + std::to_string(bos_token_id_) + ")");
                            }
                        }
                    } else if (tokenizer_config["bos_token"].is_object() &&
                               tokenizer_config["bos_token"].contains("content")) {
                        std::string bos_token_str = tokenizer_config["bos_token"]["content"].get<std::string>();
                        if (!bos_token_str.empty()) {
                            std::vector<int> bos_encoded = tokenizer_->encode(bos_token_str);
                            if (!bos_encoded.empty()) {
                                bos_token_id_ = bos_encoded[0];
                                LOG_INFO("Loaded BOS token: " + bos_token_str + " (ID: " + std::to_string(bos_token_id_) + ")");
                            }
                        }
                    }
                }
                LOG_INFO("BOS token config: add_bos_token=" + std::string(add_bos_token_ ? "true" : "false") +
                         ", bos_token_id=" + (bos_token_id_ >= 0 ? std::to_string(bos_token_id_) : "none"));
            } catch (const std::exception& e) {
                LOG_WARN("Error reading tokenizer_config.json: " + std::string(e.what()));
            }
        }

        // Detect model family
        model_config_ = detect_model_family();
        LOG_INFO("Model configuration: family=" + std::to_string(static_cast<int>(model_config_.family)) +
                 ", version=" + model_config_.version +
                 ", tool_result_role=" + model_config_.tool_result_role +
                 ", supports_thinking=" + std::to_string(model_config_.supports_thinking_mode));

        // Set max_output_tokens from model config
        max_output_tokens = model_config_.max_output_tokens;

        // Load sampling parameters from generation_config.json if available
        Models::load_generation_config(model_path_, temperature, top_p, top_k);

        initialized = true;
        LOG_INFO("TensorRT backend initialized successfully");

        // In CLI mode, add system message
        if (!g_server_mode && !session.system_message.empty()) {
            std::string formatted_system = Models::format_system_message(
                model_config_,
                session.system_message,
                ToolRegistry::instance(),
                template_node_
            );

            Response sys_resp = add_message(session, Message::SYSTEM, formatted_system, "", "", 0, 0);
            if (!sys_resp.success) {
                throw BackendError("Failed to initialize system message: " + sys_resp.error);
            }

            session.system_message_tokens = sys_resp.prompt_tokens;
            LOG_INFO("Added system message (tokens=" + std::to_string(sys_resp.prompt_tokens) + ")");
        }

    } catch (const std::exception& e) {
        throw BackendError("Failed to initialize TensorRT: " + std::string(e.what()));
    }
}

int TensorRTBackend::count_message_tokens(Message::Type type,
                                         const std::string& content,
                                         const std::string& tool_name,
                                         const std::string& tool_id) {
    if (!tokenizer_) {
        return static_cast<int>(content.length() / 4.0 + 0.5);
    }

    // Format message through template to get exact count
    if (!template_node_) {
        return tokenizer_->count_tokens(content);
    }

    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);
    auto context = minja::Context::builtins();

    // Build messages array with just this message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();

    // Convert Message::Type to role string
    std::string role;
    switch (type) {
        case Message::SYSTEM: role = "system"; break;
        case Message::USER: role = "user"; break;
        case Message::ASSISTANT: role = "assistant"; break;
        case Message::TOOL: role = "tool"; break;
        default: role = "user"; break;
    }

    msg_obj.set("role", minja::Value(role));
    msg_obj.set("content", minja::Value(content));

    if (!tool_name.empty()) {
        msg_obj.set("name", minja::Value(tool_name));
    }
    if (!tool_id.empty()) {
        msg_obj.set("tool_call_id", minja::Value(tool_id));
    }

    messages.push_back(msg_obj);
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(false));

    // Add date_string and strftime_now
    time_t now = time(nullptr);
    struct tm* tm_info = localtime(&now);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%d %b %Y", tm_info);
    context->set("date_string", minja::Value(std::string(date_buffer)));

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

    try {
        std::string rendered = (*template_ptr)->render(context);
        return tokenizer_->count_tokens(rendered);
    } catch (const std::exception& e) {
        LOG_WARN("Exception rendering message: " + std::string(e.what()));
        return tokenizer_->count_tokens(content);
    }
}

Response TensorRTBackend::add_message(Session& session, Message::Type type,
                                     const std::string& content,
                                     const std::string& tool_name,
                                     const std::string& tool_id,
                                     int prompt_tokens,
                                     int max_tokens) {
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "TensorRT backend not initialized";
        return err_resp;
    }

    // Set current session for eviction callbacks
    current_session = &session;

    LOG_DEBUG("TensorRT add_message called: type=" + std::to_string(type) +
              ", content_len=" + std::to_string(content.length()));

    // Count tokens if needed
    int message_tokens = prompt_tokens;
    if (message_tokens == 0) {
        if (type == Message::SYSTEM && !session.tools.empty()) {
            std::string formatted = Models::format_system_message(
                model_config_,
                content,
                ToolRegistry::instance(),
                template_node_
            );
            message_tokens = count_message_tokens(type, formatted, tool_name, tool_id);
        } else {
            message_tokens = count_message_tokens(type, content, tool_name, tool_id);
        }
    }

    LOG_DEBUG("Message token count: " + std::to_string(message_tokens));

    // Create message (transactional - not in session yet!)
    Message msg(type, content, message_tokens);
    msg.tool_name = tool_name;
    msg.tool_call_id = tool_id;

    // TRY to accumulate tokens FIRST
    // For non-SYSTEM messages, add generation prompt so template adds <|im_start|>assistant\n
    bool will_generate = (type != Message::SYSTEM);
    if (!tokenize_and_accumulate_message(msg, will_generate)) {
        Response err;
        err.success = false;
        err.code = Response::ERROR;
        err.error = "Failed to tokenize message";
        return err;
    }

    // SUCCESS - add to session
    session.messages.push_back(msg);
    session.total_tokens += message_tokens;

    if (type == Message::USER) {
        session.last_user_message_index = session.messages.size() - 1;
        session.last_user_message_tokens = message_tokens;
    }

    Response resp;
    resp.success = true;
    resp.code = Response::SUCCESS;

    if (type != Message::SYSTEM) {
        // Generate response
        std::string response_text = generate(session, max_tokens);

        // Parse tool calls
        std::vector<ToolParser::ToolCall> tool_calls;
        if (!response_text.empty()) {
            // Check for tool call markers
            bool has_marker = false;
            std::vector<std::string> tool_call_markers = {"<tool_call>", "```json", "{\"name\":"};
            for (const auto& marker : tool_call_markers) {
                if (response_text.find(marker) != std::string::npos) {
                    has_marker = true;
                    break;
                }
            }

            // Parse tool calls if markers present
            if (has_marker || (response_text[0] == '{' && response_text.find("\"name\"") != std::string::npos)) {
                auto tool_call = ToolParser::parse_tool_call(response_text, tool_call_markers);
                if (tool_call.has_value()) {
                    tool_calls.push_back(tool_call.value());
                }
            }
        }

        // Add assistant message
        int asst_tokens = tokenizer_->count_tokens(response_text);
        Message asst_msg(Message::ASSISTANT, response_text, asst_tokens);

        // Tokenize and accumulate assistant message
        if (!tokenize_and_accumulate_message(asst_msg)) {
            LOG_WARN("Failed to accumulate assistant message tokens");
        }

        session.messages.push_back(asst_msg);
        session.total_tokens += asst_tokens;

        session.last_assistant_message_index = session.messages.size() - 1;
        session.last_assistant_message_tokens = asst_tokens;

        resp.content = response_text;
        resp.prompt_tokens = message_tokens;
        resp.completion_tokens = asst_tokens;
        // Determine finish reason
        if (!tool_calls.empty()) {
            resp.finish_reason = "tool_calls";
        } else if (last_generation_hit_length_limit) {
            resp.finish_reason = "length";
        } else {
            resp.finish_reason = "stop";
        }
        resp.tool_calls = tool_calls;
    } else {
        resp.finish_reason = "system";
        resp.prompt_tokens = message_tokens;
    }

    LOG_DEBUG("add_message complete: prompt_tokens=" + std::to_string(resp.prompt_tokens) +
              ", completion_tokens=" + std::to_string(resp.completion_tokens) +
              ", finish_reason=" + resp.finish_reason);

    return resp;
}

Response TensorRTBackend::generate_from_session(const Session& session, int max_tokens) {
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "TensorRT backend not initialized";
        return err_resp;
    }

    LOG_DEBUG("TensorRT generate_from_session called with " + std::to_string(session.messages.size()) + " messages");

    // PREFIX CACHING: Compare incoming session with backend_session (what's in KV cache)
    size_t cached_count = backend_session.messages.size();

    LOG_DEBUG("Backend has " + std::to_string(cached_count) + " cached messages, " +
              "incoming session has " + std::to_string(session.messages.size()) + " messages");

    // Account for system message offset
    size_t backend_offset = 0;
    if (!backend_session.messages.empty() && backend_session.messages[0].type == Message::SYSTEM) {
        backend_offset = 1;
        LOG_DEBUG("Backend has system message at index 0, offsetting comparison by 1");
    }

    // Find how many messages match (prefix caching)
    size_t matching_prefix = 0;
    for (size_t i = 0; i < session.messages.size(); i++) {
        size_t backend_idx = i + backend_offset;

        if (backend_idx >= cached_count) {
            break;
        }

        const auto& cached_msg = backend_session.messages[backend_idx];
        const auto& session_msg = session.messages[i];

        if (cached_msg.get_role() == session_msg.get_role() &&
            cached_msg.content == session_msg.content) {
            matching_prefix++;
        } else {
            LOG_WARN("DIVERGENCE at session message " + std::to_string(i));
            break;
        }
    }

    LOG_DEBUG("Prefix match: " + std::to_string(matching_prefix) + " messages");

    // Check if backend has more messages than what matched
    size_t expected_backend_count = backend_offset + matching_prefix;

    if (cached_count > expected_backend_count) {
        LOG_WARN("Conversation diverged - clearing cache and restarting");

        // TensorRT: Clear accumulated tokens and reset (block reuse handles caching)
        accumulated_tokens.clear();
        backend_session.messages.clear();
        matching_prefix = 0;
        backend_offset = 0;
    }

    // Set current_session for eviction callbacks
    current_session = &backend_session;

    // Handle system message if present
    if (!session.system_message.empty() &&
        (backend_session.messages.empty() || backend_session.messages[0].type != Message::SYSTEM)) {

        LOG_DEBUG("Adding system message to backend");

        std::string formatted_system = Models::format_system_message(
            model_config_,
            session.system_message,
            ToolRegistry::instance(),
            template_node_
        );

        Message sys_msg(Message::SYSTEM, formatted_system, 0);
        sys_msg.tokens = count_message_tokens(Message::SYSTEM, formatted_system, "", "");

        if (!tokenize_and_accumulate_message(sys_msg)) {
            Response err_resp;
            err_resp.success = false;
            err_resp.code = Response::ERROR;
            err_resp.error = "Failed to tokenize system message";
            return err_resp;
        }

        backend_session.messages.push_back(sys_msg);
        backend_session.system_message = session.system_message;
        LOG_DEBUG("System message added (" + std::to_string(sys_msg.tokens) + " tokens)");
    }

    // Add NEW messages (from matching_prefix onward)
    size_t new_messages = session.messages.size() - matching_prefix;
    if (new_messages > 0) {
        LOG_DEBUG("Adding " + std::to_string(new_messages) + " new messages to backend");

        for (size_t i = matching_prefix; i < session.messages.size(); i++) {
            const auto& msg = session.messages[i];
            LOG_DEBUG("Processing message " + std::to_string(i) + ": type=" + std::to_string(static_cast<int>(msg.type)) +
                      ", content_len=" + std::to_string(msg.content.length()) +
                      ", content_preview=" + msg.content.substr(0, std::min(size_t(100), msg.content.length())));

            // Skip empty messages (shouldn't happen but server might send them)
            if (msg.content.empty() && msg.type != Message::SYSTEM) {
                LOG_WARN("Skipping empty message at index " + std::to_string(i));
                continue;
            }

            Message msg_copy = msg;
			std::cout << "Sending: " << msg.content << std::endl;

            if (msg_copy.tokens == 0) {
                msg_copy.tokens = count_message_tokens(msg_copy.type, msg_copy.content,
                                                        msg_copy.tool_name, msg_copy.tool_call_id);
            }

            // For the LAST message, add generation prompt if it will trigger generation
            bool is_last = (i == session.messages.size() - 1);
            bool will_generate = is_last && (msg_copy.type != Message::SYSTEM && msg_copy.type != Message::ASSISTANT);

            if (!tokenize_and_accumulate_message(msg_copy, will_generate)) {
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::ERROR;
                err_resp.error = "Failed to tokenize message";
                return err_resp;
            }

            backend_session.messages.push_back(msg_copy);

            if (msg.type == Message::USER) {
                backend_session.last_user_message_index = backend_session.messages.size() - 1;
                backend_session.last_user_message_tokens = msg_copy.tokens;
            } else if (msg.type == Message::ASSISTANT) {
                backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
                backend_session.last_assistant_message_tokens = msg_copy.tokens;
            }
        }

        LOG_DEBUG("Prefix caching: " + std::to_string(matching_prefix) + " cached, " +
                  std::to_string(new_messages) + " new");
    } else {
        LOG_DEBUG("All messages already in cache (100% prefix cache hit)");
    }

    // Copy tools and system_message
    backend_session.tools = session.tools;
    backend_session.system_message = session.system_message;

    // Generate response
    std::string result = generate(backend_session, max_tokens);

    Response success_resp;
    success_resp.success = true;
    success_resp.code = Response::SUCCESS;
    success_resp.content = result;
    success_resp.finish_reason = "stop";
    success_resp.prompt_tokens = 0;  // TODO: Calculate actual prompt tokens
    success_resp.completion_tokens = tokenizer_->count_tokens(result);
    return success_resp;
}

// Helper methods

std::string TensorRTBackend::render_message(const Message& msg, bool add_generation_prompt) {
    LOG_DEBUG("render_message called: template_node_=" + std::string(template_node_ ? "exists" : "null") +
              ", msg.type=" + std::to_string(static_cast<int>(msg.type)) +
              ", content_len=" + std::to_string(msg.content.length()));

    if (!template_node_) {
        throw BackendError("No chat template loaded - cannot render message. Model must provide a chat template in tokenizer_config.json");
    }

    // Use Minja template for proper rendering (including thinking tags)
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node_);
    auto context = minja::Context::builtins();

    // Build messages array with just this message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();

    std::string role = msg.get_role();
    msg_obj.set("role", minja::Value(role));
    msg_obj.set("content", minja::Value(msg.content));

    if (!msg.tool_name.empty()) {
        msg_obj.set("name", minja::Value(msg.tool_name));
    }
    if (!msg.tool_call_id.empty()) {
        msg_obj.set("tool_call_id", minja::Value(msg.tool_call_id));
    }

    messages.push_back(msg_obj);
    context->set("messages", messages);
    context->set("add_generation_prompt", minja::Value(add_generation_prompt));

    // Add date_string and strftime_now for template compatibility
    time_t now = time(nullptr);
    struct tm* tm_info = localtime(&now);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%d %b %Y", tm_info);
    context->set("date_string", minja::Value(std::string(date_buffer)));

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

    std::string result = (*template_ptr)->render(context);

    LOG_DEBUG("Rendered message (type=" + std::to_string(static_cast<int>(msg.type)) +
              ", role=" + role + ", add_generation_prompt=" +
              std::string(add_generation_prompt ? "true" : "false") + ")");
    LOG_DEBUG("Full rendered output (" + std::to_string(result.length()) + " chars): " + result);

    return result;
}

bool TensorRTBackend::tokenize_and_accumulate_message(Message& msg, bool add_generation_prompt) {
    if (!tokenizer_) {
        LOG_ERROR("Tokenizer not loaded");
        return false;
    }

    LOG_DEBUG("tokenize_and_accumulate_message: type=" + std::to_string(static_cast<int>(msg.type)) +
              ", content_len=" + std::to_string(msg.content.length()) +
              ", add_generation_prompt=" + std::string(add_generation_prompt ? "true" : "false"));

    // Render message through template
    std::string formatted_text = render_message(msg, add_generation_prompt);
	std::cout << "Formatted text: " << formatted_text << std::endl;

    // Tokenize with add_special_tokens=false to avoid unwanted BOS/EOS tokens
    std::vector<int> tokens = tokenizer_->encode(formatted_text, false);
    if (tokens.empty()) {
        LOG_ERROR("Failed to tokenize message - formatted_text length: " + std::to_string(formatted_text.length()));
        LOG_ERROR("Formatted text: " + formatted_text.substr(0, 500));
        return false;
    }

    // Update message token count to actual formatted count
    msg.tokens = tokens.size();

    // Append tokens to accumulated_tokens vector
    accumulated_tokens.insert(accumulated_tokens.end(), tokens.begin(), tokens.end());

    LOG_DEBUG("Accumulated " + std::to_string(tokens.size()) + " tokens, total accumulated: " +
              std::to_string(accumulated_tokens.size()));

    return true;
}

std::string TensorRTBackend::generate(const Session& session, int max_tokens) {
    namespace tle = tensorrt_llm::executor;
    auto* executor = static_cast<tle::Executor*>(executor_);

    // Reset ModelOutput state for new generation
    tio.reset();

    try {
        if (accumulated_tokens.empty()) {
            throw BackendError("No tokens accumulated for generation");
        }

        // Convert to int32_t
        std::vector<int32_t> input_tokens(accumulated_tokens.begin(), accumulated_tokens.end());
        LOG_DEBUG("Generating with " + std::to_string(input_tokens.size()) + " accumulated tokens");

        // Decode and log the prompt for debugging
        if (tokenizer_) {
            std::vector<int> debug_tokens(accumulated_tokens.begin(), accumulated_tokens.end());

            // Log first 10 token IDs
            std::string token_ids_str = "First tokens: ";
            for (size_t i = 0; i < std::min(size_t(10), debug_tokens.size()); i++) {
                token_ids_str += std::to_string(debug_tokens[i]) + " ";
            }
            LOG_DEBUG(token_ids_str);

            // Skip decoding for now - it seems to crash
            // std::string decoded_prompt = tokenizer_->decode(debug_tokens);
            // LOG_DEBUG("Decoded prompt (" + std::to_string(decoded_prompt.length()) + " chars): " +
            //           decoded_prompt.substr(0, std::min(size_t(500), decoded_prompt.length())));
        }

        // Track timing for prefill and decode
        auto start_time = std::chrono::high_resolution_clock::now();
        auto prefill_end_time = start_time;
        bool prefill_complete = false;

        // Track finish reason for detecting length limit
        tle::FinishReason finish_reason = tle::FinishReason::kNOT_FINISHED;

        // Create sampling config
        tle::SamplingConfig samplingConfig;
        samplingConfig.setBeamWidth(1);
        samplingConfig.setTemperature(temperature);

        // Create output config
        tle::OutputConfig outputConfig;
        outputConfig.returnLogProbs = false;
        outputConfig.returnContextLogits = false;
        outputConfig.returnGenerationLogits = false;

        // Create KV cache retention config
        size_t system_msg_tokens = 0;
        if (!backend_session.messages.empty() && backend_session.messages[0].type == Message::SYSTEM) {
            system_msg_tokens = backend_session.messages[0].tokens;
        }

        std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig> token_ranges;
        if (system_msg_tokens > 0) {
            token_ranges.push_back(
                tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, system_msg_tokens, 100)
            );
        }

        tle::KvCacheRetentionConfig retention(token_ranges, 35);

        LOG_DEBUG("Creating TensorRT request with " + std::to_string(input_tokens.size()) +
                  " tokens, max_tokens=" + std::to_string(max_tokens));

        // Create request
        int actual_max_tokens = (max_tokens > 0) ? max_tokens : 512;

        LOG_DEBUG("About to construct Request object");
        tle::Request request(
            input_tokens,
            actual_max_tokens,
            true,  // streaming
            samplingConfig,
            outputConfig,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            retention
        );
        LOG_DEBUG("Request object constructed successfully");

        // Enqueue request
        uint64_t request_id;
        try {
            request_id = executor->enqueueRequest(request);
            current_request_id_ = request_id;
            request_active_ = true;
            LOG_DEBUG("Request enqueued with ID: " + std::to_string(request_id));
        } catch (const std::exception& e) {
            LOG_ERROR("Exception enqueueing request: " + std::string(e.what()));
            throw BackendError("Failed to enqueue request: " + std::string(e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception enqueueing request");
            throw BackendError("Failed to enqueue request: unknown exception");
        }

        // Collect responses
        std::string response_text;
        std::vector<int32_t> output_tokens;

        LOG_DEBUG("Starting response collection loop");
        while (request_active_) {
            // No timeout - block until responses are ready (not polling!)
            auto responses = executor->awaitResponses(request_id);

            for (const auto& response : responses) {
                if (response.hasError()) {
                    std::string error_msg = response.getErrorMsg();
                    LOG_ERROR("Response error: " + error_msg);
                    request_active_ = false;

                    // Check if error is due to context/KV cache being full
                    std::string error_lower = error_msg;
                    std::transform(error_lower.begin(), error_lower.end(), error_lower.begin(), ::tolower);

                    if (error_lower.find("kv cache") != std::string::npos ||
                        error_lower.find("capacity") != std::string::npos ||
                        error_lower.find("context") != std::string::npos ||
                        error_lower.find("no space") != std::string::npos) {

                        // Context is full
                        if (g_server_mode) {
                            // Server mode: throw exception, client must handle context
                            throw ContextFullException("This model's maximum context length is " +
                                std::to_string(context_size) + " tokens. However, your messages resulted in too many tokens.");
                        } else {
                            // CLI mode: throw special marker for eviction-retry
                            LOG_WARN("Context full in CLI mode - eviction needed");
                            throw ContextFullException("EVICT_AND_RETRY");
                        }
                    }

                    // Other errors
                    throw BackendError("TensorRT generation failed: " + error_msg);
                }

                const auto& result = response.getResult();

                if (!result.outputTokenIds.empty() && !result.outputTokenIds[0].empty()) {
                    const auto& beam_tokens = result.outputTokenIds[0];

                    // Mark prefill completion on first token
                    if (!prefill_complete && !beam_tokens.empty()) {
                        prefill_end_time = std::chrono::high_resolution_clock::now();
                        prefill_complete = true;
                    }

                    output_tokens.insert(output_tokens.end(), beam_tokens.begin(), beam_tokens.end());

                    // Decode immediately - it's essentially free (0.2 microseconds per token)
                    std::vector<int> new_tokens(beam_tokens.begin(), beam_tokens.end());
                    std::string new_text = tokenizer_->decode(new_tokens);

                    // Debug: Log token IDs and decoded text to track think tag issues
                    if (g_debug_level > 0) {
                        std::string token_ids_str;
                        for (auto tid : new_tokens) {
                            if (!token_ids_str.empty()) token_ids_str += ",";
                            token_ids_str += std::to_string(tid);
                        }
                        LOG_DEBUG("Tokens [" + token_ids_str + "] -> \"" + new_text + "\"");
                    }

                    response_text += new_text;
                    // Only write to terminal in interactive mode, not in server mode
                    if (!g_server_mode) {
                        tio.write(new_text.c_str(), new_text.length());
                    }
                }

                if (result.isFinal) {
                    // Capture finish reason to detect if we hit length limit
                    if (!result.finishReasons.empty()) {
                        finish_reason = result.finishReasons[0];
                    }
                    LOG_DEBUG("Generation complete, finish_reason=" + std::to_string(static_cast<int>(finish_reason)));
                    request_active_ = false;
                    break;
                }
            }
        }
		std::cout << "Response: " << response_text << std::endl;

        LOG_DEBUG("Generated " + std::to_string(output_tokens.size()) + " tokens");
        LOG_DEBUG("Decoded " + std::to_string(response_text.length()) + " characters");

        // Store whether we hit length limit (for finish_reason and auto-continuation)
        last_generation_hit_length_limit = (finish_reason == tle::FinishReason::kLENGTH);
        if (last_generation_hit_length_limit) {
            LOG_DEBUG("Generation stopped due to max_tokens limit");
        }

        // Calculate and display performance metrics (like llamacpp)
        auto end_time = std::chrono::high_resolution_clock::now();

        // Prefill metrics
        if (prefill_complete) {
            auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time - start_time);
            double prefill_seconds = prefill_duration.count() / 1000.0;
            double prefill_tok_per_sec = (prefill_seconds > 0) ? (input_tokens.size() / prefill_seconds) : 0;

            int context_used = accumulated_tokens.size();  // Will be cleared soon, so use size before clear
            int context_max = context_size;

            std::cerr << "\033[90m[Prefill: " << input_tokens.size() << " tokens, "
                      << std::fixed << std::setprecision(1) << prefill_tok_per_sec << " t/s, "
                      << "context: " << context_used << "/" << context_max << "]\033[0m" << std::endl;
        }

        // Decode metrics
        if (output_tokens.size() > 0 && prefill_complete) {
            auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - prefill_end_time);
            double decode_seconds = decode_duration.count() / 1000.0;
            double decode_tok_per_sec = (decode_seconds > 0) ? (output_tokens.size() / decode_seconds) : 0;

            int context_used = accumulated_tokens.size() + output_tokens.size();
            int context_max = context_size;

            std::cerr << "\033[90m[Decode: " << output_tokens.size() << " tokens, "
                      << std::fixed << std::setprecision(1) << decode_tok_per_sec << " t/s, "
                      << "context: " << context_used << "/" << context_max << "]\033[0m" << std::endl;
        }

        // Handle <|eom_id|> token for continued tool execution
        if (model_config_.uses_eom_token) {
            size_t eom_pos = response_text.find("<|eom_id|>");
            if (eom_pos != std::string::npos) {
                response_text = response_text.substr(0, eom_pos);
                LOG_DEBUG("Found <|eom_id|> token - model expects tool execution continuation");
            }
        }

        // Strip stop tokens from final response
        for (const auto& stop_token : stop_tokens_) {
            size_t pos = response_text.find(stop_token);
            if (pos != std::string::npos) {
                response_text = response_text.substr(0, pos);
                LOG_DEBUG("Stripped stop token: " + stop_token);
                break;  // Stop at first match
            }
        }

        // Strip thinking blocks if model supports thinking mode
        if (model_config_.supports_thinking_mode && !model_config_.thinking_end_markers.empty()) {
            // Find and strip everything before the thinking end marker
            for (const auto& end_marker : model_config_.thinking_end_markers) {
                size_t end_pos = response_text.find(end_marker);
                if (end_pos != std::string::npos) {
                    // Strip thinking block and the end marker itself
                    response_text = response_text.substr(end_pos + end_marker.length());
                    LOG_DEBUG("Stripped thinking block ending with: " + end_marker);
                    break;
                }
            }
        }

        // CLEAR accumulated_tokens after generation completes
        accumulated_tokens.clear();
        LOG_DEBUG("Cleared accumulated tokens");

        return response_text;

    } catch (const std::exception& e) {
        request_active_ = false;
        accumulated_tokens.clear();
        LOG_ERROR("TensorRT generation error: " + std::string(e.what()));
        throw BackendError("TensorRT generation failed: " + std::string(e.what()));
    }
}

bool TensorRTBackend::is_ready() const {
    return initialized;
}

void TensorRTBackend::shutdown() {
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

    // Clear session data
    backend_session.messages.clear();
    accumulated_tokens.clear();

    initialized = false;
    LOG_DEBUG("TensorRTBackend shutdown");
}

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
            auto events = (*event_mgr)->getLatestEvents(std::chrono::milliseconds(100));

            for (const auto& event : events) {
                if (std::holds_alternative<tle::KVCacheRemovedData>(event.data)) {
                    const auto& removed = std::get<tle::KVCacheRemovedData>(event.data);

                    LOG_INFO("KV cache eviction detected: " +
                             std::to_string(removed.blockHashes.size()) + " blocks removed");

                    std::vector<uint64_t> block_hashes;
                    for (const auto& hash : removed.blockHashes) {
                        block_hashes.push_back(static_cast<uint64_t>(hash));
                    }

                    handle_kv_cache_removed(block_hashes);
                }
                else if (std::holds_alternative<tle::KVCacheStoredData>(event.data)) {
                    const auto& stored = std::get<tle::KVCacheStoredData>(event.data);

                    std::lock_guard<std::mutex> lock(block_map_mutex_);
                    for (const auto& block : stored.blocks) {
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
    std::lock_guard<std::mutex> lock(block_map_mutex_);

    // Calculate tokens removed (TensorRT typically uses 64 tokens per block)
    const size_t TOKENS_PER_BLOCK = 64;
    size_t tokens_removed = block_hashes.size() * TOKENS_PER_BLOCK;

    LOG_INFO("KV cache eviction: " + std::to_string(block_hashes.size()) +
             " blocks removed (~" + std::to_string(tokens_removed) + " tokens)");

    auto& messages = backend_session.messages;
    if (messages.empty()) {
        LOG_WARN("No messages in backend session");
        return;
    }

    // Calculate which messages were evicted
    // Start at index 1 (skip system message at index 0, it's priority 100)
    size_t token_sum = 0;
    int last_evicted_msg = 0;

    for (size_t i = 1; i < messages.size(); ++i) {
        token_sum += messages[i].tokens;
        if (token_sum >= tokens_removed) {
            last_evicted_msg = static_cast<int>(i);
            break;
        }
    }

    if (last_evicted_msg == 0) {
        LOG_WARN("Could not determine evicted messages");
        return;
    }

    LOG_INFO("Identified messages 1-" + std::to_string(last_evicted_msg) + " as evicted");

    // Handle open_user_question_ if it exists
    if (open_user_question_.has_value()) {
        for (int i = 1; i <= last_evicted_msg; ++i) {
            if (messages[i].type == Message::ASSISTANT) {
                LOG_INFO("Found assistant response for orphaned user question, archiving to RAG");
                ConversationTurn turn(open_user_question_->content, messages[i].content);
                RAGManager::archive_turn(turn);
                open_user_question_.reset();
                break;
            }
        }
    }

    // Scan evicted messages for user→assistant pairs
    for (int i = 1; i <= last_evicted_msg; ++i) {
        if (messages[i].type != Message::USER) {
            continue;
        }

        bool found_assistant = false;

        // Check in evicted range
        for (int j = i + 1; j <= last_evicted_msg; ++j) {
            if (messages[j].type == Message::ASSISTANT) {
                LOG_INFO("Found complete turn in evicted range, archiving to RAG");
                ConversationTurn turn(messages[i].content, messages[j].content);
                RAGManager::archive_turn(turn);
                found_assistant = true;
                break;
            }
        }

        if (found_assistant) {
            continue;
        }

        // Check in remaining messages
        for (size_t j = last_evicted_msg + 1; j < messages.size(); ++j) {
            if (messages[j].type == Message::ASSISTANT) {
                LOG_INFO("Found complete turn spanning eviction boundary, archiving to RAG");
                ConversationTurn turn(messages[i].content, messages[j].content);
                RAGManager::archive_turn(turn);
                found_assistant = true;
                break;
            }
        }

        if (!found_assistant) {
            LOG_INFO("User message has no assistant response yet, storing as orphaned");
            open_user_question_ = messages[i];
        }
    }

    // Remove evicted messages from backend_session
    for (int i = 0; i < last_evicted_msg; ++i) {
        messages.erase(messages.begin() + 1);  // Always remove index 1
    }

    // Update backend_session total_tokens
    backend_session.total_tokens = 0;
    for (const auto& msg : messages) {
        backend_session.total_tokens += msg.tokens;
    }

    LOG_INFO("Removed " + std::to_string(last_evicted_msg) +
             " messages from backend session, " + std::to_string(messages.size()) + " remaining");

    // Update current_session if set
    if (current_session) {
        // TODO: Sync eviction to current_session if needed
    }

    // Clear block mappings
    for (uint64_t hash : block_hashes) {
        block_to_tokens_.erase(hash);
    }
}
#else
// Stub implementations when TensorRT is not enabled
void TensorRTBackend::initialize(Session&) {
    throw BackendError("TensorRT backend not compiled in");
}

Response TensorRTBackend::add_message(Session&, Message::Type, const std::string&,
                                     const std::string&, const std::string&, int, int) {
    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.error = "TensorRT backend not compiled in";
    return err_resp;
}

Response TensorRTBackend::generate_from_session(const Session&, int) {
    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.error = "TensorRT backend not compiled in";
    return err_resp;
}

int TensorRTBackend::count_message_tokens(Message::Type, const std::string& content,
                                         const std::string&, const std::string&) {
    return static_cast<int>(content.length() / 4.0 + 0.5);
}
#endif
