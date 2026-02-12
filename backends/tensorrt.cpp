#include "shepherd.h"
#include "tensorrt.h"
#include "session.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "models.h"
#include "minja.hpp"
#include "nlohmann/json.hpp"
#include "config.h"
#include "rag.h"

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
            dout(1) << "PluginForcer: Loading TensorRT base plugin library..." << std::endl;
            // Load base TensorRT plugin library first (contains plugin registry)
            void* base_handle = dlopen("libnvinfer_plugin.so.10", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
            if (!base_handle) {
                dout(1) << std::string("WARNING: ") +"Could not load base TensorRT plugin library: " + std::string(dlerror()) << std::endl;
            } else {
                dout(1) << "Base TensorRT plugin library loaded" << std::endl;
            }

            dout(1) << "PluginForcer: Loading TensorRT-LLM plugin library..." << std::endl;
            // Now load TensorRT-LLM plugins
            void* handle = dlopen("libnvinfer_plugin_tensorrt_llm.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
            if (!handle) {
                const char* err = dlerror();
                if (err) {
                    std::cerr << "Could not preload TensorRT-LLM plugin library: " + std::string(err) << std::endl;
                }
            } else {
                dout(1) << "TensorRT-LLM plugin library loaded successfully!" << std::endl;

                // THIS IS THE CRITICAL STEP: Call initTrtLlmPlugins just like Python does
                // From Python: handle.initTrtLlmPlugins(None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))
                typedef bool (*InitPluginsFunc)(void*, const char*);
                InitPluginsFunc initTrtLlmPlugins = (InitPluginsFunc)dlsym(handle, "initTrtLlmPlugins");
                if (initTrtLlmPlugins) {
                    dout(1) << "Found initTrtLlmPlugins, calling it with namespace 'tensorrt_llm'..." << std::endl;
                    bool success = initTrtLlmPlugins(nullptr, "tensorrt_llm");
                    if (success) {
                        dout(1) << "initTrtLlmPlugins succeeded! Plugins should now be registered." << std::endl;
                    } else {
                        std::cerr << "initTrtLlmPlugins returned false!" << std::endl;
                    }
                } else {
                    std::cerr << "initTrtLlmPlugins not found: " + std::string(dlerror()) << std::endl;
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
    dout(1) << "TensorRT tokenizer initialized" << std::endl;
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
        std::cerr << "Failed to open tokenizer file: " + json_file << std::endl;
        return false;
    }

    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    dout(1) << "Creating tokenizer from JSON (" + std::to_string(json_blob.length()) + " bytes)..." << std::endl;

    // Use C API (which works) instead of broken C++ wrapper
    TokenizerHandle handle = tokenizers_new_from_str(json_blob.data(), json_blob.length());
    if (!handle) {
        std::cerr << "Failed to create tokenizer from JSON" << std::endl;
        return false;
    }

    dout(1) << "Tokenizer handle created successfully" << std::endl;
    tokenizer_ = handle;
    dout(1) << "Loaded tokenizer from: " + json_file << std::endl;
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
        std::cerr << "Tokenizer not loaded" << std::endl;
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
        std::cerr << "Tokenizer not loaded" << std::endl;
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
TensorRTBackend::TensorRTBackend(size_t context_size, Session& session, EventCallback callback)
    : GpuBackend(context_size, session, callback)
#ifdef ENABLE_TENSORRT
      ,backend_session(),
      current_session(nullptr),
      model_config_(ModelConfig::create_generic())
#endif
{
    // Set public Backend variables
    backend_name = "tensorrt";
    this->context_size = context_size;
    // is_gpu = true set by GpuBackend constructor

#ifdef ENABLE_TENSORRT
    tokenizer_ = std::make_unique<TensorRTTokenizer>("");
#endif
    dout(1) << "TensorRTBackend created with context_size: " + std::to_string(context_size) << std::endl;

    // Parse config
    parse_backend_config();

    // --- Full initialization (formerly in initialize()) ---
    // The initialize() code will be called here - for now just call it
    // This will be refactored to inline the code
}

TensorRTBackend::~TensorRTBackend() {
#ifdef ENABLE_TENSORRT
    shutdown();
#endif
}

void TensorRTBackend::parse_backend_config() {
    if (config->json.is_null() || config->json.empty()) {
        return;  // No config, use defaults
    }

    try {
        // Sampling parameters
        if (config->json.contains("temperature")) temperature = config->json["temperature"].get<float>();
        if (config->json.contains("top_p")) top_p = config->json["top_p"].get<float>();
        if (config->json.contains("top_k")) top_k = config->json["top_k"].get<int>();
        if (config->json.contains("min_p")) min_p = config->json["min_p"].get<float>();

        // Penalty parameters (TensorRT-LLM naming)
        if (config->json.contains("repetition_penalty")) repetition_penalty = config->json["repetition_penalty"].get<float>();
        if (config->json.contains("presence_penalty")) presence_penalty = config->json["presence_penalty"].get<float>();
        if (config->json.contains("frequency_penalty")) frequency_penalty = config->json["frequency_penalty"].get<float>();
        if (config->json.contains("length_penalty")) length_penalty = config->json["length_penalty"].get<float>();
        if (config->json.contains("no_repeat_ngram_size")) no_repeat_ngram_size = config->json["no_repeat_ngram_size"].get<int>();

        dout(1) << "Loaded TensorRT backend config: temperature=" + std::to_string(temperature) +
                  ", top_p=" + std::to_string(top_p) +
                  ", top_k=" + std::to_string(top_k) +
                  ", repetition_penalty=" + std::to_string(repetition_penalty) +
                  ", frequency_penalty=" + std::to_string(frequency_penalty) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to parse TensorRT backend config: " + std::string(e.what()) << std::endl;
    }
}

#ifdef ENABLE_TENSORRT
ModelConfig TensorRTBackend::detect_model_family() {
    // Detection priority: chat template -> config.json -> path
    // All detection logic is centralized in the Models class

    // 1. Try chat template detection (most reliable when template exists)
    ModelConfig config = Models::detect_from_chat_template(chat_template_text_, model_path_);
    if (config.family != ModelFamily::GENERIC) {
        return config;
    }

    // 2. Try config.json detection (architecture/model_type fields)
    config = Models::detect_from_config_file(model_path_);
    if (config.family != ModelFamily::GENERIC) {
        return config;
    }

    // 3. Last resort: path-based detection
    config = Models::detect_from_model_path(model_path_);
    if (config.family != ModelFamily::GENERIC) {
        return config;
    }

    dout(1) << std::string("WARNING: ") +"Could not detect model family, using generic configuration" << std::endl;
    return ModelConfig::create_generic();
}

void TensorRTBackend::initialize(Session& session) {
    if (initialized) {
        dout(1) << std::string("WARNING: ") +"TensorRTBackend already initialized" << std::endl;
        return;
    }

    // Suppress TensorRT-LLM INFO logs (they go to stdout and interfere with output)
    setenv("TLLM_LOG_LEVEL", "error", 0);  // 0 = don't overwrite if already set

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
        dout(1) << "Initializing TensorRT-LLM Executor with model: " + model_path_ << std::endl;

        // Parse config.json to extract engine build parameters
        std::string config_file = model_path_ + "/config.json";
        dout(1) << "About to open config file: " + config_file << std::endl;
        std::ifstream file(config_file);
        if (!file.is_open()) {
            throw BackendError("Failed to open config file: " + config_file);
        }

        nlohmann::json engine_config;
        try {
            file >> engine_config;
            file.close();
        } catch (const std::exception& e) {
            throw BackendError("Failed to parse config.json: " + std::string(e.what()));
        }

        // Extract world_size from mapping section
        int required_world_size = 1;
        if (engine_config.contains("pretrained_config") &&
            engine_config["pretrained_config"].contains("mapping") &&
            engine_config["pretrained_config"]["mapping"].contains("world_size")) {
            required_world_size = engine_config["pretrained_config"]["mapping"]["world_size"];
        }

        // Extract build_config parameters
        int max_seq_len = 2048;           // Default fallback
        int max_batch_size = 1;
        int max_beam_width = 1;
        int max_num_tokens = 8192;
        int max_input_len = 1024;

        if (engine_config.contains("build_config")) {
            auto& build_config = engine_config["build_config"];

            if (build_config.contains("max_seq_len")) {
                max_seq_len = build_config["max_seq_len"];
            }
            if (build_config.contains("max_batch_size")) {
                max_batch_size = build_config["max_batch_size"];
            }
            if (build_config.contains("max_beam_width")) {
                max_beam_width = build_config["max_beam_width"];
            }
            if (build_config.contains("max_num_tokens")) {
                max_num_tokens = build_config["max_num_tokens"];
            }
            if (build_config.contains("max_input_len")) {
                max_input_len = build_config["max_input_len"];
            }

            dout(1) << "Engine build config:" << std::endl;
            dout(1) << "  max_seq_len: " + std::to_string(max_seq_len) << std::endl;
            dout(1) << "  max_batch_size: " + std::to_string(max_batch_size) << std::endl;
            dout(1) << "  max_beam_width: " + std::to_string(max_beam_width) << std::endl;
            dout(1) << "  max_num_tokens: " + std::to_string(max_num_tokens) << std::endl;
            dout(1) << "  max_input_len: " + std::to_string(max_input_len) << std::endl;
        }

        // Store build config for later use in ExecutorConfig
        this->max_seq_len = max_seq_len;
        this->max_batch_size = max_batch_size;
        this->max_beam_width = max_beam_width;

        // Update context_size if auto-detect
        if (context_size == 0) {
            context_size = max_seq_len;
            dout(1) << "Model max sequence length: " + std::to_string(max_seq_len) << std::endl;
        } else {
            dout(1) << "Using explicitly configured context size: " + std::to_string(context_size) << std::endl;
        }

        dout(1) << "Model requires world_size: " + std::to_string(required_world_size) << std::endl;

        // Check if we're already running under MPI
        int current_world_size = 1;
        const char* mpi_world_env = getenv("OMPI_COMM_WORLD_SIZE");
        if (mpi_world_env) {
            current_world_size = std::stoi(mpi_world_env);
        }

        // If model requires multi-GPU but we're not under MPI, re-exec with mpirun
        if (required_world_size > 1 && current_world_size == 1) {
            dout(1) << "Model requires " + std::to_string(required_world_size) + " GPUs, re-launching with mpirun..." << std::endl;

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

            dout(1) << "Executing: " + [&]() {
                std::string cmd;
                for (const auto& arg : args) cmd += arg + " ";
                return cmd;
            }() << std::endl;

            execvp("mpirun", c_args.data());

            throw BackendError("Failed to exec mpirun: " + std::string(strerror(errno)));
        }

        // Verify we have the correct world size if running under MPI
        if (required_world_size > 1 && current_world_size != required_world_size) {
            throw BackendError("MPI world size (" + std::to_string(current_world_size) +
                             ") doesn't match model requirement (" + std::to_string(required_world_size) + ")");

        }

        dout(1) << "World size verified, creating ExecutorConfig..." << std::endl;

        // Create ExecutorConfig using parameters from engine build config
        namespace tle = tensorrt_llm::executor;

        tle::ExecutorConfig config;
        config.setMaxBeamWidth(max_beam_width);
        dout(1) << "ExecutorConfig max_beam_width set to: " + std::to_string(max_beam_width) << std::endl;

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
        dout(1) << "TensorRT scheduler configured with GUARANTEED_NO_EVICT policy" << std::endl;

        // Create Executor (loads the TensorRT engine)
        dout(1) << "About to create TensorRT-LLM Executor..." << std::endl;
        dout(1) << "Model path: " + model_path_ << std::endl;
        dout(1) << "Model path length: " + std::to_string(model_path_.length()) << std::endl;
        auto* executor = new tle::Executor(
            model_path_,
            tle::ModelType::kDECODER_ONLY,
            config
        );

        executor_ = static_cast<void*>(executor);
        dout(1) << "Executor created successfully" << std::endl;

        // Get KV cache event manager
        auto event_mgr_opt = executor->getKVCacheEventManager();
        if (event_mgr_opt.has_value()) {
            event_manager_ = new std::shared_ptr<tle::KVCacheEventManager>(*event_mgr_opt);
            dout(1) << "KV cache event monitoring enabled" << std::endl;

            // Start event monitoring thread
            monitoring_events_ = true;
            kv_event_monitor_thread_ = std::thread(&TensorRTBackend::monitor_kv_events, this);
        } else {
            dout(1) << std::string("WARNING: ") +"KV cache event manager not available" << std::endl;
        }

        // Load tokenizer
        dout(1) << "Loading tokenizer from: " + model_path_ << std::endl;
        if (!tokenizer_->load_tokenizer(model_path_)) {
            throw BackendError("Failed to load tokenizer from: " + model_path_);
        }
        dout(1) << "Tokenizer loaded" << std::endl;

        // Load chat template from tokenizer_config.json
        dout(1) << "Loading chat template..." << std::endl;
        std::string tokenizer_config_file = model_path_ + "/tokenizer_config.json";
        std::ifstream config_stream(tokenizer_config_file);
        if (config_stream.is_open()) {
            try {
                nlohmann::json tokenizer_config;
                config_stream >> tokenizer_config;
                config_stream.close();

                if (tokenizer_config.contains("chat_template")) {
                    chat_template_text_ = tokenizer_config["chat_template"].get<std::string>();
                    dout(1) << "Loaded chat template (" + std::to_string(chat_template_text_.length()) + " chars)" << std::endl;

                    // Parse with minja
                    try {
                        minja::Options options{};
                        auto parsed_template = minja::Parser::parse(chat_template_text_, options);
                        template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
                        dout(1) << "Chat template parsed successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to parse chat template: " + std::string(e.what()) << std::endl;
                        dout(1) << std::string("WARNING: ") +"Will fall back to simple format" << std::endl;
                    }
                } else {
                    // No chat_template in tokenizer_config.json - try loading from chat_template.jinja file
                    std::string jinja_file = model_path_ + "/chat_template.jinja";
                    std::ifstream jinja_stream(jinja_file);
                    bool loaded_from_jinja = false;

                    if (jinja_stream.is_open()) {
                        dout(1) << "Found chat_template.jinja, loading..." << std::endl;
                        std::stringstream buffer;
                        buffer << jinja_stream.rdbuf();
                        chat_template_text_ = buffer.str();
                        jinja_stream.close();

                        if (!chat_template_text_.empty()) {
                            dout(1) << "Loaded chat template from .jinja file (" + std::to_string(chat_template_text_.length()) + " chars)" << std::endl;

                            // Parse the jinja template
                            try {
                                minja::Options options{};
                                auto parsed_template = minja::Parser::parse(chat_template_text_, options);
                                template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
                                dout(1) << "Jinja chat template parsed successfully" << std::endl;
                                loaded_from_jinja = true;
                            } catch (const std::exception& e) {
                                std::cerr << "Failed to parse jinja chat template: " + std::string(e.what()) << std::endl;
                                dout(1) << std::string("WARNING: ") +"Will try default template fallback" << std::endl;
                            }
                        }
                    }

                    // If jinja file didn't work, fall back to default based on tokenizer_class
                    if (!loaded_from_jinja) {
                        if (tokenizer_config.contains("tokenizer_class")) {
                            std::string tokenizer_class = tokenizer_config["tokenizer_class"].get<std::string>();
                            dout(1) << "Detecting model family from tokenizer_class: " + tokenizer_class << std::endl;

                            ModelFamily family = Models::detect_from_tokenizer_class(tokenizer_class);
                            chat_template_text_ = Models::get_default_chat_template(family);

                            if (!chat_template_text_.empty()) {
                                dout(1) << "Using default chat template for detected family" << std::endl;

                                // Parse the default template
                                try {
                                    minja::Options options{};
                                    auto parsed_template = minja::Parser::parse(chat_template_text_, options);
                                    template_node_ = new std::shared_ptr<minja::TemplateNode>(parsed_template);
                                    dout(1) << "Default chat template parsed successfully" << std::endl;
                                } catch (const std::exception& e) {
                                    std::cerr << "Failed to parse default chat template: " + std::string(e.what()) << std::endl;
                                }
                            }
                        } else {
                            dout(1) << std::string("WARNING: ") +"No chat_template, jinja file, or tokenizer_class found - chat template unavailable" << std::endl;
                        }
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
                        stop_tokens.push_back(eos_token);
                        dout(1) << "Loaded stop token from tokenizer config: " + eos_token << std::endl;

                        // Encode the EOS token to get its ID
                        std::vector<int> encoded = tokenizer_->encode(eos_token, false);
                        if (!encoded.empty()) {
                            eos_token_id = encoded[0];
                            dout(1) << "EOS token ID: " + std::to_string(*eos_token_id) << std::endl;
                        }
                    }
                }

                // Load BOS token configuration for filtering
                add_bos_token = tokenizer_config.value("add_bos_token", false);
                if (tokenizer_config.contains("bos_token")) {
                    if (tokenizer_config["bos_token"].is_string()) {
                        bos_token = tokenizer_config["bos_token"].get<std::string>();
                        if (!bos_token.empty()) {
                            std::vector<int> bos_encoded = tokenizer_->encode(bos_token);
                            if (!bos_encoded.empty()) {
                                bos_token_id = bos_encoded[0];
                                dout(1) << "Loaded BOS token: " + bos_token + " (ID: " + std::to_string(bos_token_id) + ")" << std::endl;
                            }
                        }
                    } else if (tokenizer_config["bos_token"].is_object() &&
                               tokenizer_config["bos_token"].contains("content")) {
                        bos_token = tokenizer_config["bos_token"]["content"].get<std::string>();
                        if (!bos_token.empty()) {
                            std::vector<int> bos_encoded = tokenizer_->encode(bos_token);
                            if (!bos_encoded.empty()) {
                                bos_token_id = bos_encoded[0];
                                dout(1) << "Loaded BOS token: " + bos_token + " (ID: " + std::to_string(bos_token_id) + ")" << std::endl;
                            }
                        }
                    }
                }
                dout(1) << "BOS token config: add_bos_token=" + std::string(add_bos_token ? "true" : "false") +
                         ", bos_token_id=" + (bos_token_id >= 0 ? std::to_string(bos_token_id) : "none"));
            } catch (const std::exception& e) {
                dout(1) << std::string("WARNING: ") +"Error reading tokenizer_config.json: " + std::string(e.what()) << std::endl;
            }
        }

        // Detect model family
        model_config_ = detect_model_family();
        dout(1) << "Model configuration: family=" + std::to_string(static_cast<int>(model_config_.family)) +
                 ", version=" + model_config_.version +
                 ", tool_result_role=" + model_config_.tool_result_role +
                 ", supports_thinking=" + std::to_string(model_config_.supports_thinking_mode));

        // Create ChatTemplate instance based on model family
        // Pass eos_token and bos_token for models that need them in their jinja templates
        std::string eos_tok = stop_tokens.empty() ? "" : stop_tokens[0];
        chat_template_ = ChatTemplates::ChatTemplateFactory::create(
            chat_template_text_, model_config_, template_node_, eos_tok, bos_token);
        dout(1) << "Created ChatTemplate for family: " + std::to_string(static_cast<int>(model_config_.family)) << std::endl;

        // Probe template capabilities (discovers what features the template supports)
        if (chat_template_) {
            chat_template_->probe_capabilities();
            const auto& caps = chat_template_->get_capabilities();
            dout(1) << "Template capabilities: system=" + std::string(caps.supports_system_role ? "yes" : "no") +
                       ", tools=" + std::string(caps.supports_tools ? "yes" : "no") +
                       ", tool_calls=" + std::string(caps.supports_tool_calls ? "yes" : "no") +
                       ", tool_responses=" + std::string(caps.supports_tool_responses ? "yes" : "no") +
                       ", channels=" + std::string(caps.has_channels ? "yes" : "no") << std::endl;
        }

        // Build stop token sequences for generation
        // Some models use multi-token role tags (e.g., <|user|>) that need sequence-based stopping
        std::vector<std::string> stop_sequences;

        // Add role tags that indicate start of non-assistant content
        // These prevent the model from generating fake user/system turns
        if (chat_template_text_.find("<|user|>") != std::string::npos) {
            stop_sequences.push_back("<|user|>");
        }
        if (chat_template_text_.find("<|system|>") != std::string::npos) {
            stop_sequences.push_back("<|system|>");
        }
        // Llama 3.x style
        if (chat_template_text_.find("<|start_header_id|>") != std::string::npos) {
            stop_sequences.push_back("<|start_header_id|>user");
            stop_sequences.push_back("<|start_header_id|>system");
        }
        // ChatML style (Qwen, etc.)
        if (chat_template_text_.find("<|im_start|>") != std::string::npos) {
            stop_sequences.push_back("<|im_start|>user");
            stop_sequences.push_back("<|im_start|>system");
        }

        // Encode stop sequences
        for (const auto& seq : stop_sequences) {
            std::vector<int> encoded = tokenizer_->encode(seq, false);
            if (!encoded.empty()) {
                stop_token_ids.push_back(encoded);
                std::string token_str;
                for (size_t i = 0; i < encoded.size(); i++) {
                    if (i > 0) token_str += ", ";
                    token_str += std::to_string(encoded[i]);
                }
                dout(1) << "Added stop sequence: " + seq + " -> [" + token_str + "]" << std::endl;
            }
        }

        // Set max_output_tokens from model config
        max_output_tokens = model_config_.max_output_tokens;

        // Load sampling parameters from generation_config.json if available
        Models::load_generation_config(model_path_, temperature, top_p, top_k);

        initialized = true;
        dout(1) << "TensorRT backend initialized successfully" << std::endl;

        // In CLI mode, add system message
        if (!g_server_mode && !session.system_message.empty()) {
            std::string formatted_system = chat_template_->format_system_message(
                session.system_message,
                session.tools
            );

            Response sys_resp = add_message(session, Message::SYSTEM, formatted_system, "", "", 0, 0);
            if (!sys_resp.success) {
                throw BackendError("Failed to initialize system message: " + sys_resp.error);
            }

            session.system_message_tokens = sys_resp.prompt_tokens;
            dout(1) << "Added system message (tokens=" + std::to_string(sys_resp.prompt_tokens) + ")" << std::endl;
        }

    } catch (const std::exception& e) {
        throw BackendError("Failed to initialize TensorRT: " + std::string(e.what()));
    }
}

int TensorRTBackend::count_message_tokens(Message::Role role,
                                         const std::string& content,
                                         const std::string& tool_name,
                                         const std::string& tool_id) {
    if (!tokenizer_) {
        return static_cast<int>(content.length() / 4.0 + 0.5);
    }

    // Use ChatTemplate for formatting if available
    if (!chat_template_) {
        return tokenizer_->count_tokens(content);
    }

    // Create a temporary Message to format through ChatTemplate
    Message msg(role, content, 0);
    msg.tool_name = tool_name;
    msg.tool_call_id = tool_id;

    try {
        std::string formatted = chat_template_->format_message(msg);
        return tokenizer_->count_tokens(formatted);
    } catch (const std::exception& e) {
        dout(1) << std::string("WARNING: ") +"Exception formatting message for token count: " + std::string(e.what()) << std::endl;
        return tokenizer_->count_tokens(content);
    }
}

const ChatTemplates::ChatTemplateCaps* TensorRTBackend::get_chat_template_caps() const {
    if (chat_template_) {
        return &chat_template_->get_capabilities();
    }
    return nullptr;
}

// NOTE: add_message() removed - use Frontend::add_message_to_session() + generate_response() instead

void TensorRTBackend::generate_from_session(Session& session, int max_tokens) {
    dout(1) << "generate_from_session CALLED - accumulated_tokens.size=" + std::to_string(accumulated_tokens.size()) << std::endl;

    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "TensorRT backend not initialized";
        callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
    }

    dout(1) << "TensorRT generate_from_session called with " + std::to_string(session.messages.size()) + " messages" << std::endl;

    // Debug: Show last message type and if generation prompt will be added
    if (!session.messages.empty()) {
        const auto& last_msg = session.messages.back();
        dout(1) << "Last message: type=" + std::to_string(static_cast<int>(last_msg.role)) +
                  ", will add gen prompt=" + std::string(last_msg.role != Message::SYSTEM && last_msg.role != Message::ASSISTANT ? "YES" : "NO"));
    }

    // PREFIX CACHING: Compare incoming session with backend_session (what's in KV cache)
    size_t cached_count = backend_session.messages.size();

    dout(1) << "Backend has " + std::to_string(cached_count) + " cached messages, " +
              "incoming session has " + std::to_string(session.messages.size()) + " messages" << std::endl;

    // Account for system message offset
    size_t backend_offset = 0;
    if (!backend_session.messages.empty() && backend_session.messages[0].role == Message::SYSTEM) {
        backend_offset = 1;
        dout(1) << "Backend has system message at index 0, offsetting comparison by 1" << std::endl;
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
            dout(1) << std::string("WARNING: ") +"DIVERGENCE at session message " + std::to_string(i) << std::endl;
            break;
        }
    }

    dout(1) << "Prefix match: " + std::to_string(matching_prefix) + " messages" << std::endl;

    // Check if backend has more messages than what matched
    size_t expected_backend_count = backend_offset + matching_prefix;

    if (cached_count > expected_backend_count) {
        dout(1) << std::string("WARNING: ") +"Conversation diverged - clearing cache and restarting" << std::endl;

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
        (backend_session.messages.empty() || backend_session.messages[0].role != Message::SYSTEM)) {

        dout(1) << "Adding system message to backend" << std::endl;

        std::string formatted_system = chat_template_->format_system_message(
            session.system_message,
            session.tools
        );

        Message sys_msg(Message::SYSTEM, formatted_system, 0);
        sys_msg.tokens = count_message_tokens(Message::SYSTEM, formatted_system, "", "");

        if (!tokenize_and_accumulate_message(sys_msg)) {
            Response err_resp;
            err_resp.success = false;
            err_resp.code = Response::ERROR;
            err_resp.error = "Failed to tokenize system message";
            callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
        }

        backend_session.messages.push_back(sys_msg);
        backend_session.system_message = session.system_message;
        dout(1) << "System message added (" + std::to_string(sys_msg.tokens) + " tokens)" << std::endl;
    }

    // Add NEW messages (from matching_prefix onward)
    size_t new_messages = session.messages.size() - matching_prefix;
    if (new_messages > 0) {
        dout(1) << "Adding " + std::to_string(new_messages) + " new messages to backend" << std::endl;

        for (size_t i = matching_prefix; i < session.messages.size(); i++) {
            const auto& msg = session.messages[i];
            dout(1) << "Processing message " + std::to_string(i) + ": type=" + std::to_string(static_cast<int>(msg.role)) +
                      ", content_len=" + std::to_string(msg.content.length()) +
                      ", content_preview=" + msg.content.substr(0, std::min(size_t(100), msg.content.length())));

            // Skip empty messages (shouldn't happen but server might send them)
            if (msg.content.empty() && msg.role != Message::SYSTEM) {
                dout(1) << std::string("WARNING: ") +"Skipping empty message at index " + std::to_string(i) << std::endl;
                continue;
            }

            Message msg_copy = msg;
//			std::cout << "Sending: " << msg.content " << std::endl;

            if (msg_copy.tokens == 0) {
                msg_copy.tokens = count_message_tokens(msg_copy.role, msg_copy.content,
                                                        msg_copy.tool_name, msg_copy.tool_call_id);
            }

            // For the LAST message, add generation prompt if it will trigger generation
            bool is_last = (i == session.messages.size() - 1);
            bool will_generate = is_last && (msg_copy.role != Message::SYSTEM && msg_copy.role != Message::ASSISTANT);

            dout(1) << "Processing msg " + std::to_string(i) + ": type=" + std::to_string(static_cast<int>(msg_copy.type)) +
                     ", is_last=" + std::string(is_last ? "true" : "false") +
                     ", will_generate=" + std::string(will_generate ? "true" : "false"));

            if (!tokenize_and_accumulate_message(msg_copy, will_generate)) {
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::ERROR;
                err_resp.error = "Failed to tokenize message";
                callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
            }

            backend_session.messages.push_back(msg_copy);

            if (msg.role == Message::USER) {
                backend_session.last_user_message_index = backend_session.messages.size() - 1;
                backend_session.last_user_message_tokens = msg_copy.tokens;
            } else if (msg.role == Message::ASSISTANT) {
                backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
                backend_session.last_assistant_message_tokens = msg_copy.tokens;
            }
        }

        dout(1) << "Prefix caching: " + std::to_string(matching_prefix) + " cached, " +
                  std::to_string(new_messages) + " new" << std::endl;
    } else {
        dout(1) << "All messages already in cache (100% prefix cache hit)" << std::endl;
        // The KV cache contains all the context from previous messages
        // accumulated_tokens was cleared after last generation
        // We need to resend ALL tokens (TensorRT's KV cache will handle deduplication)
        dout(1) << "Rebuilding accumulated_tokens from all cached messages" << std::endl;

        // Rebuild accumulated_tokens by re-tokenizing all cached messages
        accumulated_tokens.clear();
        for (size_t i = 0; i < backend_session.messages.size(); i++) {
            Message msg_copy = backend_session.messages[i];
            // For the last message (user), add generation prompt to start assistant response
            bool is_last = (i == backend_session.messages.size() - 1);
            bool will_generate = is_last && (msg_copy.role == Message::USER);
            tokenize_and_accumulate_message(msg_copy, will_generate);
        }

        dout(1) << "Rebuilt " + std::to_string(accumulated_tokens.size()) + " tokens from " +
                  std::to_string(backend_session.messages.size()) + " cached messages" << std::endl;
    }

    // Copy tools and system_message
    backend_session.tools = session.tools;
    backend_session.system_message = session.system_message;

    // Copy sampling parameters from request
    backend_session.sampling = session.sampling;

    // Generate response
    std::string result = generate(backend_session, max_tokens, callback);

    // Add assistant message to backend_session so next turn knows what was generated
    if (!result.empty()) {
        Message assistant_msg(Message::ASSISTANT, result, 0);
        assistant_msg.tokens = tokenizer_->count_tokens(result);
        backend_session.messages.push_back(assistant_msg);
        backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
        backend_session.last_assistant_message_tokens = assistant_msg.tokens;
        dout(1) << "Added assistant message to backend cache (" + std::to_string(assistant_msg.tokens) + " tokens)" << std::endl;
    }

    // If streaming, return empty response (content already sent via callback)
    if (config->streaming) {
        Response empty_resp;
        empty_resp.success = true;
        empty_resp.code = Response::SUCCESS;
        return empty_resp;
    }

    // Non-streaming: return full response
    Response success_resp;
    success_resp.success = true;
    success_resp.code = Response::SUCCESS;
    success_resp.content = result;
    success_resp.finish_reason = "stop";
    success_resp.prompt_tokens = 0;  // TODO: Calculate actual prompt tokens
    success_resp.completion_tokens = tokenizer_->count_tokens(result);
    return success_resp;
}

void TensorRTBackend::prefill_session(Session& session) {
#ifdef ENABLE_TENSORRT
    if (!is_ready()) {
        throw std::runtime_error("TensorRT backend not initialized");
    }

    dout(1) << "TensorRT prefill_session called with " + std::to_string(session.messages.size()) + " messages" << std::endl;

    // PREFIX CACHING: Compare incoming session with backend_session (what's in KV cache)
    size_t cached_count = backend_session.messages.size();

    // Account for system message offset
    size_t backend_offset = 0;
    if (!backend_session.messages.empty() && backend_session.messages[0].role == Message::SYSTEM) {
        backend_offset = 1;
    }

    // Find how many messages match (prefix caching)
    size_t matching_prefix = 0;
    for (size_t i = 0; i < session.messages.size(); i++) {
        size_t backend_idx = i + backend_offset;
        if (backend_idx >= cached_count) break;

        const auto& cached_msg = backend_session.messages[backend_idx];
        const auto& session_msg = session.messages[i];

        if (cached_msg.get_role() == session_msg.get_role() &&
            cached_msg.content == session_msg.content) {
            matching_prefix++;
        } else {
            break;
        }
    }

    // Check if backend has more messages than what matched
    size_t expected_backend_count = backend_offset + matching_prefix;

    if (cached_count > expected_backend_count) {
        accumulated_tokens.clear();
        backend_session.messages.clear();
        matching_prefix = 0;
        backend_offset = 0;
    }

    // Set current_session for eviction callbacks
    current_session = &backend_session;

    // Handle system message if present
    if (!session.system_message.empty() &&
        (backend_session.messages.empty() || backend_session.messages[0].role != Message::SYSTEM)) {

        std::string formatted_system = chat_template_->format_system_message(
            session.system_message,
            session.tools
        );

        Message sys_msg(Message::SYSTEM, formatted_system, 0);
        sys_msg.tokens = count_message_tokens(Message::SYSTEM, formatted_system, "", "");

        if (!tokenize_and_accumulate_message(sys_msg)) {
            throw std::runtime_error("Failed to tokenize system message");
        }

        backend_session.messages.push_back(sys_msg);
        backend_session.system_message = session.system_message;
    }

    // Add new messages that aren't in cache
    for (size_t i = matching_prefix; i < session.messages.size(); i++) {
        Message msg = session.messages[i];
        bool is_last = (i == session.messages.size() - 1);
        bool needs_gen_prompt = is_last && (msg.role == Message::USER);

        if (!tokenize_and_accumulate_message(msg, needs_gen_prompt)) {
            throw std::runtime_error("Failed to tokenize message");
        }

        backend_session.messages.push_back(msg);
        backend_session.last_user_message_index = backend_session.messages.size() - 1;
        backend_session.last_user_message_tokens = msg.tokens;
    }

    // If nothing new to add but we have cached messages, rebuild tokens
    if (matching_prefix > 0 && matching_prefix == session.messages.size() && accumulated_tokens.empty()) {
        for (size_t i = 0; i < backend_session.messages.size(); i++) {
            Message msg_copy = backend_session.messages[i];
            bool is_last = (i == backend_session.messages.size() - 1);
            bool will_generate = is_last && (msg_copy.role == Message::USER);
            tokenize_and_accumulate_message(msg_copy, will_generate);
        }
    }

    // Copy tools and system_message
    backend_session.tools = session.tools;
    backend_session.system_message = session.system_message;
    backend_session.sampling = session.sampling;

    // Proactive context check - throw before generate() is called
    if (accumulated_tokens.size() > context_size) {
        throw ContextFullException("This model's maximum context length is " +
            std::to_string(context_size) + " tokens. However, your messages resulted in " +
            std::to_string(accumulated_tokens.size()) + " tokens.");
    }

    dout(1) << "TensorRT prefill_session complete: " + std::to_string(accumulated_tokens.size()) + " tokens accumulated" << std::endl;
#else
    throw std::runtime_error("TensorRT backend not compiled in");
#endif
}

void TensorRTBackend::generate_from_prefilled(Session& session, int max_tokens) {
#ifdef ENABLE_TENSORRT
    if (!is_ready()) {
        callback(CallbackEvent::ERROR, "TensorRT backend not initialized", "error", "");
        callback(CallbackEvent::STOP, "error", "", "");
        return;
    }

    dout(1) << "TensorRT generate_from_prefilled called" << std::endl;

    // Generate response
    std::string result = generate(backend_session, max_tokens, callback);

    // Add assistant message to backend_session
    if (!result.empty()) {
        Message assistant_msg(Message::ASSISTANT, result, 0);
        assistant_msg.tokens = tokenizer_->count_tokens(result);
        backend_session.messages.push_back(assistant_msg);
        backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
        backend_session.last_assistant_message_tokens = assistant_msg.tokens;
    }
#else
    callback(CallbackEvent::ERROR, "TensorRT backend not compiled in", "error", "");
    callback(CallbackEvent::STOP, "error", "", "");
#endif
}

// Helper methods

std::string TensorRTBackend::render_message(const Message& msg, bool add_generation_prompt) {
    dout(1) << "render_message called: chat_template_=" + std::string(chat_template_ ? "exists" : "null") +
              ", msg.role=" + std::to_string(static_cast<int>(msg.role)) +
              ", content_len=" + std::to_string(msg.content.length()));

    if (!chat_template_) {
        throw BackendError("No chat template loaded - cannot render message. Model must provide a chat template in tokenizer_config.json");
    }

    std::string formatted;

    // SYSTEM messages from initialize() are already formatted by format_system_message
    // They already contain <|im_start|>system...<|im_end|> - don't double-wrap
    if (msg.role == Message::SYSTEM) {
        // System messages are pre-formatted, use content directly
        formatted = msg.content;
    } else {
        // Use ChatTemplate abstraction for model-specific formatting
        formatted = chat_template_->format_message(msg);
    }

    // Add generation prompt if requested
    // This applies to user messages and tool results - anything that triggers assistant generation
    if (add_generation_prompt) {
        std::string gen_prompt = chat_template_->get_generation_prompt();
        formatted += gen_prompt;
        dout(1) << "Added generation prompt (" + std::to_string(gen_prompt.length()) + " chars)" << std::endl;
    }

    dout(1) << "Rendered message (type=" + std::to_string(static_cast<int>(msg.role)) +
              ", role=" + msg.get_role() + ", add_generation_prompt=" +
              std::string(add_generation_prompt ? "true" : "false") + "" << std::endl;

    dout(1) << "Full rendered output (" + std::to_string(formatted.length()) + " chars): " + formatted << std::endl;

    return formatted;
}

bool TensorRTBackend::tokenize_and_accumulate_message(Message& msg, bool add_generation_prompt) {
    if (!tokenizer_) {
        callback(CallbackEvent::ERROR, "Tokenizer not loaded", "", "");
        return false;
    }

    dout(1) << "tokenize_and_accumulate_message: type=" + std::to_string(static_cast<int>(msg.role)) +
              ", content_len=" + std::to_string(msg.content.length()) +
              ", add_generation_prompt=" + std::string(add_generation_prompt ? "true" : "false"));

    // Render message through template
    std::string formatted_text = render_message(msg, add_generation_prompt);
//	std::cout << "Formatted text: " << formatted_text " << std::endl;

    // Tokenize with add_special_tokens=false to avoid unwanted BOS/EOS tokens
    std::vector<int> tokens = tokenizer_->encode(formatted_text, false);
    if (tokens.empty()) {
        callback(CallbackEvent::ERROR, "Failed to tokenize message", "", "");
        dout(1) << "Formatted text: " + formatted_text.substr(0, 500) << std::endl;
        return false;
    }

    // Update message token count to actual formatted count
    msg.tokens = tokens.size();

    // Append tokens to accumulated_tokens vector
    accumulated_tokens.insert(accumulated_tokens.end(), tokens.begin(), tokens.end());

    dout(1) << "Accumulated " + std::to_string(tokens.size()) + " tokens, total accumulated: " +
              std::to_string(accumulated_tokens.size()));

    return true;
}

std::string TensorRTBackend::generate(const Session& session, int max_tokens, EventCallback callback) {
    namespace tle = tensorrt_llm::executor;
    auto* executor = static_cast<tle::Executor*>(executor_);

    dout(1) << "generate() called with callback=" + std::string(callback ? "SET" : "NULL") << std::endl;

    // Reset ModelOutput state for new generation
    tio.reset();

    try {
        if (accumulated_tokens.empty()) {
            throw BackendError("No tokens accumulated for generation");
        }

        // Convert to int32_t
        std::vector<int32_t> input_tokens(accumulated_tokens.begin(), accumulated_tokens.end());
        dout(1) << "Generating with " + std::to_string(input_tokens.size()) + " accumulated tokens" << std::endl;

        // Proactive context size check - throw before we start generation
        // This allows API server to return 400 before streaming starts
        if (input_tokens.size() + static_cast<size_t>(max_tokens) > context_size) {
            throw ContextFullException("This model's maximum context length is " +
                std::to_string(context_size) + " tokens. However, your messages resulted in " +
                std::to_string(input_tokens.size()) + " tokens plus " +
                std::to_string(max_tokens) + " max generation tokens.");
        }

        // Decode and log the prompt for debugging
        if (tokenizer_) {
            std::vector<int> debug_tokens(accumulated_tokens.begin(), accumulated_tokens.end());

            // Log first 10 token IDs
            std::string token_ids_str = "First tokens: ";
            for (size_t i = 0; i < std::min(size_t(10), debug_tokens.size()); i++) {
                token_ids_str += std::to_string(debug_tokens[i]) + " ";
            }
            dout(1) << token_ids_str << std::endl;

            // Skip decoding for now - it seems to crash
            // std::string decoded_prompt = tokenizer_->decode(debug_tokens);
            // dout(1) << "Decoded prompt (" + std::to_string(decoded_prompt.length()) + " chars): " +
            //           decoded_prompt.substr(0, std::min(size_t(500), decoded_prompt.length())));
        }

        // Track timing for prefill and decode
        auto start_time = std::chrono::high_resolution_clock::now();
        auto prefill_end_time = start_time;
        bool prefill_complete = false;

        // Track finish reason for detecting length limit
        tle::FinishReason finish_reason = tle::FinishReason::kNOT_FINISHED;

        // Create sampling config - use Session overrides if set, otherwise use backend defaults
        const auto& sp = session.sampling;
        float use_temperature = (sp.temperature >= 0.0f) ? sp.temperature : temperature;
        float use_top_p = (sp.top_p >= 0.0f) ? sp.top_p : top_p;
        int use_top_k = (sp.top_k >= 0) ? sp.top_k : top_k;
        float use_min_p = (sp.min_p >= 0.0f) ? sp.min_p : min_p;
        float use_repetition_penalty = (sp.repetition_penalty >= 0.0f) ? sp.repetition_penalty : repetition_penalty;
        float use_presence_penalty = (sp.presence_penalty > -999.0f) ? sp.presence_penalty : presence_penalty;
        float use_frequency_penalty = (sp.frequency_penalty > -999.0f) ? sp.frequency_penalty : frequency_penalty;
        float use_length_penalty = (sp.length_penalty > -999.0f) ? sp.length_penalty : length_penalty;
        int use_no_repeat_ngram = (sp.no_repeat_ngram_size >= 0) ? sp.no_repeat_ngram_size : no_repeat_ngram_size;

        dout(1) << "Sampling config: temp=" + std::to_string(use_temperature) +
                  " top_p=" + std::to_string(use_top_p) +
                  " top_k=" + std::to_string(use_top_k) +
                  " rep_penalty=" + std::to_string(use_repetition_penalty) +
                  " freq_penalty=" + std::to_string(use_frequency_penalty) +
                  " pres_penalty=" + std::to_string(use_presence_penalty));

        tle::SamplingConfig samplingConfig;
        samplingConfig.setBeamWidth(1);
        samplingConfig.setTemperature(use_temperature);
        if (use_top_p > 0.0f && use_top_p <= 1.0f) {
            samplingConfig.setTopP(use_top_p);
        }
        if (use_top_k > 0) {
            samplingConfig.setTopK(use_top_k);
        }
        if (use_min_p > 0.0f) {
            samplingConfig.setMinP(use_min_p);
        }

        // Apply penalty parameters
        if (use_repetition_penalty != 1.0f) {
            samplingConfig.setRepetitionPenalty(use_repetition_penalty);
        }
        if (use_presence_penalty != 0.0f) {
            samplingConfig.setPresencePenalty(use_presence_penalty);
        }
        if (use_frequency_penalty != 0.0f) {
            samplingConfig.setFrequencyPenalty(use_frequency_penalty);
        }
        if (use_length_penalty != 0.0f) {
            samplingConfig.setLengthPenalty(use_length_penalty);
        }
        if (use_no_repeat_ngram > 0) {
            samplingConfig.setNoRepeatNgramSize(use_no_repeat_ngram);
        }

        // Create output config
        tle::OutputConfig outputConfig;
        outputConfig.returnLogProbs = false;
        outputConfig.returnContextLogits = false;
        outputConfig.returnGenerationLogits = false;

        // Create KV cache retention config
        size_t system_msg_tokens = 0;
        if (!backend_session.messages.empty() && backend_session.messages[0].role == Message::SYSTEM) {
            system_msg_tokens = backend_session.messages[0].tokens;
        }

        std::vector<tle::KvCacheRetentionConfig::TokenRangeRetentionConfig> token_ranges;
        if (system_msg_tokens > 0) {
            token_ranges.push_back(
                tle::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, system_msg_tokens, 100)
            );
        }

        tle::KvCacheRetentionConfig retention(token_ranges, 35);

        dout(1) << "Creating TensorRT request with " + std::to_string(input_tokens.size()) +
                  " tokens, max_tokens=" + std::to_string(max_tokens));

        // Create request
        int actual_max_tokens = (max_tokens > 0) ? max_tokens : 512;

        // Convert stop_token_ids to list format for TensorRT
        std::optional<std::list<tle::VecTokens>> stop_words_opt;
        if (!stop_token_ids.empty()) {
            std::list<tle::VecTokens> stop_words_list;
            for (const auto& seq : stop_token_ids) {
                tle::VecTokens vec(seq.begin(), seq.end());
                stop_words_list.push_back(vec);
            }
            stop_words_opt = stop_words_list;
            dout(1) << "Passing " + std::to_string(stop_words_list.size()) + " stop sequences to TensorRT" << std::endl;
        }

        dout(1) << "About to construct Request object" << std::endl;
        tle::Request request(
            input_tokens,
            actual_max_tokens,
            true,  // streaming
            samplingConfig,
            outputConfig,
            eos_token_id,  // endId - EOS token ID to stop generation
            std::nullopt,  // padId
            std::nullopt,  // positionIds
            std::nullopt,  // badWords
            stop_words_opt,  // stopWords - multi-token stop sequences (e.g., <|user|>)
            std::nullopt,  // embeddingBias
            std::nullopt,  // externalDraftTokensConfig
            std::nullopt,  // pTuningConfig
            std::nullopt,  // multimodalInput
            std::nullopt,  // multimodalEmbedding
            std::nullopt,  // mRopeConfig
            std::nullopt,  // loraConfig
            std::nullopt,  // lookaheadConfig
            retention       // kvCacheRetentionConfig
        );
        dout(1) << "Request object constructed successfully" << std::endl;

        // Enqueue request
        uint64_t request_id;
        try {
            request_id = executor->enqueueRequest(request);
            current_request_id_ = request_id;
            request_active_ = true;
            dout(1) << "Request enqueued with ID: " + std::to_string(request_id) << std::endl;
        } catch (const std::exception& e) {
            callback(CallbackEvent::ERROR, "Exception enqueueing request: " + std::string(e.what()), "", "");
            throw BackendError("Failed to enqueue request: " + std::string(e.what()));
        } catch (...) {
            callback(CallbackEvent::ERROR, "Unknown exception enqueueing request", "", "");
            throw BackendError("Failed to enqueue request: unknown exception");
        }

        // Collect responses
        std::string response_text;
        std::vector<int32_t> output_tokens;
        size_t last_decoded_len = 0;  // Track decoded length for incremental output

        dout(1) << "Starting response collection loop" << std::endl;
        while (request_active_) {
            // No timeout - block until responses are ready (not polling!)
            auto responses = executor->awaitResponses(request_id);

            for (const auto& response : responses) {
                if (response.hasError()) {
                    std::string error_msg = response.getErrorMsg();
                    callback(CallbackEvent::ERROR, "Response error: " + error_msg, "", "");
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
                                std::to_string(context_size) + " tokens. However, your messages resulted in too many tokens." << std::endl;
                        } else {
                            // CLI mode: throw special marker for eviction-retry
                            dout(1) << std::string("WARNING: ") +"Context full in CLI mode - eviction needed" << std::endl;
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

                    // Decode all accumulated tokens to get correct spacing
                    // BPE/SentencePiece tokenizers encode spaces INTO tokens (e.g., "▁Place")
                    // so we must decode the full sequence and extract just the new text
                    std::string full_decoded = tokenizer_->decode(output_tokens);
                    std::string new_text;
                    if (last_decoded_len < full_decoded.length()) {
                        new_text = full_decoded.substr(last_decoded_len);
                    } else if (full_decoded.length() > 0) {
                        // Edge case: decoder returned shorter string (shouldn't happen but be safe)
                        new_text = full_decoded;
                        dout(1) << "Decoder returned shorter string than expected" << std::endl;
                    }
                    last_decoded_len = full_decoded.length();

#ifdef _DEBUG
                    // Debug: Log token IDs and decoded text to track think tag issues
                    if (g_debug_level > 2) {
                        std::string token_ids_str;
                        for (auto tid : beam_tokens) {
                            if (!token_ids_str.empty()) token_ids_str += ",";
                            token_ids_str += std::to_string(tid);
                        }
                        dout(1) << "Tokens [" + token_ids_str + "] -> \"" + new_text + "\"" << std::endl;
                    }
#endif

                    if (config->streaming) {
                        // Streaming mode: invoke callback with new text (event-based)
                        if (!callback(CallbackEvent::CONTENT, new_text, "", "")) {
                            request_active_ = false;
                            break;
                        }
                    }
                    response_text += new_text;
                }

                if (result.isFinal) {
                    // Capture finish reason to detect if we hit length limit
                    if (!result.finishReasons.empty()) {
                        finish_reason = result.finishReasons[0];
                    }
                    dout(1) << "Generation complete, finish_reason=" + std::to_string(static_cast<int>(finish_reason)) << std::endl;
                    request_active_ = false;
                    break;
                }
            }
        }
//		std::cout << "Response: " << response_text " << std::endl;

        dout(1) << "Generated " + std::to_string(output_tokens.size()) + " tokens" << std::endl;
        dout(1) << "Decoded " + std::to_string(response_text.length()) + " characters" << std::endl;

        // Store whether we hit length limit (for finish_reason and auto-continuation)
        last_generation_hit_length_limit = (finish_reason == tle::FinishReason::kLENGTH);
        if (last_generation_hit_length_limit) {
            dout(1) << "Generation stopped due to max_tokens limit" << std::endl;
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

            // Send prefill stats via callback (frontend decides whether to display)
            std::ostringstream stats;
            stats << "[Prefill: " << input_tokens.size() << " tokens, "
                  << std::fixed << std::setprecision(1) << prefill_tok_per_sec << " t/s, "
                  << "context: " << context_used << "/" << context_max << "]\n";
            callback(CallbackEvent::STATS, stats.str(), "", "");
        }

        // Decode metrics
        if (output_tokens.size() > 0 && prefill_complete) {
            auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - prefill_end_time);
            double decode_seconds = decode_duration.count() / 1000.0;
            double decode_tok_per_sec = (decode_seconds > 0) ? (output_tokens.size() / decode_seconds) : 0;

            int context_used = accumulated_tokens.size() + output_tokens.size();
            int context_max = context_size;

            // Send decode stats via callback (frontend decides whether to display)
            std::ostringstream stats2;
            stats2 << "\n[Decode: " << output_tokens.size() << " tokens, "
                  << std::fixed << std::setprecision(1) << decode_tok_per_sec << " t/s, "
                  << "context: " << context_used << "/" << context_max << "]\n";
            callback(CallbackEvent::STATS, stats2.str(), "", "");
        }

        // Handle <|eom_id|> token for continued tool execution
        if (model_config_.uses_eom_token) {
            size_t eom_pos = response_text.find("<|eom_id|>");
            if (eom_pos != std::string::npos) {
                response_text = response_text.substr(0, eom_pos);
                dout(1) << "Found <|eom_id|> token - model expects tool execution continuation" << std::endl;
            }
        }

        // Strip stop tokens from final response
        for (const auto& stop_token : stop_tokens) {
            size_t pos = response_text.find(stop_token);
            if (pos != std::string::npos) {
                response_text = response_text.substr(0, pos);
                dout(1) << "Stripped stop token: " + stop_token << std::endl;
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
                    dout(1) << "Stripped thinking block ending with: " + end_marker << std::endl;
                    break;
                }
            }
        }

        // Content extraction (for channel-based models like GPT-OSS)
        // Uses ChatTemplate's extract_content() which detects format from template
        if (chat_template_) {
            std::string extracted = chat_template_->extract_content(response_text);
            if (extracted.length() != response_text.length()) {
                dout(1) << "Content extraction: " + std::to_string(response_text.length()) +
                          " -> " + std::to_string(extracted.length()) + " chars" << std::endl;
                response_text = extracted;
            }
        }

        // CLEAR accumulated_tokens after generation completes
        accumulated_tokens.clear();
        dout(1) << "Cleared accumulated tokens" << std::endl;

        return response_text;

    } catch (const std::exception& e) {
        request_active_ = false;
        accumulated_tokens.clear();
        callback(CallbackEvent::ERROR, "TensorRT generation error: " + std::string(e.what()), "", "");
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

    // Delete executor - let destructor handle MPI coordination
    // TensorRT-LLM examples don't call shutdown() explicitly
    // The executor destructor handles cleanup and MPI_Finalize is atexit-registered
    if (executor_) {
        namespace tle = tensorrt_llm::executor;
        auto* executor = static_cast<tle::Executor*>(executor_);

        try {
            delete executor;
        } catch (const std::exception& e) {
            dout(1) << "Error during TensorRT executor cleanup: " + std::string(e.what()) << std::endl;
        }

        executor_ = nullptr;
    }

    // Clear session data
    backend_session.messages.clear();
    accumulated_tokens.clear();

    initialized = false;
    dout(1) << "TensorRTBackend shutdown" << std::endl;
}

void TensorRTBackend::monitor_kv_events() {
    namespace tle = tensorrt_llm::executor;

    if (!event_manager_) {
        dout(1) << "Event manager not initialized" << std::endl;
        return;
    }

    auto* event_mgr = static_cast<std::shared_ptr<tle::KVCacheEventManager>*>(event_manager_);
    dout(1) << "KV cache event monitoring thread started" << std::endl;

    while (monitoring_events_) {
        try {
            auto events = (*event_mgr)->getLatestEvents(std::chrono::milliseconds(100));

            for (const auto& event : events) {
                if (std::holds_alternative<tle::KVCacheRemovedData>(event.data)) {
                    const auto& removed = std::get<tle::KVCacheRemovedData>(event.data);

                    dout(1) << "KV cache eviction detected: " +
                             std::to_string(removed.blockHashes.size()) + " blocks removed" << std::endl;

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
                        dout(1) << "Block stored: hash=" + std::to_string(block.blockHash) +
                                 " tokens=" + std::to_string(block.tokens.size()));
                    }
                }
            }

        } catch (const std::exception& e) {
            dout(1) << "Error in KV cache event monitoring: " + std::string(e.what()) << std::endl;
        }
    }

    dout(1) << "KV cache event monitoring thread stopped" << std::endl;
}

void TensorRTBackend::handle_kv_cache_removed(const std::vector<uint64_t>& block_hashes) {
    std::lock_guard<std::mutex> lock(block_map_mutex_);

    // Calculate tokens removed (TensorRT typically uses 64 tokens per block)
    const size_t TOKENS_PER_BLOCK = 64;
    size_t tokens_removed = block_hashes.size() * TOKENS_PER_BLOCK;

    dout(1) << "KV cache eviction: " + std::to_string(block_hashes.size()) +
             " blocks removed (~" + std::to_string(tokens_removed) + " tokens" << std::endl;


    auto& messages = backend_session.messages;
    if (messages.empty()) {
        dout(1) << std::string("WARNING: ") +"No messages in backend session" << std::endl;
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
        dout(1) << std::string("WARNING: ") +"Could not determine evicted messages" << std::endl;
        return;
    }

    dout(1) << "Identified messages 1-" + std::to_string(last_evicted_msg) + " as evicted" << std::endl;

    // NOTE: archive_turn during eviction disabled (v2.32.0)
    // Memory extraction thread already captures anything valuable from conversations.
    open_user_question_.reset();

    // Remove evicted messages from backend_session
    for (int i = 0; i < last_evicted_msg; ++i) {
        messages.erase(messages.begin() + 1);  // Always remove index 1
    }

    // Update backend_session total_tokens
    backend_session.total_tokens = 0;
    for (const auto& msg : messages) {
        backend_session.total_tokens += msg.tokens;
    }

    dout(1) << "Removed " + std::to_string(last_evicted_msg) +
             " messages from backend session, " + std::to_string(messages.size()) + " remaining" << std::endl;

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

void TensorRTBackend::generate_from_session(Session&, int) {
    Response err_resp;
    err_resp.success = false;
    err_resp.code = Response::ERROR;
    err_resp.error = "TensorRT backend not compiled in";
    callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
}

void TensorRTBackend::prefill_session(Session&) {
    throw std::runtime_error("TensorRT backend not compiled in");
}

void TensorRTBackend::generate_from_prefilled(Session&, int) {
    callback(CallbackEvent::ERROR, "TensorRT backend not compiled in", "error", "");
    callback(CallbackEvent::STOP, "error", "", "");
}

int TensorRTBackend::count_message_tokens(Message::Role, const std::string& content,
                                         const std::string&, const std::string&) {
    return static_cast<int>(content.length() / 4.0 + 0.5);
}

const ChatTemplates::ChatTemplateCaps* TensorRTBackend::get_chat_template_caps() const {
    return nullptr;  // No chat template in stub
}
#endif
