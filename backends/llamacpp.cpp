
#include "shepherd.h"
#include "llamacpp.h"
#include "tools/tool.h"
#include "tools/tool_parser.h"
#include "nlohmann/json.hpp"
#include "models.h"
#include <sstream>
#include <fstream>
#include <regex>
#include <ctime>
#include <filesystem>

#ifdef ENABLE_LLAMACPP
#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-batch.h"
#include "llama.cpp/common/chat.h"
#endif


// LlamaCppBackend implementation
LlamaCppBackend::LlamaCppBackend(size_t max_context_tokens, Session& session, EventCallback callback)
    : Backend(max_context_tokens, session, callback),
      model_config(ModelConfig::create_generic()) {
    // Set public variables per RULES.md
    backend_name = "llamacpp";
    context_size = max_context_tokens;
    is_local = true;  // Local GPU backend

    dout(1) << "LlamaCppBackend created with context_size: " + std::to_string(context_size) << std::endl;

    // Parse config
    parse_backend_config();

    // --- Full initialization (formerly in initialize()) ---
#ifdef ENABLE_LLAMACPP
    if (initialized) {
        dout(1) << std::string("WARNING: ") +"LlamaCppBackend already initialized" << std::endl;
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

    dout(1) << "Using model path: " + full_model_path << std::endl;

    this->model_path = full_model_path;
    model_name = full_model_path;  // Set public variable

    // Suppress llama.cpp logging unless in debug mode (but always show errors)
#ifdef _DEBUG
    if (!g_debug_level)
#endif
    {
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
        dout(1) << "Metal GPU disabled by environment variable" << std::endl;
    }

    // Load actual llama.cpp model
    llama_model_params model_params = llama_model_default_params();

    if (has_gpu) {
        // Priority: environment variable > config setting > auto
        const char* gpu_layers_env = getenv("GGML_N_GPU_LAYERS");
        if (gpu_layers_env) {
            model_params.n_gpu_layers = atoi(gpu_layers_env);
            dout(1) << "Using GPU layer count from GGML_N_GPU_LAYERS env: " + std::to_string(model_params.n_gpu_layers) << std::endl;
        } else if (gpu_layers >= 0) {
            model_params.n_gpu_layers = gpu_layers;
            dout(1) << "Using GPU layer count from config: " + std::to_string(model_params.n_gpu_layers) << std::endl;
        } else {
            // -1 = auto: set to extremely high number - llama.cpp will automatically cap at actual layer count
            model_params.n_gpu_layers = INT32_MAX;
            dout(1) << "Auto GPU layers (loading all layers to GPU)" << std::endl;
        }

        // Multi-GPU support: Use pipeline_parallel or tensor_parallel config to control GPU usage
        // Note: pipeline_parallel is preferred over tensor_parallel for semantic clarity
        // Both use LLAMA_SPLIT_MODE_LAYER which distributes layers across GPUs
        int num_gpus_for_splitting = 0;

        // Determine which config to use (prefer pipeline_parallel if both specified)
        if (pipeline_parallel > 1) {
            num_gpus_for_splitting = pipeline_parallel;
        } else if (tensor_parallel == 0 || tensor_parallel > 1) {
            num_gpus_for_splitting = tensor_parallel;
        }

        if (num_gpus_for_splitting != 0) {
            // Multi-GPU support: choose split mode based on TP vs PP
            // LAYER mode = pipeline parallelism (splits layers, no P2P needed)
            // ROW mode = tensor parallelism (splits tensors, requires P2P)
            if (tensor_parallel > 1) {
                model_params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else {
                model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            }
            model_params.main_gpu = 0;

            // Configure tensor_split array to distribute model across GPUs
            int num_gpus = num_gpus_for_splitting > 0 ? num_gpus_for_splitting : 16;  // Max 16 GPUs if auto
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
                        dout(1) << "Added device " + std::to_string(i) + ": " + std::string(dev_name) << std::endl;
                    }
                }
            }
            gpu_devices.push_back(nullptr);  // NULL terminator
            model_params.devices = reinterpret_cast<ggml_backend_dev_t*>(gpu_devices.data());

            dout(1) << "Configured " + std::to_string(gpu_devices.size() - 1) + " devices for multi-GPU" << std::endl;

            if (pipeline_parallel > 1) {
                dout(1) << "Pipeline parallelism: PP=" + std::to_string(pipeline_parallel) + " GPUs with ROW split mode (tensor + layer splitting)" << std::endl;
            } else if (num_gpus_for_splitting == 0) {
                dout(1) << "Multi-GPU: AUTO (using all available GPUs with ROW split mode)" << std::endl;
            } else {
                dout(1) << "Tensor parallelism: TP=" + std::to_string(tensor_parallel) + " GPUs with ROW split mode (consider using --pp for clarity)" << std::endl;
            }
        } else {
            // No explicit TP/PP specified - auto-detect GPUs
            size_t dev_count = ggml_backend_dev_count();
            int cuda_gpu_count = 0;
            for (size_t i = 0; i < dev_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (dev) {
                    const char* dev_name = ggml_backend_dev_name(dev);
                    if (dev_name && strstr(dev_name, "CUDA") != nullptr) {
                        cuda_gpu_count++;
                    }
                }
            }

            if (cuda_gpu_count > 1) {
                // Multiple GPUs detected - auto-split layers across them
                model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
                model_params.main_gpu = 0;

                // Configure tensor_split array for equal distribution
                tensor_split.resize(128, 0.0f);
                for (int i = 0; i < cuda_gpu_count && i < 128; i++) {
                    tensor_split[i] = 1.0f;
                }
                model_params.tensor_split = tensor_split.data();

                // Configure devices array
                gpu_devices.clear();
                for (size_t i = 0; i < dev_count && (int)gpu_devices.size() < cuda_gpu_count; i++) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    if (dev) {
                        const char* dev_name = ggml_backend_dev_name(dev);
                        if (dev_name && strstr(dev_name, "CUDA") != nullptr) {
                            gpu_devices.push_back(static_cast<void*>(dev));
                        }
                    }
                }
                gpu_devices.push_back(nullptr);
                model_params.devices = reinterpret_cast<ggml_backend_dev_t*>(gpu_devices.data());

                dout(1) << "Auto-detected " + std::to_string(cuda_gpu_count) + " GPUs, using layer split mode" << std::endl;
            } else {
                // Single GPU only, no splitting
                model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
                model_params.main_gpu = 0;
                model_params.tensor_split = nullptr;
                dout(1) << "Single GPU mode (no splitting)" << std::endl;
            }
        }

        dout(1) << "GPU detected, offloading layers to GPU (n_gpu_layers=" + std::to_string(model_params.n_gpu_layers) + ")" << std::endl;
    } else {
        model_params.n_gpu_layers = 0;
        dout(1) << "GPU not available, using CPU only" << std::endl;
    }

    model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        throw BackendError("Failed to load model: " + model_path);
    }

    // Log actual layer offload count
    if (has_gpu) {
        int32_t n_layers = llama_model_n_layer(static_cast<llama_model*>(model));
        dout(1) << "Model has " + std::to_string(n_layers) + " layers, all offloaded to GPU" << std::endl;
    }

    dout(1) << "LlamaCpp model loaded successfully" << std::endl;

    // If user didn't specify context size (0), get it from the model
    if (context_size == 0) {
        int32_t model_context_size = llama_model_n_ctx_train(static_cast<llama_model*>(model));
        if (model_context_size <= 0) {
            std::cerr << "No context size specified and model has none defined" << std::endl;
            throw BackendError("Cannot determine context size: model provides no context size and none was specified");
        }
        context_size = static_cast<size_t>(model_context_size);
        dout(1) << "Using model's full context size: " + std::to_string(context_size) << std::endl;
    }
    // else: use user's specified context_size as-is

    dout(1) << "Using context size: " + std::to_string(context_size) + " tokens" << std::endl;

    // Determine optimal batch size (only if not set via --ubatch)
    if (n_batch == 512) {  // Default value means not overridden
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
        dout(1) << "Using n_batch = " + std::to_string(n_batch) + " (auto, GPU: " + (has_gpu ? "yes" : "no") + ")" << std::endl;
    } else {
        dout(1) << "Using n_batch = " + std::to_string(n_batch) + " (from config)" << std::endl;
    }

    // Create llama context with KV space callback
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(context_size);
    ctx_params.n_batch = static_cast<uint32_t>(n_batch);
    // n_ubatch must be <= n_batch
    ctx_params.n_ubatch = static_cast<uint32_t>(std::min(n_ubatch, n_batch));
    ctx_params.offload_kqv = has_gpu;
    dout(1) << "Using n_ubatch = " + std::to_string(ctx_params.n_ubatch) << std::endl;

    // Set KV cache type (both K and V use same type)
    // Valid values: f16, f32, q8_0, q4_0
    if (cache_type == "f32") {
        ctx_params.type_k = GGML_TYPE_F32;
        ctx_params.type_v = GGML_TYPE_F32;
        dout(1) << "KV cache type: f32" << std::endl;
    } else if (cache_type == "q8_0") {
        ctx_params.type_k = GGML_TYPE_Q8_0;
        ctx_params.type_v = GGML_TYPE_Q8_0;
        dout(1) << "KV cache type: q8_0" << std::endl;
    } else if (cache_type == "q4_0") {
        ctx_params.type_k = GGML_TYPE_Q4_0;
        ctx_params.type_v = GGML_TYPE_Q4_0;
        dout(1) << "KV cache type: q4_0" << std::endl;
    } else {
        // Default to f16 (best for Ampere GPUs like RTX 3090)
        ctx_params.type_k = GGML_TYPE_F16;
        ctx_params.type_v = GGML_TYPE_F16;
        dout(1) << "KV cache type: f16 (default)" << std::endl;
    }

    // Set KV cache space callback for interrupt-driven eviction
    ctx_params.kv_need_space_callback = [](uint32_t tokens_needed, void* user_data) -> uint32_t {
        auto* backend = static_cast<LlamaCppBackend*>(user_data);
        uint32_t new_head = backend->evict_to_free_space(tokens_needed);
        if (new_head == UINT32_MAX) {
            dout(1) << "Eviction succeeded, retrying operation with freed space" << std::endl;
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
        dout(1) << std::string("WARNING: ") +"Failed to initialize chat templates - tool support may be limited" << std::endl;
    } else {
        dout(1) << "Chat templates initialized successfully" << std::endl;
    }

    // Get chat template from model (no custom template support for now)
    const char* template_ptr = llama_model_chat_template(static_cast<llama_model*>(model), nullptr);
    const char* tool_use_template_ptr = llama_model_chat_template(static_cast<llama_model*>(model), "tool_use");

    if (template_ptr) {
        chat_template_text = std::string(template_ptr);
        dout(1) << "Retrieved default chat template from model (" + std::to_string(chat_template_text.length()) + " characters)" << std::endl;

        // Dump template to file for inspection
        std::ofstream template_file("/tmp/shepherd_chat_template.jinja");
        template_file << chat_template_text;
        template_file.close();
        dout(1) << "Chat template saved to /tmp/shepherd_chat_template.jinja for inspection" << std::endl;
    }

    if (tool_use_template_ptr) {
        std::string tool_use_template = std::string(tool_use_template_ptr);
        dout(1) << "Retrieved tool_use chat template from model (" + std::to_string(tool_use_template.length()) + " characters)" << std::endl;

        // Use tool_use template if it has python_tag support
        if (tool_use_template.find("<|python_tag|>") != std::string::npos) {
            chat_template_text = tool_use_template;
            dout(1) << "Using tool_use template variant for tool calling support" << std::endl;
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
        dout(1) << "Successfully parsed chat template with minja" << std::endl;
    } catch (const std::exception& e) {
        throw BackendError("Failed to parse chat template with minja: " + std::string(e.what()));
    }

    // Detect model family - priority: chat template -> config.json -> path
    model_config = Models::detect_from_chat_template(chat_template_text, model_path);
    if (model_config.family == ModelFamily::GENERIC) {
        // Try config.json in model directory
        std::filesystem::path model_dir_path = std::filesystem::path(model_path).parent_path();
        model_config = Models::detect_from_config_file(model_dir_path.string());
    }
    if (model_config.family == ModelFamily::GENERIC) {
        // Last resort: path-based detection
        model_config = Models::detect_from_model_path(model_path);
    }
    max_output_tokens = model_config.max_output_tokens;
    dout(1) << "Model configuration: family=" + std::to_string(static_cast<int>(model_config.family)) +
             ", version=" + model_config.version +
             ", tool_result_role=" + model_config.tool_result_role +
             ", uses_eom_token=" + (model_config.uses_eom_token ? "true" : "false") +
             ", uses_python_tag=" + (model_config.uses_python_tag ? "true" : "false") << std::endl;

    // Get EOS/BOS tokens from model vocabulary for ChatTemplate
    std::string eos_token_str, bos_token_str;
    {
        llama_model* mdl = static_cast<llama_model*>(model);
        const llama_vocab* vocab = llama_model_get_vocab(mdl);

        // Get EOS token
        llama_token eos_id = llama_vocab_eos(vocab);
        if (eos_id != LLAMA_TOKEN_NULL) {
            char buf[64];
            int len = llama_token_to_piece(vocab, eos_id, buf, sizeof(buf), 0, false);
            if (len > 0) {
                eos_token_str = std::string(buf, len);
                dout(1) << "EOS token: id=" << eos_id << " text=[" << eos_token_str << "]" << std::endl;
            }
        }

        // Get BOS token
        llama_token bos_id = llama_vocab_bos(vocab);
        if (bos_id != LLAMA_TOKEN_NULL) {
            char buf[64];
            int len = llama_token_to_piece(vocab, bos_id, buf, sizeof(buf), 0, false);
            if (len > 0) {
                bos_token_str = std::string(buf, len);
                dout(1) << "BOS token: id=" << bos_id << " text=[" << bos_token_str << "]" << std::endl;
            }
        }
    }

    // Create chat template instance with EOS/BOS tokens
    chat_template = ChatTemplates::ChatTemplateFactory::create(chat_template_text, model_config, template_node, eos_token_str, bos_token_str);
    dout(1) << "Created chat template for family: " + std::to_string(static_cast<int>(model_config.family)) << std::endl;

    // Probe template capabilities (discovers what features the template supports)
    if (chat_template) {
        chat_template->probe_capabilities();
        const auto& caps = chat_template->get_capabilities();
        dout(1) << "Template capabilities: system=" + std::string(caps.supports_system_role ? "yes" : "no") +
                   ", tools=" + std::string(caps.supports_tools ? "yes" : "no") +
                   ", tool_calls=" + std::string(caps.supports_tool_calls ? "yes" : "no") +
                   ", tool_responses=" + std::string(caps.supports_tool_responses ? "yes" : "no") +
                   ", channels=" + std::string(caps.has_channels ? "yes" : "no") << std::endl;
    }

    // Try to load sampling parameters from generation_config.json
    // Priority: config file values > generation_config.json > hardcoded defaults
    std::filesystem::path model_file_path(model_path);
    std::filesystem::path model_dir_path2 = model_file_path.parent_path();

    float gen_temperature = temperature;  // Start with current value
    float gen_top_p = top_p;
    int gen_top_k = top_k;

    if (Models::load_generation_config(model_dir_path2.string(), gen_temperature, gen_top_p, gen_top_k)) {
        // Only apply values that weren't explicitly set in config file
        if (!temperature_from_config) temperature = gen_temperature;
        if (!top_p_from_config) top_p = gen_top_p;
        if (!top_k_from_config) top_k = gen_top_k;
    }

    dout(1) << "LlamaCppBackend initialized with model: " + model_path << std::endl;
    initialized = true;

    // Add system message in cli mode
    if (!g_server_mode) {
        // Format system message with tools using chat template
        std::string formatted_system = chat_template->format_system_message(session.system_message, session.tools);

        // Use add_message to properly decode and add system message
        // This ensures it's in KV cache before being added to session.messages
        add_message(session, Message::SYSTEM, formatted_system, "", "", 0);

        // Update session tracking (add_message already added to session.messages)
        session.system_message_tokens = count_tokens_in_text(formatted_system);

        dout(1) << "Added system message to session" << std::endl;
    }
#else
    throw BackendError("LlamaCpp backend not compiled in");
#endif
}

void LlamaCppBackend::parse_backend_config() {
    if (config->json.is_null() || config->json.empty()) {
        return;  // No config, use defaults
    }

    try {
        if (config->json.contains("temperature")) {
            temperature = config->json["temperature"].get<float>();
            temperature_from_config = true;
        }
        if (config->json.contains("top_p")) {
            top_p = config->json["top_p"].get<float>();
            top_p_from_config = true;
        }
        if (config->json.contains("top_k")) {
            top_k = config->json["top_k"].get<int>();
            top_k_from_config = true;
        }
        if (config->json.contains("min_keep")) min_keep = config->json["min_keep"].get<int>();
        if (config->json.contains("penalty_repeat")) penalty_repeat = config->json["penalty_repeat"].get<float>();
        if (config->json.contains("penalty_freq")) penalty_freq = config->json["penalty_freq"].get<float>();
        if (config->json.contains("penalty_present")) penalty_present = config->json["penalty_present"].get<float>();
        if (config->json.contains("penalty_last_n")) penalty_last_n = config->json["penalty_last_n"].get<int>();
        if (config->json.contains("gpu_layers")) gpu_layers = config->json["gpu_layers"].get<int>();
        if (config->json.contains("context_size")) context_size = config->json["context_size"].get<size_t>();

        // Accept both full names and short names for tensor/pipeline parallel
        if (config->json.contains("tensor_parallel")) tensor_parallel = config->json["tensor_parallel"].get<int>();
        else if (config->json.contains("tp")) tensor_parallel = config->json["tp"].get<int>();

        if (config->json.contains("pipeline_parallel")) pipeline_parallel = config->json["pipeline_parallel"].get<int>();
        else if (config->json.contains("pp")) pipeline_parallel = config->json["pp"].get<int>();

        // Batch sizes: n_batch = logical, ubatch/n_ubatch = physical micro-batch
        if (config->json.contains("n_batch")) n_batch = config->json["n_batch"].get<int>();
        if (config->json.contains("ubatch")) n_ubatch = config->json["ubatch"].get<int>();
        else if (config->json.contains("n_ubatch")) n_ubatch = config->json["n_ubatch"].get<int>();

        // KV cache type
        if (config->json.contains("cache_type")) cache_type = config->json["cache_type"].get<std::string>();

        dout(1) << "Loaded llamacpp backend config: temperature=" + std::to_string(temperature) +
                  ", gpu_layers=" + std::to_string(gpu_layers) +
                  ", tensor_parallel=" + std::to_string(tensor_parallel) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to parse llamacpp backend config: " + std::string(e.what()) << std::endl;
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

int LlamaCppBackend::count_message_tokens(Message::Role role, const std::string& content, const std::string& tool_name, const std::string& tool_id) {
#ifdef ENABLE_LLAMACPP
    if (!model || !template_node) {
        // Fallback to simple text tokenization if model not initialized
        return count_tokens_in_text(content);
    }

    // Tool responses require conversation context to render through template
    // (template needs preceding assistant's tool_calls). Use simple tokenization.
    if (role == Message::TOOL_RESPONSE || role == Message::FUNCTION) {
        return count_tokens_in_text(content);
    }

    // Format message through minja template to get exact token count
    auto template_ptr = static_cast<std::shared_ptr<minja::TemplateNode>*>(template_node);
    auto context = minja::Context::builtins();

    // Add strftime_now for templates that use it (e.g., GPT-OSS)
    auto strftime_now = minja::Value::callable([](const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) -> minja::Value {
        std::string format = args.args.empty() ? "%Y-%m-%d" : args.args[0].get<std::string>();
        std::time_t now = std::time(nullptr);
        std::tm* tm_info = std::localtime(&now);
        char buffer[128];
        strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
        return minja::Value(std::string(buffer));
    });
    context->set("strftime_now", strftime_now);

    // Build messages array with just this message
    auto messages = minja::Value::array();
    auto msg_obj = minja::Value::object();

    // Convert Message::Role to role string
    std::string role_str;
    switch (role) {
        case Message::SYSTEM:
            role_str = "system";
            break;
        case Message::USER:
            role_str = "user";
            break;
        case Message::ASSISTANT:
            role_str = "assistant";
            break;
        case Message::TOOL_RESPONSE:
        case Message::FUNCTION:
            role_str = "tool";
            break;
        default:
            role_str = "user";
            break;
    }

    msg_obj.set("role", minja::Value(role_str));
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
        dout(1) << std::string("WARNING: ") +"Exception rendering message through template: " + std::string(e.what()) << std::endl;
        // Fallback to simple tokenization
        return count_tokens_in_text(content);
    }

    // Tokenize the rendered message to get exact count
    const llama_vocab* vocab = llama_model_get_vocab(static_cast<llama_model*>(model));
    std::vector<llama_token> tokens(rendered.length() + 256);
    int n_tokens = llama_tokenize(vocab, rendered.c_str(), rendered.length(),
                                   tokens.data(), tokens.size(), false, true);

    if (n_tokens < 0) {
        dout(1) << std::string("WARNING: ") +"Failed to tokenize rendered message" << std::endl;
        return count_tokens_in_text(content);
    }

    dout(1) << "count_message_tokens: role=" + std::to_string(static_cast<int>(role)) +
              ", content_len=" + std::to_string(content.length()) +
              ", rendered_len=" + std::to_string(rendered.length()) +
              ", tokens=" + std::to_string(n_tokens) << std::endl;

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

std::vector<std::string> LlamaCppBackend::get_tool_call_end_markers() const {
    return model_config.tool_call_end_markers;
}

std::vector<std::string> LlamaCppBackend::get_thinking_start_markers() const {
    return model_config.thinking_start_markers;
}

std::vector<std::string> LlamaCppBackend::get_thinking_end_markers() const {
    return model_config.thinking_end_markers;
}

const ChatTemplates::ChatTemplateCaps* LlamaCppBackend::get_chat_template_caps() const {
    if (chat_template) {
        return &chat_template->get_capabilities();
    }
    return nullptr;
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
    dout(1) << "KV cache full - need to free " + std::to_string(tokens_needed) + " tokens" << std::endl;

    if (!current_session) {
        if (callback) callback(CallbackEvent::ERROR, "No current session - cannot evict", "", "");
        return UINT32_MAX;
    }

    // Debug: Check KV cache actual usage
    int kv_used = get_context_token_count();
    int cached_tokens = 0;
    for (const auto& msg : current_session->messages) {
        cached_tokens += msg.tokens;
    }

    dout(1) << "KV cache state: used_tokens=" + std::to_string(kv_used) +
              ", max_ctx=" + std::to_string(context_size) +
              ", cached_tokens=" + std::to_string(cached_tokens) +
              ", session_messages=" + std::to_string(current_session->messages.size()) << std::endl;

    // In server mode, signal abort instead of evicting
    // Client is responsible for managing context window
    // NOTE: We cannot throw exceptions here - this callback is called from llama.cpp's C code
    // and C++ exceptions cannot propagate through C stack frames (undefined behavior)
    if (g_server_mode) {
        dout(1) << "KV cache full in server mode - signaling abort (context_full)" << std::endl;
        // Return 0 to signal abort - llama_decode will fail and we handle it at the C++ level
        context_full_in_server_mode = true;
        context_full_tokens_needed = kv_used + tokens_needed;
        return 0;  // Signal abort
    }

    // Get KV cache memory handle for eviction operations
    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_memory_t mem = llama_get_memory(ctx);

    dout(1) << "Found " + std::to_string(current_session->messages.size()) + " messages in KV cache" << std::endl;

    // Use current_session for eviction calculation
    // Cast away const for eviction operation
    Session* mutable_session = const_cast<Session*>(current_session);
    auto ranges = mutable_session->calculate_messages_to_evict(tokens_needed);

    if (ranges.empty()) {
        if (callback) callback(CallbackEvent::ERROR, "Cannot evict - no space can be freed", "", "");
        return 0;  // Signal failure: eviction cannot proceed
    }

    // Step 1: Calculate all KV positions for all ranges BEFORE doing any evictions
    // This is critical because evicting/shifting changes the KV cache state
    struct RangeInfo {
        int start_msg, end_msg;
        int start_pos, end_pos;
        int tokens;
    };
    std::vector<RangeInfo> range_infos;

    int total_tokens_removed = 0;
    int total_messages_evicted = 0;

    for (const auto& [start_msg, end_msg] : ranges) {
        // Calculate token positions for this range
        int start_pos = 0;
        for (int i = 0; i < start_msg; i++) {
            start_pos += current_session->messages[i].tokens;
        }

        int end_pos = start_pos;
        for (int i = start_msg; i <= end_msg; i++) {
            end_pos += current_session->messages[i].tokens;
        }
        end_pos--; // end_pos is inclusive

        int tokens_in_range = end_pos - start_pos + 1;

        range_infos.push_back({start_msg, end_msg, start_pos, end_pos, tokens_in_range});
        total_tokens_removed += tokens_in_range;
        total_messages_evicted += (end_msg - start_msg + 1);

        dout(3) << "EVICT range: messages[" << start_msg << "," << end_msg
                << "] = KV positions[" << start_pos << "," << end_pos << "]" << std::endl;

        dout(1) << "Evicting messages [" + std::to_string(start_msg) + ", " + std::to_string(end_msg) +
                 "] = tokens [" + std::to_string(start_pos) + ", " + std::to_string(end_pos) + "]" << std::endl;
    }

    // Step 2: Evict all ranges from KV cache in reverse order
    // Process in reverse so earlier positions don't shift before we evict later ones
    for (auto it = range_infos.rbegin(); it != range_infos.rend(); ++it) {
        llama_memory_seq_rm(mem, 0, it->start_pos, it->end_pos + 1); // +1 because llama uses exclusive end
        dout(1) << "Removed KV range [" + std::to_string(it->start_pos) + ", " + std::to_string(it->end_pos) + "]" << std::endl;
    }

    // Step 3: Shift remaining tokens down to keep positions contiguous
    // Process ranges in reverse order, shifting after each eviction
    for (auto it = range_infos.rbegin(); it != range_infos.rend(); ++it) {
        llama_memory_seq_add(mem, 0, it->end_pos + 1, -1, -it->tokens);
        dout(1) << "Shifted KV cache positions >= " + std::to_string(it->end_pos + 1) + " down by " + std::to_string(it->tokens) << std::endl;
    }

    // Step 4: Archive to RAG and remove messages from session (all ranges at once)
    if (!mutable_session->evict_messages(ranges)) {
        if (callback) callback(CallbackEvent::ERROR, "Failed to evict messages from session", "", "");
        return UINT32_MAX;
    }

    dout(1) << "Evicted " + std::to_string(total_messages_evicted) + " messages from cache" << std::endl;
    dout(1) << "Successfully evicted " + std::to_string(total_messages_evicted) + " messages (" +
             std::to_string(total_tokens_removed) + " tokens) from KV cache" << std::endl;

    // KV cache is the source of truth - query actual state
    dout(2) << "KV cache after eviction: " << get_context_token_count() << " tokens" << std::endl;
    log_token_state("After eviction");

#ifdef TEST_EVICTION_VALIDATION
    // Step 5: Optional validation - verify messages remaining in session match KV cache
    dout(1) << "Validating eviction: checking message/KV alignment" << std::endl;
    int expected_kv_tokens = 0;
    for (const auto& msg : current_session->messages) {
        expected_kv_tokens += msg.tokens;
    }
    int actual_kv_tokens = get_context_token_count();
    if (expected_kv_tokens != actual_kv_tokens) {
        if (callback) callback(CallbackEvent::ERROR, "Eviction validation FAILED: expected " +
                  std::to_string(expected_kv_tokens) + " tokens in KV cache but found " +
                  std::to_string(actual_kv_tokens), "", "");
    } else {
        dout(1) << "Eviction validation PASSED: " + std::to_string(actual_kv_tokens) + " tokens match" << std::endl;
    }
#endif

    // Return the new head position (where first freed space begins) as required by callback API
    // Calculate position of first range
    int first_range_pos = 0;
    if (!ranges.empty()) {
        auto [start_msg, end_msg] = ranges[0];
        for (int i = 0; i < start_msg; i++) {
            first_range_pos += current_session->messages[i].tokens;
        }
    }
    dout(1) << "Eviction complete - returning new head position: " + std::to_string(first_range_pos) << std::endl;
    return static_cast<uint32_t>(first_range_pos);
#else
    std::cerr << "llama.cpp not enabled" << std::endl;
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
    dout(1) << "LlamaCppBackend shutdown complete" << std::endl;
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

std::string LlamaCppBackend::generate(int max_tokens, EventCallback callback) {
    dout(1) << "=== GENERATE START ===" << std::endl;
    if (!is_ready()) {
        throw std::runtime_error("LlamaCpp backend not initialized");
    }

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
                dout(1) << "Extracted " + std::to_string(params.preserved_tokens.size()) + " preserved tokens from template:" << std::endl;

                // Separate start and end markers
                std::vector<std::string> start_markers;
                std::vector<std::string> end_markers;

                for (const auto& marker : params.preserved_tokens) {
                    dout(1) << "  - " + marker << std::endl;

                    if (marker.size() >= 3 && marker[0] == '<' && marker[1] == '/') {
                        // This is a closing tag
                        end_markers.push_back(marker);
                    } else if (marker.size() >= 2 && marker[0] == '<') {
                        // This is an opening tag or self-contained tag
                        start_markers.push_back(marker);
                    }
                }

                tool_call_markers = start_markers;
                model_config.tool_call_start_markers = start_markers;
                model_config.tool_call_end_markers = end_markers;

                dout(1) << "Separated into " + std::to_string(start_markers.size()) + " start markers and " +
                        std::to_string(end_markers.size()) + " end markers" << std::endl;
            } else {
                dout(1) << "No preserved_tokens from template - checking vocabulary" << std::endl;

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
                    dout(1) << "Found <|python_tag|> in model vocabulary (token " + std::to_string(tokens[0]) + ")" << std::endl;
                } else {
                    dout(1) << "Model does not have <|python_tag|> special token (tokenized to " + std::to_string(n_tokens) + " tokens)" << std::endl;
                }
            }

            have_tool_markers = true;
        } catch (const std::exception& e) {
            dout(1) << "Exception during tool marker extraction: " + std::string(e.what()) << std::endl;
            have_tool_markers = true;
        }

        // If no tool markers were found, add common fallback markers
        if (tool_call_markers.empty()) {
            dout(1) << "No tool markers found from template, adding fallback markers" << std::endl;
            tool_call_markers.push_back("<tool_call");
            tool_call_markers.push_back("<function_call");
            model_config.tool_call_start_markers = tool_call_markers;
            model_config.tool_call_end_markers = {"</tool_call>", "</function_call>"};
        }
    }

    // All messages are already decoded in KV cache from add_*_message() calls
    // We just need to run the generation loop now

#ifdef _DEBUG
    // In debug mode, show what's in the KV cache
    if (g_debug_level && current_session) {
        dout(1) << "=== MESSAGES IN KV CACHE ===" << std::endl;
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
            dout(1) << line << std::endl;
        }
        dout(1) << "=== END KV CACHE ===" << std::endl;
    }
#endif

    dout(1) << "Running generation (messages already cached)" << std::endl;

    // Before generation, we need to add the generation prompt to KV cache
    // Use chat template to get the generation prompt
    std::string generation_prompt = chat_template->get_generation_prompt();

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
            dout(1) << "Decoding generation prompt (" + generation_prompt + "): " + std::to_string(n_tokens) + " tokens" << std::endl;

            for (size_t i = 0; i < prompt_tokens.size(); i += n_batch) {
                int batch_size = std::min(n_batch, static_cast<int>(prompt_tokens.size() - i));
                llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

                if (llama_decode(ctx, batch) != 0) {
                    if (callback) callback(CallbackEvent::ERROR, "Failed to decode generation prompt at position " + std::to_string(i), "", "");
                }
            }
        }
    }

    // Suppress streaming in server mode (output goes to server.log)
    // Tools are OK with streaming - the terminal filter will hide <tool_call> tags
    bool suppress_stream = g_server_mode;
    std::string raw_response = run_inference("", max_tokens, suppress_stream, callback);  // Empty prompt since everything is cached
    dout(1) << "Got raw response length: " + std::to_string(raw_response.length()) << std::endl;
    // Log first 500 chars to avoid terminal truncation issues
    std::string debug_content = raw_response.length() > 500 ? raw_response.substr(0, 500) + "..." : raw_response;
    dout(1) << "Raw response content: " + debug_content << std::endl;

    // Always write full response to debug file for inspection
    {
        std::ofstream debug_file("/tmp/shepherd_response_debug.txt", std::ios::app);
        debug_file << "=== Response at " << std::time(nullptr) << " ===\n";
        debug_file << raw_response << "\n";
        debug_file << "=== End (length: " << raw_response.length() << ") ===\n\n";
        debug_file.close();
    }

    // Content extraction (for channel-based models like GPT-OSS)
    // Uses ChatTemplate's extract_content() which detects format from template
    if (chat_template) {
        std::string extracted = chat_template->extract_content(raw_response);
        if (extracted.length() != raw_response.length()) {
            dout(1) << "Content extraction: " + std::to_string(raw_response.length()) +
                      " -> " + std::to_string(extracted.length()) + " chars" << std::endl;
            raw_response = extracted;
        }
    }

    // CRITICAL FIX: After generation, add the closing tag to KV cache
    // The generation prompt added assistant_start_tag, and run_inference() generated tokens,
    // but we never added the assistant_end_tag closing tag! This causes malformed context on next generate()
    int n_closing = 0;
    std::string assistant_end_tag = chat_template->get_assistant_end_tag();
    if (!assistant_end_tag.empty()) {
        // Tokenize and add the closing tag to KV cache
        const llama_vocab* vocab = llama_model_get_vocab(mdl);
        std::vector<llama_token> closing_tokens(16);
        n_closing = llama_tokenize(vocab, assistant_end_tag.c_str(),
                                    assistant_end_tag.length(),
                                    closing_tokens.data(), closing_tokens.size(), false, true);

        if (n_closing > 0) {
            closing_tokens.resize(n_closing);
            llama_batch closing_batch = llama_batch_get_one(closing_tokens.data(), n_closing);
            if (llama_decode(ctx, closing_batch) != 0) {
                dout(1) << std::string("WARNING: ") +"Failed to decode closing tag into KV cache" << std::endl;
            } else {
                dout(1) << "Added closing tag to KV cache: " + assistant_end_tag << std::endl;
            }
        }
    }

    // Store prompt token count for server to return (like API backends do)
    // This is the total number of tokens in the KV cache before generation
    // Calculate total tokens from current session messages
    // Store the actual tokens added to KV cache for this assistant message
    // This includes: generation_prompt + generated_tokens + closing_tag
    // Note: last_completion_tokens is set by run_inference() to n_generated
    last_assistant_kv_tokens = n_tokens + last_completion_tokens + n_closing;

    // Return response directly - main will handle tool parsing and cleanup
    return raw_response;
}

std::string LlamaCppBackend::run_inference(const std::string& prompt_text, int max_tokens, bool suppress_streaming, EventCallback callback) {
#ifdef ENABLE_LLAMACPP
    // Reset cancellation flag at start of generation
    g_generation_cancelled = false;

    reset_output_state();

    // Note: TerminalIO reset removed

    if (!model || !model_ctx) {
        if (callback) callback(CallbackEvent::ERROR, "llama.cpp model or context not initialized", "", "");
        return "Error: Model not initialized";
    }

    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_model* mdl = static_cast<llama_model*>(this->model);
    const llama_vocab* vocab = llama_model_get_vocab(mdl);

    // Compute effective sampling parameters - use session overrides if set, otherwise backend defaults
    // This mirrors TensorRT's approach for per-request parameter support
    float use_temperature = (current_session && current_session->temperature >= 0.0f) ? current_session->temperature : temperature;
    float use_top_p = (current_session && current_session->top_p >= 0.0f) ? current_session->top_p : top_p;
    int use_top_k = (current_session && current_session->top_k >= 0) ? current_session->top_k : top_k;
    float use_penalty_repeat = (current_session && current_session->repetition_penalty >= 0.0f) ? current_session->repetition_penalty : penalty_repeat;
    float use_penalty_freq = (current_session && current_session->frequency_penalty > -999.0f) ? current_session->frequency_penalty : penalty_freq;
    float use_penalty_present = (current_session && current_session->presence_penalty > -999.0f) ? current_session->presence_penalty : penalty_present;

    // Initialize sampler chain with effective sampling parameters
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Add samplers in the recommended order (from llama.cpp examples)
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(use_top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(use_top_p, min_keep));

    // Add repetition penalties BEFORE temperature to discourage repetitive patterns
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        penalty_last_n,        // last n tokens to penalize
        use_penalty_repeat,    // repetition penalty (1.0 = disabled)
        use_penalty_freq,      // frequency penalty (0.0 = disabled)
        use_penalty_present)); // presence penalty (0.0 = disabled)

    llama_sampler_chain_add(sampler, llama_sampler_init_temp(use_temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));           // greedy sampling from dist

    dout(1) << "Sampling params: temperature=" + std::to_string(use_temperature) +
              ", top_p=" + std::to_string(use_top_p) +
              ", top_k=" + std::to_string(use_top_k) +
              ", min_keep=" + std::to_string(min_keep) +
              ", penalty_repeat=" + std::to_string(use_penalty_repeat) +
              ", penalty_freq=" + std::to_string(use_penalty_freq) +
              ", penalty_present=" + std::to_string(use_penalty_present) +
              ", penalty_last_n=" + std::to_string(penalty_last_n) << std::endl;

    // If prompt_text is empty, everything is already in KV cache
    // Just skip straight to generation
    if (!prompt_text.empty()) {
        // Legacy path: tokenize and decode prompt
        // This is kept for backward compatibility but shouldn't be used with stateful KV cache
        dout(1) << std::string("WARNING: ") +"run_inference() called with non-empty prompt - this is wasteful with stateful KV cache" << std::endl;

        std::vector<llama_token> prompt_tokens(prompt_text.length() + 256);
        int n_prompt_tokens = llama_tokenize(vocab, prompt_text.c_str(), prompt_text.length(),
                                             prompt_tokens.data(), prompt_tokens.size(), false, true);

        if (n_prompt_tokens < 0) {
            if (callback) callback(CallbackEvent::ERROR, "Failed to tokenize input text", "", "");
            llama_sampler_free(sampler);
            return "Error: Tokenization failed";
        }

        prompt_tokens.resize(n_prompt_tokens);
        dout(1) << "Evaluating " + std::to_string(n_prompt_tokens) + " prompt tokens" << std::endl;

        // Check if prompt alone is larger than entire context window
        if (n_prompt_tokens > static_cast<int>(context_size)) {
            if (callback) callback(CallbackEvent::ERROR, "Prompt too large for context: " +
                      std::to_string(n_prompt_tokens) + " tokens exceeds max " + std::to_string(context_size), "", "");
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
                dout(1) << "Generation cancelled during prompt processing" << std::endl;
                llama_sampler_free(sampler);
                return "";
            }

            int batch_size = std::min(n_batch, static_cast<int>(prompt_tokens.size() - i));

            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch) != 0) {
                if (callback) callback(CallbackEvent::ERROR, "Failed to evaluate prompt tokens at position " + std::to_string(i), "", "");
                llama_sampler_free(sampler);
                return "Error: Evaluation failed";
            }
        }
    } else {
        dout(1) << "Prompt already cached, skipping tokenization/decoding" << std::endl;
    }

    // Generate tokens
    std::string response;
    int n_generated = 0;

    // Note: TerminalIO marker update removed
    bool cancelled_by_escape = false;

    // Calculate max generation tokens: context_size - system - current_user
    // These token counts include ALL template overhead (saved when messages were decoded)
    int available_for_generation = static_cast<int>(context_size) - system_formatted_tokens - current_user_formatted_tokens;

    // Use explicit max_tokens if provided, otherwise use all available space
    int max_gen_tokens = max_tokens > 0 ? max_tokens : available_for_generation;

    dout(1) << "Generation limits: available=" + std::to_string(available_for_generation) +
              " (context=" + std::to_string(context_size) +
              " - system=" + std::to_string(system_formatted_tokens) +
              " - user=" + std::to_string(current_user_formatted_tokens) +
              "), max_gen_tokens=" + std::to_string(max_gen_tokens) << std::endl;

    // Start timing for t/s measurement
    auto gen_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_gen_tokens; i++) {
        // Check for cancellation (from SIGUSR1 signal)
        if (g_generation_cancelled) {
            dout(1) << "Generation cancelled by signal" << std::endl;
            break;
        }

        // TODO: Escape key cancellation removed with TerminalIO

        // Sample next token
        llama_token next_token = llama_sampler_sample(sampler, ctx, -1);

        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        dout(3) << "Sampled token: " << next_token << " (vocab size: " << n_vocab << ")" << std::endl;
        if (next_token < 0 || next_token >= n_vocab) {
            if (callback) callback(CallbackEvent::ERROR, "Invalid token sampled: " + std::to_string(next_token) + " (vocab size: " + std::to_string(n_vocab) + ")", "", "");
            break;
        }

        // Check for end of generation using llama.cpp's native EOG detection
        // This handles all model-specific end tokens automatically
        if (llama_vocab_is_eog(vocab, next_token)) {
            dout(1) << "End of generation token detected" << std::endl;
            break;
        }

        // Accept the token
        llama_sampler_accept(sampler, next_token);

        // Convert token to text (filter special tokens with false parameter)
        char token_str[256];
        int token_len = llama_token_to_piece(vocab, next_token, token_str, sizeof(token_str), 0, false);
        dout(3) << "Token " << next_token << " -> len=" << token_len << std::endl;

        if (token_len > 0) {
            // Accumulate for final response
            response.append(token_str, token_len);

            // Route through unified output filter (includes channel parsing for GPT-OSS)
            std::string delta(token_str, token_len);
            if (!process_output(delta)) {
                // Callback returned false or channel parser signaled stop
                break;
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
                dout(1) << "Token decode failed (likely eviction), retrying" << std::endl;
            }
        }

        if (!decode_ok) {
            if (callback) callback(CallbackEvent::ERROR, "Failed to evaluate generated token after retries", "", "");
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

    // Send performance stats via callback (frontend decides whether to display)
    if (n_generated > 0 && callback) {
        int context_used = get_context_token_count();
        int context_max = context_size;

        std::ostringstream stats;
        stats << "\n[Decode: " << n_generated << " tokens, "
              << std::fixed << std::setprecision(1) << tokens_per_second << " t/s, "
              << "context: " << context_used << "/" << context_max << "]\n";
        callback(CallbackEvent::STATS, stats.str(), "", "");
    }

    dout(1) << "Generation (decode): " + std::to_string(n_generated) + " tokens in " +
             std::to_string(seconds) + "s (" + std::to_string(tokens_per_second) + " t/s" << std::endl;


    // Store completion token count for server to return (like API backends do)
    last_completion_tokens = n_generated;

    // Check if we hit the length limit (generated exactly max_gen_tokens)
    last_generation_hit_length_limit = (n_generated == max_gen_tokens);
    if (last_generation_hit_length_limit) {
        dout(1) << "Generation stopped due to max_tokens limit (" + std::to_string(max_gen_tokens) + " tokens)" << std::endl;
    }

    // Set cancellation flag if escape was pressed
    if (cancelled_by_escape) {
        g_generation_cancelled = true;
    }

    // Flush any remaining output from the filter
    flush_output();

    return response;
#else
    // Fallback when llama.cpp not available
    std::cerr << "llama.cpp not compiled in" << std::endl;
    return "Error: llama.cpp not available";
#endif
}





std::map<std::string, std::any> LlamaCppBackend::parse_json_to_args(const std::string& args_str) {
    std::map<std::string, std::any> args;

    dout(1) << "Parsing tool arguments: " + args_str << std::endl;

    // Try JSON parsing first
    if (args_str.find('{') != std::string::npos && args_str.find('}') != std::string::npos) {
        try {
            // Extract "key": "value" patterns from JSON-like format
            std::regex json_regex("\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\""  );
            std::sregex_iterator iter(args_str.begin(), args_str.end(), json_regex);
            std::sregex_iterator end;

            for (; iter != end; ++iter) {
                std::smatch match = *iter;
                args[match[1].str()] = match[2].str();
            }

            if (!args.empty()) {
                dout(1) << "Successfully parsed JSON-style arguments" << std::endl;
                return args;
            }
        } catch (...) {
            dout(1) << "JSON parsing failed, trying other formats" << std::endl;
        }
    }

    // Try simple key=value format
    std::regex kv_regex("(\\w+)\\s*=\\s*\"([^\"]*)\""  );
    std::sregex_iterator iter(args_str.begin(), args_str.end(), kv_regex);
    std::sregex_iterator end;

    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        args[match[1].str()] = match[2].str();
    }

    if (!args.empty()) {
        dout(1) << "Successfully parsed key=value arguments" << std::endl;
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
        dout(1) << "Using fallback argument parsing" << std::endl;
    }

    return args;
}

std::string LlamaCppBackend::render_message(
    const std::vector<Message>& all_messages,
    size_t target_index,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) {
#ifdef ENABLE_LLAMACPP
    if (!chat_template) {
        dout(1) << "No chat template available, falling back to simple format" << std::endl;
        if (target_index < all_messages.size()) {
            return all_messages[target_index].get_role() + ": " + all_messages[target_index].content + "\n\n";
        }
        return "";
    }

    // Use incremental rendering with full conversation context
    try {
        std::string formatted = chat_template->format_message_incremental(
            all_messages, target_index, tools, add_generation_prompt);

        dout(1) << "Rendered message with context (index=" + std::to_string(target_index) +
                   ", " + std::to_string(formatted.length()) + " chars)" << std::endl;
        return formatted;
    } catch (const std::exception& e) {
        // Template rendering failed - fall back to simple format
        // This can happen with templates that have unsupported features
        dout(1) << "Template rendering failed: " + std::string(e.what()) + ", using fallback" << std::endl;
        if (target_index < all_messages.size()) {
            const auto& msg = all_messages[target_index];
            return "<|im_start|>" + msg.get_role() + "\n" + msg.content + "<|im_end|>\n";
        }
        return "";
    }
#else
    if (target_index < all_messages.size()) {
        return all_messages[target_index].get_role() + ": " + all_messages[target_index].content + "\n\n";
    }
    return "";
#endif
}

bool LlamaCppBackend::format_and_decode_message(
    std::vector<Message>& all_messages,
    size_t target_index,
    const std::vector<Session::Tool>& tools,
    bool add_generation_prompt) {
#ifdef ENABLE_LLAMACPP
    if (!model || !model_ctx) {
        if (callback) callback(CallbackEvent::ERROR, "Model or context not initialized", "", "");
        return false;
    }

    if (target_index >= all_messages.size()) {
        if (callback) callback(CallbackEvent::ERROR, "Target index " + std::to_string(target_index) + " out of bounds (size=" + std::to_string(all_messages.size()) + ")", "", "");
        return false;
    }

    const Message& msg = all_messages[target_index];

    // Render message through template with full conversation context
    // Exception: assistant messages without tool_calls use raw content to avoid think tag stripping
    std::string rendered_msg;
    if (msg.get_role() == "assistant" && msg.tool_calls_json.empty()) {
        // Use raw content for regular assistant messages - no template processing
        rendered_msg = msg.content;
    } else {
        // Render through the template with full conversation context
        // This is needed for: user messages, tool responses, and assistant messages with tool_calls
        rendered_msg = render_message(all_messages, target_index, tools, add_generation_prompt);
    }

    llama_context* ctx = static_cast<llama_context*>(model_ctx);
    llama_model* mdl = static_cast<llama_model*>(model);
    const llama_vocab* vocab = llama_model_get_vocab(mdl);

    // Tokenize the rendered message
    std::vector<llama_token> msg_tokens(rendered_msg.length() + 256);
    int n_tokens = llama_tokenize(vocab, rendered_msg.c_str(), rendered_msg.length(),
                                   msg_tokens.data(), msg_tokens.size(), false, true);

    if (n_tokens < 0) {
        if (callback) callback(CallbackEvent::ERROR, "Failed to tokenize message", "", "");
        return false;
    }
    msg_tokens.resize(n_tokens);

    // Update the message's token count - avoids redundant count_message_tokens() call
    all_messages[target_index].tokens = n_tokens;

    dout(1) << "Decoding " + std::to_string(n_tokens) + " tokens for message at index " +
               std::to_string(target_index) << std::endl;

    // Check if message alone is larger than entire context window
    if (n_tokens > static_cast<int>(context_size)) {
        if (callback) callback(CallbackEvent::ERROR, "Message too large for context: " +
                  std::to_string(n_tokens) + " tokens exceeds max " + std::to_string(context_size), "", "");
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
    const int MAX_DECODE_RETRIES = 1;
    int retry_count = 0;

    context_full_in_server_mode = false;
    context_full_tokens_needed = 0;

    while (retry_count <= MAX_DECODE_RETRIES) {
        bool decode_failed = false;

        for (size_t i = 0; i < msg_tokens.size(); i += n_batch) {
            int batch_size = std::min(n_batch, static_cast<int>(msg_tokens.size() - i));

            llama_batch batch = llama_batch_get_one(msg_tokens.data() + i, batch_size);

            if (llama_decode(ctx, batch) != 0) {
                if (context_full_in_server_mode) {
                    throw ContextFullException("This model's maximum context length is " +
                        std::to_string(context_size) + " tokens. However, your messages resulted in " +
                        std::to_string(context_full_tokens_needed) + " tokens.");
                }

                if (retry_count < MAX_DECODE_RETRIES) {
                    dout(1) << "WARNING: Decode failed at position " + std::to_string(i) + ", retrying" << std::endl;
                    decode_failed = true;
                    break;
                } else {
                    if (callback) callback(CallbackEvent::ERROR, "Failed to decode message tokens after " + std::to_string(MAX_DECODE_RETRIES + 1) + " attempts", "", "");
                    return false;
                }
            }
        }

        if (!decode_failed) {
            break;
        }

        retry_count++;
    }

    // End timing
    auto prefill_end_time = std::chrono::high_resolution_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end_time - prefill_start_time);
    double prefill_seconds = prefill_duration.count() / 1000.0;
    double prefill_tokens_per_second = (n_tokens > 0 && prefill_seconds > 0) ? (n_tokens / prefill_seconds) : 0.0;

    dout(2) << "KV cache after decode: " << get_context_token_count() << " tokens" << std::endl;

    // Send performance stats via callback (frontend decides whether to display)
    if (callback) {
        int context_used = get_context_token_count();
        int context_max = context_size;
        std::ostringstream stats;
        stats << "[Prefill: " << n_tokens << " tokens, "
              << std::fixed << std::setprecision(1) << prefill_tokens_per_second << " t/s, "
              << "context: " << context_used << "/" << context_max << "]\n";
        callback(CallbackEvent::STATS, stats.str(), "", "");
    }

    return true;
#else
    return false;
#endif
}

void LlamaCppBackend::generate_from_session(Session& session, int max_tokens) {
#ifdef ENABLE_LLAMACPP
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "LlamaCpp backend not initialized";
        callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
    }

    dout(1) << "LlamaCpp generate_from_session called with " + std::to_string(session.messages.size()) + " messages" << std::endl;

    // GPT-OSS HARMONY FIX: Drop analysis content from completed assistant turns
    // vLLM's auto_drop_analysis_messages() does this - we need to match that behavior
    // For channel-based models, assistant messages contain <|channel|>analysis<|message|>...
    // followed by <|channel|>final<|message|>content<|end|>
    // We only want to keep the final content for completed turns (any assistant message
    // that's followed by another message)
    if (chat_template && chat_template->get_capabilities().has_channels) {
        for (size_t i = 0; i < session.messages.size(); i++) {
            auto& msg = session.messages[i];

            // Only process assistant messages that are followed by another message (completed turns)
            if (msg.role == Message::ASSISTANT && i + 1 < session.messages.size()) {
                // Check if this message contains analysis channel content
                if (msg.content.find("<|channel|>analysis") != std::string::npos) {
                    // Extract only the final channel content
                    std::string filtered = chat_template->extract_content(msg.content);

                    if (filtered != msg.content) {
                        dout(1) << "GPT-OSS: Filtered analysis from completed turn " << i
                                << " (" << msg.content.length() << " -> " << filtered.length() << " chars)" << std::endl;
                        msg.content = filtered;
                    }
                }
            }
        }
    }

    // PREFIX CACHING: Compare incoming session with backend_session (what's in KV cache)
    size_t cached_count = backend_session.messages.size();

    dout(1) << "Backend has " + std::to_string(cached_count) + " cached messages, " +
              "incoming session has " + std::to_string(session.messages.size()) + " messages" << std::endl;

    // Account for system message offset: backend_session may have SYSTEM at [0], but session doesn't
    size_t backend_offset = 0;
    if (!backend_session.messages.empty() && backend_session.messages[0].role == Message::SYSTEM) {
        backend_offset = 1;
        dout(1) << "Backend has system message at index 0, offsetting comparison by 1" << std::endl;
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
            dout(1) << std::string("WARNING: ") +"DIVERGENCE at session message " + std::to_string(i) + " (backend index " + std::to_string(backend_idx) + "):" << std::endl;
            dout(1) << std::string("WARNING: ") +"  Cached: [" + cached_msg.get_role() + "] " + cached_msg.content.substr(0, 50) + "..." << std::endl;
            dout(1) << std::string("WARNING: ") +"  Session: [" + session_msg.get_role() + "] " + session_msg.content.substr(0, 50) + "..." << std::endl;
            break;
        }
    }

    dout(1) << "Prefix match: " + std::to_string(matching_prefix) + " messages (out of " + std::to_string(session.messages.size()) + " in session)" << std::endl;

    // Check if backend has more messages than what matched
    // Keep: system message (if present) + matching_prefix messages
    size_t expected_backend_count = backend_offset + matching_prefix;

    if (cached_count > expected_backend_count) {
        dout(1) << std::string("WARNING: ") +"Conversation diverged - clearing " + std::to_string(cached_count - expected_backend_count) + " cached messages" << std::endl;

        // Clear KV cache from divergence point onward
        llama_context* ctx = static_cast<llama_context*>(model_ctx);
        llama_memory_t mem = llama_get_memory(ctx);

        // Calculate token position to clear from
        int clear_from_pos = 0;
        for (size_t i = 0; i < expected_backend_count; i++) {
            clear_from_pos += backend_session.messages[i].tokens;
        }

        llama_memory_seq_rm(mem, 0, clear_from_pos, -1);
        dout(1) << "Cleared KV cache from token position " + std::to_string(clear_from_pos) << std::endl;

        // Remove diverged messages from backend_session
        while (backend_session.messages.size() > expected_backend_count) {
            backend_session.messages.pop_back();
        }
    }

    // Set current_session for eviction callbacks
    current_session = &backend_session;

    // Handle system message if present (server mode sends it separately)
    if (!session.system_message.empty() &&
        (backend_session.messages.empty() || backend_session.messages[0].role != Message::SYSTEM)) {

        dout(1) << "Adding system message to KV cache" << std::endl;

        // Format system message with tools using chat template
        std::string formatted_system = chat_template->format_system_message(session.system_message, session.tools);
        dout(1) << "Formatted system prompt with " + std::to_string(session.tools.size()) + " tools" << std::endl;

        // Create system message (token count will be set by format_and_decode_message)
        Message sys_msg(Message::SYSTEM, formatted_system, 0);

        // Decode to KV cache using full-conversation rendering
        std::vector<Message> sys_messages = {sys_msg};
        if (!format_and_decode_message(sys_messages, 0, session.tools, false)) {
            Response err_resp;
            err_resp.success = false;
            err_resp.code = Response::ERROR;
            err_resp.finish_reason = "error";
            err_resp.error = "Failed to decode system message to KV cache";
            callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
        }

        // Add to backend_session (token count was set by format_and_decode_message)
        backend_session.messages.push_back(sys_messages[0]);
        backend_session.system_message = session.system_message;
        dout(1) << "System message added to KV cache (" + std::to_string(sys_messages[0].tokens) + " tokens)" << std::endl;
    }

    // Add NEW messages (from matching_prefix onward)
    size_t new_messages = session.messages.size() - matching_prefix;
    if (new_messages > 0) {
        dout(1) << "Adding " + std::to_string(new_messages) + " new messages to KV cache" << std::endl;

        for (size_t i = matching_prefix; i < session.messages.size(); i++) {
            const auto& msg = session.messages[i];

            // Decode to KV cache - always use full-conversation rendering
            // Build all_messages from backend_session + messages up to current
            std::vector<Message> all_messages(backend_session.messages.begin(), backend_session.messages.end());
            for (size_t j = matching_prefix; j <= i; j++) {
                all_messages.push_back(session.messages[j]);
            }
            size_t target_idx = all_messages.size() - 1;
            bool decode_ok = format_and_decode_message(all_messages, target_idx, session.tools, false);
            if (!decode_ok) {
                Response err_resp;
                err_resp.success = false;
                err_resp.code = Response::ERROR;
                err_resp.finish_reason = "error";
                err_resp.error = "Failed to decode message to KV cache";
                callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
            }

            // Successfully decoded - add to backend_session
            // Token count was set by format_and_decode_message
            Message msg_copy = all_messages[target_idx];
            backend_session.messages.push_back(msg_copy);

            // Update last_user/last_assistant for eviction protection
            if (msg.role == Message::USER) {
                backend_session.last_user_message_index = backend_session.messages.size() - 1;
                backend_session.last_user_message_tokens = msg_copy.tokens;
            } else if (msg.role == Message::ASSISTANT) {
                backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
                backend_session.last_assistant_message_tokens = msg_copy.tokens;
            }

            dout(1) << "Decoded message " + std::to_string(i) + " to KV cache" << std::endl;
        }

        dout(1) << "Prefix caching: " + std::to_string(matching_prefix) + " cached, " +
                  std::to_string(new_messages) + " new, total " + std::to_string(session.messages.size()) << std::endl;
    } else {
        dout(1) << "All messages already in KV cache (100% prefix cache hit)" << std::endl;
    }

    // Copy tools to backend_session
    backend_session.tools = session.tools;
    backend_session.system_message = session.system_message;

    // Copy sampling parameters from request (like TensorRT does)
    backend_session.temperature = session.temperature;
    backend_session.top_p = session.top_p;
    backend_session.top_k = session.top_k;
    backend_session.min_p = session.min_p;
    backend_session.repetition_penalty = session.repetition_penalty;
    backend_session.presence_penalty = session.presence_penalty;
    backend_session.frequency_penalty = session.frequency_penalty;

    // Update member variables used by generate() for token calculations
    // System message tokens
    if (!backend_session.messages.empty() && backend_session.messages[0].role == Message::SYSTEM) {
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

    dout(1) << "Server mode generation - system_tokens=" + std::to_string(system_formatted_tokens) +
              ", user_tokens=" + std::to_string(current_user_formatted_tokens) << std::endl;

    // Generate response (pass streaming callback if provided)
    std::string result = generate(max_tokens, callback);

    // Update session token counts from KV cache (session is source of truth)
    int kv_tokens = get_context_token_count();
    session.total_tokens = kv_tokens;
    session.last_prompt_tokens = kv_tokens - last_assistant_kv_tokens;
    session.last_assistant_message_tokens = last_assistant_kv_tokens;

    dout(1) << "generate_from_session token update: kv_tokens=" << kv_tokens
            << ", last_assistant_kv_tokens=" << last_assistant_kv_tokens
            << ", session.total_tokens=" << session.total_tokens
            << ", session.last_prompt_tokens=" << session.last_prompt_tokens << std::endl;

    // Add assistant response to backend_session to keep it in sync with KV cache
    // Without this, next request's prefix matching fails because backend_session
    // only has the user message but KV cache has user + assistant tokens
    if (!result.empty()) {
        Message assistant_msg(Message::ASSISTANT, result, last_assistant_kv_tokens);
        backend_session.messages.push_back(assistant_msg);
        backend_session.last_assistant_message_index = backend_session.messages.size() - 1;
        backend_session.last_assistant_message_tokens = last_assistant_kv_tokens;
        dout(1) << "Added assistant message to backend_session (" << last_assistant_kv_tokens << " tokens)" << std::endl;
    }

    // Content already delivered via callback during generate()
    // Signal completion
    callback(CallbackEvent::STOP, "stop", "", "");
#else
    callback(CallbackEvent::ERROR, "LlamaCpp backend not compiled in", "error", "");
    callback(CallbackEvent::STOP, "error", "", "");
#endif
}


void LlamaCppBackend::add_message(Session& session, Message::Role role, const std::string& content,
                                      const std::string& tool_name, const std::string& tool_id,
                                      int max_tokens) {
#ifdef ENABLE_LLAMACPP
    if (!is_ready()) {
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "LlamaCpp backend not initialized";
        callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
    }

    // Set current session for eviction callbacks
    current_session = &session;

    dout(1) << "LlamaCpp add_message called: role=" + std::to_string(static_cast<int>(role)) +
              ", content_len=" + std::to_string(content.length()) << std::endl;

    // Count tokens for this message
    int message_tokens = 0;
    {
        if (role == Message::SYSTEM && !session.tools.empty()) {
            // Format system message with tools using chat template
            std::string formatted_content = chat_template->format_system_message(content, session.tools);
            message_tokens = count_message_tokens(role, formatted_content, tool_name, tool_id);
        } else {
            message_tokens = count_message_tokens(role, content, tool_name, tool_id);
        }
    }

    dout(1) << "Message token count: " + std::to_string(message_tokens) << std::endl;

    // Create the message object (NOT in session yet)
    Message msg(role, content, message_tokens);
    msg.tool_name = tool_name;
    msg.tool_call_id = tool_id;

    // TRY to decode to KV cache FIRST - always use full-conversation rendering
    // Build messages array with context: session messages + new message
    std::vector<Message> all_messages(session.messages.begin(), session.messages.end());
    all_messages.push_back(msg);
    size_t target_index = all_messages.size() - 1;
    bool decode_success = format_and_decode_message(all_messages, target_index, session.tools, false);
    // Update msg.tokens from the formatted message
    if (decode_success) {
        msg.tokens = all_messages[target_index].tokens;
    }

    if (!decode_success) {
        // Session is unchanged - no rollback needed
        Response err_resp;
        err_resp.success = false;
        err_resp.code = Response::ERROR;
        err_resp.finish_reason = "error";
        err_resp.error = "Failed to decode message to KV cache";
        callback(CallbackEvent::ERROR, err_resp.error, "error", ""); callback(CallbackEvent::STOP, "error", "", ""); return;
    }

    // SUCCESS - now add message to session
    session.messages.push_back(msg);
    int new_message_index = static_cast<int>(session.messages.size()) - 1;

    // Update session total tokens
    session.total_tokens += message_tokens;

    // Track for context preservation
    if (role == Message::USER) {
        session.last_user_message_index = new_message_index;
        session.last_user_message_tokens = message_tokens;
    }

    // Fire USER event callback when prompt is accepted
    // This displays the user prompt before generation starts
    if (callback && role == Message::USER && !content.empty()) {
        callback(CallbackEvent::USER_PROMPT, content, "", "");
    }

    // Generate response (unless this is a system message)
    Response resp;
    resp.success = true;
    resp.code = Response::SUCCESS;

    if (role != Message::SYSTEM) {
        // Update member variables used by generate() for token calculations
        // System message tokens (first message if it's a SYSTEM type)
        if (!session.messages.empty() && session.messages[0].role == Message::SYSTEM) {
            system_formatted_tokens = session.messages[0].tokens;
        } else {
            system_formatted_tokens = 0;
        }

        // Last user message tokens
        if (role == Message::USER) {
            // Current message being added
            current_user_formatted_tokens = message_tokens;
        } else if (session.last_user_message_index >= 0) {
            // Previous user message
            current_user_formatted_tokens = session.last_user_message_tokens;
        } else {
            current_user_formatted_tokens = 0;
        }

        // User prompt echo moved to frontend - backend never outputs directly

        // Generate with callback if provided (streaming), otherwise accumulate
        std::string response_text = generate(max_tokens, callback);

        // Add assistant message to session (generation was successful)
        Message assistant_msg(Message::ASSISTANT, response_text, last_assistant_kv_tokens);

        // Populate tool_calls_json from tool calls detected during streaming
        if (!pending_tool_calls.empty()) {
            nlohmann::json tc_array = nlohmann::json::array();
            for (const auto& tc : pending_tool_calls) {
                nlohmann::json tc_obj;
                tc_obj["id"] = tc.tool_call_id;
                tc_obj["type"] = "function";
                tc_obj["function"]["name"] = tc.name;
                tc_obj["function"]["arguments"] = tc.raw_json;
                tc_array.push_back(tc_obj);
            }
            assistant_msg.tool_calls_json = tc_array.dump();
            dout(1) << "Populated tool_calls_json: " << assistant_msg.tool_calls_json << std::endl;
        }

        session.messages.push_back(assistant_msg);

        // Update session total tokens with assistant response
        session.total_tokens += last_assistant_kv_tokens;

        // Update tracking
        session.last_assistant_message_index = static_cast<int>(session.messages.size()) - 1;
        session.last_assistant_message_tokens = last_assistant_kv_tokens;

        // Determine finish reason
        std::string finish_reason = "stop";
        if (g_generation_cancelled) {
            finish_reason = "cancelled";
        } else if (last_generation_hit_length_limit) {
            finish_reason = "length";
        }

#if 0
        // Send tool calls via callback - now handled by output() filter
        for (const auto& tc : tool_calls) {
            callback(CallbackEvent::TOOL_CALL, tc.raw_json, tc.name, tc.tool_call_id);
        }
#endif

        // Signal completion
        callback(CallbackEvent::STOP, finish_reason, "", "");
    } else {
        // System message - no generation, just success
        callback(CallbackEvent::STOP, "system", "", "");
    }

    dout(1) << "add_message complete" << std::endl;
#else
    callback(CallbackEvent::ERROR, "LlamaCpp backend not compiled in", "error", "");
    callback(CallbackEvent::STOP, "error", "", "");
#endif
}

