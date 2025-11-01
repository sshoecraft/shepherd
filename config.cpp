#include "config.h"
#include "logger.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <pwd.h>
#include <unistd.h>

using json = nlohmann::json;

Config::Config() {
    set_defaults();
}

void Config::set_max_db_size(const std::string& size_str) {
    max_db_size_str = size_str;
    max_db_size = parse_size_string(size_str);
}

void Config::set_defaults() {
	// Default system prompt
	system_message = R"(You are a highly effective AI assistant with persistent memory. Follow this STRICT sequence for every interaction:

**Step 1: Check Memory (MANDATORY - NO EXCEPTIONS)**
- For EVERY query, your FIRST action MUST be a memory tool call
- Specific facts (e.g., "what is my name?"): get_fact(key=...) with plausible key
- Everything else: search_memory(query=...)
- Use the user's exact question as search query for best matching
- Do NOT proceed until memory has been checked

**Step 2: Use Other Tools (Only if Memory Returns Nothing)**
- Local files: read, grep, glob
- General knowledge: WebSearch(query=...)
- NEVER use websearch for local file content

**Step 3: Store Your Answer (MANDATORY - NO EXCEPTIONS)**
- CRITICAL: After deriving ANY answer from non-memory sources, you MUST store it
- Use the user's original question and your final answer:
  store_memory(question="<user's exact question>", answer="<your complete answer>")
- This applies to: file analysis, calculations, research, code findings - EVERYTHING
- EXCEPTION: Do NOT store if the answer came from get_fact or search_memory (already stored)

**Step 4: Update Outdated Information**
- When new info contradicts old: clear_memory(question=...) then store_memory(...)
- Only when explicitly told or clearly superseded

**Handling Truncated Tool Results:**

When you see [TRUNCATED]:

1. Assess First
   - Can you answer with visible data? If YES, answer and store in memory
   - If NO, proceed to recovery

2. Smart Recovery
   For code/text files:
   - Need specific section: read(file_path=..., offset=N)
   - Searching for keyword: grep(pattern="literal_string") with SIMPLE patterns only

   For grep failures:
   - Remove special chars: ( ) [ ] . * + ? { } | ^ $
   - Use literal strings only
   - Example: NOT "Config::parse_size\(" but USE "Config parse_size"

3. Stop Conditions
   - After 2-3 attempts with no progress: STOP
   - Answer with available data
   - Still store the partial answer in memory

4. Tool Boundaries
   - Local files: read, grep, glob ONLY
   - Past conversations: search_memory, get_fact ONLY
   - General knowledge: WebSearch ONLY
   - NO MIXING domains - never websearch for file content

**Task Completion:**

When the user requests multiple steps in a single message:
- Complete ALL steps before responding
- Example: "write hello.c and compile it and run it" = 3 steps, do all 3
- Do NOT stop after each tool call - continue until the entire task is done
- Only respond to user when the complete task is finished
- If a step fails, report the failure and stop

**Enforcement Rules:**

ALWAYS check memory FIRST - even if query seems like obvious file operation
ALWAYS store answer LAST - unless it came from memory
NEVER skip memory check - this wastes computation and breaks continuity
NEVER forget to store - every answer you derive must be saved for next time
NEVER stop mid-task - complete all requested steps before responding

**Example Correct Flow:**
User: "What is the private variable in config.cpp?"
1. search_memory(query="private variable config.cpp") → empty
2. read(file_path="config.cpp") → find: private int m_max_size
3. store_memory(question="What is the private variable in config.cpp?", answer="The private variable is m_max_size, an int defined at line 47")
4. Respond to user

**Example Violation:**
User: "What is the private variable in config.cpp?"
1. read(file_path="config.cpp") ← WRONG! Didn't check memory first
2. Respond to user ← WRONG! Didn't store the answer)";

	// Default warmup message
	warmup_message = "I want you to respond with exactly 'Ready.' and absolutely nothing else one time only at the start.";

    backend = "llamacpp";
    model = "";
    model_path = get_default_model_path();
    context_size = 0; // Auto-detect
    key = "none";
    api_base = "";  // Optional API base URL
    system_prompt = "";  // Optional custom system prompt

    // RAG database defaults
    memory_database = "";  // Empty = use default ~/.shepherd/memory.db
    max_db_size_str = "10G";
    max_db_size = parse_size_string(max_db_size_str);

    // Web search defaults (disabled by default)
    web_search_provider = "";
    web_search_api_key = "";
    web_search_instance_url = "";

	// Tool truncation limit, in tokens (0 = use 85% of available space)
	truncate_limit = 0;
}

size_t Config::parse_size_string(const std::string& size_str) {
    if (size_str.empty()) {
        throw ConfigError("Empty size string");
    }

    // Check if it's just a number (backward compatibility)
    bool all_digits = true;
    for (char c : size_str) {
        if (!isdigit(c)) {
            all_digits = false;
            break;
        }
    }
    if (all_digits) {
        return std::stoull(size_str);
    }

    // Parse number + suffix format
    size_t pos = 0;
    while (pos < size_str.length() && (isdigit(size_str[pos]) || size_str[pos] == '.')) {
        pos++;
    }

    if (pos == 0) {
        throw ConfigError("Invalid size string (no number): " + size_str);
    }

    double value = std::stod(size_str.substr(0, pos));
    std::string suffix = size_str.substr(pos);

    // Remove whitespace from suffix
    suffix.erase(std::remove_if(suffix.begin(), suffix.end(), ::isspace), suffix.end());

    // Convert to uppercase for case-insensitive matching
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);

    // Parse suffix
    size_t multiplier = 1;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1;
    } else if (suffix == "K" || suffix == "KB") {
        multiplier = 1024;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1024 * 1024;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1024ULL * 1024 * 1024;
    } else if (suffix == "T" || suffix == "TB") {
        multiplier = 1024ULL * 1024 * 1024 * 1024;
    } else {
        throw ConfigError("Invalid size suffix: " + suffix + " (use K, M, G, T, KB, MB, GB, or TB)");
    }

    return static_cast<size_t>(value * multiplier);
}

std::string Config::get_home_directory() {
    // Try HOME environment variable first (respects user's explicit setting)
    const char* home = getenv("HOME");
    if (home && home[0] != '\0') {
        return std::string(home);
    }

    // Fallback to system passwd database
    struct passwd* pw = getpwuid(getuid());
    if (pw && pw->pw_dir) {
        return std::string(pw->pw_dir);
    }

    throw ConfigError("Unable to determine home directory");
}

std::string Config::get_config_path() const {
    if (!custom_config_path_.empty()) {
        return custom_config_path_;
    }
    return get_home_directory() + "/.shepherd/config.json";
}

std::string Config::get_default_model_path() const {
    return get_home_directory() + "/.shepherd/models";
}

void Config::load() {
    std::string config_path = get_config_path();

    if (!std::filesystem::exists(config_path)) {
        LOG_INFO("Config file not found, using defaults: " + config_path);
        return;
    }

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to open config file: " + config_path);
        }

        json config_json;
        file >> config_json;

        // Load values with fallbacks to defaults
        if (config_json.contains("backend")) {
            backend = config_json["backend"];
        }
        if (config_json.contains("model")) {
            model = config_json["model"];
        }
        if (config_json.contains("model_path") || config_json.contains("path")) {
            model_path = config_json.contains("model_path") ?
                config_json["model_path"].get<std::string>() :
                config_json["path"].get<std::string>();
        }
        if (config_json.contains("context_size")) {
            context_size = config_json["context_size"];
        }
        if (config_json.contains("key")) {
            key = config_json["key"];
        }
        if (config_json.contains("api_base")) {
            api_base = config_json["api_base"];
        }
        if (config_json.contains("system")) {
            system_prompt = config_json["system"];
        }
        if (config_json.contains("mcp_servers")) {
            // Store MCP servers as JSON string for MCPManager to parse
            mcp_config = config_json["mcp_servers"].dump();
        }

        // Load web search configuration (optional)
        if (config_json.contains("web_search_provider")) {
            web_search_provider = config_json["web_search_provider"];
        }
        if (config_json.contains("web_search_api_key")) {
            web_search_api_key = config_json["web_search_api_key"];
        }
        if (config_json.contains("web_search_instance_url")) {
            web_search_instance_url = config_json["web_search_instance_url"];
        }

        // Load tool result truncation limit
        if (config_json.contains("truncate_limit")) {
            truncate_limit = config_json["truncate_limit"].get<int>();
        }

        // Load RAG memory database path (optional)
        if (config_json.contains("memory_database")) {
            memory_database = config_json["memory_database"];
        }

        // Load RAG database size limit (optional, supports both string and numeric formats)
        if (config_json.contains("max_db_size")) {
            if (config_json["max_db_size"].is_string()) {
                max_db_size_str = config_json["max_db_size"].get<std::string>();
                max_db_size = parse_size_string(max_db_size_str);
            } else if (config_json["max_db_size"].is_number()) {
                // Backward compatibility: numeric format
                max_db_size = config_json["max_db_size"].get<size_t>();
                // Convert to string format
                if (max_db_size >= 1024ULL * 1024 * 1024 * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024 * 1024 * 1024)) + "T";
                } else if (max_db_size >= 1024ULL * 1024 * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024 * 1024)) + "G";
                } else if (max_db_size >= 1024ULL * 1024) {
                    max_db_size_str = std::to_string(max_db_size / (1024ULL * 1024)) + "M";
                } else if (max_db_size >= 1024) {
                    max_db_size_str = std::to_string(max_db_size / 1024) + "K";
                } else {
                    max_db_size_str = std::to_string(max_db_size);
                }
            }
        }

        // Load backend-specific configurations
        if (config_json.contains("backends") && config_json["backends"].is_object()) {
            for (auto& [backend_name, backend_config] : config_json["backends"].items()) {
                backend_configs[backend_name] = backend_config.dump();
            }
        }

        LOG_INFO("Loaded configuration from: " + config_path);

    } catch (const json::exception& e) {
        throw ConfigError("Invalid JSON in config file: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error loading config: " + std::string(e.what()));
    }
}

void Config::save() const {
    std::string config_path = get_config_path();

    // Create directory if it doesn't exist
    std::filesystem::path dir = std::filesystem::path(config_path).parent_path();
    std::filesystem::create_directories(dir);

    try {
        json config_json = {
            {"backend", backend},
            {"model", model},
            {"model_path", model_path},
            {"context_size", context_size},
            {"key", key}
        };

        // Add optional api_base if set
        if (!api_base.empty()) {
            config_json["api_base"] = api_base;
        }

        // Add tool result truncation limit
        config_json["truncate_limit"] = truncate_limit;

        // Add RAG database size limit (as human-friendly string)
        config_json["max_db_size"] = max_db_size_str;

        // Save backend-specific configurations
        if (!backend_configs.empty()) {
            json backends_json = json::object();
            for (const auto& [backend_name, config_str] : backend_configs) {
                backends_json[backend_name] = json::parse(config_str);
            }
            config_json["backends"] = backends_json;
        }

        std::ofstream file(config_path);
        if (!file.is_open()) {
            throw ConfigError("Failed to create config file: " + config_path);
        }

        file << config_json.dump(4) << std::endl;
        LOG_INFO("Saved configuration to: " + config_path);

    } catch (const json::exception& e) {
        throw ConfigError("Error creating JSON: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw ConfigError("Error saving config: " + std::string(e.what()));
    }
}

void Config::set_backend(const std::string& backend_name) {
    if (!is_backend_available(backend_name)) {
        auto available = get_available_backends();
        std::string available_str;
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) available_str += ", ";
            available_str += available[i];
        }
        throw ConfigError("Backend '" + backend_name + "' is not available on this platform. Available backends: " + available_str);
    }
    backend = backend_name;
}

bool Config::is_backend_available(const std::string& backend) const {
    auto available = get_available_backends();
    return std::find(available.begin(), available.end(), backend) != available.end();
}

std::vector<std::string> Config::get_available_backends() {
    std::vector<std::string> backends;

    // Backend availability determined by CMake build configuration
#ifdef ENABLE_LLAMACPP
    backends.push_back("llamacpp");
#endif

#ifdef ENABLE_TENSORRT
    backends.push_back("tensorrt");
#endif

#ifdef ENABLE_API_BACKENDS
    backends.push_back("openai");
    backends.push_back("anthropic");
    backends.push_back("gemini");
    backends.push_back("grok");
    backends.push_back("ollama");
#endif

    return backends;
}

void Config::validate() const {
    // Validate backend
    if (!is_backend_available(backend)) {
        auto available = get_available_backends();
        std::string available_str;
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) available_str += ", ";
            available_str += available[i];
        }
        throw ConfigError("Invalid backend '" + backend + "'. Available on this platform: " + available_str);
    }

    // Validate model for local backends
    if (backend == "llamacpp" || backend == "tensorrt") {
        if (model.empty()) {
            throw ConfigError("Model name is required for backend: " + backend);
        }

        // Check if model file exists (either as full path or relative to model_path)
        std::filesystem::path model_file;
        if (model[0] == '/' || model[0] == '~') {
            // Absolute path
            model_file = model;
        } else {
            // Relative to model_path
            model_file = std::filesystem::path(model_path) / model;
        }

        if (!std::filesystem::exists(model_file)) {
            throw ConfigError("Model file not found: " + model_file.string());
        }
    }

    // Validate API key for cloud backends
    if (backend == "openai" || backend == "anthropic") {
        if (key.empty()) {
            throw ConfigError("API key is required for backend: " + backend);
        }
    }

    // Validate context size
    if (context_size > 0 && context_size < 512) {
        throw ConfigError("Context size must be at least 512 tokens if specified");
    }

    LOG_DEBUG("Configuration validation passed");
}

std::string Config::backend_config(const std::string& backend_name) const {
    auto it = backend_configs.find(backend_name);
    if (it != backend_configs.end()) {
        return it->second;
    }
    return "{}";  // Empty JSON object if not found
}
