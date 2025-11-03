#include "shepherd.h"
#include "api_tools.h"
#include "../tools/tool.h"
#include "../logger.h"

APITools& APITools::instance() {
    static APITools api_tools;
    return api_tools;
}

bool APITools::initialize() {
    LOG_INFO("Initializing API Tools...");

    std::string config_path = std::string(getenv("HOME")) + "/.shepherd/config.json";

    try {
        // Load API tool configurations
        std::vector<APIToolEntry> tool_configs = APIToolConfig::load(config_path);

        if (tool_configs.empty()) {
            LOG_INFO("No API tools configured");
            return true;
        }

        return initialize(tool_configs);

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load API tools configuration: " + std::string(e.what()));
        return false;
    }
}

bool APITools::initialize(const std::vector<APIToolEntry>& tool_configs) {
    LOG_INFO("Initializing API Tools with " + std::to_string(tool_configs.size()) + " tools");

    bool any_success = false;

    for (const auto& tool_config : tool_configs) {
        LOG_INFO("Registering API tool: " + tool_config.name);
        if (register_tool(tool_config)) {
            any_success = true;
            LOG_INFO("Successfully registered API tool: " + tool_config.name);
        } else {
            LOG_WARN("Failed to register API tool: " + tool_config.name);
        }
    }

    if (any_success) {
        LOG_INFO("API Tools initialized with " +
                 std::to_string(total_tools) + " tools");
    } else {
        LOG_DEBUG("No API tools could be initialized");
    }

    return any_success;
}

bool APITools::register_tool(const APIToolEntry& tool_config) {
    try {
        // Create adapter
        auto adapter = std::make_unique<APIToolAdapter>(tool_config);
        std::string tool_name = adapter->unsanitized_name();

        // Register with ToolRegistry
        auto& registry = ToolRegistry::instance();
        registry.register_tool(std::move(adapter));

        LOG_DEBUG("Registered API tool: " + tool_name);
        total_tools++;

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Exception registering API tool " + tool_config.name + ": " + e.what());
        return false;
    }
}

void APITools::shutdown() {
    LOG_INFO("Shutting down API Tools...");
    total_tools = 0;
    LOG_INFO("API Tools shutdown complete");
}

std::vector<std::string> APITools::get_tool_names() const {
    std::vector<std::string> names;
    std::string config_path = std::string(getenv("HOME")) + "/.shepherd/config.json";
    std::vector<APIToolEntry> tools = APIToolConfig::load(config_path);

    for (const auto& tool : tools) {
        names.push_back(tool.name);
    }

    return names;
}
