#pragma once

#include "../tools/tool.h"
#include "../provider.h"
#include <string>
#include <vector>
#include <memory>

class Tools;

/// @brief API tool configuration entry
/// Used to create API tool adapters from provider configs
struct APIToolEntry {
    std::string name;
    std::string backend;
    std::string model;
    std::string api_key;
    std::string api_base;
    size_t context_size = 0;
    int max_tokens = 0;
};

/// @brief Adapter that wraps an API backend as a Shepherd Tool
/// This allows one backend to call another backend as a tool
class APIToolAdapter : public Tool {
public:
    explicit APIToolAdapter(const APIToolEntry& entry, Tools* tools = nullptr);

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;

private:
    APIToolEntry config;
    Tools* tools_ptr;  // Optional pointer to Tools for populating sub-session
};

/// @brief Convert ProviderConfig to APIToolEntry
APIToolEntry provider_to_tool_entry(const ProviderConfig* config, const std::string& provider_name);

/// @brief Register all API providers as tools (except the active one)
/// @param tools Tools instance to register with
/// @param active_provider Name of the currently active provider (will be skipped)
void register_provider_tools(Tools& tools, const std::string& active_provider);

/// @brief Register a single provider as a tool
/// @param tools Tools instance to register with
/// @param provider_name Name of the provider to register
void register_provider_as_tool(Tools& tools, const std::string& provider_name);

/// @brief Unregister a provider tool
/// @param tools Tools instance to unregister from
/// @param provider_name Name of the provider to unregister
void unregister_provider_tool(Tools& tools, const std::string& provider_name);
