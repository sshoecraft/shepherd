#pragma once

#include "../tools/tool.h"
#include "../provider.h"
#include "../session.h"
#include "../backend.h"
#include <string>
#include <vector>
#include <memory>

class Tools;

/// @brief Adapter that wraps an API backend as a Shepherd Tool
/// This allows one backend to call another backend as a tool.
/// Mirrors the CLI pattern: creates backend once via Provider.connect(),
/// reuses it for all calls.
class APIToolAdapter : public Tool {
public:
    explicit APIToolAdapter(const Provider& provider, Tools* tools = nullptr);
    ~APIToolAdapter() = default;

    std::string unsanitized_name() const override;
    std::string description() const override;
    std::string parameters() const override;
    std::vector<ParameterDef> get_parameters_schema() const override;
    std::map<std::string, std::any> execute(const std::map<std::string, std::any>& args) override;

private:
    Provider provider;
    Tools* tools_ptr;

    // Persistent backend and session (created once, reused)
    Session tool_session;
    std::unique_ptr<Backend> backend;

    // Callback state
    std::string accumulated_content;
    std::vector<ToolParser::ToolCall> pending_tool_calls;
    bool cb_success = true;
    std::string cb_error;

    // Lazy initialization
    bool connected = false;
    void ensure_connected();
    void populate_session_tools();
};

/// @brief Register all API providers as tools (except the active one)
void register_provider_tools(Tools& tools, const std::string& active_provider);

/// @brief Register a single provider as a tool
void register_provider_as_tool(Tools& tools, const std::string& provider_name);

/// @brief Unregister a provider tool
void unregister_provider_tool(Tools& tools, const std::string& provider_name);
