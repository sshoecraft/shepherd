#pragma once

#include "frontend.h"
#include "session.h"
#include "backends/backend.h"
#include "tools/tools.h"
#include <string>
#include <memory>
#include <termios.h>

// Forward declaration
typedef struct Replxx Replxx;

// CLI class handles user interaction and tool execution
class CLI : public Frontend {
public:
    CLI();
    ~CLI();

    // Frontend interface
    void init(Session& session,
              bool no_mcp = false,
              bool no_tools = false,
              const std::string& provider_name = "") override;
    int run(Session& session) override;

    // Tool management
    Tools tools;

    // Output functions (handle colors based on mode)
    void show_tool_call(const std::string& name, const std::string& params);
    void show_tool_result(const std::string& result);
    void show_error(const std::string& error);
    void show_cancelled();

    // Message I/O using terminal I/O system
    void send_message(const std::string& message);
    std::string receive_message(const std::string& prompt = ">");

    // Slash command handler
    bool handle_slash_commands(const std::string& input, Session& session);

    // Public state - direct access, no getters
    bool eof_received;
    bool generation_cancelled;
};