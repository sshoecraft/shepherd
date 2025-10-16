#pragma once

// Single include file for all tools

#include "tool.h"
#include "memory_tools.h"
#include "filesystem_tools.h"
#include "command_tools.h"
#include "json_tools.h"
#include "http_tools.h"
#include "mcp_resource_tools.h"
#include "core_tools.h"

// Register all tools at once
inline void register_all_tools() {
    register_memory_tools();
    register_core_tools();
    // DISABLED: Only memory tool is active
    // register_filesystem_tools();
    // register_command_tools();
    // register_json_tools();
    // register_http_tools();
    // register_mcp_resource_tools();
}