#pragma once

// ============================================================================
// Shepherd Core Header
// ============================================================================
// This file contains core functionality used across the Shepherd codebase.
// Include this in all .cpp files to get access to:
// - Global system flags
// - Common logging facilities
// - Standard library headers used throughout the codebase
// ============================================================================

// Standard library headers (used in 50%+ of source files)
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <filesystem>

// Core Shepherd headers
#include "logger.h"
#include "debug.h"

// ============================================================================
// Global System Flags
// ============================================================================
// These are defined in main.cpp and accessible throughout the entire system

// Debug mode flag - enables verbose logging and debug output
extern bool g_debug_mode;

// Debug level (0=off, 1-9=increasing verbosity) - used by dprintf() macro
extern int g_debug_level;

// Server mode flag - true when running as HTTP API server (--server)
// When true, the system operates in stateless request/response mode
extern bool g_server_mode;

// ============================================================================
// Common Utilities
// ============================================================================

namespace shepherd {
    // Get current Unix timestamp in seconds
    inline int64_t get_current_timestamp() {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
