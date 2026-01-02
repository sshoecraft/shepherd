#pragma once

// ============================================================================
// Shepherd Core Header
// ============================================================================
// This file contains core functionality used across the Shepherd codebase.
// Include this in all .cpp files to get access to:
// - Global system flags
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
#include <atomic>

#if 0
// Make json a core type
#ifndef __JSON_DEFINED
#include "nlohmann/json.hpp"
using json = nlohmann::json;
#define __JSON_DEFINED
#endif
#endif

// Core Shepherd headers
#include "config.h"
#include "session.h"

// Config - avail to all as a global
extern std::unique_ptr<Config> config;

// Global command-line arguments
void get_global_args(int& argc, char**& argv);

// ============================================================================
// Global System Flags
// ============================================================================
// These are defined in main.cpp and accessible throughout the entire system

#ifdef _DEBUG
// Debug level (0=off, 1-9=increasing verbosity) - used by dout() macro
extern int g_debug_level;
#endif

// Verbose mode flag - enables verbose logging
extern bool g_verbose_mode;

// Server mode flag - true when running as HTTP API server (--server)
extern bool g_server_mode;

// Cancellation flag (atomic for thread safety)
extern std::atomic<bool> g_generation_cancelled;

// Scheduler disable flag (--nosched)
extern bool g_disable_scheduler;

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

#ifdef _DEBUG
// Debug output stream - returns cerr if level <= g_debug_level, else null stream
std::ostream& dout(int level);
#else
// No-op macro that discards all stream operations in non-debug builds
#define dout(level) if(false) std::cerr
#endif
