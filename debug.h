/*
 * Debug level support for Shepherd
 * Provides fine-grained debug output control with dprintf macro
 */

#ifndef __SHEPHERD_DEBUG_H
#define __SHEPHERD_DEBUG_H

#include "logger.h"
#include <sstream>

// Global debug level variable (defined in main.cpp)
extern int g_debug_level;

// Debug level printf macro - only logs if current debug level >= specified level
// Usage: dprintf(1, "Major event: %s", event_name);
//        dprintf(5, "Loop iteration %d", i);
#define dprintf(level, format, ...) \
    do { \
        if (g_debug_level >= level) { \
            std::ostringstream __debug_oss; \
            __debug_oss << __FILE__ << "(" << __LINE__ << ") " << __FUNCTION__ << ": "; \
            char __debug_buf[4096]; \
            snprintf(__debug_buf, sizeof(__debug_buf), format, ##__VA_ARGS__); \
            __debug_oss << __debug_buf; \
            LOG_DEBUG(__debug_oss.str()); \
        } \
    } while(0)

// Simplified version without file/line info for cleaner output
#define dprintf_clean(level, format, ...) \
    do { \
        if (g_debug_level >= level) { \
            char __debug_buf[4096]; \
            snprintf(__debug_buf, sizeof(__debug_buf), format, ##__VA_ARGS__); \
            LOG_DEBUG(std::string(__debug_buf)); \
        } \
    } while(0)

#endif /* __SHEPHERD_DEBUG_H */
