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

#define dprintf(level,format,args...) if (g_debug_level >= level) { char msg[4096]; snprintf(msg,sizeof(msg),"%s(%d) %s: " format,__FILE__,__LINE__, __FUNCTION__, ## args); int end = strlen(msg)-1; if (msg[end] == '\n') msg[end] = 0; LOG_DEBUG(msg); }

#endif /* __SHEPHERD_DEBUG_H */
