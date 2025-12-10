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

// Forward declare to check if TUI is active
extern class TUIScreen* g_tui_screen;

#define dprintf(level,format,args...) \
	do { \
		if (g_debug_level >= level && !g_tui_screen) { \
			fprintf(stderr, "[DEBUG] %s(%d) %s: " format "\n",__FILE__,__LINE__, __FUNCTION__, ## args); \
		} \
	} while(0)

#endif /* __SHEPHERD_DEBUG_H */
