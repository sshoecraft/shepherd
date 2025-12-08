#pragma once

#include "thread_queue.h"
#include <string>

// Global output queue for streaming tokens from backend thread to main thread
// Backends push tokens here during generation, main loop drains and displays
extern ThreadQueue<std::string> g_output_queue;
