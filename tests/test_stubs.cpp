// Test stubs for globals defined in main.cpp
// These are needed when testing code that references these globals

#include <atomic>

// Server mode flag - false for tests
bool g_server_mode = false;

// Generation cancellation flag
std::atomic<bool> g_generation_cancelled{false};
