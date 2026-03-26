#pragma once

#include "frontend.h"
#include <string>

// JSON line-protocol frontend for machine-to-machine integration
// Reads JSON from stdin (one per line), writes JSON to stdout (one per line)
// Tools execute locally (like CLI/TUI)
//
// Input:  {"type": "user", "content": "read the file foo.txt"}
// Output: {"type": "tool_use", "name": "read_file", "params": {...}, "id": "call_1"}
//         {"type": "tool_result", "name": "read_file", "id": "call_1", "success": true, "summary": "Read 20 lines"}
//         {"type": "text", "content": "Here's the content..."}
//         {"type": "end_turn", "turns": 1, "cost_usd": 0.03, "total_tokens": 1234}
class JsonFrontend : public Frontend {
public:
    JsonFrontend();
    ~JsonFrontend();

    void init(const FrontendFlags& flags) override;
    int run(Provider* cmdline_provider = nullptr) override;

private:
    FrontendFlags init_flags;

    // Turn counter for end_turn stats
    int turn_count = 0;

    // Output a single JSON line to stdout (compact, flushed immediately)
    void emit_json(const nlohmann::json& j);

    // Compute cost in USD for a turn using provider pricing
    float compute_turn_cost(int prompt_tokens, int completion_tokens);
};
