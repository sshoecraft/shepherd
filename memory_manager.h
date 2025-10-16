#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

enum class MemoryTier {
    GPU_VRAM,    // Hot - immediate access (recent context)
    SYSTEM_RAM,  // Warm - fast access (extended context)
    STORAGE      // Cold - slow access (archived context)
};

struct ContextBlock {
    std::string id;
    std::vector<int> tokens;
    MemoryTier tier;
    uint64_t timestamp;
    double importance_score;
    size_t access_count;
};

class MemoryManager {
public:
    MemoryManager();
    ~MemoryManager();

    // Context management
    bool store_context(const std::string& id, const std::vector<int>& tokens,
                      MemoryTier preferred_tier = MemoryTier::GPU_VRAM);
    std::vector<int> retrieve_context(const std::string& id);
    bool promote_context(const std::string& id, MemoryTier new_tier);
    bool demote_context(const std::string& id, MemoryTier new_tier);

    // Memory management
    void optimize_memory_usage();
    size_t get_memory_usage(MemoryTier tier) const;
    size_t get_available_memory(MemoryTier tier) const;
    void set_memory_limits(MemoryTier tier, size_t limit_bytes);

    // Statistics
    void print_memory_stats() const;
    double calculate_importance_score(const ContextBlock& block) const;

private:
    // Internal methods
    void move_context(const std::string& id, MemoryTier from, MemoryTier to);
    void cleanup_expired_contexts();
    void update_access_patterns(const std::string& id);

    // Memory tracking
    std::unordered_map<std::string, ContextBlock> contexts_;
    std::unordered_map<MemoryTier, size_t> memory_usage_;
    std::unordered_map<MemoryTier, size_t> memory_limits_;

    // Configuration
    uint64_t max_age_seconds_;
    size_t min_importance_threshold_;
    bool auto_optimization_enabled_;
};