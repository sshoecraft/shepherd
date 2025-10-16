#include "memory_manager.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cmath>

MemoryManager::MemoryManager()
    : max_age_seconds_(3600)  // 1 hour
    , min_importance_threshold_(10)
    , auto_optimization_enabled_(true) {

    // Set default memory limits (in bytes)
    memory_limits_[MemoryTier::GPU_VRAM] = 16ULL * 1024 * 1024 * 1024;  // 16GB
    memory_limits_[MemoryTier::SYSTEM_RAM] = 64ULL * 1024 * 1024 * 1024; // 64GB
    memory_limits_[MemoryTier::STORAGE] = 1ULL * 1024 * 1024 * 1024 * 1024; // 1TB

    // Initialize usage tracking
    memory_usage_[MemoryTier::GPU_VRAM] = 0;
    memory_usage_[MemoryTier::SYSTEM_RAM] = 0;
    memory_usage_[MemoryTier::STORAGE] = 0;

    std::cout << "Memory manager initialized with hierarchical storage" << std::endl;
}

MemoryManager::~MemoryManager() {
    std::cout << "Memory manager shutting down..." << std::endl;
    print_memory_stats();
}

bool MemoryManager::store_context(const std::string& id,
                                 const std::vector<int>& tokens,
                                 MemoryTier preferred_tier) {
    try {
        ContextBlock block;
        block.id = id;
        block.tokens = tokens;
        block.tier = preferred_tier;
        block.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        block.access_count = 1;
        block.importance_score = calculate_importance_score(block);

        // Calculate memory required
        size_t required_memory = tokens.size() * sizeof(int);

        // Check if we have space in preferred tier
        if (get_available_memory(preferred_tier) < required_memory) {
            // Try to make space by demoting less important contexts
            optimize_memory_usage();

            // If still no space, use next tier
            if (get_available_memory(preferred_tier) < required_memory) {
                if (preferred_tier == MemoryTier::GPU_VRAM) {
                    preferred_tier = MemoryTier::SYSTEM_RAM;
                } else if (preferred_tier == MemoryTier::SYSTEM_RAM) {
                    preferred_tier = MemoryTier::STORAGE;
                }
                block.tier = preferred_tier;
            }
        }

        // Store the context
        contexts_[id] = std::move(block);
        memory_usage_[preferred_tier] += required_memory;

        std::cout << "Stored context '" << id << "' in tier "
                  << static_cast<int>(preferred_tier)
                  << " (" << tokens.size() << " tokens)" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to store context '" << id << "': " << e.what() << std::endl;
        return false;
    }
}

std::vector<int> MemoryManager::retrieve_context(const std::string& id) {
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        return {}; // Context not found
    }

    // Update access patterns
    update_access_patterns(id);

    // Consider promoting frequently accessed contexts
    auto& block = it->second;
    if (block.access_count > 5 && block.tier != MemoryTier::GPU_VRAM) {
        promote_context(id, MemoryTier::GPU_VRAM);
    }

    return block.tokens;
}

bool MemoryManager::promote_context(const std::string& id, MemoryTier new_tier) {
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        return false;
    }

    MemoryTier current_tier = it->second.tier;
    if (current_tier <= new_tier) {
        return true; // Already in better or same tier
    }

    size_t required_memory = it->second.tokens.size() * sizeof(int);
    if (get_available_memory(new_tier) >= required_memory) {
        move_context(id, current_tier, new_tier);
        std::cout << "Promoted context '" << id << "' to tier "
                  << static_cast<int>(new_tier) << std::endl;
        return true;
    }

    return false; // Not enough space
}

bool MemoryManager::demote_context(const std::string& id, MemoryTier new_tier) {
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        return false;
    }

    MemoryTier current_tier = it->second.tier;
    if (current_tier >= new_tier) {
        return true; // Already in worse or same tier
    }

    move_context(id, current_tier, new_tier);
    std::cout << "Demoted context '" << id << "' to tier "
              << static_cast<int>(new_tier) << std::endl;
    return true;
}

void MemoryManager::optimize_memory_usage() {
    if (!auto_optimization_enabled_) {
        return;
    }

    cleanup_expired_contexts();

    // Sort contexts by importance score
    std::vector<std::pair<std::string, ContextBlock*>> sortable_contexts;
    for (auto& [id, block] : contexts_) {
        sortable_contexts.emplace_back(id, &block);
    }

    std::sort(sortable_contexts.begin(), sortable_contexts.end(),
              [this](const auto& a, const auto& b) {
                  return calculate_importance_score(*a.second) >
                         calculate_importance_score(*b.second);
              });

    // Demote less important contexts from higher tiers
    for (auto& [id, block_ptr] : sortable_contexts) {
        auto& block = *block_ptr;

        if (block.tier == MemoryTier::GPU_VRAM &&
            block.importance_score < min_importance_threshold_) {
            demote_context(id, MemoryTier::SYSTEM_RAM);
        } else if (block.tier == MemoryTier::SYSTEM_RAM &&
                   block.importance_score < min_importance_threshold_ / 2) {
            demote_context(id, MemoryTier::STORAGE);
        }
    }
}

size_t MemoryManager::get_memory_usage(MemoryTier tier) const {
    auto it = memory_usage_.find(tier);
    return (it != memory_usage_.end()) ? it->second : 0;
}

size_t MemoryManager::get_available_memory(MemoryTier tier) const {
    size_t limit = memory_limits_.at(tier);
    size_t used = get_memory_usage(tier);
    return (used < limit) ? (limit - used) : 0;
}

void MemoryManager::set_memory_limits(MemoryTier tier, size_t limit_bytes) {
    memory_limits_[tier] = limit_bytes;
    std::cout << "Set memory limit for tier " << static_cast<int>(tier)
              << " to " << (limit_bytes / (1024 * 1024)) << " MB" << std::endl;
}

void MemoryManager::print_memory_stats() const {
    std::cout << "\n=== Memory Statistics ===" << std::endl;

    const char* tier_names[] = {"GPU_VRAM", "SYSTEM_RAM", "STORAGE"};

    for (int i = 0; i < 3; ++i) {
        MemoryTier tier = static_cast<MemoryTier>(i);
        size_t used = get_memory_usage(tier);
        size_t limit = memory_limits_.at(tier);
        double usage_percent = (limit > 0) ? (100.0 * used / limit) : 0.0;

        std::cout << tier_names[i] << ": "
                  << std::fixed << std::setprecision(1)
                  << (used / (1024.0 * 1024.0)) << " MB / "
                  << (limit / (1024.0 * 1024.0)) << " MB ("
                  << usage_percent << "%)" << std::endl;
    }

    std::cout << "Total contexts: " << contexts_.size() << std::endl;
}

double MemoryManager::calculate_importance_score(const ContextBlock& block) const {
    uint64_t current_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Factors: recency, access frequency, size
    double recency_factor = 1.0 / (1.0 + (current_time - block.timestamp) / 3600.0);
    double frequency_factor = std::log(1.0 + block.access_count);
    double size_factor = std::log(1.0 + block.tokens.size() / 1000.0);

    return recency_factor * frequency_factor * size_factor;
}

void MemoryManager::move_context(const std::string& id, MemoryTier from, MemoryTier to) {
    auto it = contexts_.find(id);
    if (it == contexts_.end()) {
        return;
    }

    size_t memory_size = it->second.tokens.size() * sizeof(int);

    // Update memory usage
    memory_usage_[from] -= memory_size;
    memory_usage_[to] += memory_size;

    // Update context tier
    it->second.tier = to;
}

void MemoryManager::cleanup_expired_contexts() {
    uint64_t current_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    auto it = contexts_.begin();
    while (it != contexts_.end()) {
        if ((current_time - it->second.timestamp) > max_age_seconds_ &&
            it->second.access_count <= 1) {

            size_t memory_size = it->second.tokens.size() * sizeof(int);
            memory_usage_[it->second.tier] -= memory_size;

            std::cout << "Cleaned up expired context: " << it->first << std::endl;
            it = contexts_.erase(it);
        } else {
            ++it;
        }
    }
}

void MemoryManager::update_access_patterns(const std::string& id) {
    auto it = contexts_.find(id);
    if (it != contexts_.end()) {
        it->second.access_count++;
        it->second.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        it->second.importance_score = calculate_importance_score(it->second);
    }
}