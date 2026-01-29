#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <set>
#include <csignal>
#include <functional>
#include "nlohmann/json.hpp"

class Scheduler {
public:
    struct ScheduleEntry {
        std::string id;
        std::string name;
        std::string cron;
        std::string prompt;
        bool enabled;
        std::string last_run;   // ISO 8601 format
        std::string created;    // ISO 8601 format

        nlohmann::json to_json() const;
        static ScheduleEntry from_json(const nlohmann::json& j);
    };

    // Parsed cron expression for matching
    struct CronExpr {
        std::set<int> minutes;      // 0-59
        std::set<int> hours;        // 0-23
        std::set<int> days;         // 1-31
        std::set<int> months;       // 1-12
        std::set<int> weekdays;     // 0-6 (0=Sunday)
    };

    Scheduler();
    ~Scheduler();

    // Persistence
    void load();
    void save();

    // CRUD operations
    std::string add(const std::string& name, const std::string& cron, const std::string& prompt);
    bool remove(const std::string& id_or_name);
    bool enable(const std::string& id_or_name);
    bool disable(const std::string& id_or_name);
    ScheduleEntry* get(const std::string& id_or_name);
    const ScheduleEntry* get(const std::string& id_or_name) const;
    std::vector<ScheduleEntry> list() const;

    // Runtime control
    void start();
    void stop();
    bool is_running() const { return running; }

    // Callback for when a schedule fires
    using FireCallback = std::function<void(const std::string& prompt)>;
    void set_fire_callback(FireCallback cb);

    // Cron utilities
    static bool parse_cron(const std::string& cron, CronExpr& expr);
    static bool matches_time(const CronExpr& expr, const std::tm& tm);
    static std::string next_run_time(const std::string& cron);
    static std::string format_next_run(const std::string& cron);

    // Validation
    static bool validate_cron(const std::string& cron, std::string& error);

    // Config path
    static std::string get_config_path();

private:
    std::vector<ScheduleEntry> schedules;
    std::atomic<bool> running;
    mutable std::mutex mutex;
    std::string config_path;
    FireCallback fire_callback;

    // SIGALRM handler (static for signal compatibility)
    static void alarm_handler(int sig);
    static Scheduler* instance;  // For signal handler access
    static volatile sig_atomic_t alarm_pending;  // Flag set by signal handler

public:
    // Check if alarm fired and process (call from main loop)
    void poll();

private:
    void check_and_fire();
    std::string generate_id();
    std::string current_iso_time() const;
    std::string current_minute_key() const;

    // Find entry by id or name (internal, caller must hold lock)
    ScheduleEntry* find_entry(const std::string& id_or_name);
    const ScheduleEntry* find_entry(const std::string& id_or_name) const;

    // Parse a single cron field
    static bool parse_cron_field(const std::string& field, int min, int max, std::set<int>& values);
};

// Common scheduler command implementation (takes parsed args)
// Returns 0 on success, 1 on error
// callback: function to emit output
int handle_sched_args(const std::vector<std::string>& args,
                      std::function<void(const std::string&)> callback);

// Handle scheduler slash commands (/sched) - interactive version
// Returns true if command was handled, false otherwise
bool handle_sched_command(const std::string& input);
