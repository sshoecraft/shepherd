#include "shepherd.h"
#include "scheduler.h"
#include <fstream>

#include <filesystem>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>

using json = nlohmann::json;

// Static instance pointer for signal handler
Scheduler* Scheduler::instance = nullptr;
volatile sig_atomic_t Scheduler::alarm_pending = 0;

// ScheduleEntry JSON serialization
json Scheduler::ScheduleEntry::to_json() const {
    return json{
        {"id", id},
        {"name", name},
        {"cron", cron},
        {"prompt", prompt},
        {"enabled", enabled},
        {"last_run", last_run},
        {"created", created}
    };
}

Scheduler::ScheduleEntry Scheduler::ScheduleEntry::from_json(const json& j) {
    ScheduleEntry entry;
    entry.id = j.value("id", "");
    entry.name = j.value("name", "");
    entry.cron = j.value("cron", "");
    entry.prompt = j.value("prompt", "");
    entry.enabled = j.value("enabled", true);
    entry.last_run = j.value("last_run", "");
    entry.created = j.value("created", "");
    return entry;
}

Scheduler::Scheduler() : running(false) {
    config_path = get_config_path();
}

Scheduler::~Scheduler() {
    stop();
}

std::string Scheduler::get_config_path() {
    std::string config_dir;

    // Use XDG_CONFIG_HOME if set, otherwise ~/.config
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config && xdg_config[0] != '\0') {
        config_dir = xdg_config;
    } else {
        const char* home = std::getenv("HOME");
        if (home) {
            config_dir = std::string(home) + "/.config";
        } else {
            config_dir = "/tmp";
        }
    }

    return config_dir + "/shepherd/schedule.json";
}

void Scheduler::load() {
    std::lock_guard<std::mutex> lock(mutex);

    schedules.clear();

    if (!std::filesystem::exists(config_path)) {
        dout(1) << "Schedule file does not exist: " + config_path << std::endl;
        return;
    }

    try {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            dout(1) << "Could not open schedule file: " + config_path << std::endl;
            return;
        }

        json j;
        file >> j;

        if (j.contains("schedules") && j["schedules"].is_array()) {
            for (const auto& item : j["schedules"]) {
                schedules.push_back(ScheduleEntry::from_json(item));
            }
        }

        dout(1) << "Loaded " + std::to_string(schedules.size()) + " schedules" << std::endl;
    } catch (const std::exception& e) {
        dout(1) << "Error loading schedules: " + std::string(e.what()) << std::endl;
    }
}

void Scheduler::save() {
    // Note: caller should hold lock, or this is called from within locked context
    try {
        // Ensure directory exists
        std::filesystem::path path(config_path);
        std::filesystem::create_directories(path.parent_path());

        json j;
        j["schedules"] = json::array();
        for (const auto& entry : schedules) {
            j["schedules"].push_back(entry.to_json());
        }

        std::ofstream file(config_path);
        if (file.is_open()) {
            file << j.dump(2);
            dout(1) << "Saved " + std::to_string(schedules.size()) + " schedules" << std::endl;
        }
    } catch (const std::exception& e) {
        dout(1) << "Error saving schedules: " + std::string(e.what()) << std::endl;
    }
}

std::string Scheduler::generate_id() {
    // Generate a short random ID
    static const char chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);

    std::string id;
    for (int i = 0; i < 8; ++i) {
        id += chars[dis(gen)];
    }
    return id;
}

std::string Scheduler::current_iso_time() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

std::string Scheduler::current_minute_key() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M");
    return oss.str();
}

std::string Scheduler::add(const std::string& name, const std::string& cron, const std::string& prompt) {
    std::lock_guard<std::mutex> lock(mutex);

    // Validate cron expression
    std::string error;
    if (!validate_cron(cron, error)) {
        return "";  // Invalid cron
    }

    // Check for duplicate name
    for (const auto& entry : schedules) {
        if (entry.name == name) {
            return "";  // Duplicate name
        }
    }

    ScheduleEntry entry;
    entry.id = generate_id();
    entry.name = name;
    entry.cron = cron;
    entry.prompt = prompt;
    entry.enabled = true;
    entry.last_run = "";
    entry.created = current_iso_time();

    schedules.push_back(entry);
    save();

    return entry.id;
}

bool Scheduler::remove(const std::string& id_or_name) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = std::remove_if(schedules.begin(), schedules.end(),
        [&](const ScheduleEntry& e) {
            return e.id == id_or_name || e.name == id_or_name;
        });

    if (it != schedules.end()) {
        schedules.erase(it, schedules.end());
        save();
        return true;
    }
    return false;
}

bool Scheduler::enable(const std::string& id_or_name) {
    std::lock_guard<std::mutex> lock(mutex);

    ScheduleEntry* entry = find_entry(id_or_name);
    if (entry) {
        entry->enabled = true;
        save();
        return true;
    }
    return false;
}

bool Scheduler::disable(const std::string& id_or_name) {
    std::lock_guard<std::mutex> lock(mutex);

    ScheduleEntry* entry = find_entry(id_or_name);
    if (entry) {
        entry->enabled = false;
        save();
        return true;
    }
    return false;
}

Scheduler::ScheduleEntry* Scheduler::find_entry(const std::string& id_or_name) {
    for (auto& entry : schedules) {
        if (entry.id == id_or_name || entry.name == id_or_name) {
            return &entry;
        }
    }
    return nullptr;
}

const Scheduler::ScheduleEntry* Scheduler::find_entry(const std::string& id_or_name) const {
    for (const auto& entry : schedules) {
        if (entry.id == id_or_name || entry.name == id_or_name) {
            return &entry;
        }
    }
    return nullptr;
}

Scheduler::ScheduleEntry* Scheduler::get(const std::string& id_or_name) {
    std::lock_guard<std::mutex> lock(mutex);
    return find_entry(id_or_name);
}

const Scheduler::ScheduleEntry* Scheduler::get(const std::string& id_or_name) const {
    std::lock_guard<std::mutex> lock(mutex);
    return find_entry(id_or_name);
}

std::vector<Scheduler::ScheduleEntry> Scheduler::list() const {
    std::lock_guard<std::mutex> lock(mutex);
    return schedules;
}

void Scheduler::alarm_handler(int sig) {
    (void)sig;  // Unused
    // Just set the flag - don't do any real work in signal handler
    // (mutex/CV operations are not async-signal-safe)
    alarm_pending = 1;

    // Re-arm the alarm for next minute
    if (instance && instance->running) {
        // Simple calculation - just set for 60 seconds
        // poll() will handle the actual timing
        alarm(60);
    }
}

void Scheduler::poll() {
    if (!alarm_pending) return;
    alarm_pending = 0;

    if (running) {
        check_and_fire();
    }
}

void Scheduler::start() {
    if (running) return;

    instance = this;
    running = true;

    // Set up signal handler for SIGALRM
    // Note: Don't use SA_RESTART - we want SIGALRM to interrupt blocking reads
    // so the main loop can check for scheduled prompts
    struct sigaction sa;
    sa.sa_handler = alarm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, nullptr);

    // Calculate seconds until next minute boundary + 1 second
    auto now = std::chrono::system_clock::now();
    auto next_min = std::chrono::ceil<std::chrono::minutes>(now);
    auto now_sec = std::chrono::time_point_cast<std::chrono::seconds>(now);

    // If we're right at a minute boundary, wait for next minute
    if (now_sec == std::chrono::time_point_cast<std::chrono::seconds>(next_min)) {
        next_min += std::chrono::minutes(1);
    }

    auto wake_time = next_min + std::chrono::seconds(1);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(wake_time - now);
    unsigned int secs = static_cast<unsigned int>(duration.count());
    if (secs == 0) secs = 1;

    alarm(secs);

    dout(1) << "Scheduler started with SIGALRM (next check in " + std::to_string(secs) + "s)" << std::endl;
}

void Scheduler::stop() {
    if (!running) return;

    running = false;
    alarm(0);  // Cancel pending alarm
    instance = nullptr;

    // Restore default signal handler
    signal(SIGALRM, SIG_DFL);

    dout(1) << "Scheduler stopped" << std::endl;
}

void Scheduler::set_fire_callback(FireCallback cb) {
    std::lock_guard<std::mutex> lock(mutex);
    fire_callback = std::move(cb);
}

void Scheduler::check_and_fire() {
    std::lock_guard<std::mutex> lock(mutex);

    auto now = std::time(nullptr);
    std::tm tm_now = *std::localtime(&now);
    std::string minute_key = current_minute_key();

    bool save_needed = false;

    for (auto& entry : schedules) {
        if (!entry.enabled) continue;

        // Check if already fired this minute
        if (!entry.last_run.empty() && entry.last_run.substr(0, 16) == minute_key) {
            continue;
        }

        // Parse and match cron
        CronExpr expr;
        if (!parse_cron(entry.cron, expr)) {
            continue;
        }

        if (matches_time(expr, tm_now)) {
            dout(1) << "Firing schedule: " + entry.name + " with prompt: " + entry.prompt.substr(0, 50) << std::endl;

            if (fire_callback) {
                fire_callback(entry.prompt);
            } else {
                dout(1) << "No fire callback set, prompt not injected" << std::endl;
            }

            // Update last_run
            entry.last_run = current_iso_time();
            save_needed = true;
        }
    }

    if (save_needed) {
        save();
    }
}

// Cron parsing implementation

bool Scheduler::parse_cron_field(const std::string& field, int min, int max, std::set<int>& values) {
    values.clear();

    // Handle wildcard
    if (field == "*") {
        for (int i = min; i <= max; ++i) {
            values.insert(i);
        }
        return true;
    }

    // Handle step (*/n or n-m/s)
    size_t slash_pos = field.find('/');
    if (slash_pos != std::string::npos) {
        std::string range_part = field.substr(0, slash_pos);
        std::string step_str = field.substr(slash_pos + 1);

        int step;
        try {
            step = std::stoi(step_str);
        } catch (...) {
            return false;
        }

        if (step <= 0) return false;

        int start = min;
        int end = max;

        if (range_part != "*") {
            // Handle range with step (n-m/s)
            size_t dash = range_part.find('-');
            if (dash != std::string::npos) {
                try {
                    start = std::stoi(range_part.substr(0, dash));
                    end = std::stoi(range_part.substr(dash + 1));
                } catch (...) {
                    return false;
                }
            } else {
                try {
                    start = std::stoi(range_part);
                } catch (...) {
                    return false;
                }
            }
        }

        for (int i = start; i <= end; i += step) {
            if (i >= min && i <= max) {
                values.insert(i);
            }
        }
        return !values.empty();
    }

    // Handle list (n,m,o)
    if (field.find(',') != std::string::npos) {
        std::istringstream iss(field);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Each token could be a single value or a range
            size_t dash = token.find('-');
            if (dash != std::string::npos) {
                int start, end;
                try {
                    start = std::stoi(token.substr(0, dash));
                    end = std::stoi(token.substr(dash + 1));
                } catch (...) {
                    return false;
                }
                for (int i = start; i <= end; ++i) {
                    if (i >= min && i <= max) {
                        values.insert(i);
                    }
                }
            } else {
                try {
                    int val = std::stoi(token);
                    if (val >= min && val <= max) {
                        values.insert(val);
                    }
                } catch (...) {
                    return false;
                }
            }
        }
        return !values.empty();
    }

    // Handle range (n-m)
    size_t dash = field.find('-');
    if (dash != std::string::npos) {
        int start, end;
        try {
            start = std::stoi(field.substr(0, dash));
            end = std::stoi(field.substr(dash + 1));
        } catch (...) {
            return false;
        }
        for (int i = start; i <= end; ++i) {
            if (i >= min && i <= max) {
                values.insert(i);
            }
        }
        return !values.empty();
    }

    // Single value
    try {
        int val = std::stoi(field);
        if (val >= min && val <= max) {
            values.insert(val);
            return true;
        }
    } catch (...) {
        return false;
    }

    return false;
}

bool Scheduler::parse_cron(const std::string& cron, CronExpr& expr) {
    std::istringstream iss(cron);
    std::vector<std::string> fields;
    std::string field;

    while (iss >> field) {
        fields.push_back(field);
    }

    if (fields.size() != 5) {
        return false;
    }

    if (!parse_cron_field(fields[0], 0, 59, expr.minutes)) return false;
    if (!parse_cron_field(fields[1], 0, 23, expr.hours)) return false;
    if (!parse_cron_field(fields[2], 1, 31, expr.days)) return false;
    if (!parse_cron_field(fields[3], 1, 12, expr.months)) return false;
    if (!parse_cron_field(fields[4], 0, 6, expr.weekdays)) return false;

    return true;
}

bool Scheduler::matches_time(const CronExpr& expr, const std::tm& tm) {
    // tm_min: 0-59, tm_hour: 0-23, tm_mday: 1-31, tm_mon: 0-11, tm_wday: 0-6
    if (expr.minutes.find(tm.tm_min) == expr.minutes.end()) return false;
    if (expr.hours.find(tm.tm_hour) == expr.hours.end()) return false;
    if (expr.days.find(tm.tm_mday) == expr.days.end()) return false;
    if (expr.months.find(tm.tm_mon + 1) == expr.months.end()) return false;  // tm_mon is 0-11
    if (expr.weekdays.find(tm.tm_wday) == expr.weekdays.end()) return false;

    return true;
}

bool Scheduler::validate_cron(const std::string& cron, std::string& error) {
    CronExpr expr;
    if (!parse_cron(cron, expr)) {
        error = "Invalid cron expression. Expected 5 fields: minute hour day month weekday";
        return false;
    }
    return true;
}

std::string Scheduler::next_run_time(const std::string& cron) {
    CronExpr expr;
    if (!parse_cron(cron, expr)) {
        return "";
    }

    // Start from current time + 1 minute
    auto now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);
    tm.tm_sec = 0;
    tm.tm_min += 1;
    std::mktime(&tm);  // Normalize

    // Search up to 1 year ahead
    for (int i = 0; i < 525600; ++i) {  // 365 * 24 * 60 minutes
        if (matches_time(expr, tm)) {
            std::ostringstream oss;
            oss << std::put_time(&tm, "%Y-%m-%d %H:%M");
            return oss.str();
        }

        // Advance by 1 minute
        tm.tm_min += 1;
        std::mktime(&tm);  // Normalize
    }

    return "never";
}

std::string Scheduler::format_next_run(const std::string& cron) {
    return next_run_time(cron);
}

// Common scheduler command implementation
int handle_sched_args(const std::vector<std::string>& args,
                      std::function<void(const std::string&)> callback) {
	// Create and load scheduler
	Scheduler scheduler;
	scheduler.load();

	// Helper to format schedule entry
	auto fmt_entry = [](const Scheduler::ScheduleEntry& entry) {
		std::string status = entry.enabled ? "enabled " : "disabled";
		char buf[256];
		snprintf(buf, sizeof(buf), "  [%s]  %-20s  %-16s  %s\n",
			status.c_str(), entry.name.c_str(), ("\"" + entry.cron + "\"").c_str(), entry.prompt.c_str());
		return std::string(buf);
	};

	// No args shows list
	if (args.empty()) {
		auto entries = scheduler.list();
		if (entries.empty()) {
			callback("No schedules configured\n");
		} else {
			callback("Schedules:\n");
			for (const auto& entry : entries) {
				callback(fmt_entry(entry));
			}
		}
		return 0;
	}

	// Determine if first arg is an action (action-first) or a name (name-first)
	std::string first_arg = args[0];
	bool action_first = (first_arg == "list" || first_arg == "add" || first_arg == "next" ||
	                     first_arg == "help" || first_arg == "--help" || first_arg == "-h");

	std::string name;
	std::string subcmd;

	if (action_first) {
		subcmd = first_arg;
		name = (args.size() >= 2) ? args[1] : "";
	} else {
		// Name-first: sched NAME [action]
		name = first_arg;
		subcmd = (args.size() >= 2) ? args[1] : "help";  // No action = show help
	}

	if (subcmd == "help" || subcmd == "--help" || subcmd == "-h") {
		if (!name.empty()) {
			callback("Usage: shepherd sched " + name + " <action>\n"
			    "\nActions:\n"
			    "  show     - Show schedule details\n"
			    "  enable   - Enable schedule\n"
			    "  disable  - Disable schedule\n"
			    "  remove   - Remove schedule\n");
		} else {
			callback("Usage: shepherd sched <name> <action>\n"
			    "\nActions (after name):\n"
			    "  show     - Show schedule details\n"
			    "  enable   - Enable schedule\n"
			    "  disable  - Disable schedule\n"
			    "  remove   - Remove schedule\n"
			    "\nOther commands:\n"
			    "  list     - List all schedules\n"
			    "  add <name> \"<cron>\" \"<prompt>\" - Add schedule\n"
			    "  next     - Show next runs\n");
		}
		return 0;
	}

	if (subcmd == "list") {
		auto entries = scheduler.list();
		if (entries.empty()) {
			callback("No schedules configured\n");
		} else {
			callback("Schedules:\n");
			for (const auto& entry : entries) {
				callback(fmt_entry(entry));
			}
		}
		return 0;
	}

	if (subcmd == "add") {
		// Action-first: sched add NAME "cron" "prompt"
		if (args.size() < 4) {
			callback("Usage: shepherd sched add <name> \"<cron>\" \"<prompt>\"\n"
			    "\nCron format: minute hour day month weekday\n"
			    "  minute:  0-59\n"
			    "  hour:    0-23\n"
			    "  day:     1-31\n"
			    "  month:   1-12\n"
			    "  weekday: 0-6 (0=Sunday)\n"
			    "\nSpecial characters: * (any), - (range), , (list), / (step)\n"
			    "\nExamples:\n"
			    "  shepherd sched add daily-summary \"0 9 * * *\" \"Give me a summary\"\n"
			    "  shepherd sched add hourly \"0 * * * *\" \"What time is it?\"\n");
			return 0;
		}

		std::string add_name = args[1];
		std::string cron = args[2];
		std::string prompt = args[3];

		std::string error;
		if (!Scheduler::validate_cron(cron, error)) {
			callback("Error: " + error + "\n");
			return 1;
		}

		std::string id = scheduler.add(add_name, cron, prompt);
		if (id.empty()) {
			callback("Error: Failed to add schedule (name may already exist)\n");
			return 1;
		}

		callback("Added schedule '" + add_name + "' (id: " + id + ")\n");
		callback("Next run: " + Scheduler::format_next_run(cron) + "\n");
		return 0;
	}

	if (subcmd == "remove") {
		if (name.empty()) {
			callback("Usage: shepherd sched <name> remove\n");
			return 1;
		}

		if (scheduler.remove(name)) {
			callback("Removed schedule '" + name + "'\n");
			return 0;
		} else {
			callback("Error: Schedule not found: " + name + "\n");
			return 1;
		}
	}

	if (subcmd == "enable") {
		if (name.empty()) {
			callback("Usage: shepherd sched <name> enable\n");
			return 1;
		}

		if (scheduler.enable(name)) {
			callback("Enabled schedule '" + name + "'\n");
			return 0;
		} else {
			callback("Error: Schedule not found: " + name + "\n");
			return 1;
		}
	}

	if (subcmd == "disable") {
		if (name.empty()) {
			callback("Usage: shepherd sched <name> disable\n");
			return 1;
		}

		if (scheduler.disable(name)) {
			callback("Disabled schedule '" + name + "'\n");
			return 0;
		} else {
			callback("Error: Schedule not found: " + name + "\n");
			return 1;
		}
	}

	if (subcmd == "show") {
		if (name.empty()) {
			callback("Usage: shepherd sched <name> show\n");
			return 1;
		}

		const auto* entry = scheduler.get(name);
		if (!entry) {
			callback("Error: Schedule not found: " + name + "\n");
			return 1;
		}

		callback("Schedule: " + entry->name + "\n");
		callback("  ID:       " + entry->id + "\n");
		callback("  Cron:     " + entry->cron + "\n");
		callback("  Prompt:   " + entry->prompt + "\n");
		callback("  Enabled:  " + std::string(entry->enabled ? "yes" : "no") + "\n");
		callback("  Created:  " + entry->created + "\n");
		callback("  Last run: " + (entry->last_run.empty() ? "never" : entry->last_run) + "\n");
		callback("  Next run: " + Scheduler::format_next_run(entry->cron) + "\n");
		return 0;
	}

	if (subcmd == "next") {
		// Action-first: sched next [name]
		if (name.empty()) {
			// Show next run for all schedules
			auto entries = scheduler.list();
			if (entries.empty()) {
				callback("No schedules configured\n");
			} else {
				callback("Next scheduled runs:\n");
				for (const auto& entry : entries) {
					if (entry.enabled) {
						callback("  " + entry.name + ": " + Scheduler::format_next_run(entry.cron) + "\n");
					}
				}
			}
			return 0;
		}

		// Show next for specific schedule (action-first: sched next NAME)
		const auto* entry = scheduler.get(name);
		if (!entry) {
			callback("Error: Schedule not found: " + name + "\n");
			return 1;
		}

		callback(entry->name + ": " + Scheduler::format_next_run(entry->cron) + "\n");
		return 0;
	}

	callback("Unknown sched command: " + subcmd + "\n");
	callback("Use 'shepherd sched help' to see available commands\n");
	return 1;
}

// Handle scheduler slash commands (interactive version)
bool handle_sched_command(const std::string& input) {
	// Tokenize the input
	std::istringstream iss(input);
	std::string cmd;
	iss >> cmd;

	// Parse remaining arguments - handle quoted strings
	std::vector<std::string> args;
	std::string token;
	bool in_quotes = false;
	std::string quoted_arg;

	// Get rest of line after command
	std::string rest;
	std::getline(iss, rest);

	// Parse handling quoted strings
	for (size_t i = 0; i < rest.size(); ++i) {
		char c = rest[i];
		if (c == '"') {
			if (in_quotes) {
				args.push_back(quoted_arg);
				quoted_arg.clear();
				in_quotes = false;
			} else {
				in_quotes = true;
			}
		} else if (std::isspace(c) && !in_quotes) {
			if (!token.empty()) {
				args.push_back(token);
				token.clear();
			}
		} else {
			if (in_quotes) {
				quoted_arg += c;
			} else {
				token += c;
			}
		}
	}
	if (!token.empty()) args.push_back(token);
	if (!quoted_arg.empty()) args.push_back(quoted_arg);

	// Call common implementation with stdout output
	auto out = [](const std::string& msg) { std::cout << msg; };
	int result = handle_sched_args(args, out);
	return result == 0;
}
