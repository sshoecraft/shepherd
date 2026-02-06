#include "shepherd.h"
#include "scheduler_tools.h"
#include "tools.h"
#include "../scheduler.h"
#include "../config.h"

#include <sstream>

// ListSchedulesTool

std::string ListSchedulesTool::unsanitized_name() const {
    return "list_schedules";
}

std::string ListSchedulesTool::description() const {
    return "List all scheduled prompts with their cron expressions and status";
}

std::string ListSchedulesTool::parameters() const {
    return "";
}

std::vector<ParameterDef> ListSchedulesTool::get_parameters_schema() const {
    return {};
}

std::map<std::string, std::any> ListSchedulesTool::execute(const std::map<std::string, std::any>& args) {
    (void)args;
    std::map<std::string, std::any> result;

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    auto schedules = scheduler.list();

    if (schedules.empty()) {
        result["content"] = std::string("No schedules configured");
        result["summary"] = std::string("0 schedules");
        return result;
    }

    std::ostringstream oss;
    oss << "Schedules (" << schedules.size() << "):\n\n";

    for (const auto& entry : schedules) {
        oss << "- " << entry.name << " [" << entry.id << "]\n";
        oss << "  Cron: " << entry.cron << "\n";
        oss << "  Status: " << (entry.enabled ? "enabled" : "disabled") << "\n";
        oss << "  Prompt: " << entry.prompt.substr(0, 100);
        if (entry.prompt.length() > 100) oss << "...";
        oss << "\n";
        if (!entry.last_run.empty()) {
            oss << "  Last run: " << entry.last_run << "\n";
        }
        oss << "  Next: " << Scheduler::format_next_run(entry.cron) << "\n";
        oss << "\n";
    }

    result["content"] = oss.str();
    result["summary"] = std::to_string(schedules.size()) + " schedule(s)";
    return result;
}

// AddScheduleTool

std::string AddScheduleTool::unsanitized_name() const {
    return "add_schedule";
}

std::string AddScheduleTool::description() const {
    return "Add a new scheduled prompt with a cron expression. Cron format: minute hour day month weekday (e.g., '0 9 * * *' for 9am daily)";
}

std::string AddScheduleTool::parameters() const {
    return "name, cron, prompt";
}

std::vector<ParameterDef> AddScheduleTool::get_parameters_schema() const {
    return {
        {"name", "string", "Unique name for the schedule", true, "", "", {}},
        {"cron", "string", "Cron expression (5 fields: minute hour day month weekday)", true, "", "", {}},
        {"prompt", "string", "The prompt to inject when the schedule fires", true, "", "", {}}
    };
}

std::map<std::string, std::any> AddScheduleTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    if (config && config->is_read_only()) {
        result["error"] = std::string("Cannot modify schedules in read-only mode");
        return result;
    }

    std::string name = tool_utils::get_string(args, "name");
    std::string cron = tool_utils::get_string(args, "cron");
    std::string prompt = tool_utils::get_string(args, "prompt");

    if (name.empty()) {
        result["error"] = std::string("name is required");
        return result;
    }
    if (cron.empty()) {
        result["error"] = std::string("cron expression is required");
        return result;
    }
    if (prompt.empty()) {
        result["error"] = std::string("prompt is required");
        return result;
    }

    // Validate cron expression
    std::string error;
    if (!Scheduler::validate_cron(cron, error)) {
        result["error"] = std::string("Invalid cron expression: ") + error;
        return result;
    }

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    // Check if name already exists
    if (scheduler.get(name) != nullptr) {
        result["error"] = std::string("Schedule with name '" + name + "' already exists");
        return result;
    }

    std::string id = scheduler.add(name, cron, prompt);

    result["id"] = id;
    result["content"] = "Created schedule '" + name + "' (id: " + id + ")\nCron: " + cron + "\nNext run: " + Scheduler::format_next_run(cron);
    result["summary"] = "Schedule '" + name + "' created";
    return result;
}

// RemoveScheduleTool

std::string RemoveScheduleTool::unsanitized_name() const {
    return "remove_schedule";
}

std::string RemoveScheduleTool::description() const {
    return "Remove a scheduled prompt by name or ID";
}

std::string RemoveScheduleTool::parameters() const {
    return "name_or_id";
}

std::vector<ParameterDef> RemoveScheduleTool::get_parameters_schema() const {
    return {
        {"name_or_id", "string", "The name or ID of the schedule to remove", true, "", "", {}}
    };
}

std::map<std::string, std::any> RemoveScheduleTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    if (config && config->is_read_only()) {
        result["error"] = std::string("Cannot modify schedules in read-only mode");
        return result;
    }

    std::string name_or_id = tool_utils::get_string(args, "name_or_id");

    if (name_or_id.empty()) {
        result["error"] = std::string("name_or_id is required");
        return result;
    }

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    if (!scheduler.remove(name_or_id)) {
        result["error"] = std::string("Schedule '" + name_or_id + "' not found");
        return result;
    }

    result["content"] = "Removed schedule '" + name_or_id + "'";
    result["summary"] = "Schedule removed";
    return result;
}

// EnableScheduleTool

std::string EnableScheduleTool::unsanitized_name() const {
    return "enable_schedule";
}

std::string EnableScheduleTool::description() const {
    return "Enable a disabled scheduled prompt";
}

std::string EnableScheduleTool::parameters() const {
    return "name_or_id";
}

std::vector<ParameterDef> EnableScheduleTool::get_parameters_schema() const {
    return {
        {"name_or_id", "string", "The name or ID of the schedule to enable", true, "", "", {}}
    };
}

std::map<std::string, std::any> EnableScheduleTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    if (config && config->is_read_only()) {
        result["error"] = std::string("Cannot modify schedules in read-only mode");
        return result;
    }

    std::string name_or_id = tool_utils::get_string(args, "name_or_id");

    if (name_or_id.empty()) {
        result["error"] = std::string("name_or_id is required");
        return result;
    }

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    if (!scheduler.enable(name_or_id)) {
        result["error"] = std::string("Schedule '" + name_or_id + "' not found");
        return result;
    }

    result["content"] = "Enabled schedule '" + name_or_id + "'";
    result["summary"] = "Schedule enabled";
    return result;
}

// DisableScheduleTool

std::string DisableScheduleTool::unsanitized_name() const {
    return "disable_schedule";
}

std::string DisableScheduleTool::description() const {
    return "Disable a scheduled prompt without removing it";
}

std::string DisableScheduleTool::parameters() const {
    return "name_or_id";
}

std::vector<ParameterDef> DisableScheduleTool::get_parameters_schema() const {
    return {
        {"name_or_id", "string", "The name or ID of the schedule to disable", true, "", "", {}}
    };
}

std::map<std::string, std::any> DisableScheduleTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    if (config && config->is_read_only()) {
        result["error"] = std::string("Cannot modify schedules in read-only mode");
        return result;
    }

    std::string name_or_id = tool_utils::get_string(args, "name_or_id");

    if (name_or_id.empty()) {
        result["error"] = std::string("name_or_id is required");
        return result;
    }

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    if (!scheduler.disable(name_or_id)) {
        result["error"] = std::string("Schedule '" + name_or_id + "' not found");
        return result;
    }

    result["content"] = "Disabled schedule '" + name_or_id + "'";
    result["summary"] = "Schedule disabled";
    return result;
}

// GetScheduleTool

std::string GetScheduleTool::unsanitized_name() const {
    return "get_schedule";
}

std::string GetScheduleTool::description() const {
    return "Get details of a specific scheduled prompt";
}

std::string GetScheduleTool::parameters() const {
    return "name_or_id";
}

std::vector<ParameterDef> GetScheduleTool::get_parameters_schema() const {
    return {
        {"name_or_id", "string", "The name or ID of the schedule to retrieve", true, "", "", {}}
    };
}

std::map<std::string, std::any> GetScheduleTool::execute(const std::map<std::string, std::any>& args) {
    std::map<std::string, std::any> result;

    std::string name_or_id = tool_utils::get_string(args, "name_or_id");

    if (name_or_id.empty()) {
        result["error"] = std::string("name_or_id is required");
        return result;
    }

    Scheduler scheduler(config ? config->scheduler_name : "default");
    scheduler.load();

    const auto* entry = scheduler.get(name_or_id);
    if (!entry) {
        result["error"] = std::string("Schedule '" + name_or_id + "' not found");
        return result;
    }

    std::ostringstream oss;
    oss << "Schedule: " << entry->name << "\n";
    oss << "ID: " << entry->id << "\n";
    oss << "Cron: " << entry->cron << "\n";
    oss << "Status: " << (entry->enabled ? "enabled" : "disabled") << "\n";
    oss << "Prompt: " << entry->prompt << "\n";
    oss << "Created: " << entry->created << "\n";
    if (!entry->last_run.empty()) {
        oss << "Last run: " << entry->last_run << "\n";
    }
    oss << "Next run: " << Scheduler::format_next_run(entry->cron) << "\n";

    result["content"] = oss.str();
    result["summary"] = "Schedule: " + entry->name;
    return result;
}

// Registration

void register_scheduler_tools(Tools& tools) {
    tools.register_tool(std::make_unique<ListSchedulesTool>());
    tools.register_tool(std::make_unique<AddScheduleTool>());
    tools.register_tool(std::make_unique<RemoveScheduleTool>());
    tools.register_tool(std::make_unique<EnableScheduleTool>());
    tools.register_tool(std::make_unique<DisableScheduleTool>());
    tools.register_tool(std::make_unique<GetScheduleTool>());

    dout(1) << "Registered scheduler tools: list_schedules, add_schedule, remove_schedule, enable_schedule, disable_schedule, get_schedule" << std::endl;
}
