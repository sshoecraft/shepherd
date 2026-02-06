# Scheduler Module

## Overview

The scheduler module provides cron-like scheduling for LLM prompts. Scheduled prompts are injected into the input queue when their scheduled time arrives, allowing automated interactions with the LLM.

## Architecture

### Files

- `scheduler.h` - Class declaration and data structures
- `scheduler.cpp` - Implementation

### Key Components

1. **ScheduleEntry** - Data structure for a single schedule
   - `id` - Unique identifier (8-char random string)
   - `name` - User-friendly name
   - `cron` - Cron expression (5 fields)
   - `prompt` - The prompt to inject
   - `enabled` - Whether schedule is active
   - `last_run` - ISO 8601 timestamp of last execution
   - `created` - ISO 8601 timestamp of creation

2. **CronExpr** - Parsed cron expression with sets of valid values for each field

3. **Scheduler** - Main class handling persistence, CRUD operations, and SIGALRM-based execution

### Timer Model (v2.7.0+)

- Uses SIGALRM signal instead of background thread
- `alarm()` set to fire at each minute boundary + 1 second
- Signal handler checks all enabled schedules against current time
- Injects matching prompts via callback to frontend's `add_input()`
- Thread-safe access via mutex
- No polling or sleep loops - efficient timer-based wakeup

### Storage

Schedules are stored in `config.json` under a `schedulers[]` array. Each entry is a named scheduler with its own list of schedules:

```json
{
  "schedulers": [
    {
      "name": "default",
      "schedules": [
        {
          "id": "abc12345",
          "name": "daily-summary",
          "cron": "0 9 * * *",
          "prompt": "Give me a summary",
          "enabled": true,
          "last_run": "2025-12-03T09:00:00",
          "created": "2025-12-01T10:00:00"
        }
      ]
    },
    {
      "name": "moltbook",
      "schedules": [...]
    }
  ]
}
```

### Named Schedulers (v2.26.0+)

Each shepherd instance can use a specific named scheduler via the `--scheduler` flag:

```bash
shepherd --scheduler default           # Uses "default" scheduler (the default)
shepherd --cliserver --scheduler moltbook --port 8003  # Uses "moltbook" scheduler
```

This allows different instances to run independent schedules. A CLI server for a specific task can have its own schedules that don't fire on other instances.

The `--scheduler` flag also works with the `sched` subcommand:

```bash
shepherd sched --scheduler moltbook list
shepherd sched --scheduler moltbook add feed_check "*/5 * * * *" "Check feed"
```

### Migration

On first load, if no `schedulers[]` exists in config.json but a legacy `schedule.json` file is found, its entries are automatically migrated into a "default" scheduler and the old file is renamed to `schedule.json.migrated`.

## Cron Format

Standard 5-field cron expression:

```
minute hour day month weekday
  │      │    │    │     │
  │      │    │    │     └─ 0-6 (0=Sunday)
  │      │    │    └─────── 1-12
  │      │    └──────────── 1-31
  │      └───────────────── 0-23
  └──────────────────────── 0-59
```

### Supported Syntax

- `*` - Any value
- `N` - Specific value
- `N-M` - Range (inclusive)
- `N,M,O` - List
- `*/N` - Step (every N)
- `N-M/S` - Range with step

## CLI Interface

### Command Line (`shepherd sched`)

```bash
shepherd sched                    # List all schedules
shepherd sched list               # List all schedules
shepherd sched add <name> "<cron>" "<prompt>"
shepherd sched remove <name|id>
shepherd sched enable <name|id>
shepherd sched disable <name|id>
shepherd sched show <name|id>
shepherd sched next [name|id]
```

### Interactive (`/sched`)

Same commands available as slash commands within the CLI:

```
/sched
/sched list
/sched add daily "0 9 * * *" "Good morning!"
/sched remove daily
/sched enable daily
/sched disable daily
/sched show daily
/sched next
```

## Behavior

- Scheduler runs in CLI, TUI, and CLI Server modes
- Missed schedules (shepherd wasn't running) are skipped silently
- `last_run` prevents double-firing within the same minute
- Schedules persist across sessions
- Scheduled prompts are injected into the input queue and processed automatically

## LLM Tools

The model can manage schedules using these tools:

| Tool | Description |
|------|-------------|
| `list_schedules` | List all scheduled prompts with their status |
| `add_schedule` | Create a new scheduled prompt |
| `remove_schedule` | Delete a schedule by name or ID |
| `enable_schedule` | Enable a disabled schedule |
| `disable_schedule` | Disable a schedule without removing it |
| `get_schedule` | Get details of a specific schedule |

## Version History

- **2.6.0** - Initial implementation of scheduler functionality
- **2.6.1** - Added get_time and get_date tools
- **2.7.0** - Replaced worker thread with SIGALRM-based timer
- **2.22.0** - Implemented prompt injection via callback, added scheduler management tools, CLI Server support
- **2.26.0** - Named schedulers: schedules moved from standalone `schedule.json` into `config.json` under `schedulers[]` array. Added `--scheduler <name>` flag for per-instance schedule isolation. Auto-migration of legacy `schedule.json`. Read-only mode guards for Key Vault config.
