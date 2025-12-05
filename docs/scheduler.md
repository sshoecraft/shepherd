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

3. **Scheduler** - Main class handling persistence, CRUD operations, and the background worker thread

### Thread Model

- Background worker thread started in `run_cli()`
- Wakes up at each minute boundary + 1 second
- Checks all enabled schedules against current time
- Injects matching prompts into `tio.add_input()`
- Thread-safe access via mutex

### Storage

Schedules are persisted to `~/.config/shepherd/schedule.json`:

```json
{
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
}
```

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

- Scheduler only runs when CLI is active (not in server mode)
- Missed schedules (shepherd wasn't running) are skipped silently
- `last_run` prevents double-firing within the same minute
- Schedules persist across sessions
- Scheduled prompts are injected into the input queue and processed after the next user input

## Version History

- **2.6.0** - Initial implementation of scheduler functionality
- **2.6.1** - Added get_time and get_date tools
