# Shepherd Project Notes

## Version Management
- Version is defined in **CMakeLists.txt** (line 3): `VERSION x.y.z`
- `version.h` is auto-generated from `version.h.in` via cmake's `configure_file()`
- Do NOT manually edit `version.h` - edit CMakeLists.txt and rebuild

## MCP/SMCP Servers
- Initialization is parallelized using std::thread (as of v2.25.0)
- Child processes use `prctl(PR_SET_PDEATHSIG, SIGTERM)` to auto-terminate when parent dies
- Shutdown is non-blocking - no waiting on child processes

## Build
```bash
make            # build
make config     # reconfigure cmake options (reads ~/.shepherd_opts)
```

## Git Rules - CRITICAL
**DO NOT USE GIT COMMANDS. EVER. NOT UNDER ANY CIRCUMSTANCES.**

- NO `git checkout` - NEVER
- NO `git reset` - NEVER  
- NO `git revert` - NEVER
- NO `git clean` - NEVER
- NO `git stash` - NEVER
- NO `git restore` - NEVER
- NO ANY git command that modifies files or state - NEVER

The ONLY exception is if the user EXPLICITLY says "use git to..." or "run git..." or otherwise directly authorizes a specific git command.

Read-only commands (git status, git diff, git log, git show) are OK but DO NOT run any git command that changes anything.
