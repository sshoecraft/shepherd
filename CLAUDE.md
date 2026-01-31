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
cd build && cmake .. && make
```
