#pragma once

// Get the original command-line arguments passed to main()
// Used by backends that need to re-exec (e.g., TensorRT with MPI)
void get_global_args(int& argc, char**& argv);
