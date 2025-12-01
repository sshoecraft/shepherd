#ifndef CUDA_GLIBC_FIX_H
#define CUDA_GLIBC_FIX_H

// Workaround for glibc 2.38+ / CUDA 12.9 incompatibility
#define __THROW
#include <cuda_runtime.h>
#undef __THROW

#endif
