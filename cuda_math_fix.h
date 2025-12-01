#ifndef CUDA_MATH_FIX_H
#define CUDA_MATH_FIX_H

// Work around for glibc 2.38+ incompatibility with CUDA
#ifdef __cplusplus
extern "C" {
#endif

#undef cospi
#undef sinpi
#undef cospif
#undef sinpif

#ifdef __cplusplus
}
#endif

#endif // CUDA_MATH_FIX_H
