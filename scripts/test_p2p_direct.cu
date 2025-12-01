#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CU error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

int main() {
    CHECK_CU(cuInit(0));

    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    printf("Found %d devices\n\n", deviceCount);

    if (deviceCount < 2) {
        printf("Need 2 GPUs\n");
        return 1;
    }

    // Enable P2P
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(1, 0));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(0, 0));
    printf("P2P enabled\n\n");

    float *d0, *d1;
    float h_src[10], h_dst[10];

    // Alloc on GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&d0, 10 * sizeof(float)));

    // Alloc on GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d1, 10 * sizeof(float)));

    // Get device pointers
    CUdeviceptr cu_d0 = (CUdeviceptr)d0;
    CUdeviceptr cu_d1 = (CUdeviceptr)d1;

    printf("d0 ptr: 0x%llx\n", (unsigned long long)d0);
    printf("d1 ptr: 0x%llx\n\n", (unsigned long long)d1);

    // ========== TEST GPU 0 -> GPU 1 ==========
    printf("=== TEST: GPU 0 -> GPU 1 ===\n");

    for (int i = 0; i < 10; i++) h_src[i] = 100.0f + i;

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(d0, h_src, 10*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemset(d1, 0, 10*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Use cuMemcpyDtoD
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CU(cuMemcpyDtoD(cu_d1, cu_d0, 10*sizeof(float)));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_dst, d1, 10*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // ========== TEST GPU 1 -> GPU 0 ==========
    printf("=== TEST: GPU 1 -> GPU 0 ===\n");

    for (int i = 0; i < 10; i++) h_src[i] = 200.0f + i;

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(d1, h_src, 10*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemset(d0, 0, 10*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Use cuMemcpyDtoD from GPU 1's context
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CU(cuMemcpyDtoD(cu_d0, cu_d1, 10*sizeof(float)));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_dst, d0, 10*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // ========== TEST GPU 1 -> GPU 0 using cudaMemcpy with cudaMemcpyDefault ==========
    printf("=== TEST: GPU 1 -> GPU 0 (cudaMemcpyDefault) ===\n");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemset(d0, 0, 10*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(d0, d1, 10*sizeof(float), cudaMemcpyDefault));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_dst, d0, 10*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst: ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d0));
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaFree(d1));

    printf("Done\n");
    return 0;
}
