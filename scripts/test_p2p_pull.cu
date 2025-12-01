#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Kernel to copy from src to dst
__global__ void copy_kernel(float *dst, float *src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

int main() {
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
    int n = 10;

    // Alloc on GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&d0, n * sizeof(float)));

    // Alloc on GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&d1, n * sizeof(float)));

    // ========== TEST 1: GPU 0 kernel reads from GPU 1 (GPU 0 pulls) ==========
    printf("=== TEST 1: GPU 0 kernel READS from GPU 1 memory (pull) ===\n");

    // Put data on GPU 1
    for (int i = 0; i < n; i++) h_src[i] = 100.0f + i;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(d1, h_src, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemset(d0, 0, n*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 0 kernel reads from d1 (GPU 1) and writes to d0 (GPU 0)
    copy_kernel<<<1, n>>>(d0, d1, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_dst, d0, n*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src (GPU1): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst (GPU0): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // ========== TEST 2: GPU 1 kernel reads from GPU 0 (GPU 1 pulls) ==========
    printf("=== TEST 2: GPU 1 kernel READS from GPU 0 memory (pull) ===\n");

    // Put data on GPU 0
    for (int i = 0; i < n; i++) h_src[i] = 200.0f + i;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(d0, h_src, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemset(d1, 0, n*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 1 kernel reads from d0 (GPU 0) and writes to d1 (GPU 1)
    copy_kernel<<<1, n>>>(d1, d0, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_dst, d1, n*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src (GPU0): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst (GPU1): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // ========== TEST 3: GPU 0 kernel writes to GPU 1 (GPU 0 pushes) ==========
    printf("=== TEST 3: GPU 0 kernel WRITES to GPU 1 memory (push) ===\n");

    // Put data on GPU 0
    for (int i = 0; i < n; i++) h_src[i] = 300.0f + i;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(d0, h_src, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemset(d1, 0, n*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 0 kernel reads from d0 (local) and writes to d1 (GPU 1)
    CHECK_CUDA(cudaSetDevice(0));
    copy_kernel<<<1, n>>>(d1, d0, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_dst, d1, n*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src (GPU0): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst (GPU1): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("\n  %s\n\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // ========== TEST 4: GPU 1 kernel writes to GPU 0 (GPU 1 pushes) ==========
    printf("=== TEST 4: GPU 1 kernel WRITES to GPU 0 memory (push) ===\n");

    // Put data on GPU 1
    for (int i = 0; i < n; i++) h_src[i] = 400.0f + i;
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMemcpy(d1, h_src, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemset(d0, 0, n*sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 1 kernel reads from d1 (local) and writes to d0 (GPU 0)
    CHECK_CUDA(cudaSetDevice(1));
    copy_kernel<<<1, n>>>(d0, d1, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_dst, d0, n*sizeof(float), cudaMemcpyDeviceToHost));

    printf("  src (GPU1): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("\n  dst (GPU0): ");
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
