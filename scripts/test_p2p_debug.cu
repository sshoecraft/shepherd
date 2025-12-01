#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define GPU_0 0
#define GPU_1 1
#define TEST_SIZE (1024 * 1024)  // 1 MB

__global__ void fill_kernel(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value + idx;
    }
}

__global__ void verify_kernel(float* data, int n, float expected_base, int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float expected = expected_base + idx;
        if (data[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

int main(int argc, char *argv[]) {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n\n", device_count);

    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs for P2P test\n");
        return EXIT_FAILURE;
    }

    // Enable P2P
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(GPU_1, 0));
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(GPU_0, 0));
    printf("P2P access enabled\n\n");

    size_t num_floats = TEST_SIZE / sizeof(float);
    float *d_buf0, *d_buf1;
    int *d_errors0, *d_errors1;
    int h_errors;

    // Allocate on GPU 0
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMalloc(&d_buf0, TEST_SIZE));
    CHECK_CUDA(cudaMalloc(&d_errors0, sizeof(int)));

    // Allocate on GPU 1
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMalloc(&d_buf1, TEST_SIZE));
    CHECK_CUDA(cudaMalloc(&d_errors1, sizeof(int)));

    int blocks = (num_floats + 255) / 256;

    // ==================== TEST 1: GPU 0 -> GPU 1 using cudaMemcpyPeer ====================
    printf("=== TEST 1: cudaMemcpyPeer GPU 0 -> GPU 1 ===\n");

    // Fill GPU 0 buffer with 1000 + idx
    CHECK_CUDA(cudaSetDevice(GPU_0));
    fill_kernel<<<blocks, 256>>>(d_buf0, num_floats, 1000.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 1 buffer
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemset(d_buf1, 0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // P2P copy
    CHECK_CUDA(cudaMemcpyPeer(d_buf1, GPU_1, d_buf0, GPU_0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify on GPU 1
    CHECK_CUDA(cudaMemset(d_errors1, 0, sizeof(int)));
    verify_kernel<<<blocks, 256>>>(d_buf1, num_floats, 1000.0f, d_errors1);
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Errors: %d\n\n", h_errors);

    // ==================== TEST 2: GPU 1 -> GPU 0 using cudaMemcpyPeer ====================
    printf("=== TEST 2: cudaMemcpyPeer GPU 1 -> GPU 0 ===\n");

    // Fill GPU 1 buffer with 2000 + idx
    CHECK_CUDA(cudaSetDevice(GPU_1));
    fill_kernel<<<blocks, 256>>>(d_buf1, num_floats, 2000.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Clear GPU 0 buffer
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMemset(d_buf0, 0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // P2P copy
    CHECK_CUDA(cudaMemcpyPeer(d_buf0, GPU_0, d_buf1, GPU_1, TEST_SIZE));
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify on GPU 0
    CHECK_CUDA(cudaMemset(d_errors0, 0, sizeof(int)));
    verify_kernel<<<blocks, 256>>>(d_buf0, num_floats, 2000.0f, d_errors0);
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors0, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Errors: %d\n\n", h_errors);

    // ==================== TEST 3: Direct P2P write from GPU 0 kernel to GPU 1 ====================
    printf("=== TEST 3: Direct kernel write GPU 0 -> GPU 1 memory ===\n");

    // Clear GPU 1 buffer
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemset(d_buf1, 0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 0 kernel writes directly to GPU 1 memory
    CHECK_CUDA(cudaSetDevice(GPU_0));
    fill_kernel<<<blocks, 256>>>(d_buf1, num_floats, 3000.0f);  // Writing to d_buf1 which is on GPU 1!
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify on GPU 1
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemset(d_errors1, 0, sizeof(int)));
    verify_kernel<<<blocks, 256>>>(d_buf1, num_floats, 3000.0f, d_errors1);
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Errors: %d\n\n", h_errors);

    // ==================== TEST 4: Direct P2P write from GPU 1 kernel to GPU 0 ====================
    printf("=== TEST 4: Direct kernel write GPU 1 -> GPU 0 memory ===\n");

    // Clear GPU 0 buffer
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMemset(d_buf0, 0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // GPU 1 kernel writes directly to GPU 0 memory
    CHECK_CUDA(cudaSetDevice(GPU_1));
    fill_kernel<<<blocks, 256>>>(d_buf0, num_floats, 4000.0f);  // Writing to d_buf0 which is on GPU 0!
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify on GPU 0
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMemset(d_errors0, 0, sizeof(int)));
    verify_kernel<<<blocks, 256>>>(d_buf0, num_floats, 4000.0f, d_errors0);
    CHECK_CUDA(cudaMemcpy(&h_errors, d_errors0, sizeof(int), cudaMemcpyDeviceToHost));
    printf("  Errors: %d\n\n", h_errors);

    // ==================== TEST 5: Simple direct test ====================
    printf("=== TEST 5: Simple P2P copy test ===\n");
    float h_src[10], h_dst[10];

    // Initialize source on GPU 0
    for (int i = 0; i < 10; i++) h_src[i] = 100.0f + i;
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMemcpy(d_buf0, h_src, 10 * sizeof(float), cudaMemcpyHostToDevice));

    // Clear dest on GPU 1
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemset(d_buf1, 0, 10 * sizeof(float)));

    // P2P copy GPU 0 -> GPU 1
    CHECK_CUDA(cudaMemcpyPeer(d_buf1, GPU_1, d_buf0, GPU_0, 10 * sizeof(float)));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back from GPU 1
    CHECK_CUDA(cudaMemcpy(h_dst, d_buf1, 10 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU 0 -> GPU 1: src=");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("... dst=");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("... %s\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // Now test GPU 1 -> GPU 0
    for (int i = 0; i < 10; i++) h_src[i] = 200.0f + i;
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemcpy(d_buf1, h_src, 10 * sizeof(float), cudaMemcpyHostToDevice));

    // Clear dest on GPU 0
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMemset(d_buf0, 0, 10 * sizeof(float)));

    // P2P copy GPU 1 -> GPU 0
    CHECK_CUDA(cudaMemcpyPeer(d_buf0, GPU_0, d_buf1, GPU_1, 10 * sizeof(float)));
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back from GPU 0
    CHECK_CUDA(cudaMemcpy(h_dst, d_buf0, 10 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU 1 -> GPU 0: src=");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_src[i]);
    printf("... dst=");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_dst[i]);
    printf("... %s\n", (h_dst[0] == h_src[0]) ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaFree(d_buf0));
    CHECK_CUDA(cudaFree(d_errors0));
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaFree(d_buf1));
    CHECK_CUDA(cudaFree(d_errors1));

    printf("\nDone.\n");
    return 0;
}
