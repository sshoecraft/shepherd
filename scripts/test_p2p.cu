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

int main(int argc, char *argv[]) {
    int device_count;
    int can_access_0_to_1, can_access_1_to_0;
    cudaDeviceProp prop0, prop1;

    // Check device count
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);

    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs for P2P test\n");
        return EXIT_FAILURE;
    }

    // Get device properties
    CHECK_CUDA(cudaGetDeviceProperties(&prop0, GPU_0));
    CHECK_CUDA(cudaGetDeviceProperties(&prop1, GPU_1));

    printf("\nGPU %d: %s\n", GPU_0, prop0.name);
    printf("GPU %d: %s\n", GPU_1, prop1.name);

    // Check P2P access capability
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_0_to_1, GPU_0, GPU_1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_1_to_0, GPU_1, GPU_0));

    printf("\nP2P Access Capability:\n");
    printf("  GPU %d -> GPU %d: %s\n", GPU_0, GPU_1,
           can_access_0_to_1 ? "YES" : "NO");
    printf("  GPU %d -> GPU %d: %s\n", GPU_1, GPU_0,
           can_access_1_to_0 ? "YES" : "NO");

    if (!can_access_0_to_1 || !can_access_1_to_0) {
        printf("\nP2P access not fully supported between GPU %d and GPU %d\n",
               GPU_0, GPU_1);
        return EXIT_FAILURE;
    }

    // Enable P2P access
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(GPU_1, 0));

    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(GPU_0, 0));

    printf("\nP2P access enabled successfully\n");

    // Allocate memory on both GPUs
    float *d_buf0, *d_buf1;
    float *h_src, *h_dst;

    h_src = (float *)malloc(TEST_SIZE);
    h_dst = (float *)malloc(TEST_SIZE);

    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize source buffer
    size_t num_floats = TEST_SIZE / sizeof(float);
    for (size_t i = 0; i < num_floats; i++) {
        h_src[i] = (float)i;
    }

    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaMalloc(&d_buf0, TEST_SIZE));
    CHECK_CUDA(cudaMemcpy(d_buf0, h_src, TEST_SIZE, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMalloc(&d_buf1, TEST_SIZE));

    // Test P2P copy: GPU 0 -> GPU 1
    printf("\nTesting P2P copy: GPU %d -> GPU %d (%zu bytes)...\n",
           GPU_0, GPU_1, (size_t)TEST_SIZE);

    CHECK_CUDA(cudaMemcpyPeer(d_buf1, GPU_1, d_buf0, GPU_0, TEST_SIZE));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify the copy
    CHECK_CUDA(cudaMemcpy(h_dst, d_buf1, TEST_SIZE, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < num_floats; i++) {
        if (h_dst[i] != h_src[i]) {
            if (errors < 10) {
                printf("  Mismatch at index %zu: expected %f, got %f\n",
                       i, h_src[i], h_dst[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("P2P copy GPU %d -> GPU %d: PASSED\n", GPU_0, GPU_1);
    } else {
        printf("P2P copy GPU %d -> GPU %d: FAILED (%d errors)\n",
               GPU_0, GPU_1, errors);
    }

    // Test P2P copy: GPU 1 -> GPU 0
    printf("\nTesting P2P copy: GPU %d -> GPU %d (%zu bytes)...\n",
           GPU_1, GPU_0, (size_t)TEST_SIZE);

    // Modify data on GPU 1
    for (size_t i = 0; i < num_floats; i++) {
        h_src[i] = (float)(num_floats - i);
    }
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaMemcpy(d_buf1, h_src, TEST_SIZE, cudaMemcpyHostToDevice));

    // P2P copy back
    CHECK_CUDA(cudaMemcpyPeer(d_buf0, GPU_0, d_buf1, GPU_1, TEST_SIZE));
    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify
    CHECK_CUDA(cudaMemcpy(h_dst, d_buf0, TEST_SIZE, cudaMemcpyDeviceToHost));

    errors = 0;
    for (size_t i = 0; i < num_floats; i++) {
        if (h_dst[i] != h_src[i]) {
            if (errors < 10) {
                printf("  Mismatch at index %zu: expected %f, got %f\n",
                       i, h_src[i], h_dst[i]);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("P2P copy GPU %d -> GPU %d: PASSED\n", GPU_1, GPU_0);
    } else {
        printf("P2P copy GPU %d -> GPU %d: FAILED (%d errors)\n",
               GPU_1, GPU_0, errors);
    }

    // Bandwidth test
    printf("\nP2P Bandwidth Test:\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int iterations = 100;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpyPeer(d_buf1, GPU_1, d_buf0, GPU_0, TEST_SIZE));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double gb = (double)TEST_SIZE * iterations / (1024.0 * 1024.0 * 1024.0);
    double seconds = ms / 1000.0;
    printf("  GPU %d -> GPU %d: %.2f GB/s\n", GPU_0, GPU_1, gb / seconds);

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpyPeer(d_buf0, GPU_0, d_buf1, GPU_1, TEST_SIZE));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("  GPU %d -> GPU %d: %.2f GB/s\n", GPU_1, GPU_0, gb / seconds);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaSetDevice(GPU_0));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(GPU_1));
    CHECK_CUDA(cudaFree(d_buf0));

    CHECK_CUDA(cudaSetDevice(GPU_1));
    CHECK_CUDA(cudaDeviceDisablePeerAccess(GPU_0));
    CHECK_CUDA(cudaFree(d_buf1));

    free(h_src);
    free(h_dst);

    printf("\nP2P test completed successfully\n");
    return EXIT_SUCCESS;
}
