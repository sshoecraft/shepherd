#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            return 0;                                                          \
        }                                                                      \
    } while (0)

#define TEST_SIZE (1024 * 1024)  // 1 MB

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int test_p2p_pair(int gpu0, int gpu1) {
    int can_access_0_to_1, can_access_1_to_0;
    cudaDeviceProp prop0, prop1;

    // Get device properties
    CHECK_CUDA(cudaGetDeviceProperties(&prop0, gpu0));
    CHECK_CUDA(cudaGetDeviceProperties(&prop1, gpu1));

    printf("\nTesting GPU %d (%s) <-> GPU %d (%s)\n",
           gpu0, prop0.name, gpu1, prop1.name);

    // Check P2P access capability
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_0_to_1, gpu0, gpu1));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access_1_to_0, gpu1, gpu0));

    printf("  P2P Capability: GPU %d -> GPU %d: %s, GPU %d -> GPU %d: %s\n",
           gpu0, gpu1, can_access_0_to_1 ? "YES" : "NO",
           gpu1, gpu0, can_access_1_to_0 ? "YES" : "NO");

    if (!can_access_0_to_1 || !can_access_1_to_0) {
        printf("  P2P not supported - skipping\n");
        return 0;
    }

    // Enable P2P access
    cudaError_t err0, err1;
    cudaSetDevice(gpu0);
    err0 = cudaDeviceEnablePeerAccess(gpu1, 0);

    cudaSetDevice(gpu1);
    err1 = cudaDeviceEnablePeerAccess(gpu0, 0);

    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("  P2P enable FAILED: %s / %s\n",
               cudaGetErrorString(err0), cudaGetErrorString(err1));
        cudaGetLastError(); // Clear error
        return 0;
    }

    printf("  P2P access enabled successfully\n");

    // Allocate memory
    float *d_buf0, *d_buf1;

    cudaSetDevice(gpu0);
    CHECK_CUDA(cudaMalloc(&d_buf0, TEST_SIZE));

    cudaSetDevice(gpu1);
    CHECK_CUDA(cudaMalloc(&d_buf1, TEST_SIZE));

    // Test P2P copy
    cudaSetDevice(gpu0);
    CHECK_CUDA(cudaMemcpy(d_buf1, d_buf0, TEST_SIZE, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("  P2P copy test: PASSED\n");

    // Bandwidth test
    const int iterations = 100;
    double start = get_time();
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaMemcpy(d_buf1, d_buf0, TEST_SIZE, cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = get_time();

    double bandwidth = (TEST_SIZE * iterations) / (end - start) / 1e9;
    printf("  P2P Bandwidth GPU %d -> GPU %d: %.2f GB/s\n", gpu0, gpu1, bandwidth);

    // Cleanup
    cudaFree(d_buf0);
    cudaFree(d_buf1);

    // Disable P2P
    cudaSetDevice(gpu0);
    cudaDeviceDisablePeerAccess(gpu1);
    cudaSetDevice(gpu1);
    cudaDeviceDisablePeerAccess(gpu0);

    return 1;
}

int main(int argc, char *argv[]) {
    int device_count;

    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);
    printf("==========================================\n");

    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs for P2P test\n");
        return EXIT_FAILURE;
    }

    // Test all pairs
    int success_count = 0;
    for (int i = 0; i < device_count; i++) {
        for (int j = i + 1; j < device_count; j++) {
            if (test_p2p_pair(i, j)) {
                success_count++;
            }
        }
    }

    printf("\n==========================================\n");
    printf("P2P working on %d out of %d possible pairs\n",
           success_count, (device_count * (device_count - 1)) / 2);

    return EXIT_SUCCESS;
}
