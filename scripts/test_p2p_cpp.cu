#include <cuda_runtime.h>
#include "cuda_math_fix.h"
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d GPUs\n", deviceCount);

    if (deviceCount < 2) {
        printf("Need at least 2 GPUs\n");
        return 1;
    }

    // Check P2P access between GPU 0 and GPU 1
    int canAccessPeer01, canAccessPeer10;
    cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0);

    printf("GPU 0 -> GPU 1: %s\n", canAccessPeer01 ? "YES" : "NO");
    printf("GPU 1 -> GPU 0: %s\n", canAccessPeer10 ? "YES" : "NO");

    if (canAccessPeer01 && canAccessPeer10) {
        printf("\nEnabling P2P access...\n");
        cudaSetDevice(0);
        cudaError_t err01 = cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaError_t err10 = cudaDeviceEnablePeerAccess(0, 0);

        if (err01 == cudaSuccess && err10 == cudaSuccess) {
            printf("P2P access ENABLED successfully!\n");
        } else {
            printf("Failed to enable P2P: %s, %s\n",
                   cudaGetErrorString(err01), cudaGetErrorString(err10));
        }
    } else {
        printf("\nP2P NOT supported between these GPUs\n");
    }

    return 0;
}
