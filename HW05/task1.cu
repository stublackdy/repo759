#include <cstdio>
#include <cuda_runtime.h>

__global__ void factorialKernel() {
    const int a = static_cast<int>(threadIdx.x) + 1;
    int result = 1;
    for (int i = 2; i <= a; ++i) {
        result *= i;
    }
    std::printf("%d!=%d\n", a, result);
}

int main() {
    factorialKernel<<<1, 8>>>();

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }

    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Kernel execution failed: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }
    return 0;
}
