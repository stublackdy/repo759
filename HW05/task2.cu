#include <cstdio>
#include <random>
#include <cuda_runtime.h>

__global__ void fillArray(int *dA, int a) {
    const int x = static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.x);
    const int index = y * static_cast<int>(blockDim.x) + x;
    dA[index] = a * x + y;
}

static bool checkCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

int main() {
    // A fixed seed makes the pseudorandom input reproducible.
    std::mt19937 generator(17);
    std::uniform_int_distribution<int> distribution(1, 100);
    const int a = distribution(generator);

    int hA[16] = {};
    int *dA = nullptr;
    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&dA), sizeof(hA)),
                   "cudaMalloc")) {
        return 1;
    }

    fillArray<<<2, 8>>>(dA, a);
    if (!checkCuda(cudaGetLastError(), "Kernel launch") ||
        !checkCuda(cudaDeviceSynchronize(), "Kernel execution") ||
        !checkCuda(cudaMemcpy(hA, dA, sizeof(hA), cudaMemcpyDeviceToHost),
                   "cudaMemcpy")) {
        cudaFree(dA);
        return 1;
    }
    if (!checkCuda(cudaFree(dA), "cudaFree")) {
        return 1;
    }

    for (int i = 0; i < 16; ++i) {
        std::printf("%s%d", i == 0 ? "" : " ", hA[i]);
    }
    std::printf("\n");
    return 0;
}
