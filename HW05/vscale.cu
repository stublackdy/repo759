#include <cstddef>
#include <cuda_runtime.h>
#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (i < n) {
        b[i] = a[i] * b[i];
    }
}
