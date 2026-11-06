#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "vscale.cuh"

// The required build uses 512. Only the comparison build overrides this to 16.
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#endif
static_assert(THREADS_PER_BLOCK == 512 || THREADS_PER_BLOCK == 16);

static void requireCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

// Release partially constructed resources as well as normally completed ones.
struct CudaResources {
    float *a = nullptr;
    float *b = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ~CudaResources() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (b) cudaFree(b);
        if (a) cudaFree(a);
    }
};

static unsigned int parseSize(int argc, char **argv) {
    if (argc != 2) {
        throw std::invalid_argument("Usage: ./task3 n (positive integer)");
    }
    const std::string text(argv[1]);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("n must be a positive integer");
    }
    const unsigned long long value = std::stoull(text);
    if (value == 0 || value > std::numeric_limits<unsigned int>::max() ||
        value > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("n is outside the supported range");
    }
    return static_cast<unsigned int>(value);
}

int main(int argc, char **argv) {
    try {
        const unsigned int n = parseSize(argc, argv);
        const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
        const unsigned int blocks = static_cast<unsigned int>(
            (static_cast<std::size_t>(n) + THREADS_PER_BLOCK - 1) /
            THREADS_PER_BLOCK);
        std::vector<float> a(n), b(n);
        std::mt19937 generator(17);
        std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
        std::uniform_real_distribution<float> distB(0.0f, 1.0f);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = distA(generator);
            b[i] = distB(generator);
        }

        CudaResources device;
        requireCuda(cudaMalloc(reinterpret_cast<void **>(&device.a), bytes),
                    "cudaMalloc(a)");
        requireCuda(cudaMalloc(reinterpret_cast<void **>(&device.b), bytes),
                    "cudaMalloc(b)");
        requireCuda(cudaMemcpy(device.a, a.data(), bytes, cudaMemcpyHostToDevice),
                    "copy a to device");
        requireCuda(cudaMemcpy(device.b, b.data(), bytes, cudaMemcpyHostToDevice),
                    "copy b to device");

        // Reuse the host a buffer for a full CPU reference, outside GPU timing.
        for (std::size_t i = 0; i < n; ++i) {
            a[i] *= b[i];
        }

        // Warm up once, then restore b so the measured call computes a*b once.
        vscale<<<blocks, THREADS_PER_BLOCK>>>(device.a, device.b, n);
        requireCuda(cudaGetLastError(), "warm-up launch");
        requireCuda(cudaDeviceSynchronize(), "warm-up execution");
        requireCuda(cudaMemcpy(device.b, b.data(), bytes, cudaMemcpyHostToDevice),
                    "restore b before timing");
        requireCuda(cudaDeviceSynchronize(), "finish input copies");

        requireCuda(cudaEventCreate(&device.start), "create start event");
        requireCuda(cudaEventCreate(&device.stop), "create stop event");
        requireCuda(cudaEventRecord(device.start), "record start event");
        vscale<<<blocks, THREADS_PER_BLOCK>>>(device.a, device.b, n);
        requireCuda(cudaGetLastError(), "timed kernel launch");
        requireCuda(cudaEventRecord(device.stop), "record stop event");
        requireCuda(cudaEventSynchronize(device.stop), "wait for stop event");
        float milliseconds = 0.0f;
        requireCuda(cudaEventElapsedTime(&milliseconds, device.start, device.stop),
                    "measure elapsed time");

        requireCuda(cudaMemcpy(b.data(), device.b, bytes, cudaMemcpyDeviceToHost),
                    "copy result to host");
        for (std::size_t i = 0; i < n; ++i) {
            const float tolerance = 1.0e-6f + 1.0e-6f * std::fabs(a[i]);
            if (!std::isfinite(b[i]) || std::fabs(b[i] - a[i]) > tolerance) {
                throw std::runtime_error("CPU reference mismatch at index " +
                                         std::to_string(i));
            }
        }
        std::printf("%.9g\n%.9g\n%.9g\n", milliseconds, b.front(), b.back());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
