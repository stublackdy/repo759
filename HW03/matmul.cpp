#include "matmul.h"

#include <algorithm>

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    // Each iteration owns one complete output row, including its initialization.
    // A and B are read-only; different threads never update the same C element.
#pragma omp parallel for default(none) shared(A, B, C, n) schedule(static)
    for (std::size_t i = 0; i < n; ++i) {
        std::fill_n(C + i * n, n, 0.0f);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
