#include "convolution.h"

void convolve(const float* image, float* output, std::size_t n,
              const float* mask, std::size_t m) {
    const auto radius = static_cast<std::ptrdiff_t>(m / 2);
    const auto side = static_cast<std::ptrdiff_t>(n);

    // Each iteration owns one output row. The input and mask are read-only.
#pragma omp parallel for default(none) shared(image, output, n, mask, m, radius, side) schedule(static)
    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; ++i) {
                const auto row = static_cast<std::ptrdiff_t>(x) +
                                 (static_cast<std::ptrdiff_t>(i) - radius);
                const bool row_inside = row >= 0 && row < side;
                for (std::size_t j = 0; j < m; ++j) {
                    const auto col = static_cast<std::ptrdiff_t>(y) +
                                     (static_cast<std::ptrdiff_t>(j) - radius);
                    const bool col_inside = col >= 0 && col < side;

                    float value = 0.0f;  // Both coordinates outside: corner padding.
                    if (row_inside && col_inside) {
                        value = image[static_cast<std::size_t>(row) * n +
                                      static_cast<std::size_t>(col)];
                    } else if (row_inside || col_inside) {
                        value = 1.0f;  // Exactly one coordinate outside: edge padding.
                    }
                    // Use the mask directly, without flipping it.
                    sum += mask[i * m + j] * value;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
