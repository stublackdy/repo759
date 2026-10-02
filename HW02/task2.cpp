#include "convolution.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <system_error>

namespace {
bool parse_dimension(const char *text, std::size_t &value) {
    const char *end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    // Check the square's byte count before performing multiplication.
    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    return result.ec == std::errc{} && result.ptr == end && value > 0 &&
           value <= max_elements / value;
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n m\n";
        return 1;
    }

    std::size_t n = 0;
    std::size_t m = 0;
    if (!parse_dimension(argv[1], n) || !parse_dimension(argv[2], m) || m % 2 == 0) {
        std::cerr << "n and m must be positive integers with valid matrix sizes; m must be odd.\n";
        return 1;
    }

    float *image = nullptr;
    float *mask = nullptr;
    float *output = nullptr;
    try {
        image = new float[n * n];
        mask = new float[m * m];
        output = new float[n * n]{};

        std::mt19937 generator(759);
        std::uniform_real_distribution<float> image_distribution(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mask_distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < n * n; ++i) {
            image[i] = image_distribution(generator);
        }
        for (std::size_t i = 0; i < m * m; ++i) {
            mask[i] = mask_distribution(generator);
        }

        const auto start = std::chrono::high_resolution_clock::now();
        convolve(image, output, n, mask, m);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        std::cout << std::setprecision(9) << elapsed.count() << '\n'
                  << output[0] << '\n'
                  << output[n * n - 1] << '\n';
    } catch (const std::bad_alloc &) {
        delete[] image;
        delete[] mask;
        delete[] output;
        std::cerr << "Unable to allocate the matrices; check sizes and available memory.\n";
        return 1;
    }

    delete[] image;
    delete[] mask;
    delete[] output;
    return 0;
}
