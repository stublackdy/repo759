#include "matmul.h"

#include <charconv>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace {
bool parse_positive(const char* text, std::size_t& value) {
    const std::string_view input(text);
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    return result.ec == std::errc{} && result.ptr == input.data() + input.size() && value > 0;
}
}  // namespace

int main(int argc, char* argv[]) {
    std::size_t n = 0;
    std::size_t threads = 0;
    if (argc != 3 || !parse_positive(argv[1], n) ||
        !parse_positive(argv[2], threads) || threads > 20) {
        std::cerr << "Usage: ./task1 n t (n > 0, 1 <= t <= 20)\n";
        return 1;
    }
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (n > max_elements / n) {
        std::cerr << "Matrix size is too large.\n";
        return 1;
    }

    try {
        const std::size_t count = n * n;
        std::vector<float> A(count), B(count), C(count);
        // The fixed seed keeps inputs identical across thread counts on Euler.
        // Binary fractions also allow exact reference comparisons for n = 1024.
        std::mt19937 generator(759);
        std::uniform_int_distribution<int> distribution(-16, 16);
        for (std::size_t i = 0; i < count; ++i) {
            A[i] = distribution(generator) / 16.0f;
            B[i] = distribution(generator) / 16.0f;
        }

        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));

        const auto start = std::chrono::high_resolution_clock::now();
        mmul(A.data(), B.data(), C.data(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        // Required order: first element, last element, elapsed milliseconds.
        std::cout << std::setprecision(12) << C.front() << '\n'
                  << C.back() << '\n' << elapsed.count() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
