#include "scan.h"

#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <system_error>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n\n";
        return 1;
    }

    std::size_t n = 0;
    const char *end = argv[1] + std::strlen(argv[1]);
    const auto result = std::from_chars(argv[1], end, n);
    if (result.ec != std::errc{} || result.ptr != end || n == 0 ||
        n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        std::cerr << "n must be a positive integer with a valid array size.\n";
        return 1;
    }

    try {
        // Smart pointers release both arrays automatically on exit.
        std::unique_ptr<float[]> arr(new float[n]);
        std::unique_ptr<float[]> output(new float[n]{});

        // A fixed seed makes repeated runs reproducible in the same environment.
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        for (std::size_t i = 0; i < n; ++i) {
            arr[i] = distribution(generator);
        }

        // Only the scan call is timed, not allocation, initialization or printing.
        const auto start = std::chrono::high_resolution_clock::now();
        scan(arr.get(), output.get(), n);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        // Required output: milliseconds, first element, last element.
        std::cout << std::setprecision(9) << elapsed.count() << '\n'
                  << output[0] << '\n'
                  << output[n - 1] << '\n';
    } catch (const std::bad_alloc &) {
        std::cerr << "Unable to allocate the arrays; check n and available memory.\n";
        return 1;
    }

    return 0;
}
