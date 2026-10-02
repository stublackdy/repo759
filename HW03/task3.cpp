#include "msort.h"

#include <algorithm>
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
    std::size_t threshold = 0;
    if (argc != 4 || !parse_positive(argv[1], n) ||
        !parse_positive(argv[2], threads) || threads > 20 ||
        !parse_positive(argv[3], threshold) ||
        n > std::numeric_limits<std::size_t>::max() / sizeof(int)) {
        std::cerr << "Usage: ./task3 n t ts (n > 0, 1 <= t <= 20, ts > 0)\n";
        return 1;
    }

    try {
        std::vector<int> arr(n);
        std::mt19937 generator(759);
        std::uniform_int_distribution<int> distribution(-1000, 1000);
        for (int& value : arr) value = distribution(generator);
        // Retain the input for a complete correctness check outside timing.
        auto expected = arr;
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));

        const auto start = std::chrono::high_resolution_clock::now();
        msort(arr.data(), n, threshold);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        // std::sort is only a reference, not part of the msort implementation.
        std::sort(expected.begin(), expected.end());
        if (arr != expected) {
            std::cerr << "Full sorting result differs from std::sort.\n";
            return 1;
        }
        std::cout << std::setprecision(12) << arr.front() << '\n'
                  << arr.back() << '\n' << elapsed.count() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
