#include "convolution.h"

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
        std::cerr << "Usage: ./task2 n t (n > 0, 1 <= t <= 20)\n";
        return 1;
    }
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (n > max_elements / n) {
        std::cerr << "Image size is too large.\n";
        return 1;
    }

    try {
        constexpr std::size_t m = 3;
        std::vector<float> image(n * n), mask(m * m), output(n * n);
        std::mt19937 generator(759);
        std::uniform_real_distribution<float> image_distribution(-10.0f, 10.0f);
        std::uniform_real_distribution<float> mask_distribution(-1.0f, 1.0f);
        for (float& value : image) {
            value = image_distribution(generator);
        }
        for (float& value : mask) {
            value = mask_distribution(generator);
        }

        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(threads));

        const auto start = std::chrono::high_resolution_clock::now();
        convolve(image.data(), output.data(), n, mask.data(), m);
        const auto stop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = stop - start;

        // Required order: first element, last element, elapsed milliseconds.
        std::cout << std::setprecision(12) << output.front() << '\n'
                  << output.back() << '\n' << elapsed.count() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
