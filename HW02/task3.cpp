#include "matmul.h"

#include <array>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr unsigned int n = 1024;
    constexpr std::size_t count = static_cast<std::size_t>(n) * n;

    try {
        std::vector<double> A(count), B(count);
        std::array<std::vector<double>, 4> C;
        for (auto &matrix : C) {
            matrix.resize(count);
        }

        // Binary fractions keep all products and partial sums exactly
        // representable here, making a full exact comparison meaningful.
        std::mt19937 generator(759);
        std::uniform_int_distribution<int> distribution(-16, 16);
        for (std::size_t i = 0; i < count; ++i) {
            A[i] = distribution(generator) / 16.0;
            B[i] = distribution(generator) / 16.0;
        }

        const auto measure = [](auto multiply) {
            const auto start = std::chrono::high_resolution_clock::now();
            multiply();
            const auto stop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double, std::milli> elapsed = stop - start;
            return elapsed.count();
        };

        // Raw-pointer and vector interfaces use the same input storage.
        // Each function includes the same initialization of its output.
        std::array<double, 4> times{};
        times[0] = measure([&] { mmul1(A.data(), B.data(), C[0].data(), n); });
        times[1] = measure([&] { mmul2(A.data(), B.data(), C[1].data(), n); });
        times[2] = measure([&] { mmul3(A.data(), B.data(), C[2].data(), n); });
        times[3] = measure([&] { mmul4(A, B, C[3].data(), n); });

        // Check every element, outside all timed intervals.
        for (std::size_t version = 1; version < C.size(); ++version) {
            if (C[version] != C[0]) {
                std::cerr << "Matrix results disagree for mmul" << version + 1 << '\n';
                return 1;
            }
        }

        // Exactly nine lines: n, then time and last element for mmul1..4.
        std::cout << std::setprecision(12) << n << '\n';
        for (std::size_t version = 0; version < C.size(); ++version) {
            std::cout << times[version] << '\n' << C[version].back() << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
