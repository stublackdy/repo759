#include <charconv>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <system_error>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    int n = 0;
    const char* end = argv[1] + std::strlen(argv[1]);
    const auto result = std::from_chars(argv[1], end, n);
    if (result.ec != std::errc{} || result.ptr != end || n < 0) {
        std::cerr << "N must be a nonnegative integer in the int range.\n";
        return 1;
    }

    // Print the last element separately to avoid trailing spaces.
    for (int i = 0; i < n; ++i) {
        std::printf("%d ", i);
    }
    std::printf("%d\n", n);

    for (int i = n; i > 0; --i) {
        std::cout << i << ' ';
    }
    std::cout << 0 << '\n';

    return 0;
}
