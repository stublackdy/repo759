#include "msort.h"

#include <memory>

namespace {
// Non-recursive serial algorithm for small subarrays [begin, end).
void insertion_sort(int* arr, std::size_t begin, std::size_t end) {
    for (std::size_t i = begin + 1; i < end; ++i) {
        const int value = arr[i];
        std::size_t j = i;
        while (j > begin && arr[j - 1] > value) {
            arr[j] = arr[j - 1];
            --j;
        }
        arr[j] = value;
    }
}

void merge_sort(int* arr, int* buffer, std::size_t begin,
                std::size_t end, std::size_t threshold) {
    const std::size_t length = end - begin;
    if (length < 2) return;
    if (length < threshold) {
        insertion_sort(arr, begin, end);
        return;
    }

    const std::size_t middle = begin + length / 2;
    // Copy pointer values and bounds into each task; the underlying arrays
    // remain shared. Sibling tasks access disjoint ranges in both arrays.
#pragma omp task default(none) firstprivate(arr, buffer, begin, middle, threshold)
    {
        merge_sort(arr, buffer, begin, middle, threshold);
    }
#pragma omp task default(none) firstprivate(arr, buffer, middle, end, threshold)
    {
        merge_sort(arr, buffer, middle, end, threshold);
    }
#pragma omp taskwait

    // Each child has also waited for its own children before returning.
    // Consequently both complete sorted halves are ready to merge here.
    std::size_t left = begin;
    std::size_t right = middle;
    std::size_t out = begin;
    while (left < middle && right < end) {
        if (arr[left] <= arr[right]) {
            buffer[out++] = arr[left++];
        } else {
            buffer[out++] = arr[right++];
        }
    }
    while (left < middle) buffer[out++] = arr[left++];
    while (right < end) buffer[out++] = arr[right++];
    for (std::size_t i = begin; i < end; ++i) arr[i] = buffer[i];
}
}  // namespace

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n < 2) return;
    if (n < threshold) {
        insertion_sort(arr, 0, n);
        return;
    }
    // Allocate once, outside the parallel region, and reuse disjoint slices.
    const std::unique_ptr<int[]> storage(new int[n]);
    int* buffer = storage.get();
#pragma omp parallel default(none) shared(arr, buffer, n, threshold)
    {
#pragma omp single
        {
            merge_sort(arr, buffer, 0, n, threshold);
        }
    }
}
