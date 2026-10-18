// starts the program, prints information about the GPU, and then runs all the five tests
#include "common.h"
#include "tests.h"

int main(void) {
    // prints the GPU info, and returns the theoretical max value for memory speed,
    // print_gpu_info() defined in common.cu
    double theoretical_max = print_gpu_info();

    // print test outputs
    printf("-----------------------------------\n");
    printf("*** Max Bandwidth ***\n");
    test_max_bandwidth(theoretical_max);

    printf("-----------------------------------\n");
    printf("*** Memory Stride ***\n");
    test_memory_stride(theoretical_max);

    printf("-----------------------------------\n");
    printf("*** Memory Alignment ***\n");
    test_memory_alignment(theoretical_max);

    printf("-----------------------------------\n");
    printf("*** Constant Memory ***\n");
    test_constant_memory();

    printf("-----------------------------------\n");
    printf("*** Cache Latency ***\n");
    test_cache_latency();

    return 0;
}