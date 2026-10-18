// measures how long a memory read takes as the array size grows from 4KB to all the way through
// 128MB. Shows how fast the L1 cache, L2 cache, and the main memory are.
#include "common.h"

// smallest array size that fits in the L1 cache
#define MIN_BYTES (4 * 1024)

// largest array size, and is too big for an L2 cache
#define MAX_BYTES (128 * 1024 * 1024)

// each read jumps 128 bytes, which is 32 elements, because then every read uses a different cache
// line, which is a 128 byte block
#define STRIDE 32

// number of reads timed for every single array size, where the array size starts from 4KB, and
// the next size is the double of that, and so on, where 128 MB is the last size.
#define STEPS 10000

// one thread reads the array, each value read is the index of the next element to read
__global__ void dependent_reads(unsigned int* array, int elements_visited,
    unsigned int *out_i, long long *out_cycles) {
    
    // first element
    unsigned int i = 0;

    // read the array 3 times as warmup
    for (int j = 0; j < 3 * elements_visited; j++) {
        i = array[i];
    }

    // back to the first element
    i = 0;

    // counts the clock cycle of the GPU before the timed reads
    long long start = clock64();

    // timed reads
    for (int j = 0; j < STEPS; j++) {
        i = array[i];
    }

    // counts the clock cycle of the GPU after the timed reads
    long long stop = clock64();

    // saves the amount of cycles
    *out_cycles = stop - start;

    // saves the last index, so that the compiler won't delete the loop
    *out_i = i;
}

// runs the kernel for each array size, saves how many cycles a read takes
void test_cache_latency(void) {
    // CPU and GPU arrays of indexes
    unsigned int *host_array;
    unsigned int *device_array;

    // kernel's GPU results
    unsigned int *device_index;
    long long *device_cycles;

    // allocate CPU and GPU arrays with the largest size
    host_array = (unsigned int*) malloc(MAX_BYTES);
    CHECK_ERROR(cudaMalloc(&device_array, MAX_BYTES));

    // allocate GPU memory for the results of the kernel
    CHECK_ERROR(cudaMalloc(&device_index, sizeof(unsigned int)));
    CHECK_ERROR(cudaMalloc(&device_cycles, sizeof(long long)));

    // open the output file to store the results
    FILE *file = open_csv("output/cache_latency.csv");

    // goes to csv file
    fprintf(file, "Array Size (KB),Read Latency (cycles)\n");
    // goes to terminal
    printf("Array Size (KB), Read Latency (cycles)\n");

    // goes from 4KB to 128MB with doubling the array size in each round
    for (size_t bytes = MIN_BYTES; bytes <= MAX_BYTES; bytes *= 2) {
        // number of elements in the array
        size_t num_elements = bytes / sizeof(unsigned int);

        // the reads happen in every 32nd, which is STRIDEth, element, and this
        // represents the total elements that get visited
        int elements_visited = num_elements / STRIDE;

        // fills the host_array, each element holds the index that is 32 elements
        // ahead than the current position, and the last index points to 0
        for (size_t i = 0; i < num_elements; i++) {
            size_t jump_index = STRIDE + i;
            host_array[i] = jump_index % num_elements;
        }

        // copy the index values in CPU to the GPU
        CHECK_ERROR(cudaMemcpy(device_array, host_array, bytes, cudaMemcpyHostToDevice));

        // Reads need to happen one after another, so launch with one block and one thread 
        dependent_reads<<<1, 1>>>(device_array, elements_visited, device_index, device_cycles);
        CHECK_ERROR(cudaGetLastError());

        // variable to store GPU device_index and device_cycles values
        unsigned int last_index;
        long long cycles;

        // brings the results from GPU back to the CPU
        CHECK_ERROR(cudaMemcpy(&last_index, device_index, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_ERROR(cudaMemcpy(&cycles, device_cycles, sizeof(long long), cudaMemcpyDeviceToHost));

        // average number of cycles each read took
        double latency = (double) cycles / STEPS;

        // converts array size in bytes to kb
        int k_bytes = (int) (bytes / 1024);

        // calculates the index where the reads "should" finish
        size_t total_dist = STEPS * STRIDE;
        size_t expected_index = total_dist % num_elements;

        // checks that the reads ended at the expected last index
        if (last_index != expected_index) {
            fprintf(stderr, "Last index %u is incorrect at the array size of %d KB\n", last_index, k_bytes);
            exit(EXIT_FAILURE);
        }

        // saves the result to the CSV file
        fprintf(file, "%d,%.2f\n", k_bytes, latency);
        // shows the result in the terminal
        printf("%d, %.2f\n", k_bytes, latency);
    }

    // closes the CSV file
    fclose(file);

    // frees CPU and GPU memory
    free(host_array);
    CHECK_ERROR(cudaFree(device_array));
    CHECK_ERROR(cudaFree(device_index));
    CHECK_ERROR(cudaFree(device_cycles));
}