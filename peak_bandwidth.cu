// measures the fastest memory speed a kernel can actually reach and compares the value to the theoretical max based on GPU info
#include "common.h"

// array size of 128MB for each, much larger than L2 cache, so that every run reads and writes to main memory instead
#define MAX_BYTES (128 * 1024 * 1024)

// 128 threads per block = 4 warps of 32 threads
#define THREADS_PER_BLOCK 128

// threads copy each element from device source to device destination
__global__ void memory_copy(float *source, float *destination, int size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size) {
        destination[i] = source[i];
    }
}

// times the kernel to measure the memory speed, compares it with the theoretical max, saves the result, and verifies the copy of each element
void test_peak_bandwidth(double peak) {
    // CPU arrays and GPU arrays
    float *host_source;
    float *host_destination;
    float *device_source;
    float *device_destination;

    // array for keeping track of timed GPU runs in ms 
    float times[10];

    // number of floats that fit into 128MB
    size_t max_size = MAX_BYTES / sizeof(float);

    // allocates memory for the CPU arrays
    host_source = (float *) malloc(MAX_BYTES);
    host_destination = (float *) malloc(MAX_BYTES);

    // fills in the CPU source array 
    for (size_t i = 0; i < max_size; i++) {
        host_source[i] = i;
    }

    // allocates GPU memory for the GPU arrays
    CHECK_ERROR(cudaMalloc(&device_source, MAX_BYTES));
    CHECK_ERROR(cudaMalloc(&device_destination, MAX_BYTES));
    // Copy the CPU source array data to the GPU source array
    CHECK_ERROR(cudaMemcpy(device_source, host_source, MAX_BYTES, cudaMemcpyHostToDevice));

    // opens the output file which will take results data later on
    FILE *file = open_csv("output/peak_bandwidth.csv");

    // goes to csv file
    fprintf(file, "Array size (bytes),Time (ms),Bandwidth (GB/s),Percent of Theoretical Max\n");
    // goes to terminal
    printf("Array size (bytes),Time (ms),Bandwidth (GB/s),Percent of Theoretical Max\n");

    // calculates the total number of blocks needed to make sure every element can get a thread
    int blocks = (max_size) / THREADS_PER_BLOCK;
    if (max_size % THREADS_PER_BLOCK != 0) {
        blocks++;
    }

    // does 3 warmup runs that aren't timed to make sure first few slow runs don't affect the timed values
    for (int i = 0; i < 3; i++) {
        memory_copy<<<blocks, THREADS_PER_BLOCK>>>(device_source, device_destination, max_size);
    }
    CHECK_ERROR(cudaGetLastError());

    // creates the start and stop events used for recording timestamps before and after kernel runs, and measure runtime
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));

    // times 10 kernel runs, records the start and stop, saves the time between
    for (int i = 0; i < 10; i++) {
        CHECK_ERROR(cudaEventRecord(start));
        memory_copy<<<blocks, THREADS_PER_BLOCK>>>(device_source, device_destination, max_size);
        CHECK_ERROR(cudaEventRecord(stop));
        CHECK_ERROR(cudaEventSynchronize(stop));
        CHECK_ERROR(cudaEventElapsedTime(&times[i], start, stop));
    }

    // destroys the start and stop events since not needed anymore
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    // median time is the chosen time for a stable value
    float time = median_time(times, 10);

    // total bytes is 128 MB for read and 128 MB for write
    size_t total_bytes = 2 * MAX_BYTES;
    double bandwidth = calc_bandwidth(total_bytes, time);
    double real_vs_theoretical = (bandwidth / peak) * 100;

    // saves the result to the CSV file
    fprintf(file, "%d,%.2f,%.2f,%.2f\n", MAX_BYTES, time, bandwidth, real_vs_theoretical);
    // shows the result to the terminal
    printf("%d, %.2f, %.2f, %.2f\n", MAX_BYTES, time, bandwidth, real_vs_theoretical);

    // brings GPU destination array values to CPU destination array
    CHECK_ERROR(cudaMemcpy(host_destination, device_destination, MAX_BYTES, cudaMemcpyDeviceToHost));

    // checks if kernel copied all the elements from device_source to device_destination correctly
    for (size_t i = 0; i < max_size; i++) {
        if (host_destination[i] != host_source[i]) {
            fprintf(stderr, "Copy is incorrect at the index %d\n", (int) i);
            exit(EXIT_FAILURE);
        }
    }

    // closes the CSV file
    fclose(file);

    // frees GPU and CPU memory
    CHECK_ERROR(cudaFree(device_source));
    CHECK_ERROR(cudaFree(device_destination));
    free(host_source);
    free(host_destination);
}