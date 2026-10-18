// measures the drop in memory speed when due to an offset a warp reads from device source and
// writes to device destination. The offset causes the data to not start at 32 byte sector aligned
// position, so data can spill to an additional sector, causing lower memory speed in read and write.

#include "common.h"

// array size of 128MB for each, much larger than L2 cache, so that every run reads and writes to main
// memory instead
#define MAX_BYTES (128 * 1024 * 1024)

// 128 threads per block = 4 warps of 32 threads
#define THREADS_PER_BLOCK 128

// threads copy each element at their index + offset from device source to device destination
__global__ void memory_copy_offset(float *source, float *destination, int size, int offset) {
    int i = threadIdx.x + blockDim.x * blockIdx.x + offset;

    if (i < size) {
        destination[i] = source[i];
    }
}

// runs the kernel for offset values 0 through 31, measuring the memory speed and verifying the results
void test_memory_alignment(double theoretical_max) {
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
    FILE *file = open_csv("output/memory_alignment.csv");

    // goes to csv file
    fprintf(file, "Offset,Time (ms),Bandwidth (GB/s),Percent of Theoretical Max\n");
    // goes to terminal
    printf("Offset, Time (ms), Bandwidth (GB/s), Percent of Theoretical Max\n");

    // creates the start and stop events used for recording timestamps before and after kernel runs,
    // and measure runtime
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));

    // runs the test for each offset from 0 to 31
    for (int offset = 0; offset < 32; offset++) {
        // calculates the number of threads needed
        int num_threads = max_size - offset;

        // calculates the number of blocks needed for every thread to fit in
        int blocks = num_threads / THREADS_PER_BLOCK;
        if (num_threads % THREADS_PER_BLOCK != 0) {
            blocks++;
        }

        // resets the GPU destination array to 0, so old results from the previous offset don't
        // affect the new check
        CHECK_ERROR(cudaMemset(device_destination, 0 , MAX_BYTES));

        // does 3 warmup runs that aren't timed to make sure first few slow runs don't affect
        // the timed values
        for (int i = 0; i < 3; i++) {
            memory_copy_offset<<<blocks, THREADS_PER_BLOCK>>>(device_source, device_destination, max_size, offset);
        }
        CHECK_ERROR(cudaGetLastError());

        // times 10 kernel runs, records the start and stop, saves the time between
        for (int i = 0; i < 10; i++) {
            CHECK_ERROR(cudaEventRecord(start));
            memory_copy_offset<<<blocks, THREADS_PER_BLOCK>>>(device_source, device_destination, max_size, offset);
            CHECK_ERROR(cudaEventRecord(stop));
            CHECK_ERROR(cudaEventSynchronize(stop));
            CHECK_ERROR(cudaEventElapsedTime(&times[i], start, stop));
        }

        
        // median time is the chosen time for a stable value
        float time = median_time(times, 10);

        // each thread reads 4 bytes, and writes 4 bytes, so multiply by 2
        size_t total_bytes = 2 * sizeof(float) * num_threads;

        // calculates the memory speed in GB/s
        double bandwidth = calc_bandwidth(total_bytes, time);

        // calculates the  memory speed as a percentage of the theoretical maximum value
        double real_vs_theoretical = (bandwidth / theoretical_max) * 100;

        // saves the result to the CSV file
        fprintf(file, "%d,%.2f,%.2f,%.2f\n", offset, time, bandwidth, real_vs_theoretical);
        // shows the result to the terminal
        printf("%d, %.2f, %.2f, %.2f\n", offset, time, bandwidth, real_vs_theoretical);

        // brings GPU destination array values to CPU destination array
        CHECK_ERROR(cudaMemcpy(host_destination, device_destination, MAX_BYTES, cudaMemcpyDeviceToHost));

        // checks if kernel copied every element from the index offset until the end of the
        // array correctly
        for (size_t i = offset; i < max_size; i++) {
            if (host_destination[i] != host_source[i]) {
                fprintf(stderr, "Copy is incorrect at the index %d\n", (int) i);
                exit(EXIT_FAILURE);
            }
        }
    }

    // destroys the start and stop events since not needed anymore
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    // closes the CSV file
    fclose(file);

    // frees GPU and CPU memory
    CHECK_ERROR(cudaFree(device_source));
    CHECK_ERROR(cudaFree(device_destination));
    free(host_source);
    free(host_destination);
}