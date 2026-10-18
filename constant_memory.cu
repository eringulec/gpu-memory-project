// measures how much slower the constant memory is when the 32 threads in a warp read different elements
// rather than the same elements of the constant values array at the same time. To explain, when all
// the threads ask for the same element the constant memory reads it for one time and the value is sent
// to the 32 threads at once. But, if each thread is looking for different elements in the constant
// array, then reads happen one at a time, making memory read slower.
#include "common.h"

// number of floats in the constant memory, 4 KB, smaller than 8 KB constant cache
#define CONST_SIZE 1024

// defines how many times each thread reads the constant memory. Since a single read is too fast to
// measure, due to costs like launching the kernel, 1000 reads per thread enables the difference to
// be measured more clearly due to significant time spent on reading
#define READS_PER_THREAD 1000

// 128 threads per block = 4 warps of 32 threads
#define THREADS_PER_BLOCK 128

// total threads launched, a value enough to make all the SMs work at the same time, so no SM gets
// left out of doing work
#define TOTAL_THREADS (1024 * 1024)

// an array in the constant memory, values are filled from the CPU
__constant__ float constant_values[CONST_SIZE];

// times the constant_read kernel with two cases of either threads reading the same element
// (read_different = 0), or different elements (read_different = 1). Returns the median time
float time_constant_kernel(float *device_out, float *host_out, int blocks, int read_different);

// each thread in the kernel reads the constant_values array in the constant memory READS_PER_THREAD
// times, and adds the values to the sum. The sum is written to the device_out array because the compiler
// removes work in which the result isn't stored. So, without writing to the device_out array nothing
// could be measured. And, the sums are also used for verifying the results at the end.
// If read_different = 0, all threads in a warp read the same element, if read_different = 1, each thread
// reads a different element. 
__global__ void constant_read(float *out, int size, int read_different) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < size) {
        float sum = 0;

        for (int j = 0; j < READS_PER_THREAD; j++) {
            if (read_different) {
                sum += constant_values[(threadIdx.x + j) % CONST_SIZE];
            } else {
                sum += constant_values[j % CONST_SIZE];
            }
        }
        out[i] = sum;
    }
}

// times the kernel for same and different element reads from the constant memory, compares the
// two times, saves the results to a file, shows them in the terminal, and verifies the results
void test_constant_memory(void) {
    // CPU array holding the values that the constant_values array will get in the constant memory
    float* host_values;

    // CPU and GPU arrays for the kernel results
    float* host_out;
    float* device_out;

    // allocates memory for the CPU array
    host_values = (float*) malloc(CONST_SIZE * sizeof(float));
    
    // fills all the values of the array with 1.0, so that later on in the check it can be
    // verified that each thread's sum of reads, in the amount of READS_PER_THREAD, is equal
    // to the value READS_PER_THREAD
    for (int i = 0; i < CONST_SIZE; i++) {
        host_values[i] = 1.0;
    }

    // copies the CPU host_values array content into an array in the constant memory on the GPU
    CHECK_ERROR(cudaMemcpyToSymbol(constant_values, host_values, CONST_SIZE * sizeof(float)));

    // allocates memory for the CPU result array
    host_out = (float*) malloc(TOTAL_THREADS * sizeof(float));

    // allocates memory for the GPU result array
    CHECK_ERROR(cudaMalloc(&device_out, TOTAL_THREADS * sizeof(float)));

    // opens the output file which will take results data later on
    FILE *file = open_csv("output/constant_memory.csv");

    // calculates the total number of blocks needed to make sure every element can get a thread
    int blocks = (TOTAL_THREADS) / THREADS_PER_BLOCK;
    if (TOTAL_THREADS % THREADS_PER_BLOCK != 0) {
        blocks++;
    }

    // times the kernel when all threads in a warp read the same element in the constant array
    float time_read_same = time_constant_kernel(device_out, host_out, blocks, 0);

    // times the kernel when each thread reads a different element in the constant array
    float time_read_different = time_constant_kernel(device_out, host_out, blocks, 1);

    // calculates how much slower it is for threads to read different elements vs the same element
    // in the constant array.
    double different_vs_same = time_read_different / time_read_same;

    // goes to csv file
    fprintf(file, "Same element (ms),Different elements (ms),Different vs Same\n");

    // saves the results to the CSV file
    fprintf(file, "%.4f,%.4f,%.4f\n", time_read_same, time_read_different, different_vs_same);

    // shows the result to the terminal
    printf("Same element: %.4f ms\n", time_read_same);
    printf("Different elements: %.4f ms\n", time_read_different);
    printf("Constant memory is %.1fx slower when each thread in a warp reads a different element "
           "compared to when threads read the same element\n",
           different_vs_same);

    // closes the CSV file
    fclose(file);

    // frees GPU and CPU memory
    CHECK_ERROR(cudaFree(device_out));
    free(host_values);
    free(host_out);
}

float time_constant_kernel(float *device_out, float *host_out, int blocks, int read_different) {
    // creates the start and stop events for timing the kernel
    cudaEvent_t start, stop;
    CHECK_ERROR(cudaEventCreate(&start));
    CHECK_ERROR(cudaEventCreate(&stop));

    // array for time in ms for each timed run
    float times[10];

    // resets the GPU result array to 0, so the previous call's results don't affect the
    // current call
    CHECK_ERROR(cudaMemset(device_out, 0, TOTAL_THREADS * sizeof(float)));

    // does 3 warmup runs that aren't timed to make sure first few slow runs don't affect
    // the timed values
    for (int i = 0; i < 3; i++) {
        constant_read<<<blocks, THREADS_PER_BLOCK>>>(device_out, TOTAL_THREADS, read_different);
    }
    CHECK_ERROR(cudaGetLastError());

    // times 10 kernel runs, records the start and stop, saves the time between
    for (int i = 0; i < 10; i++) {
        CHECK_ERROR(cudaEventRecord(start));
        constant_read<<<blocks, THREADS_PER_BLOCK>>>(device_out, TOTAL_THREADS, read_different);
        CHECK_ERROR(cudaEventRecord(stop));
        CHECK_ERROR(cudaEventSynchronize(stop));
        CHECK_ERROR(cudaEventElapsedTime(&times[i], start, stop));
    }

    // destroys the start and stop events since not needed anymore
    CHECK_ERROR(cudaEventDestroy(start));
    CHECK_ERROR(cudaEventDestroy(stop));

    // median time is the chosen time for a stable value
    float time = median_time(times, 10);

    // brings GPU values to the CPU
    CHECK_ERROR(cudaMemcpy(host_out, device_out, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost));

    // checks that each thread has its sum equal to READS_PER_THREAD
    for (size_t i = 0; i < TOTAL_THREADS; i++) {
        if (host_out[i] != READS_PER_THREAD) {
            fprintf(stderr, "Result is incorrect at the index %d\n", (int) i);
            exit(EXIT_FAILURE);
        }
    }

    return time;
}