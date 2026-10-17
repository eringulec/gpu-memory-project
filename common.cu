// function definitions for helpers declared in common.h
#include "common.h"

// read GPU values and print, calculate the max memory bandwidth and return
double print_gpu_info(void) {
    cudaDeviceProp prop;
    CHECK_ERROR(cudaGetDeviceProperties(&prop, 0));
    printf("GPU name: %s\n", prop.name);
    printf("SM amount: %d\n", prop.multiProcessorCount);
    printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache size: %d bytes\n", prop.l2CacheSize);

    double transfers_per_sec = (prop.memoryClockRate * 1000.0) * 2; // khz to Hz, and * 2 because twice data per tick
    double bytes_per_transfer = prop.memoryBusWidth / 8.0;
    double bytes_per_sec = transfers_per_sec * bytes_per_transfer;
    double peak = bytes_per_sec / 1e9;

    printf("Peak bandwidth: %.2f GB/s\n", peak);
    return peak;
}

// convert bytes accessed and time in ms into GB/s
double calc_bandwidth(size_t bytes, float ms) {
    double gb = bytes / 1e9;
    double sec = ms / 1000;
    return gb / sec;
}

// sort the time values and return the median value
float median_time(float *times, int count) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = 0; j < count - 1; j++) {
            if (times[j] > times[j + 1]) {
                float temp = times[j];
                times[j] = times[j + 1];
                times[j + 1] = temp;
            }
        }
    }
    return times[count / 2];
}

// open the file for writing, if it fails stop the program
FILE* open_csv(const char *file_name) {
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        fprintf(stderr, "Can't open %s\n", file_name);
        exit(EXIT_FAILURE);
    }
    return file;
}

// if a CUDA call fails print error message with details and stop the program
void check_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s) at %s:%d\n",
                cudaGetErrorName(err),
                cudaGetErrorString(err),
                file,
                line);
        exit(EXIT_FAILURE);
    }
}