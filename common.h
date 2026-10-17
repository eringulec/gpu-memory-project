// helpers for all the tests: checking CUDA errors, GPU info, bandwidth and median math, CSV files
#pragma once

#include <stdio.h>
#include <stdlib.h>

// checks CUDA calls and prints information about where it failed
#define CHECK_ERROR(a) check_error((a), __FILE__, __LINE__)

// prints GPU information and returns the possible max bandwidth in GB/s
double print_gpu_info(void);

// turns bytes and time in ms into GB/s
double calc_bandwidth(size_t bytes, float ms);

// returns the median time value of the times by sorting the array
float median_time(float *times, int count);

// opens the file in the output folder for writing
FILE* open_csv(const char *file_name);

// prints the error occurred and in which file and line it happened at then stops the program
void check_error(cudaError_t err, const char* file, int line);