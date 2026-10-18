// declares five tests that main.cu runs with each test in a separate .cu file
#pragma once

// measures the fastest memory speed a kernel actually reaches and compares the value
// to the theoretical max calculated based on the GPU's memory clock rate and bus width.
void test_max_bandwidth(double theoretical_max);

// measures the drop in memory speed when the threads in the same warp read and write addresses that
// are stride elements apart in the GPU memory. GPU memory doesn't send values one at a time but in
// 32 byte sectors. When threads use close addresses, a sector holds data for several threads, whereas
// when the addresses are farther apart, fewer threads share a sector. Thus, there are more unused
// bytes in sectors on average, and to move the same amount of data more sectors need to be used,
// lowering the memory speed.
void test_memory_stride(double theoretical_max);

// measures the drop in memory speed when due to an offset a warp reads from device source and
// writes to device destination. The offset causes the data to not start at 32 byte sector aligned
// position, so data can spill to an additional sector, causing lower memory speed in read and write.
void test_memory_alignment(double theoretical_max);

// measures how much slower the constant memory is when the 32 threads in a warp read different elements
// rather than the same elements of the constant values array at the same time. To explain, when all
// the threads ask for the same element the constant memory reads it for one time and the value is sent
// to the 32 threads at once. But, if each thread is looking for different elements in the constant
// array, then reads happen one at a time, making memory read slower.
void test_constant_memory(void);

// measures how long a memory read takes as the array size grows from 4KB to all the way through
// 128MB. Shows how fast the L1 cache, L2 cache, and the main memory are.
void test_cache_latency(void);

