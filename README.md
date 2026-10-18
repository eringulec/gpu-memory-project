# GPU Memory Project

Author: Erin Gulec ([GitHub](https://github.com/eringulec))

Five CUDA tests are used to find out how GPU memory performs in different situations:
- max_bandwidth
- memory_stride
- memory_alignment
- constant_memory
- cache_latency

## GPU Test Equipment 

All tests were run on Georgia Tech ICE GPU cluster with the purpose of class material review personal project.

|||
|---|---|
| GPU model | NVIDIA H100 80GB HBM3 |
| SMs | 132 |
| Memory bus width | 5120 bits|
| L2 cache | 50 MB |
| Theoretical maximum bandwidth | 3352 GB/s |



## GPU Test Results

### Max Bandwidth

Setup: Each thread copies one element, and the elements are side by side in the memory.

Experiment Question: How fast can the memory go?

| Time | Bandwidth | % of theoretical max |
|---|---|---|
| 0.16 ms | 1644 GB/s | 49% |

Explanation: The "Max Bandwidth" test copies a 128 MB array. Since 128 MB is larger than the 50 MB L2 cache, data comes from the GPU main memory. Data shows that the measured bandwidth reached 49% of the theoretical max value, likely because each thread loads only one 4 byte float, so this was not enough to use the full memory bandwidth.

Full Results: [max_bandwidth.csv](output/max_bandwidth.csv)



### Memory Stride

Setup: Each thread copies every "stride"-th element, stride goes from 1 through 32.

Experiment Question: What happens if neighboring threads read data far apart than their neighboring data?

| Stride | Bandwidth (GB/s) | % of theoretical max |
|---|---|---|
| 1 | 1648 | 49.2 |
| 2 | 1012 | 30.2 |
| 4 | 503 | 15.0 |
| 8 | 252 | 7.5 |
| 32 | 110 | 3.3 |

Explanation: The GPU reads and writes memory in 32 byte sectors, but if we use a stride, as the stride gets larger, more space gets wasted in each sector. We can see that the bandwidth drops from 1648 GB/s when stride is 1 to 110 GB/s when stride is 32. So, only every "stride"-th element in each sector is used, and the rest of the sector gets wasted.

Full Results: [memory_stride.csv](output/memory_stride.csv)



### Memory Alignment

Setup: Each thread starts to read data at a shifted position, but side by side unlike the stride test. Offset goes from 0 through 31, and all the elements at index below the offset are ignored in memory reads and writes.

Experiment Question: Does memory access get slower when data doesn't begin at a 32 byte memory sector due to an offset?

| Offset | Bandwidth (GB/s) |
|---|---|
| 0 | 1647 |
| 1 | 1646 |
| 16 | 1647 |
| 31 | 1646 |

Explanation: Bandwidth stayed about the same throughout all the results for all the offset values, with the full result showing the bandwidth value ranging from 1641.61 to 1647.09 GB/s. Therefore, starting off the beginning of a sector due to an offset didn't have any visible effect on the NVIDIA H100.

Full Results: [memory_alignment.csv](output/memory_alignment.csv)



### Constant Memory

Setup: Around a million threads read from a 1024 element constant memory array 1000 times. There are two cases, in the first case all threads in the warp read the same element, in the second case each thread reads a different element.

Experiment Question: How much slower is the constant memory when all the 32 threads in a warp read a different element of an array in the constant memory compared to the same element case?

| Read (type) | Time |
|---|---|
| Same element | 0.065 ms |
| Different elements | 8.147 ms |

Explanation: When all the threads ask for the same element, the constant memory reads it for one time and the value is sent to all the 32 threads at once. But, when each thread asks for different elements in the constant array, then reads happen one at a time, making memory read slower. As the test result, we see that the different element case was approximately 125-126 times slower.

Full Results: [constant_memory.csv](output/constant_memory.csv)



### Cache Latency

Setup: A thread reads an array 10000 times with each read telling the element to read next. The whole process is repeated for array sizes ranging from 4 KB to 128 MB, where the size doubles each time.

Experiment Question: How long does a read take from each section of GPU memory?

| Array (size) | Latency (cycles) | From |
|---|---|---|
| 4 KB - 128 KB | 40 - 73 | L1 |
| 256 KB - 32 MB | 267 - 370 | L2 |
| 64 MB - 128 MB | 689 - 690 | GPU main memory |

Explanation: Arrays with small size fit in the L1 cache, and read takes 40 to 73 cycles, showing that L1 cache is fast. But after the size gets too large for the L1 cache to use, L2 cache provides the reads and ranges from 267 to 370 cycles based on the result. When the size gets even bigger, GPU main memory provides the reads, taking 689-690 cycles.

Full Results: [cache_latency.csv](output/cache_latency.csv)

