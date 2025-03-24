The optimzations available for Reduction parallel pattern includes:
- Reduction trees
- A simple reduction kernel
- Minimizing control divergence
- Minimizing memory divergence
- Minimizing global memory accesses
- Hierarchical reduction for arbitrary input length
- Thread coarsening for reduced overhead


Reduction trees:

This is an optimization technique where the reduction is performed in the sense that it forms a tree like structure. In this structure,instead of reducing
sequentially, the threads work in parallel on pairs of elements which in turn would reduce the amount of operations. The parallel reduction would progress
downwards  and the tasks are done in each step horizontally with less operations being done each time. In turn, this would allow for faster execution due 
to better parallelization and fewer points of synchronization with is step it takes in the tree structure.

A simple reduction kernel:

This is an optimization technique where each thread would be responsible for processesing a a different portion of the dataset and reduces it using basic 
operations.This makes it possible for all threads to collaborate across the entire grid unlike other optimizations. Hence, it provides a baseline parallel 
reduction method that can later be enhanced with other optimizations. In turn, it allows for simplicity and ease of implementation that compliments other
optimization techniques.

Minimizing control divergence:

This is an optimization technique that ensures that all threads in a warp or GPU would follow the same execution path which would not waste inactive 
threads. When threads take different execution paths, there is a possibility some threads become idle or inactive  while other threads are active in the 
warp which is reducing efficiency since it is wasting threads that are inactive. Hence, this technique would improve resource utilization where it would
not waste the inactive threads hence not wasting executing resources. This in turn will increases throughput by keeping all threads active for proper 
resource utilization.

Minimizing memory divergence:

This is an optimization technique that ensures that threads access memory in a coalesced or aligned manner within each warp to improve efficiency.
If threads access scattered memory locations that are adjacent to each other, memory access becomes inefficient since adjacent threads do not access
adjacent locations which in turn increases latency.However, this optimization allows for the threads are coalesced for efficient memory access. This 
in turn would reduces memory access time and as such improve overall performance.

Minimizing global memory accesses:

In this optimization technique, shared memory would be used or registers instead of repeatedly accessing global memory which is much slower. Global memory 
accesses are slow, while shared memory and registers are much faster which can also be used to compliment other optimizations which in turn would 
reduce latency and has a higher bandwidth than with global memory. Execution speed would be improved in the long run.

Hierarchical reduction for arbitrary input length:

This optimization technique is a flexible reduction approach that divides work across multiple levels of reduction. It ensures that large input sizes do not 
cause load imbalance and are processed efficiently. The idea being shown here is to partition the input array into properly sized segments for a block so 
that the block can execute a reduction tree and gather their results for a final output. This scales well for large datasets and prevents performance from 
decreasing.

Thread coarsening for reduced overhead:

In this optimization technique, each thread processes multiple elements instead of just one, which in turn reduces the number of kernel launches and 
synchronization points. If each thread is assigned to one element, it can slow down the execution time. Hence, with this optimization technique, there 
are fewer threads to manage  which in turn reduces the number of global memory access which improves or optimizes the algorithm. It will overall
improve GPU utilization and reduces kernel launch overhead.

Optimization Rationale:

After reviewing the list of optimizations availble and the profiling results from the basic algorithm. Thre selcted optimization were Tiling to minimize global memeory access and thread coarsenign to reduce overhead. The reason we chose these two in particular is because of the synergey between these two optimizations. As in our previous labs when implementing just Tiling into our code, the results equal a decrease in execution time but an increase in effective memmory bandwidth where as the expected resutls of implementing thread coarsening equal a decrease in effect memmory and an increase in execution time. These results led our group to the conclusion that implementing both these optimization together may result in an overall reduction in both execution time and effective memmory bandwidth.

Detailed analysis of the performance improvement relating to the results (readme)**
include the summary table from Appendix B

| Optimization# | Short Description | Execution time (ms) | Memory Bandwidth (GB/s) | Step Speedup | Cumulative Speedup |
| ------------- | ----------------- | ------------------- | ------------------------ | ------------ | ------------------- |
| 1             | Basic algorithm with no optimization    |1.05 ms                   | 3.81 bytes/s                      | 0          | 0                 |
| 2             | Coursening only algorithm               | 1.04 ms                  | 3.83 bytes/s                      | 19.52      | 19.52             |
| 3             | Fully optimized algorithm               | 0.79 ms                  | 5.04 bytes/s                      | 18.81      | 18.81             |




Simple Algorithm Output:

![image](https://github.com/meng3540/parallel-patterns-group-6-mikaeel-josh-hudson/blob/main/Optimizations/Tests/Profiling%20Results%20For%20Basic%20Algorithm/Simple%20Algorithm%20results.png)

One optimization(coursening) output:

![image](https://github.com/meng3540/parallel-patterns-group-6-mikaeel-josh-hudson/blob/main/Optimizations/Tests/Profiling%20Results%20For%20Thread%20Coarsening/Coursening%20Only%20results.png)

Full optimization(two optimization) output
![image](https://github.com/meng3540/parallel-patterns-group-6-mikaeel-josh-hudson/blob/main/Optimizations/Tests/Profiling%20Results%20For%20Tiling%20and%20Thread%20Coarsening/Fully%20optimized%20results.png)


Detailed analysis of the performance improvement relating to the results:
With the simple algorithm taht has no optimization applied, the algorithm runs at 1.05 milliseconds and has a memory bandwidth of 3.81 bytes/s. Since there are no optimizations yet, the step speedup and cumulative speedup are both zero, meaning we have no performance improvement in this case. For the algorithm with one optimization, we’ve applied a "coursening" optimization, which reduces the execution time slightly from 1.05 ms to 1.04 ms (about a 1% improvement). The memory bandwidth also sees a small increase. The big change here, though, is the step speedup of 19.52,which means that the algorithm is now much more efficient in terms of the number of operations it can perform per unit of time. Even though the execution time doesn’t change drastically, this optimization still boosts performance significantly. Now with full optimization, we see a much more noticeable reduction in execution time from 1.04 ms down to 0.79 ms as there is less of a need to access global memory hence less time taken for that. That’s a 24% improvement, which is pretty significant. Memory bandwidth also jumps quite a bit, from 3.83 bytes/s to 5.04 bytes/s, meaning the algorithm is now handling data more effectively. The step speedup drops slightly however to 18.81 compared to the previous optimization’s 19.52, but overall a significant improvement more than the basic algorithm and one optimization implementation.
