Welcome to The group 6 repository

here are some FAQs

1) What are parallel patterns in computing? 
  
Parallel patterns are common methods or techniques used to solve problems regarding processing speed and efficiency, by performing multiple calculations or processes simultaneously. Parallel patterns help effectivley and efficiently utilize multicore processors (like the GPU) when executing algorithims. 
  
2) What is their significance?- Explain in the context of typical applications they are used in. 

Their significance is the processing speed they offer since they allow several pieces of data to reach their destination simultaneously. This can be especially seen in gaming computers when playing game programs as there's a large amount of data that needs to be processed for proper play. GPUs in these computers would need to process multiple thing like calculations, rendering, image processing to ensure smooth gameplay especially when the game presents a vast environment to play in. Hence it plays a role in high performance tasks like gaming through performing efficiently for real time responses.


3) How is heterogeneous GPU-CPU computing useful in solving a parallel pattern?

Hetrogenouse GPU-CPU computing is useful because it uses the strengths of the CPU and GPU. GPU are great at data operations because of parallel process cabability and a high throughput where as CPU are great at task operations and scheduling, combining these two results in more efficient processing of a wider range of tasks within a parallel pattern.

Which parallel pattern is group 6 using: Reduction!

Reduction is a parallel computation pattern used to combine a collection of values into a single result using an associative operation such as summation, multiplication, or finding the maximum/minimum. The process involves partitioning the data across multiple processing units, computing partial results in parallel, and then aggregating those partial results to produce the final output. Common operations in reduction include summation, product calculation, and logical operations like AND/OR. For example, in OpenMP, reduction can be implemented using a reduction(+:sum) clause to sum an array efficiently. In CUDA, a tree-based approach is often used to minimize synchronization overhead. Optimizations such as tree-based reduction, hierarchical reduction, and atomic operations help improve performance by reducing computation time and avoiding race conditions. Reduction is widely used in parallel computing to accelerate data aggregation tasks, making it essential for applications like big data processing, machine learning, and scientific computing
