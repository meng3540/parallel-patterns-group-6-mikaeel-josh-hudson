Welcome to The group 6 repository

Here are some FAQs

1) What are parallel patterns in computing? 
  
Parallel patterns are common methods or techniques used to solve problems regarding processing speed and efficiency, by performing multiple calculations or processes simultaneously. Parallel patterns help effectivley and efficiently utilize multicore processors (like the GPU) when executing algorithims. 
  
2) What is their significance?- Explain in the context of typical applications they are used in. 

Their significance is the processing speed they offer since they allow several pieces of data to reach their destination simultaneously. This can be especially seen in gaming computers when playing game programs as there's a large amount of data that needs to be processed for proper play. GPUs in these computers would need to process multiple thing like calculations, rendering, image processing to ensure smooth gameplay especially when the game presents a vast environment to play in. Hence it plays a role in high performance tasks like gaming through performing efficiently for real time responses.


3) How is heterogeneous GPU-CPU computing useful in solving a parallel pattern?

Hetrogenouse GPU-CPU computing is useful because it uses the strengths of the CPU and GPU. GPU are great at data operations because of parallel process cabability and a high throughput where as CPU are great at task operations and scheduling, combining these two results in more efficient processing of a wider range of tasks within a parallel pattern.

Which parallel pattern is group 6 using: Reduction!

Imagine you and your friends have a big pile of apples, and you want to count them all quickly. Instead of one person counting every apple, you split the apples into smaller piles and give each friend a pile to count. Once everyone finishes counting their own pile, they tell the numbers to one person, who adds them all together to get the final total.

This is how reduction works in computers! Instead of counting apples, computers add numbers, find the biggest one, or do other calculations—really fast! This helps with things like video games, weather predictions, and even robots learning new things. 

Reduction is a parallel computation pattern used to combine a collection of values into a single result using an associative operation such as summation, multiplication, or finding the maximum/minimum. Such as the provided image:


<div align="center">
  <img src="https://github.com/user-attachments/assets/c5c1be50-dbeb-45b1-9fd0-bc159949a08c">
</div>


The basic reduction algorithm in parallel computing works by breaking a large problem into smaller parts, processing them in parallel, and then combining the results efficiently. First, the input data is divided into smaller chunks, with each processing unit (such as a thread or core) handling a portion of the data. Each unit then computes a partial result independently, applying an operation like summation, multiplication, or finding the maximum value. Once the partial results are obtained, they are combined iteratively in a structured manner, often using a tree-based approach, where pairs of values are reduced step by step until only a single final result remains.

Common operations in reduction include summation, product calculation, and logical operations like AND/OR. 

In CUDA, a tree-based approach is often used to minimize synchronization overhead. Reduction is widely used in parallel computing to accelerate data aggregation tasks, making it essential for applications like big data processing, machine learning, and scientific computing

