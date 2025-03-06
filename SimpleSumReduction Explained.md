The algorithm implemented in the provided CUDA code is a simple sum reduction. Sum reduction is a common parallel algorithm used to sum all elements of an array. The goal is to reduce the array to a single sum value efficiently using parallel processing.

ALGORITHIM EXPLAINATION:
1.	Initialization: Allocate memory for the input array A and the output variable C on both the host (CPU) and the device (GPU). Initialize the input array A with values.
2.	Memory Transfer: Copy the input array A from the host to the device.
3.	Kernel Execution: Launch the CUDA kernel to perform the sum reduction on the GPU. The kernel uses a parallel reduction technique to sum the elements of the array.
4.	Memory Transfer Back: Copy the result from the device back to the host.
5.	Cleanup: Free the allocated memory on both the host and the device.

KERNEL BREAKDOWN: 
  - The kernel takes an input array of any length and will output a single value which is the sum of the entire array. 
  - An index value is calculated which will represent a thread in the code, each thread will work on a pair of elements
  - The reduction loop: A new variable is created called 'stride', this represents the amount of space in between the 2 elements being added. This is needed to properly assign the correct index in the kernel, the stride is the for loop variable, which doubles every iteration and will increase up to the x dimension of the block. The addition will only be executed if the selected thread is a multiple of the stride x 2 (so the same values are not added), this ensures that the spacing of the addition is proper and will not leave values out or get used more then once. In each iteration, threads add elements that are 'stride' positions apart.
  - Synchronization: __syncthreads() ensures all threads complete their operations before moving to the next iteration.
  - Result storage: After all interations are complete the first thread (thread 0) writes the final sum to the output and the kernel is now complete. variable.

HOST CODE:
The host code sets up the environment and launches the kernel:
- Memory Allocation: Allocate memory for the input and output arrays on both the host and the device.
- Memory Copy: Copy the input array from the host to the device.
- Kernel Launch: Launch the kernel with the calculated grid and block dimensions.
- Timing: Use CUDA events to measure the kernel execution time.
- Result Copy: Copy the result from the device back to the host.
- Cleanup: Free the allocated memory and destroy the CUDA events.

ERRORS:
Things to avoid/consider when implementing this algorithim:
- Changes must be made to reduce a matrix, as the kernel only considers the X dimension of blocks. Solving this problem, the group edited the matrix population host code from Lab 4 to sum-reduce an array as a demonstration of a basic algorithim instead of a matrix.

