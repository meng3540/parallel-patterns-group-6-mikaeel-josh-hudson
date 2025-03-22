#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Compute sum reduction
__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    // Perform reduction in shared memory
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads(); // Ensure all threads have completed the current stride
    }
    // Write the result of the reduction to the output
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// Function to print a matrix
void printMatrix(float* matrix, int numRows, int numColumns, const char* name) {
    printf("\nMatrix %s:\n", name);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            printf("%.1f ", matrix[i * numColumns + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv)
{
    float* hostA; // The A matrix on the host
    float* hostC; // The output C matrix on the host
    float* deviceA; // Device copy of A matrix
    float* deviceC; // Device result of sum reduction
    int numARows = 1; // Number of rows in the matrix A
    int numAColumns = 64; // Number of columns in the matrix A
    int numCRows = 1; // Number of rows in the matrix C
    int numCColumns = 1; // Number of columns in the matrix C

    // Calculate the size in bytes for matrix A and C
    int sizeA = (numARows * numAColumns) * sizeof(float);
    int sizeC = (numCRows * numCColumns) * sizeof(float);

    // Allocate and initialize the A matrix on the host
    hostA = (float*)malloc(sizeA);
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            hostA[i * numAColumns + j] = j + 1; // Initialize with values 1, 2, 3, ..., 64
        }
    }

    // Allocate the C matrix on the host
    hostC = (float*)malloc(sizeC);

    // Print the A matrix
    printMatrix(hostA, numARows, numAColumns, "A");

    printf("\nThe dimensions of A are %d x %d\n", numARows, numAColumns);
    printf("The dimensions of C are %d x %d\n", numCRows, numCColumns);

    // Allocate memory on the GPU for matrices A and C
    cudaMalloc((void**)&deviceA, sizeA);
    cudaMalloc((void**)&deviceC, sizeof(float));

    // Copy the A matrix from the host to the device
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);

    // Initialize the block dimensions
    int blockSize = numAColumns; // Number of threads per block
    int gridSize = (numARows * numAColumns + blockSize - 1) / blockSize; // Number of blocks

    printf("The block dimensions are %d\n", blockSize);
    printf("The grid dimensions are %d\n", gridSize);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the GPU Kernel
    SimpleSumReductionKernel << <gridSize, blockSize >> > (deviceA, deviceC);
    cudaDeviceSynchronize();

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Calculate memory bandwidth
    int totalBytes = sizeA + sizeC;
    float bandwidth = (totalBytes) / (elapsedTime * 1e3); // Bytes/s
    printf("Effective memory bandwidth: %f bytes/s\n", bandwidth);

    // Copy the result from the device back to the host
    float result;
    cudaMemcpy(&result, deviceC, sizeof(float), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceC);

    // Print the result
    printf("\nResult: %f\n", result);

    // Free host memory
    free(hostA);
    free(hostC);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
