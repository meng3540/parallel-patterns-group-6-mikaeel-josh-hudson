#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 256
#define BLOCK_SIZE 16
#define COARSENING_FACTOR 2

// Compute sum reduction using tiling and thread coarsening
__global__ void TiledSumReductionKernel(float* input, float* output, int numElements) {
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * COARSENING_FACTOR + threadIdx.x;

    // Load elements into shared memory with thread coarsening
    float sum = 0.0f;
    for (int j = 0; j < COARSENING_FACTOR; j++) {
        if (i + j * blockDim.x < numElements) {
            sum += input[i + j * blockDim.x];
        }
    }
    sharedData[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of the reduction for this block to the output
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
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
    int numAColumns = N; // Number of columns in the matrix A
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
    cudaMalloc((void**)&deviceC, sizeof(float) * numAColumns);

    // Copy the A matrix from the host to the device
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);

    // Initialize the block dimensions
    int blockSize = BLOCK_SIZE; // Number of threads per block
    int gridSize = (numAColumns + blockSize * COARSENING_FACTOR - 1) / (blockSize * COARSENING_FACTOR); // Number of blocks

    printf("The block dimensions are %d\n", blockSize);
    printf("The grid dimensions are %d\n", gridSize);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch the GPU Kernel
    TiledSumReductionKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(deviceA, deviceC, numAColumns);
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
    float* partialResults = (float*)malloc(sizeof(float) * gridSize);
    cudaMemcpy(partialResults, deviceC, sizeof(float) * gridSize, cudaMemcpyDeviceToHost);

    // Sum the partial results on the host
    float result = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        result += partialResults[i];
    }

    // Free the GPU memory
    cudaFree(deviceA);
    cudaFree(deviceC);

    // Print the result
    printf("\nResult: %f\n", result);

    // Free host memory
    free(hostA);
    free(hostC);
    free(partialResults);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
