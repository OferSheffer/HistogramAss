#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error; }



// 300,000 indices; 2,000 total threads; 2 blocks: 1,000 threads per block;
// Each thread in charge of 150 contigeous indices
#define THREAD_BLOCK_SIZE 150	
#define NO_BLOCKS  2            
#define THREADS_PER_BLOCK 1000

// Helper function for using CUDA to add vectors in parallel.
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Histogram code
__global__ void threadedHistKernel(int *threadedHist, int *arr, const int blockSize, const int valRange, const int threadBlockSize)
{
	int val,
		bid = blockIdx.x,
		tid = threadIdx.x,
		pid = bid*blockSize + tid;  //positional ID

									// each thread takes info from its given info and increases the relevant position on the threadedHist
	for (int i = 0; i < threadBlockSize; i++)
	{
		val = arr[pid*threadBlockSize + i];
		threadedHist[valRange*pid + val]++;

	}
}

__global__ void sumThreadedResultsKernel(long *dev_hist, int *dev_threadedHist, const int valRange, const int Blocks)
{
	//e.g. tid from 0 to valRange-1, blocks = THREADS_PER_BLOCK * NO_BLOCKS
	int tid = threadIdx.x;

	for (int bl = 0; bl < Blocks; bl++)
	{
		dev_hist[tid] += dev_threadedHist[bl*valRange + tid];
	}
}

cudaError_t histogramWithCuda(long* hist, const int* largeArr, const int arrSize, const int histSize)
{
	int  *dev_arr = 0;
	long *dev_hist = 0;
	int  *dev_threadedHist = 0;
	cudaError_t cudaStatus;

	// memory init block
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffers
		cudaStatus = cudaMalloc((void**)&dev_arr, arrSize * sizeof(int)); CHKMAL_ERROR;
		cudaStatus = cudaMalloc((void**)&dev_hist, histSize * sizeof(long)); CHKMAL_ERROR;
		cudaStatus = cudaMalloc((void**)&dev_threadedHist, THREADS_PER_BLOCK * NO_BLOCKS * histSize * sizeof(int)); CHKMAL_ERROR;    // each thread gets a "private" 

		// Copy input / memSet (Host to Device)
		cudaStatus = cudaMemcpy(dev_arr, largeArr, arrSize * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;
		cudaStatus = cudaMemcpy(dev_hist, hist, histSize * sizeof(int), cudaMemcpyHostToDevice); CHKMEMCPY_ERROR;

		cudaStatus = cudaMemset((void*)dev_threadedHist, 0, THREADS_PER_BLOCK * NO_BLOCKS * histSize * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemset failed!\n");
			goto Error;
		}

	}

	// *** phase 1 ***
	// Launch a kernel on the GPU with one thread for every THREAD_BLOCK_SIZE elements.
	threadedHistKernel << <NO_BLOCKS, THREADS_PER_BLOCK >> >(dev_threadedHist, dev_arr, THREADS_PER_BLOCK, histSize, THREAD_BLOCK_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "threadedHistKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;

	// *** phase 2 ***
	sumThreadedResultsKernel << <1, histSize >> >(dev_hist, dev_threadedHist, histSize, THREADS_PER_BLOCK * NO_BLOCKS);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sumThreadedResultsKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaDeviceSynchronize(); CHKSYNC_ERROR;

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(hist, dev_hist, histSize * sizeof(int), cudaMemcpyDeviceToHost); CHKMEMCPY_ERROR;

Error:
	cudaFree(dev_arr);
	cudaFree(dev_hist);
	cudaFree(dev_threadedHist);

	return cudaStatus;

}







// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
