#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>

#define CHKMAL_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMalloc failed!"); goto Error; }
#define CHKMEMCPY_ERROR if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed!"); goto Error; }
#define CHKSYNC_ERROR	if (cudaStatus != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed! Error code %d\n", cudaStatus); goto Error; }


// arrSize indices; THREADS_PER_BLOCK * NO_BLOCKS total threads;
// Each thread in charge of THREAD_BLOCK_SIZE contigeous indices
#define NO_BLOCKS  5       
#define THREADS_PER_BLOCK 1000

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
	if (arrSize % (THREADS_PER_BLOCK * NO_BLOCKS) != 0) {
		fprintf(stderr, "histogramWithCuda launch failed:\n"
			"Array size (%d) modulo Total threads (%d) != 0.\n"
			"Try changing number of threads.\n", arrSize, (THREADS_PER_BLOCK * NO_BLOCKS));
		goto Error;
	}

	const int THREAD_BLOCK_SIZE = arrSize / (THREADS_PER_BLOCK * NO_BLOCKS);
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
		cudaStatus = cudaMalloc((void**)&dev_threadedHist, THREADS_PER_BLOCK * NO_BLOCKS * histSize * sizeof(int)); CHKMAL_ERROR;    // each thread gets a "private" histogram

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
