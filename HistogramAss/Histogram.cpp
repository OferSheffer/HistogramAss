#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Histogram.h"

#define MASTER 0
#define NO_OMP_THREADS 4		// OMP: 4 core laptop

int main(int argc, char *argv[])
{
	const int ARR_SIZE = 300000;
	const int MY_ARR_SIZE = ARR_SIZE / 2;
	const int VALUES_RANGE = 256;
	int numprocs, myid;
	long hist[VALUES_RANGE] = { 0 };
	int* myLargeArr;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;

	if (myid == MASTER)
	{
		//init MASTER variables
		
		int* largeArr = (int*)calloc(ARR_SIZE, sizeof(int));

		//init largeArr values:
		srand(time(NULL));
		for (long i = 0; i < ARR_SIZE; i++)
			largeArr[i] = rand() % VALUES_RANGE;
			//largeArr[i] = 1;
		myLargeArr = largeArr;

	
		// TODO send half the array to slave for calculations


	}
	else  // *** SLAVE ***
	{
	}

	// use openMP for first half
	// TODO apply same code in slave
	int* ompHistogram = (int*)calloc(VALUES_RANGE, sizeof(int));
	int* ompCounterArr = (int*)calloc(NO_OMP_THREADS * VALUES_RANGE, sizeof(int));   // each thread given pseudo-private VALUES_RANGE

	#pragma omp parallel for
	for (int i = 0; i < MY_ARR_SIZE/2; ++i)
		ompCounterArr[omp_get_thread_num()*VALUES_RANGE + myLargeArr[i]]++;			// each thread collects histogram data

	#pragma omp parallel for num_threads(NO_OMP_THREADS)
	for (int i = 0; i < VALUES_RANGE; ++i)
		for (int ix = 0; ix < NO_OMP_THREADS; ix++)
		{
			ompHistogram[i] += ompCounterArr[ix*VALUES_RANGE + i];					// threads aggregate histogram data for specific histogram values
		}

	printf("OMP Histogram sample = {%d,%d,%d,%d,%d}\n",
		ompHistogram[0], ompHistogram[1], ompHistogram[2], ompHistogram[3], ompHistogram[4]);


	// use CUDA for second half
	cudaError_t cudaStatus = histogramWithCuda(hist, &(myLargeArr[MY_ARR_SIZE / 2]), MY_ARR_SIZE/2, VALUES_RANGE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("Histogram sample = {%d,%d,%d,%d,%d}\n",
		hist[0], hist[1], hist[2], hist[3], hist[4]);




	// TODO receive results from slave


	// TODO combine results with openMP


	// TODO display final histogram


	free(myLargeArr);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	

	MPI_Finalize();
	return 0;
}
