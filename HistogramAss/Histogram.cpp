#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Histogram.h"

#define MASTER 0
#define SLAVE  1
#define NO_OMP_THREADS 4		// OMP: 4 core laptop
#define VALUES_RANGE   256

int main(int argc, char *argv[])
{
	const int ARR_SIZE = 100000;
	const int MY_ARR_SIZE = ARR_SIZE / 2;
	int numprocs, myid;
	long hist[VALUES_RANGE] = { 0 };
	int* myLargeArr;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;

	if (numprocs != 2) {
		printf("Please run with %d processes.\n", 2); fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (myid == MASTER)
	{
		//init MASTER variables
		int* largeArr = (int*)calloc(ARR_SIZE, sizeof(int));

		//init largeArr values:
		srand(time(NULL));
		for (long i = 0; i < ARR_SIZE; i++)
			largeArr[i] = rand() % VALUES_RANGE;
			//largeArr[i] = 1;   // alternative test value

		myLargeArr = largeArr;	 // Master works with original array

		// send half the array to slave for calculations
		MPI_Send(&(largeArr[MY_ARR_SIZE]), MY_ARR_SIZE, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);

	}
	else  // *** SLAVE ***
	{
		// recv half the array to slave for calculations
		myLargeArr = (int*)calloc(MY_ARR_SIZE, sizeof(int));
		MPI_Recv(myLargeArr, MY_ARR_SIZE, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
	}

	// use openMP for first half
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

	// use CUDA for second half
	cudaError_t cudaStatus = histogramWithCuda(hist, &(myLargeArr[MY_ARR_SIZE / 2]), MY_ARR_SIZE/2, VALUES_RANGE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	#pragma omp parallel for
	for (int i = 0; i < VALUES_RANGE; ++i)
	{
		hist[i] += ompHistogram[i];					// threads aggregate local OMP & CUDA results
	}

	free(ompCounterArr);
	free(ompHistogram);

	if (myid == MASTER)
	{
		// receive results from slave
		int* recvHist = (int*)calloc(VALUES_RANGE, sizeof(int));
		MPI_Recv(recvHist, VALUES_RANGE, MPI_INT, SLAVE, 0, MPI_COMM_WORLD, &status);

		// combine results with openMP
		#pragma omp parallel for
		for (int i = 0; i < VALUES_RANGE; ++i)
		{
			hist[i] += recvHist[i];					// threads aggregate local OMP & CUDA results
		}

		// display final histogram
		printf("Histogram:\n");
		for (int i = 0; i < VALUES_RANGE; i++)
			printf("%d ", hist[i]);
		printf("\n");

		free(recvHist);
	}
	else  // *** SLAVE ***
	{
		// send results to master
		MPI_Send(hist, VALUES_RANGE, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
	}

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
