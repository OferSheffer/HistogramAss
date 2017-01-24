#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Histogram.h"

int main(int argc, char *argv[])
{
	int numprocs, myid;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;




	const int ARR_SIZE = 300000;
	const int VALUES_RANGE = 256;
	int* largeArr = (int*)malloc(ARR_SIZE * sizeof(int));
	for (int i = 0; i < ARR_SIZE; i++)
	{
		largeArr[i] = 0;
	}
	long hist[VALUES_RANGE] = { 0 };

	srand(time(NULL));
	for (long i = 0; i < ARR_SIZE; i++)
		largeArr[i] = rand() % VALUES_RANGE;
	//largeArr[i] = 1;



	cudaError_t cudaStatus = histogramWithCuda(hist, largeArr, ARR_SIZE, VALUES_RANGE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("Histogram sample = {%d,%d,%d,%d,%d}\n",
		hist[0], hist[1], hist[2], hist[3], hist[4]);

	free(largeArr);

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
