#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Histogram.h"

#define MASTER 0

int main(int argc, char *argv[])
{
	int numprocs, myid;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Status status;

	if (myid == MASTER)
	{
		//init variables
		const int ARR_SIZE = 300000;
		const int VALUES_RANGE = 256;
		int* largeArr = (int*)calloc(ARR_SIZE, sizeof(int));
		long hist[VALUES_RANGE] = { 0 };

		srand(time(NULL));
		for (long i = 0; i < ARR_SIZE; i++)
			largeArr[i] = rand() % VALUES_RANGE;
		//largeArr[i] = 1;


		int*  valCountArr = (int*)calloc(VALUES_RANGE, sizeof(int));
		int*  dataArr = (int*)malloc(ARR_SIZE * sizeof(int));
		int** counterMat = (int**)malloc(4 * sizeof(int*)); //init to my 4 core laptop
		for (int i = 0; i < 4; i++)
		{
			counterMat[i] = (int*)calloc(VALUES_RANGE, sizeof(int));
		}

		// TODO send half the array to slave for calculations



		// TODO use openMP for first half




		// TODO use CUDA for second half
		cudaError_t cudaStatus = histogramWithCuda(hist, largeArr, ARR_SIZE, VALUES_RANGE);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		printf("Histogram sample = {%d,%d,%d,%d,%d}\n",
			hist[0], hist[1], hist[2], hist[3], hist[4]);




		// TODO receive results from slave


		// TODO combine results with openMP


		// TODO display final histogram


		free(largeArr);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}

	MPI_Finalize();
	return 0;
}
