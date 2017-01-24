#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

cudaError_t histogramWithCuda(long* hist, const int* largeArr, const int histSize, const int arrSize);