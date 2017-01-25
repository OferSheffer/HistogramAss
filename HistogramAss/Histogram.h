#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t histogramWithCuda(long* hist, const int* largeArr, const int histSize, const int arrSize);