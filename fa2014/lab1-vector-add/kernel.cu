/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__global__ 
void vecAddKernel(float* A, float* B, float* C, int n) 
{

    // Calculate global thread index based on the block and thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Use global index to determine which elements to read, add, and write
    if(i < n)
    {
        C[i] = A[i] + B[i];
    }

}

