/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/
    __shared__ float scan_array[BLOCK_SIZE];
    scan_array[threadIdx.x] = in[threadIdx.x];
    __syncthreads();

    int stride = 1;
    while (stride < BLOCK_SIZE)
    {
    	int index = (threadIdx.x + 1) * stride * 2 - 1;
    	if (index < BLOCK_SIZE)
    	{
    		scan_array[index] += scan_array[index - stride];
    		stride *= 2;
    	}
    	__syncthreads();
    }
}
