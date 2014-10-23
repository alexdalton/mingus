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
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    if ((start + tx) < size)
    {
        partialSum[tx] = in[start + tx];
    }  
    else
    {
        partialSum[tx] = 0.0f;
    }
    if ((start + blockDim.x + tx) < size)
    {
        partialSum[blockDim.x + tx] = in[start + blockDim.x + tx];
    }
    else
    {
        partialSum[blockDim.x + tx] = 0.0f;
    }
    __syncthreads();

    int stride = 1;
    while (stride < 2 * BLOCK_SIZE)
    {
    	int index = (tx + 1) * stride * 2 - 1;
    	if (index < 2* BLOCK_SIZE)
    	{
    		partialSum[index] += partialSum[index - stride];
    	}
    	stride *= 2;
    	__syncthreads();
    }

    if (tx == 0)
        out[blockIdx.x] = partialSum[2 * BLOCK_SIZE - 1];

}
