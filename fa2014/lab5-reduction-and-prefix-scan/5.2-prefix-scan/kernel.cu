/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

__global__ void scan(float *sums, float *out, float *in, unsigned size)
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
        partialSum[tx] = in[start + tx];
    else
        partialSum[tx] = 0.0f;
    if ((start + blockDim.x + tx) < size)
        partialSum[blockDim.x + tx] = in[start + blockDim.x + tx];
    else
        partialSum[blockDim.x + tx] = 0.0f;
    __syncthreads();

    int stride = 1;
    while (stride < 2 * BLOCK_SIZE)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
        {
                partialSum[index] += partialSum[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    
    if (tx == 0)
    {
        sums[blockIdx.x] = partialSum[2 * BLOCK_SIZE - 1];
        partialSum[2 * BLOCK_SIZE - 1] = 0;
    }

    stride = BLOCK_SIZE;
    while(stride > 0)
    {
        int index = (tx + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE)
        {
            float temp = partialSum[index];
            partialSum[index] += partialSum[index - stride];
            partialSum[index - stride] = temp;
        }
        stride /= 2;
        __syncthreads();
    }

    if ((start + tx) < size)
        out[start + tx] = partialSum[tx];
    if ((start + blockDim.x + tx) < size)
        out[start + blockDim.x + tx] = partialSum[blockDim.x + tx]; 

}

__global__ void addSums(float *sums, float *out, unsigned size)
{
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int tx = threadIdx.x;
    if ((start + tx) < size)
        out[start + tx] += sums[blockIdx.x];
    if ((start + blockDim.x + tx) < size)
        out[start + blockDim.x + tx] += sums[blockIdx.x];
}


/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
    float * sums;
    dim3 dim_grid, dim_block;
    unsigned out_elements;
    out_elements = in_size / (BLOCK_SIZE << 1);
    if (in_size % (BLOCK_SIZE<<1)) out_elements++;

    cudaMalloc((void**)&sums, out_elements*sizeof(float));

    dim_block.x = BLOCK_SIZE;
    dim_block.y = dim_block.z = 1;
    dim_grid.x = out_elements;
    dim_grid.y = dim_grid.z = 1;

    if (in_size <= 2 * BLOCK_SIZE)
        scan<<<dim_grid, dim_block>>>(sums, out, in, in_size);
    else
    {
        scan<<<dim_grid, dim_block>>>(sums, out, in, in_size);
        float * sums_scanned;
        cudaMalloc((void**)&sums_scanned, out_elements*sizeof(float));
        preScan(sums_scanned, sums, out_elements);
        addSums<<<dim_grid, dim_block>>>(sums_scanned, out, in_size);
        cudaFree(sums_scanned);
    }
    cudaFree(sums);
}

