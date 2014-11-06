/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#define BLOCK_SIZE 512

__device__ void atomicIncrementInt8(uint8_t* address)
{
    unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(size_t)address & 3];
    unsigned int old, assumed, add_, new_;
    uint8_t val = 0x01;
    old = *base_address;
    do {
        assumed = old;
        add_ = (uint8_t)__byte_perm(old, 0, ((size_t)address & 3)) + val;
        if (add_ > 255)
            add_ = 255;
        new_ = __byte_perm(old, add_, sel);
        old = atomicCAS(base_address, assumed, new_);
    } while (assumed != old);
}

__global__ void histo_kernel(unsigned int* input, uint8_t* bins, unsigned int num_elements,
                             unsigned int num_bins)
{
    /* Version 1
    unsigned int start = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 6;
    for (int i = start; i < start + 6; i++)
    {
        if (i < num_elements)
            atomicIncrementInt8(&bins[input[i]]);
    }*/

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (i < num_elements)
    {
        atomicIncrementInt8(&bins[input[i]]);
        i += stride;
    }

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) 
{
    /* Version 1
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((num_elements - 1) / (6 * BLOCK_SIZE) + 1, 1, 1);*/

    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((num_elements + 1) / (BLOCK_SIZE * 2) + 1, 1, 1);
    histo_kernel<<<dimGrid, dimBlock>>>(input, bins, num_elements, num_bins);    
}
