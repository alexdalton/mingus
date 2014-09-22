/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    // base matrix multiply
    /*int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < m) && (col < n))
    {
      float pvalue = 0;
	  for (int i = 0; i < k; ++i)
	  {
		  pvalue += A[row*k + i] * B[i*n + col];
	  }
	  C[row*n + col] = pvalue;
    }*/
    __shared__ float ds_M[TILE_SIZE][TILE_SIZE];
    __shared__ float ds_N[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float pvalue = 0;
    for(int i = 0; i < ((k - 1) / TILE_SIZE + 1); ++i)
    {
            ds_M[threadIdx.y][threadIdx.x] = A[row * k + i * TILE_SIZE + threadIdx.x];
            ds_N[threadIdx.y][threadIdx.x] = B[col + (i * TILE_SIZE + threadIdx.y) * n];
            __syncthreads();

            int loopCount;
            int numLeft = k - (i * TILE_SIZE);
            if( numLeft < TILE_SIZE)
                loopCount = numLeft;
            else
                loopCount = TILE_SIZE;

            for(int j = 0; j < loopCount; ++j)
            {
                pvalue += ds_M[threadIdx.y][j] * ds_N[j][threadIdx.x];
            }
            __syncthreads();
    }

    if((row < m) && (col < n))
        C[row*n + col] = pvalue;
}

void tiledSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    const unsigned int grid_x = (n - 1) / TILE_SIZE + 1; 
    const unsigned int grid_y = (m - 1) / TILE_SIZE + 1;
    dim3 DimGrid(grid_x, grid_y, 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);


    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<DimGrid, DimBlock>>>(m, n, k, A, B, C);
}

