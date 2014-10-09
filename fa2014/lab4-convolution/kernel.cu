/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

/***********************************************************************
* Determine input and output indexes of each thread                    *
* Load a tile of the input image to shared memory                      *
* Apply the filter on the input image tile                             *
* Write the compute values to the output image at the correct indexes  *
* Use OUTPUT_TILE_SIZE and INPUT_TILE_SIZE defined in support.h        *
************************************************************************/
__global__ void convolution(Matrix N, Matrix P)
{
    __shared__ float N_s[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and column for output tile
    int row_o = blockIdx.y * OUTPUT_TILE_SIZE + ty;
    int col_o = blockIdx.x * OUTPUT_TILE_SIZE + tx;

    // row and column for input tile
    int row_i = row_o - FILTER_SIZE / 2;
    int col_i = col_o - FILTER_SIZE / 2;

    // each thread loads one element from input matrix into shared memory
    if ((row_i >= 0) && (row_i < N.height) && (col_i >= 0) && (col_i < N.width))
    {
        N_s[ty][tx] = N.elements[row_i * N.width + col_i];
    }
    else
    {
        N_s[ty][tx] = 0.0f;
    }

    // sync all threads so shared memory is ready for convolutions calculation
    __syncthreads();

    float output = 0.0f;
    if ((ty < OUTPUT_TILE_SIZE) && (tx < OUTPUT_TILE_SIZE))
    {
        int i, j;
        // compute the convolution for the index in the output tile
        for (i = 0; i < FILTER_SIZE; i++)
        {
            for (j = 0; j < FILTER_SIZE; j++)
            {
                output += M_c[i][j] * N_s[i + ty][j + tx];
            }
        }
        // if output indices are in range write to output matrix
        if ((row_o < P.height) && (col_o < P.width))
        {
            P.elements[row_o * P.width + col_o] = output;
        }
    }
    // sync all threads to finish so next block can go
    __syncthreads();
}

