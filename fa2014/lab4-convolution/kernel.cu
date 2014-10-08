/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P) {

	/************************************************************************
     * Determine input and output indexes of each thread                    *
     * Load a tile of the input image to shared memory                      *
     * Apply the filter on the input image tile                             *
     * Write the compute values to the output image at the correct indexes  *
     * Use OUTPUT_TILE_SIZE and INPUT_TILE_SIZE defined in support.h        *
	 ************************************************************************/

    //INSERT KERNEL CODE HERE
}

