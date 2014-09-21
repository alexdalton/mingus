/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#include "support.h"

void  kernelTimer(int matArow, int matBcol, int matBrow, const float *A_d, const float *B_d, float *C_d){
    Timer timer;
    cudaError_t cuda_ret;
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    tiledSgemm('N', 'N', matArow, matBcol, matBrow, 1.0f, \
                A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);

    cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
}
