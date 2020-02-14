#include "../host/inc/matrixMult.h"  


#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

   
__attribute__((reqd_work_group_size(TILE_SIZE,TILE_SIZE,1)))
__attribute__((num_simd_work_items(SIMD_WORK_ITEMS)))

__kernel void mmul(int M, int N, int K,  __global float* A, __global float* B, __global float* restrict C) {
        
        // Thread identifiers
        int row = get_local_id(0); 
        int col = get_local_id(1); 

        int globalRow = TILE_SIZE*get_group_id(0) + row; 
        int globalCol = TILE_SIZE*get_group_id(1) + col; 
             
        __local float A_loc[TILE_SIZE][TILE_SIZE];
        __local float B_loc[TILE_SIZE][TILE_SIZE];
     
        
        float acc = 0.0f;
        
        
        int numTiles = K/TILE_SIZE;
        for (int t=0; t<numTiles; t++) {
     
            
            int tiledRow = TILE_SIZE*t + row;
            int tiledCol = TILE_SIZE*t + col;

            A_loc[col][row] = A[tiledCol*M + globalRow];
            B_loc[col][row] = B[globalCol*K + tiledRow];
     
            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
     
            
            #pragma unroll
            for (int k=0; k<TILE_SIZE; k++) {
                acc += Asub[k][row] * Bsub[col][k];
            }
     
            // Synchronise before loading the next tile
            barrier(CLK_LOCAL_MEM_FENCE);
        }
     
        // Store the final result in C
        C[globalCol*M + globalRow] = acc;
    }