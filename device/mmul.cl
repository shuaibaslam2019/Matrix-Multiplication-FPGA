#include "../host/inc/matrixMult.h"  


#ifndef SIMD_WORK_ITEMS
#define SIMD_WORK_ITEMS 4 // default value
#endif

   // Tiled and coalesced version

__attribute__((reqd_work_group_size(TILE_SIZE,TILE_SIZE,1)))
__attribute__((num_simd_work_items(SIMD_WORK_ITEMS)))

__kernel void mmul(int M, int N, int K,  __global float* restrict A, __global float* restrict B, __global float* restrict C) {
        
        // Thread identifiers
        int row = get_local_id(0); // Local row ID (max: TILE_SIZE)
        int col = get_local_id(1); // Local col ID (max: TILE_SIZE)

        int globalRow = TILE_SIZE*get_group_id(0) + row; // Row ID of C (0..M)
        int globalCol = TILE_SIZE*get_group_id(1) + col; // Col ID of C (0..N)
     
        // Local memory to fit a tile of TILE_SIZE*TILE_SIZE elements of A and B
        __local float Asub[TILE_SIZE][TILE_SIZE];
        __local float Bsub[TILE_SIZE][TILE_SIZE];
     
        // Initialise the accumulation register
        float acc = 0.0f;
        
        // Loop over all tiles
        int numTiles = K/TILE_SIZE;
        for (int t=0; t<numTiles; t++) {
     
            // Load one tile of A and B into local memory
            int tiledRow = TILE_SIZE*t + row;
            int tiledCol = TILE_SIZE*t + col;

            Asub[col][row] = A[tiledCol*M + globalRow];
            Bsub[col][row] = B[globalCol*K + tiledRow];
     
            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
     
            // Perform the computation for a single tile
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