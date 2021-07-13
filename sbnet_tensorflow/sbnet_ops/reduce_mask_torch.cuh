/*

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

#pragma once

#include "cuda_helpers.h"
#include "reduce_mask_torch.cuh"
#include <cuda_runtime.h>

//
// Mask is NHW1
// This tensor can be quite small, say highest res is (64,1920,1200,1)
// Could be as small as (64,32,32,1) or even (64,7,7,1) for last ImageNet layer
//
// One possible work partition strategy assuming larger batch:
// wrap HW around tNTHREADS, run N blocks, one block per batch,
// reduce the count inside each block
// use atomicAdd to reduce the total number of blocks
// 
// For small batch inference it's going to be better to have a HW-blocked kernel.
// This works for, say, 1x1920x1200x1 block size 
// Sometimes it's going to be difficult to utilize the GPU.
// For instance how do we partition a 1x7x7 with block size 1?
// 
// Perhaps we can do N*bCntH*bCntW blocks and wrap the threads around block pixels?
// there's going to be some duplication in reads/BW waste but the inputs should be small anyway
// N*BCH*BCW blocks kernel: blockIdx.x=[0, N)
// tNTHREADS is tHb*tWb
//
// blockDim.x = tbH*tbW
// gridDim = (x=bCntW, y=bCntH, z=N)
// So basically run a CUDA block per sparsity block
// threadIdx.x = intra-block w+h*W, rounded up to 32 (warpLanes)
//


#define FULL_MASK 0xFFFFFFFF

#include <stdint.h>


void reduce_mask_cuda_wrapper(
        float* mask,                  // Mask array.
        int N,                          // Batch dimension of the mask.
        int H,                          // Height of the mask.
        int W,                          // Width of the mask.
        float threshold,                // Threshold for being active.
        int bOffsH0,                    // Block padding offset height, negative.
        int bOffsW0,                    // Block padding offset width, negative.
        int bSzH,                       // Block size height.
        int bSzW,                       // Block size width.
        int bStrH,                      // Block stride, height.
        int bStrW,                      // Block stride, width.
        int bCntH,                      // Number of blocks, height.
        int bCntW,                      // Number of blocks, width.
        unsigned int numBins,           // number of bins in binCounts
        unsigned int binSize,           // maximum size of each counter bin
        int16_t* activeBlockIndices,      // triples of [n, ih, iw] indices for active blocks.
        int32_t* binCounts,                 // Number of indices of active blocks.
        bool avgPool                    // true for avg pooling, false for max pooling
        );