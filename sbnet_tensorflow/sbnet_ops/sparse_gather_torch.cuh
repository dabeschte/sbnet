#pragma once

#include "cuda_helpers.h"
#include "cuda_runtime.h"
#include "op_utils_torch.h"
#include <iostream>
#include <algorithm>

using std::cout;
using std::endl;

#define COMPUTE_R1(RR) ((RR) < 7 ? ((RR) == 1 ? 1 : 2) : 4)

namespace {
struct LaunchParams {
    dim3 block, grid;
    int shmemSize;
    int bSzH1;
    int fittingC1;
    enum { MAX_SHMEM = 24*1024 };
    LaunchParams(int C, int bSzH, int bSzW, int numActive)
    {
        fittingC1 = std::min(32, C);
        bSzH1 = COMPUTE_R1(bSzH);
        while ((shmemSize = (fittingC1+1)*bSzH1*bSzW*sizeof(float)) > MAX_SHMEM)
            fittingC1--;
        assert(fittingC1 >= 1);
        assert(bSzH1*bSzW*(fittingC1+1)*sizeof(float) <= MAX_SHMEM);
        block = dim3(512, 1, 1);
        grid = dim3(numActive, DIVUP(C, fittingC1), DIVUP(bSzH, bSzH1));
    }
};
}

void sparse_gather_cuda_wrapper(
    const float* x, int N, int H, int W, int C,
    float* y,
    int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
    int numActive, const short* activeBlockIndices, bool transpose
);

void sparse_scatter_cuda_wrapper(
    const float* x, int N, int H, int W, int C,
    float* y,
    int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
    int numActive, const short* activeBlockIndices, bool add, bool transpose, bool atomic
);

void copy_tensor_cuda_wrapper(float* dst, const float* src, int count);