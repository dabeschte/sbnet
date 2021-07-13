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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <vector>
#include <cuda_runtime.h>

#include "op_utils_torch.h"
#include "sparse_gather_torch.h"
#include "sparse_gather_torch.cuh"
#include <stdint.h>
#include <torch/extension.h>
#include "reduce_mask_torch.cuh"

std::vector<torch::Tensor> reduce_mask_wrapper(
    torch::Tensor mask,
    std::vector<int32_t> bcount_dynamic,
    std::vector<int32_t> bsize_dynamic,
    std::vector<int32_t> bstride_dynamic,
    std::vector<int32_t> boffset_dynamic,
    float tol,
    bool avg_bool
)
{
    // Grabs input shape.
    int N = mask.size(0);
    // TODO Mathias CHECK if we can make the mask 4 dimensional like in delta conv
    // changed shape indices according to pytorch standards --> but using channels last to make kernels behave correctly
    int H = mask.size(2);
    int W = mask.size(3);

    int bCntH = bcount_dynamic[0];
    int bCntW = bcount_dynamic[1];
    int bSzH = bsize_dynamic[0];
    int bSzW = bsize_dynamic[1];
    int bStrH = bstride_dynamic[0];
    int bStrW = bstride_dynamic[1];
    int bOffsH0 = boffset_dynamic[0];
    int bOffsW0 = boffset_dynamic[1];
    printf("cnt=%d, %d, sz=%d, %d, str=%d, %d, offs=%d, %d  N=%d, H=%d, W=%d\n",
          bCntH, bCntW, bSzH, bSzW, bStrH, bStrW, bOffsH0, bOffsW0, N, H, W);
    fflush(stdout);

    // Initializes output.
    // TODO: try to find a way not to redo the allocation in Compute
    int maxIndices = N * bCntH * bCntW;
    torch::Tensor activeBlockIndices = torch::empty({ maxIndices, 3 }, torch::TensorOptions().dtype(torch::kInt16).device(mask.device()).layout(mask.layout()));

    unsigned int numBins = 1;
    unsigned int binSize = (maxIndices + numBins - 1) / numBins;
    torch::Tensor binCounts = torch::zeros({ numBins }, torch::TensorOptions().dtype(torch::kInt32).device(mask.device()).layout(mask.layout()));

    // TODO continue porting to PyTorch

    reduce_mask_cuda_wrapper(
        mask.data_ptr<float>(),                    // Mask array.
        N,                                        // Batch dimension of the mask.
        H,                                        // Height of the mask.
        W,                                        // Width of the mask.
        tol,                                     // Threshold for being active.
        bOffsH0,                                  // Block padding offset height.
        bOffsW0,                                  // Block padding offset width.
        bSzH,                                     // Block size height.
        bSzW,                                     // Block size width.
        bStrH,                                    // Block stride, height.
        bStrW,                                    // Block stride, width.
        bCntH,                                    // Number of blocks, height.
        bCntW,                                    // Number of blocks, width.
        numBins,
        binSize,
        activeBlockIndices.data_ptr<int16_t>(), // Indices of active blocks.
        binCounts.data_ptr<int32_t>(),           // Counts per bin of active blocks.
        avg_bool
    );

    int readBack_ = 0;

    cudaMemcpy(&readBack_, binCounts.data_ptr<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (readBack_ == 0) {
        cudaMemset(activeBlockIndices.data_ptr<int16_t>(), 0, sizeof(int16_t)*3);
        // cudaMemset(activeBlockIndices.data_ptr<int16_t>(), 0, sizeof(int16_t)*3*maxIndices);
        readBack_ = 1;
    }

    return {activeBlockIndices, binCounts};
}

using std::cout;
using std::endl;

torch::Tensor sparse_gather_wrapper(
    torch::Tensor x, 
    torch::Tensor binCounts, 
    torch::Tensor activeBlockIndices, 
    std::vector<int32_t> bsize_dynamic, 
    std::vector<int32_t> bstride_dynamic, 
    std::vector<int32_t> boffset_dynamic,
    int numActive, 
    bool transpose
) {
    int bSzH = bsize_dynamic[0];
    int bSzW = bsize_dynamic[1];
    int bStrH = bstride_dynamic[0];
    int bStrW = bstride_dynamic[1];
    int bOffsH0 = boffset_dynamic[0];
    int bOffsW0 = boffset_dynamic[1];
    // changed shape indices according to pytorch standards --> but using channels last to make kernels behave correctly
    int N = x.size(0);
    int H = x.size(2);
    int W = x.size(3);
    int C = x.size(1);

    int32_t bin0Count = binCounts[0].item<int32_t>();
    int yShapeArr[] = { bin0Count, C, bSzH, bSzW };
    if (transpose)
    {
        // output is NCHW for tranposed version
        yShapeArr[1] = C;
        yShapeArr[2] = bSzH;
        yShapeArr[3] = bSzW;
    }

    torch::Tensor y = torch::empty({yShapeArr[0],yShapeArr[1],yShapeArr[2],yShapeArr[3]}, torch::TensorOptions().dtype(x.dtype()).device(x.device()).layout(x.layout()));
    y = y.contiguous(at::MemoryFormat::ChannelsLast);

    sparse_gather_cuda_wrapper(
        x.data_ptr<float>(), N, H, W, C,
        y.data_ptr<float>(),
        bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
        bin0Count, (const short*)activeBlockIndices.data_ptr<int16_t>(),
        transpose);

    return y;
}


// REGISTER_OP("SparseScatterVar")
//     .Attr("T: {float}")
//     .Attr("add: bool")
//     .Attr("atomic: bool = false")
//     .Attr("transpose: bool = false")
//     .Input("x: T") // Dimensions: bin_counts[0]*bsize[0]*bsize[1]*C
//     .Input("bin_counts: int32")
//     .Input("active_block_indices: int16")
//     .Input("ybase: Ref(T)") // ybase values will be overwritten with scatters from x
//     .Input("dynamic_bsize: int32")
//     .Input("dynamic_bstride: int32")
//     .Input("dynamic_boffset: int32")
//     .Output("y: Ref(T)"); // Dimensions: NHWC, scatter will write on top of current y content

// REGISTER_OP("SparseScatter")
//     .Attr("T: {float}")
//     .Attr("add: bool")
//     .Attr("atomic: bool = false")
//     .Attr("transpose: bool = false")
//     .Input("x: T") // Dimensions: bin_counts[0]*bsize[0]*bsize[1]*C
//     .Input("bin_counts: int32")
//     .Input("active_block_indices: int16")
//     .Input("ybase: T") // ybase values will be copied to output and overwritten with scatters from x
//     .Input("dynamic_bsize: int32")
//     .Input("dynamic_bstride: int32")
//     .Input("dynamic_boffset: int32")
//     .Output("y: T"); // Dimensions: NHWC, scatter will write on top of ybase content

torch::Tensor sparse_scatter_wrapper(
    torch::Tensor x,
    torch::Tensor binCounts,
    torch::Tensor activeBlockIndices,
    torch::Tensor ybase,
    std::vector<int32_t> bsize_dynamic,
    std::vector<int32_t> bstride_dynamic,
    std::vector<int32_t> boffset_dynamic,
    bool add,
    bool atomic,
    bool transpose
)
{
    cudaStream_t stream = 0;
    // changed shape indices according to pytorch standards --> but using channels last to make kernels behave correctly
    // Grabs input shape.
    int N = ybase.size(0);
    int H = ybase.size(2);
    int W = ybase.size(3);
    int C = ybase.size(1);

    int bSzH = bsize_dynamic[0];
    int bSzW = bsize_dynamic[1];
    int bStrH = bstride_dynamic[0];
    int bStrW = bstride_dynamic[1];
    int bOffsH0 = boffset_dynamic[0];
    int bOffsW0 = boffset_dynamic[1];

    // TODO Mathias check if y is pre allocated
    bool use_var = true;

    // read the number of active blocks from bin_counts input that is expected to be always in host mem
    int32_t bin0Count = binCounts[0].item<int32_t>();

    // TODO: verify sizes of x match { bin0Count, bSzH_, bSzW_, C };
    // TODO: try to find a way not to redo the allocation in Compute
    float* outData = nullptr;
    torch::Tensor y;
    if (use_var) {
        // TODO Mathias what does this do???
        outData = ybase.data_ptr<float>();
        y = ybase;
    } else {
        // Initializes output.
        y = torch::zeros_like(ybase);
        float* destData = y.data_ptr<float>();
        int sz = y.numel();
        const float* srcData = ybase.data_ptr<float>();
        // TODO @Mathias is this only a memcpy?
        cudaMemcpy(destData, srcData, sizeof(float)*sz, cudaMemcpyDeviceToDevice);
        outData = y.data_ptr<float>();
    }

    // Splat/add x on top of y
    sparse_scatter_cuda_wrapper(
        x.data_ptr<float>(), N, H, W, C,
        outData,
        bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
        bin0Count, (const short*)activeBlockIndices.data_ptr<int16_t>(),
        add, transpose, atomic
    );

    return y;
}

// template<typename Device>
// class CudaTimerStart : public OpKernel
// {
// public:
//     explicit CudaTimerStart(OpKernelConstruction *context) : OpKernel(context)
//     {
//         // creating a persistent start event in constructor doesn't seem to work because destructor gets called prematurely
//     }

//     void Compute(OpKernelContext *context) override
//     {
//         //const cudaStream_t *stream = (cudaStream_t*)CopyTensorFunctor<Device, float>().getStream(context->eigen_device<Device>()); 
//         //printf("recording start event=%x\n", reinterpret_cast<int64>(event));
//         //printf("start stream=%d\n", stream ? *stream : 0);
//         // AP: don't use streams because TF can split subgraphs so we want to sync on all streams

//         Tensor* output = nullptr;
//         AllocatorAttributes hostAttr; hostAttr.set_on_host(true);
//         OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output, hostAttr));

//         cudaEvent_t event;
//         gpuErrorCheck(cudaEventCreate(&event));
//         //gpuErrorCheck(cudaEventRecord(event, stream ? *stream : 0));
//         gpuErrorCheck(cudaEventRecord(event));
//         output->scalar<int64>()() = reinterpret_cast<int64>(event);
//     }
// };

// template<typename Device>
// class CudaTimerEnd : public OpKernel
// {
// public:
//     explicit CudaTimerEnd(OpKernelConstruction *context) : OpKernel(context)
//     {
//         gpuErrorCheck(cudaEventCreate(&event_));
//     }

//     virtual ~CudaTimerEnd() override
//     {
//         //printf("Destroying end event\n");
//         gpuErrorCheck(cudaEventDestroy(event_));
//     }

//     void Compute(OpKernelContext *context) override
//     {
//         //const cudaStream_t *stream = (cudaStream_t*)CopyTensorFunctor<Device, float>().getStream(context->eigen_device<Device>()); 
//         //printf("end stream=%d\n", stream ? *stream : 0);
//         // AP: don't use streams because TF can split subgraphs so we want to sync on all streams
//         gpuErrorCheck(cudaEventRecord(event_));//, stream ? *stream : 0));
//         gpuErrorCheck(cudaEventSynchronize(event_));

//         Tensor* output = nullptr;
//         AllocatorAttributes hostAttr; hostAttr.set_on_host(true);
//         OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output, hostAttr));
//         cudaEvent_t startEvent = reinterpret_cast<cudaEvent_t>(context->input(0).scalar<int64>()());
//         //printf("startEvent handle=%x\n", startEvent);
//         float time;
//         gpuErrorCheck(cudaEventElapsedTime(&time, startEvent, event_)); // ms
//         output->scalar<float>()() = time;
//         //printf("TIME=%.2f\n", time);
//         gpuErrorCheck(cudaEventDestroy(startEvent));
//     }
// private:
//     cudaEvent_t event_;
// };


// REGISTER_KERNEL_BUILDER(Name("CudaTimerStart").Device(DEVICE_GPU).HostMemory("start_event"), CudaTimerStart<GPUDevice>);
// REGISTER_KERNEL_BUILDER(Name("CudaTimerEnd").Device(DEVICE_GPU).HostMemory("dt").HostMemory("start_event"), CudaTimerEnd<GPUDevice>);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("reduce_mask_wrapper", &reduce_mask_wrapper, "SBNet reduce mask (CUDA)");
    m.def("sparse_gather_wrapper", &sparse_gather_wrapper, "SBNet sparse gather (CUDA)");
    m.def("sparse_scatter_wrapper", &sparse_scatter_wrapper, "SBNet sparse scatter (CUDA)");
}