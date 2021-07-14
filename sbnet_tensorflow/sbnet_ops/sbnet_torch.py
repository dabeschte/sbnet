import torch
from sbnet.cuda import reduce_mask_wrapper, sparse_scatter_wrapper, sparse_gather_wrapper
from torch._C import memory_format

def reduce_mask(mask, bcount, bsize, bstride, boffset, tol, avg_bool=False):
    return reduce_mask_wrapper(
        mask, 
        bcount,
        bsize,
        bstride,
        boffset,
        tol, 
        avg_bool
        )

def sparse_gather(x, bcount, activeBlockIndices, bsize, bstride, boffset, numActive=0, transpose=False) -> torch.Tensor:
    return sparse_gather_wrapper(
        x, 
        bcount,
        activeBlockIndices,
        bsize,
        bstride,
        boffset,
        numActive, 
        transpose
        )

def sparse_scatter(x, bcount, activeBlockIndices, ybase, bsize, bstride, boffset, add=False, atomic=False, transpose=False) -> torch.Tensor:
    return sparse_scatter_wrapper(
            x, 
            bcount,
            activeBlockIndices,
            ybase,
            bsize,
            bstride,
            boffset,
            add, 
            atomic,
            transpose
        )



from torch.nn.common_types import _size_2_t
import torch.nn as nn
import torch

class SBNetConv(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros', 
            threshold=0.0,
            tol = 0.0,
            blockSize=(16,16),
            blockStride=None,
            blockOffset=None,
            mask_out = False,
            add_to_prev_out = False
    ):
        super(SBNetConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.tol = tol
        self.threshold = threshold
        self.blockSize = blockSize
        if blockStride is None:
            blockStride = (blockSize[0] - 2*padding[0], blockSize[1] - 2*padding[1])
        self.blockStride = blockStride
        if blockOffset is None:
            blockOffset = (-padding[0], -padding[1])
        self.blockOffset = blockOffset
        self.first_iter = True
        self.blockCount = 0
        self.inBlockParams = {"bsize": self.blockSize, "boffset": self.blockOffset, "bstride": self.blockStride}
        self.outBlockParams = {"bsize": self.blockStride, "boffset": (0,0), "bstride": self.blockStride}
        self.mask_out = mask_out
        self.transpose = False
        self.use_native_conv = stride[0] != 1 or stride[1] != 1
        
        self.add_to_prev_out = add_to_prev_out
        self.prev_out = None

        self.padding_original = (padding[0], padding[1])
        if not self.use_native_conv:
            self.padding = (0,0)

    def forward(self, input):
        mask = None
        if type(input) == tuple:
            input, mask = input

        w, h = input.shape[2], input.shape[3]
        if self.first_iter:
            def divup(a, b):
                return (a+b-1) // b

            self.blockCount = divup(w, self.blockStride[0]), divup(h, self.blockStride[1])

        
        if self.use_native_conv:
            return super(SBNetConv, self).forward(input)

        if mask is None:
            mask = torch.max(input, dim=1, keepdim=True)[0]

        active_block_indices, bin_counts = reduce_mask(mask, self.blockCount, tol=0.5, **self.inBlockParams)
        blockStack = sparse_gather(
            input, bin_counts, active_block_indices, transpose=self.transpose, **self.inBlockParams)

        print(blockStack.is_contiguous(memory_format=torch.channels_last))

        convBlocks = super(SBNetConv, self).forward(blockStack)

        hout = (h - self.kernel_size[0]+1) // self.stride[0] + 2 * self.padding_original[0] 
        wout = (h - self.kernel_size[0]+1) // self.stride[0] + 2 * self.padding_original[0] 
        out_y = torch.empty((input.shape[0], self.out_channels, hout, wout), 
            dtype=input.dtype, device=input.device
        ).contiguous(memory_format=torch.channels_last)

        y = sparse_scatter(
            convBlocks, bin_counts, active_block_indices,
            out_y, transpose=self.transpose, add=False, atomic=False, **self.outBlockParams)

        if self.add_to_prev_out:
            if self.prev_out is None:
                self.prev_out = torch.zeros_like(y)
            self.prev_out += y
            y = self.prev_out

        if self.mask_out:
            return y, mask

        return y



class SBNetConvOperator(nn.Module):
    def __init__(
            self,
            weight,
            bias,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            padding_mode: str = 'zeros', 
            threshold=0.0,
            tol = 0.0,
            blockSize=(16,16),
            blockStride=None,
            blockOffset=None,
            mask_out = False,
            add_to_prev_out = False
    ):
        super(SBNetConvOperator, self).__init__()
        self.weight = weight
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.dilation = (dilation, dilation) if type(dilation) == int else dilation
        self.groups = groups
        self.padding_mode = padding_mode


        self.threshold = threshold
        self.tol = tol
        self.blockSize = blockSize
        if blockStride is None:
            blockStride = (blockSize[0] - 2*self.padding[0], blockSize[1] - 2*self.padding[1])
        self.blockStride = blockStride
        if blockOffset is None:
            blockOffset = (-self.padding[0], -self.padding[1])
        self.blockOffset = blockOffset
        self.first_iter = True
        self.blockCount = 0
        self.inBlockParams = {"bsize": self.blockSize, "boffset": self.blockOffset, "bstride": self.blockStride}
        self.outBlockParams = {"bsize": self.blockStride, "boffset": (0,0), "bstride": self.blockStride}
        self.mask_out = mask_out
        self.transpose = False
        self.use_native_conv = self.stride[0] != 1 or self.stride[1] != 1
        
        self.add_to_prev_out = add_to_prev_out
        self.prev_out = None

        self.padding_original = (self.padding[0], self.padding[1])
        if not self.use_native_conv:
            self.padding = (0,0)

    def forward(self, input):
        mask = None
        if type(input) == tuple:
            input, mask = input

        w, h = input.shape[2], input.shape[3]
        if self.first_iter:
            def divup(a, b):
                return (a+b-1) // b

            self.blockCount = divup(w, self.blockStride[0]), divup(h, self.blockStride[1])

        bias = None if self.add_to_prev_out and self.prev_out is not None else self.bias
        
        if self.use_native_conv:
            out = torch.conv2d(input, self.weight, bias, self.stride, self.padding, self.dilation, self.groups)
            if self.add_to_prev_out:
                if self.prev_out is None:
                    self.prev_out = out.detach().clone()
                else:
                    self.prev_out += out
                out = self.prev_out
            return out

        if mask is None:
            mask = torch.max(input, dim=1, keepdim=True)[0]

        active_block_indices, bin_counts = reduce_mask(mask, self.blockCount, tol=0.5, **self.inBlockParams)
        blockStack = sparse_gather(
            input, bin_counts, active_block_indices, transpose=self.transpose, **self.inBlockParams)

        # print(blockStack.is_contiguous(memory_format=torch.channels_last))

        convBlocks = torch.conv2d(blockStack, self.weight, bias, self.stride, self.padding, self.dilation, self.groups)

        hout = (h - self.kernel_size[0]+1) // self.stride[0] + 2 * self.padding_original[0] 
        wout = (h - self.kernel_size[0]+1) // self.stride[0] + 2 * self.padding_original[0] 
        
        out_y = None
        if self.add_to_prev_out:
            out_y = self.prev_out
        
        if out_y is None:
            out_y = torch.zeros((input.shape[0], self.out_channels, hout, wout), 
                dtype=input.dtype, device=input.device
            ).contiguous(memory_format=torch.channels_last)

        y = sparse_scatter(
            convBlocks, bin_counts, active_block_indices,
            out_y, transpose=self.transpose, add=self.add_to_prev_out, atomic=True, **self.outBlockParams)

        if self.add_to_prev_out and self.prev_out is None:
            self.prev_out = out_y.detach().clone()

        # if self.add_to_prev_out:
        #     if self.prev_out is None:
        #         self.prev_out = torch.zeros_like(y)
        #     self.prev_out += y
        #     y = self.prev_out

        if self.mask_out:
            return y, mask

        return y


