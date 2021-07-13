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
            mask_out = False
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

        if self.mask_out:
            return y, mask

        return y