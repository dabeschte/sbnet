import matplotlib.pyplot as plt
import numpy as np
import sbnet_ops
import torch
from sbnet_ops import reduce_mask, sparse_gather, sparse_scatter


def divup(a, b):
    return (a+b-1) // b


def step_by_step():
    # Specify input tensor dimensions and block-sparsity parameters
    batch = 4
    # hw = 64
    hw = 256
    channels = 64
    blockSize = [16, 16]
    blockStride = [14, 14]
    blockOffset = [0, 0]
    blockCount = [divup(hw, blockStride[0]), divup(hw, blockStride[1])]

    # build kwargs to simplify op calls
    inBlockParams = {"bsize": blockSize, "boffset": blockOffset, "bstride": blockStride}
    outBlockParams = {"bsize": [blockSize[0]-2, blockSize[1]-2], "boffset": blockOffset, "bstride": blockStride}

    device = "cuda:0"

    # create a random mask representing attention/a priori sparsity
    # threshold the mask to a specified percentile sparsity
    # mask = np.random.randn(batch, blockCount[0], blockCount[1], 1).astype(np.float32)
    mask = np.random.randn(batch, blockCount[0], blockCount[1], 1).astype(np.float32) * 0

    mask[:, :blockCount[0] // 2, :blockCount[1] // 2] = 100.0
    mask[:, blockCount[0] // 2:, blockCount[1] // 2:] = 100.0

    # mask = np.ones((batch, blockCount[0], blockCount[1], channels)).astype(np.float32)
    # threshold = np.percentile(mask, 99)
    # threshold = np.percentile(mask, 60)
    threshold = 99
    # threshold = 0.0
    sparseMask = np.greater(mask, threshold).astype(np.float32)

    # upsample the mask to full resolution
    upsampledMask = sparseMask.repeat(blockStride[0], axis=1).repeat(blockStride[1], axis=2)
    upsampledMask = torch.from_numpy(upsampledMask).to(device=device)

    # mask = np.random.randn(batch, hw, hw, 1).astype(np.float32)
    # mask[:, :hw//2, :hw//2] = mask.max()
    # upsampledMask = torch.from_numpy(mask).to(device)

    # create a random input tensor
    # x = tf.constant( np.random.randn(batch, hw, hw, channels).astype(np.float32) )
    x = torch.from_numpy(np.random.randn(batch, hw, hw, channels).astype(np.float32)).to(device)
    x_torch = torch.transpose(torch.transpose(x, 2, 3), 1, 2).clone()
    # x_torch[torch.repeat_interleave(upsampledMask[:,None,:hw,:hw,0] < 0.5, channels, dim=1)] = 0.0

    # # create a random weight tensor
    # w = tf.constant( np.random.randn(3, 3, channels, channels).astype(np.float32) )
    w = torch.from_numpy(np.random.randn(channels, channels, 3, 3).astype(np.float32)).to(device)


    mask = torch.from_numpy(mask).to(device)

    # # reduce the mask to indices by using a fused pooling+indexing operation
    # indices = sbnet_module.reduce_mask(mask, blockCount, tol=0.5, **inBlockParams)
    # active_block_indices, bin_counts = reduce_mask(mask, blockCount, tol=0.5, **inBlockParams)
    # active_block_indices, bin_counts = reduce_mask(mask, blockCount, tol=-10.0, **inBlockParams)
    # active_block_indices, bin_counts = reduce_mask(upsampledMask, blockCount, tol=0, **inBlockParams)
    active_block_indices, bin_counts = reduce_mask(upsampledMask, blockCount, tol=0.5, **inBlockParams)

    print("---- 0")

    # stack active overlapping tiles to batch dimension
    blockStack = sparse_gather(
        x, bin_counts, active_block_indices, transpose=True, **inBlockParams)

    print("---- 1")

    # blockStack_torch = torch.transpose(torch.transpose(blockStack, 1, 2), 2, 3)
    # perform dense convolution on a sparse stack of tiles
    pad = 0
    convBlocks = torch.conv2d(blockStack, w, padding=pad)
    y_target = torch.conv2d(x_torch, w, padding=pad)

    # convBlocks = torch.transpose(torch.transpose(convBlocks_torch, 2, 3), 1, 2)

    print("---- 2")
    # write/scatter the tiles back on top of original tensor
    # note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
    validX = torch.zeros_like(x[:, 1:hw-1, 1:hw-1, :])
    # validX = x.clone()
    y = sparse_scatter(
        convBlocks, bin_counts, active_block_indices,
        validX, transpose=True, add=False, atomic=False, **outBlockParams)

    y = torch.transpose(torch.transpose(y, 2, 3), 1, 2)

    diff = y_target - y

    def scale_img(x, xmin=None, xmax=None):
        if xmin is None:
            xmin = x.min()
        if xmax is None:
            xmax = x.max()
        x += xmin
        x /= (xmax - xmin)
        x *= 255
        return x.astype(np.uint8)

    plt.figure(1)
    plt.subplot(221)
    plt.imshow(torch.mean(upsampledMask,dim=-1)[0].cpu().numpy())
    plt.title("mask")
    plt.subplot(223)
    plt.imshow(torch.mean(y_target,dim=1)[0].cpu().numpy())
    plt.title("target")
    plt.subplot(224)
    plt.imshow(torch.mean(y,dim=1)[0].cpu().numpy())
    plt.title("result")
    plt.subplot(222)
    plt.imshow(scale_img(torch.mean(diff,dim=1)[0].cpu().numpy(), 0, 1))
    plt.title("diff")
    # plt.imshow(torch.mean(x_torch,dim=1)[0].cpu().numpy())
    # plt.title("input")
    plt.show()

    print("---- 3")
    a = 10
    print(blockStack)

    print("")


def conv_test(res=(256,256), kernel_size=(3,3), cin=32, cout=48, padding=(1,1), stride=(1,1), dilation=(1,1)):
    device = "cuda:0"
    # x = torch.ones((1, cin, res[0], res[1]), dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
    x = torch.arange(1*cin*res[0]*res[1], dtype=torch.float32, device=device).reshape((1, cin, res[0], res[1])).contiguous(memory_format=torch.channels_last)
    x /= x.numel()
    # w = torch.ones((cout, cin, kernel_size[0], kernel_size[1]), dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
    w = torch.arange(cout*cin*kernel_size[0]*kernel_size[1], dtype=torch.float32, device=device).reshape((cout, cin, kernel_size[0], kernel_size[1])).contiguous(memory_format=torch.channels_last)
    w /= w.numel()

    mask = torch.ones_like(x[:,:1,:,:], device=device)
    # mask[:, :res[0]//2, res[1]//4:res[1]*3//4] = 0

    threshold = 0.5
    tol = 0.5

    conv = sbnet_ops.SBNetConv(cin, cout, kernel_size, stride, padding, dilation, bias=False,
        threshold=threshold, tol=tol
    ).to(device=device)
    conv.weight.data = w

    # x_torch = torch.transpose(torch.transpose(x, 2, 3), 1, 2)
    x_torch = x
    x_torch[torch.repeat_interleave(mask[:,:1,:,:], cin, dim=1) < threshold] = 0.0

    y = conv(x)
    y_target = torch.conv2d(x_torch, w, stride=stride, padding=padding, dilation=dilation)

    # y_target = torch.transpose(torch.transpose(y_target, 1, 2), 2, 3)
    diff = y-y_target
    print(diff.abs().sum())

    import matplotlib.pyplot as plt
    # plt.imshow(torch.mean(mask[0], dim=-1).cpu().numpy())
    plt.imshow((torch.sum(diff[0], dim=0) / torch.sum(y_target[0], dim=0)).cpu().numpy())
    plt.show()

    print("")


conv_test(stride=(1,1))