module manual_conv

include("../utils.jl")

export KAN_conv2D

using LinearAlgebra
using Statistics
using Flux, CUDA, KernelAbstractions, Tullio
using .UTILS: unfold

function calc_out_dims(matrix, kernel_size, stride, dilation, padding)
    n, m, n_channels, batch_size = size(matrix)
    h_out = floor(Int, (n + 2 * padding - kernel_size[1] - (kernel_size[1] - 1) * (dilation - 1)) / stride) + 1
    w_out = floor(Int, (m + 2 * padding - kernel_size[2] - (kernel_size[2] - 1) * (dilation - 1)) / stride) + 1
    return h_out, w_out, n_channels, batch_size
end

function KAN_conv2D(matrix, kernel, kernel_size, stride=1, dilation=1, padding=0)
    n_channels, batch_size = size(matrix, 3), size(matrix, 4)
    patches = unfold(matrix, kernel_size; stride=stride, padding=padding, dilation=dilation)
    h, w = size(patches, 3), size(patches, 4)
    patches = reshape(patches, prod(kernel_size) * n_channels, h * w, batch_size)
    return reshape(kernel(patches), h, w, :, batch_size)
    end
end