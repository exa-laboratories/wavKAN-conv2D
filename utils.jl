module UTILS

export loss_fcn, BIC, UnitGaussianNormaliser, unit_encode, unit_decode, MinMaxNormaliser, minmax_encode, minmax_decode, log_csv, get_grid, node_mul_1D, node_mul_2D, unfold

using Statistics
using Flux
using CUDA, KernelAbstractions, Tullio
using NNlib

p = parse(Float32, get(ENV, "p", "2.0"))

function loss_fcn(m, x, y)
    return sum(abs.(m(x) .- y).^p)
end

function BIC(model, x, loss)
    n = size(x)[end] # Number of samples
    k = sum(length, Flux.params(model)) # Number of parameters
    return 2 * loss + k * log(n)
end

eps = Float32(1e-5)

### Normalisers really help on this dataset! ###
struct UnitGaussianNormaliser{T<:AbstractFloat}
    μ::T
    σ::T
    ε::T
end

function unit_encode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return (x .- normaliser.μ) ./ (normaliser.σ .+ normaliser.ε)
end

function unit_decode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return x .* (normaliser.σ .+ normaliser.ε) .+ normaliser.μ
end

function UnitGaussianNormaliser(x::AbstractArray)
    data_mean = Statistics.mean(x)
    data_std = Statistics.std(x)
    return UnitGaussianNormaliser(data_mean, data_std, eps)
end

struct MinMaxNormaliser{T<:AbstractFloat}
    min::T
    max::T
end

function minmax_encode(normaliser::MinMaxNormaliser, x::AbstractArray)
    return (x .- normaliser.min) ./ (normaliser.max - normaliser.min)
end

function minmax_decode(normaliser::MinMaxNormaliser, x::AbstractArray)
    return x .* (normaliser.max - normaliser.min) .+ normaliser.min
end

function MinMaxNormaliser(x::AbstractArray)
    data_min = minimum(x)
    data_max = maximum(x)
    return MinMaxNormaliser(data_min, data_max)
end

# Log the loss to CSV
function log_csv(epoch, train_loss, test_loss, BIC, time, file_name)
    open(file_name, "a") do file
        write(file, "$epoch,$time,$train_loss,$test_loss,$BIC\n")
    end
end

# Creates grids for spectral convolutions (x, y, 1, batch_size) -> (3, x, y, batch_size)
nx, ny = 32, 32
X = Float32.(range(0,1,nx))
Y = Float32.(range(0,1,ny))
X = reshape(X, 1, nx, 1, 1)
Y = reshape(Y, 1, 1, ny, 1)

function get_grid(x)
    batch_size = size(x, 4)
    gridx = repeat(X, 1, 1, ny, batch_size)
    gridy = repeat(Y, 1, nx, 1, batch_size)
    grid = cat(gridx, gridy, dims=1) |> gpu
    x_reshaped = @tullio y[c, w, h, b] := x[w, h, c, b]
    return vcat(x_reshaped, grid)
    
end

struct slicer
    dh
    dw
    sh
    sw
    out_h
    out_w
end

function create_slicer(dh, dw, sh, sw, out_h, out_w)
    return slicer(dh, dw, sh, sw, out_h, out_w)
end

function (s::slicer)(input, i, j)
    h_start = (i - 1) * s.dh + 1
    w_start = (j - 1) * s.dw + 1
    h_indices = h_start:s.sh:(h_start + s.sh * (s.out_h - 1))
    w_indices = w_start:s.sw:(w_start + s.sw * (s.out_w - 1))
    slice = input[h_indices, w_indices, :, :]
    return reshape(slice, 1, size(slice)...)
end

function unfold(input, kernel_size; stride=1, padding=0, dilation=1)
    H, W, C, N = size(input)
    kh, kw = kernel_size
    sh, sw = stride isa NTuple ? stride : (stride, stride)
    ph, pw = padding isa NTuple ? padding : (padding, padding)
    dh, dw = dilation isa NTuple ? dilation : (dilation, dilation)

    out_h = div(H + 2ph - (dh * (kh - 1) + 1), sh) + 1
    out_w = div(W + 2pw - (dw * (kw - 1) + 1), sw) + 1

    slice = create_slicer(dh, dw, sh, sw, out_h, out_w)
    padded_input = NNlib.pad_circular(input, (ph, ph, pw, pw))

    output = zeros(0, kw, out_h, out_w, C, N) |> gpu

    for i in 1:kh
        inner_output = zeros(0, out_h, out_w, C, N) |> gpu
        for j in 1:kw
            inner_output = cat(inner_output, slice(padded_input, i, j), dims=1)
        end
        inner_output = reshape(inner_output, 1, size(inner_output)...)
        output = cat(output, inner_output, dims=1)
    end
    return output
end

# This is a node of the KAN. It sums wavelets in a manner presented by the wavKAN paper.
function node_mul_1D(y, w)
    output = @tullio out[i, o, b] := w[i, o] * y[i, o, b]
    return reshape(sum(output, dims=1), size(w)[2], size(y)[end])
end

function node_mul_2D(y, w)
    output = @tullio out[i, o, l, b] := w[i, o] * y[i, o, l, b]
    return reshape(sum(output, dims=1), size(w)[2], size(y)[3], size(y)[end])
end

end