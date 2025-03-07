module layers

include("./wavelets/mexican_hat.jl")
include("./wavelets/morlet.jl")
include("./wavelets/derivative_of_gaussian.jl")
include("./wavelets/shannon.jl")
include("./wavelets/meyer.jl")
include("manual_conv.jl")
include("manual_convTranspose.jl")

export KANdense, KAN_Conv, KAN_conv2D

using Flux, CUDA, KernelAbstractions
using Flux: Dense, BatchNorm
using NNlib
using .MexicanHat: MexicanHatWavelet
using .Morlet: MorletWavelet
using .DoG: DoGWavelet
using .Shannon: ShannonWavelet
using .Meyer: MeyerWavelet
using .manual_conv: KAN_conv2D
using .manual_convTranspose: KAN_convTranspose2D

bool_2D = parse(Bool, get(ENV, "2D", "false"))

act_mapping = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu,
    "selu" => NNlib.selu,
)

wavelet_mapping = Dict(
    "MexicanHat" => MexicanHatWavelet,
    "Morlet" => MorletWavelet,
    "DerivativeOfGaussian" => DoGWavelet,
    "Shannon" => ShannonWavelet,
    "Meyer" => MeyerWavelet
)

struct KANdense_layer
    transform
    output_layer
    norm
    scale::AbstractArray
    translation::AbstractArray
    reshape_fcn
    norm_permute
end

function KANdense(input_size::Int64, output_size::Int64, wavelet_name, base_activation, norm=false)
    wavelet_weights = Flux.kaiming_uniform(input_size, output_size)
    wavelet = wavelet_mapping[wavelet_name](wavelet_weights)
    activation = act_mapping[base_activation]
    # output_layer = Flux.Dense(input_size, output_size, activation)
    output_layer = nothing

    if norm
        norm_layer = bool_2D ? Flux.BatchNorm(output_size) : Flux.LayerNorm(output_size)
    else
        norm_layer = identity
    end

    translation = Flux.zeros32(input_size, output_size)
    scale = Flux.ones32(input_size, output_size)

    # RNO takes 1D input, else transformer uses 2D input
    reshape_fcn = bool_2D ? x -> repeat(reshape(x, size(x, 1), 1, size(x, 2), size(x, 3)), 1, size(translation, 2), 1, 1) : x -> repeat(reshape(x, size(x, 1), 1, size(x, 2)), 1, size(translation, 2), 1)
    norm_permute = bool_2D ? x -> reshape(x, size(x, 2), size(x, 1), size(x, 3)) : x -> x

    return KANdense_layer(wavelet, output_layer, norm_layer, scale, translation, reshape_fcn, norm_permute)
end

function (l::KANdense_layer)(x) 
    x_expanded = l.reshape_fcn(x)

    translation_expanded = repeat(l.translation, 1, 1, size(x_expanded)[3:end]...)
    scale_expanded = repeat(l.scale, 1, 1, size(x_expanded)[3:end]...)
    x_expanded = (x_expanded - translation_expanded) ./ scale_expanded 

    y = l.transform(x_expanded)
    #z = l.output_layer(x)
    out = y #+ z
    out = l.norm(l.norm_permute(out))
    return l.norm_permute(out)
end

Flux.@functor KANdense_layer

struct KAN_Convolution
    dense_kernel
    kernel_size
    stride
    dilation
    padding
    norm
end

function KAN_Conv(in_channels::Int64, out_channels::Int64, kernel_size, wavelet, base_activation, stride=1, dilation=0, padding=0, norm=false)
    dense_kernel = KANdense(prod(kernel_size)*in_channels, out_channels, wavelet, base_activation, norm)
    return KAN_Convolution(dense_kernel, kernel_size, stride, dilation, padding, norm)
end

function (c::KAN_Convolution)(x)
    x = KAN_conv2D(x, c.dense_kernel, c.kernel_size, c.stride, c.dilation, c.padding)
    return x
end

Flux.@functor KAN_Convolution

struct KAN_ConvolutionTranspose
    dense_kernel
    kernel_size
    stride
    dilation
    padding
    norm
end

function KAN_ConvTranspose(in_channels::Int64, out_channels::Int64, kernel_size, wavelet, base_activation, stride=1, dilation=0, padding=0, norm=false)
    dense_kernel = KANdense(prod(kernel_size)*in_channels, out_channels, wavelet, base_activation, norm)
    return KAN_ConvolutionTranspose(dense_kernel, kernel_size, stride, dilation, padding, norm)
end

function (c::KAN_ConvolutionTranspose)(x)
    x = KAN_convTranspose2D(x, c.dense_kernel, c.kernel_size, c.stride, c.dilation, c.padding)
    return x
end

Flux.@functor KAN_ConvolutionTranspose

end