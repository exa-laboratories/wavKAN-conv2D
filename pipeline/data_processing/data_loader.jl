module loaders

include("./data_reader.jl")
include("../../utils.jl")
using .MATFileLoader: load_visco_data
using .UTILS: UnitGaussianNormaliser, unit_encode, MinMaxNormaliser, minmax_encode
using Flux
using CUDA

function get_darcy_loader(batch_size=32)
    a_train, u_train, a_test, u_test = load_darcy_data()

    a_normaliser = UnitGaussianNormaliser(a_train)
    u_normaliser = UnitGaussianNormaliser(u_train)

    # Normalise
    a_train = unit_encode(a_normaliser, a_train)
    u_train = unit_encode(u_normaliser, u_train)
    a_test = unit_encode(a_normaliser, a_test)
    u_test = unit_encode(u_normaliser, u_test)

    a_train = reshape(a_train, size(a_train)..., 1)
    u_train = reshape(u_train, size(u_train)..., 1)
    a_test = reshape(a_test, size(a_test)..., 1)
    u_test = reshape(u_test, size(u_test)..., 1)

    a_train = permutedims(a_train, [2, 3, 4, 1])
    u_train = permutedims(u_train, [2, 3, 4, 1])
    a_test = permutedims(a_test, [2, 3, 4, 1])
    u_test = permutedims(u_test, [2, 3, 4, 1])
    
    train_loader = Flux.DataLoader((a_train, u_train) |> gpu, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((a_test, u_test) |> gpu, batchsize=batch_size, shuffle=false)

    return train_loader, test_loader
end
end

