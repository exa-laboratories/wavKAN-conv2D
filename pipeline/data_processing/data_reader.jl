module MATFileLoader

export load_darcy_data

using MAT

function load_darcy_data()
    train_matfile = matread("2D_DarcyFlow_Data/Darcy_2D_data_train.mat")
    test_matfile = matread("2D_DarcyFlow_Data/Darcy_2D_data_test.mat")

    a_train = Float32.(train_matfile["a_field"])
    u_train = Float32.(train_matfile["u_field"])
    a_test = Float32.(test_matfile["a_field"])
    u_test = Float32.(test_matfile["u_field"])

    return a_train, u_train, a_test, u_test
end 
end