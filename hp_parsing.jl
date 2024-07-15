module hyperparams

export set_hyperparams

using ConfParser

function set_CNN_params()
    conf = ConfParse("MLP_CNN/CNN_config.ini")
    parse_conf!(conf)

    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = retrieve(conf, "Optimizer", "step_rate")
    ENV["decay"] = retrieve(conf, "Optimizer", "gamma")
    ENV["LR"] = retrieve(conf, "Optimizer", "learning_rate")
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = retrieve(conf, "Architecture", "activation")
    ENV["hidden_dim"] = retrieve(conf, "Architecture", "hidden_dim")
    ENV["batch_size"] = retrieve(conf, "Dataloader", "batch_size")
    ENV["num_epochs"] = retrieve(conf, "Pipeline", "num_epochs")
    ENV["optimizer"] = retrieve(conf, "Optimizer", "type")

    return nothing
end

function set_FNO_params()
    conf = ConfParse("MLP_FNO/FNO_config.ini")
    parse_conf!(conf)

    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = retrieve(conf, "Optimizer", "step_rate")
    ENV["decay"] = retrieve(conf, "Optimizer", "gamma")
    ENV["LR"] = retrieve(conf, "Optimizer", "learning_rate")
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["activation"] = retrieve(conf, "Architecture", "activation")
    ENV["width"] = retrieve(conf, "Architecture", "channel_width")
    ENV["modes1"] = retrieve(conf, "Architecture", "modes1")
    ENV["modes2"] = retrieve(conf, "Architecture", "modes2")
    ENV["num_blocks"] = retrieve(conf, "Architecture", "num_hidden_blocks")
    ENV["batch_size"] = retrieve(conf, "Dataloader", "batch_size")
    ENV["num_epochs"] = retrieve(conf, "Pipeline", "num_epochs")
    ENV["optimizer"] = retrieve(conf, "Optimizer", "type")

    return nothing
end

function set_KAN_CNN_params()
    conf = ConfParse("wavKAN_CNN/KAN_CNN_config.ini")
    parse_conf!(conf)

    ENV["p"] = retrieve(conf, "Loss", "p")
    ENV["step"] = retrieve(conf, "Optimizer", "step_rate")
    ENV["decay"] = retrieve(conf, "Optimizer", "gamma")
    ENV["LR"] = retrieve(conf, "Optimizer", "learning_rate")
    ENV["min_LR"] = retrieve(conf, "Optimizer", "min_lr")
    ENV["hidden_dim"] = retrieve(conf, "Architecture", "hidden_dim")
    ENV["batch_size"] = retrieve(conf, "Dataloader", "batch_size")
    ENV["num_epochs"] = retrieve(conf, "Pipeline", "num_epochs")
    ENV["optimizer"] = retrieve(conf, "Optimizer", "type")
    ENV["norm"] = retrieve(conf, "Architecture", "norm")

    encoder_wavelet_names = [
        retrieve(conf, "EncoderWavelets", "wav_one"),
        retrieve(conf, "EncoderWavelets", "wav_two"),
        retrieve(conf, "EncoderWavelets", "wav_three")
    ]

    encoder_activations = [
        retrieve(conf, "EncoderActivations", "act_one"),
        retrieve(conf, "EncoderActivations", "act_two"),
        retrieve(conf, "EncoderActivations", "act_three")
    ]

    decoder_wavelet_names = [
        retrieve(conf, "DecoderWavelets", "wav_one"),
        retrieve(conf, "DecoderWavelets", "wav_two"),
        retrieve(conf, "DecoderWavelets", "wav_three"),
        retrieve(conf, "DecoderWavelets", "wav_four")
    ]

    decoder_activations = [
        retrieve(conf, "DecoderActivations", "act_one"),
        retrieve(conf, "DecoderActivations", "act_two"),
        retrieve(conf, "DecoderActivations", "act_three"),
        retrieve(conf, "DecoderActivations", "act_four")
    ]

    return encoder_wavelet_names, encoder_activations, decoder_wavelet_names, decoder_activations
end

function set_hyperparams(model_name)

    if model_name == "CNN"
        out = set_CNN_params()
    elseif model_name == "FNO"
        out = set_FNO_params()
    elseif model_name = "KAN_CNN"
        out = set_KAN_CNN_params()
    else
        println("Invalid model name")
    end

    return out
end

end