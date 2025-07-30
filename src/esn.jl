"""
    ESNHyperparams <: AbstractESNHyperparams

Hyperparameters for an Echo State Network.    

# Fields: 

* `reservoir_size` number of nodes in the reservoir
* `spectral radius` spectral radius of the reservoir matrix
* `sparsity` regulates number of connections in the reservoir
* `input_scale` scaling of the input layer
* `ridge_param`

"""

abstract type AbstractESNHyperparams end

"""
Hyperparameters for an Echo State Network.
"""
struct ESNHyperparams <: AbstractESNHyperparams
    reservoir_size::Int64
    spectral_radius::Float64
    sparsity::Float64
    input_scale::Float64
    ridge_param::Float64
end

"""
    train_esn!(esn, y, ridge_param)

Given an Echo State Network, train it on the target sequence y_target and return the optimised output weights Wₒᵤₜ.
"""
function train_esn!(esn::ESN, y_target::AbstractMatrix, ridge_param::Float64)
    training_method = StandardRidge(ridge_param)
    return train(esn, y_target, training_method)
end

"""
    cross_validate_esn(train_data, val_data, param_grid)

Do a grid search on the given param_grid to find the optimal hyperparameters.
"""
function cross_validate_esn(train_data::AbstractMatrix, val_data::AbstractMatrix, param_grid::Vector)
    best_loss = Inf
    best_params = nothing

    # We want to predict one step ahead, so the input signal is equal to the target signal from the previous step
    # i.e. the sequence is shifted by one step
    u_train = train_data[:, 1:end-1]
    y_train = train_data[:, 2:end]
        
    for hyperparams in param_grid        
        # Unpack the hyperparams struct
        (; reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = hyperparams

        # Generate and train an ESN
        esn = ESN(
            u_train,
            size(train_data, 1),
            reservoir_size;
            reservoir=rand_sparse(; radius=spectral_radius, sparsity=sparsity),
            input_layer=scaled_rand(; scaling=input_scale),
        )
        Wₒᵤₜ = train_esn!(esn, y_train, ridge_param)

        # Evaluate the loss on the validation set
        steps_to_predict = size(val_data, 2)
        prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
        loss = sum(abs2, prediction - val_data)
        
        # Keep track of the best hyperparameter values
        if loss < best_loss
            best_loss = loss
            best_params = hyperparams
            println(best_params)
            println("Validation loss = $best_loss")
        end
    end
    
    # Retrain the model using the optimal hyperparameters on both the training and validation data
    # This is necessary because we don't want errors incurred during validation to affect the test error
    (;reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = best_params
    data = hcat(train_data, val_data)
    u = data[:, 1:end-1]
    y = data[:, 2:end]
    esn = ESN(
        u,
        size(train_data, 1),
        reservoir_size;
        reservoir=rand_sparse(; radius=spectral_radius, sparsity=sparsity),
        input_layer=scaled_rand(; scaling=input_scale),
    )
    Wₒᵤₜ = train_esn!(esn, y, ridge_param)
    
    return esn, Wₒᵤₜ, best_loss
end

"""
    esn_eval_pred(esn::ESN, W_out, data::Matrix)

    given an ESN, its output layer W_out and a data matrix, evaluate the prediction of the ESN on the given data.
"""
function esn_eval_pred(esn::ESN, W_out, data::AbstractMatrix)
    steps_to_predict = size(data,2)
    prediction = esn(Generative(steps_to_predict), W_out)
    return prediction[1,:]
end

"""
    plot_esn_prediction(esn, Wₒᵤₜ, test_data, data_name::String)

Given an Echo State Network, plot its predictions versus the given test set.
data_name is used to label the plot correctly
"""
function plot_esn_prediction(esn::ESN, W_out, data::AbstractMatrix, data_name::String)
    prediction = esn_eval_pred(esn, W_out, data)
    
    label = ["actual" "predicted"]
    times =  collect(0:size(data,2))[1:end-1]

    plot(times, [data[1,:], prediction], label=label, ylabel="ONI", xlabel="Months", title="Prediction of ENSO using an ESN on $data_name")
end

"""
    create_param_grid(param_ranges:Array)

Given an Array of the parameter ranges, create a parameter grid.
"""
function create_param_grid(reservoir_sizes::Array,
    spectral_radii::Array,
    sparsities::Array,
    input_scales::Array,
    ridge_values::Array)

    param_grid = []

    # Take the Cartesian product of the possible values
    for params in Iterators.product(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
        push!(param_grid, ESNHyperparams(params...))
    end

    return param_grid
end