"""
Hyperparameters for an Echo State Network.
"""
struct ESNHyperparams
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
function train_esn!(esn::ESN, y_target::Matrix, ridge_param::Float64)
    training_method = StandardRidge(ridge_param)
    return train(esn, y_target, training_method)
end

"""
    cross_validate_esn(train_data, val_data, param_grid)

Do a grid search on the given param_grid to find the optimal hyperparameters.
"""
function cross_validate_esn(train_data::Matrix, val_data::Matrix, param_grid::Vector{ESNHyperparams})
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
            #@printf "Validation loss = %.1e\n" best_loss
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
    
    return esn, Wₒᵤₜ
end

"""
    plot_prediction(esn, Wₒᵤₜ, test_data, λ_max)

Given an Echo State Network, plot its predictions versus the given test set.
"""
function plot_prediction(esn::ESN, Wₒᵤₜ, test_data::Matrix)
    steps_to_predict = size(test_data, 2)
    prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
    
    label = ["actual" "predicted"]
    times =  collect(0:steps_to_predict)[1:end-1]

    p1 = plot(times, [test_data[1, :], prediction[1, :]], label = label, ylabel = "x(t)")
    p2 = plot(times, [test_data[2, :], prediction[2, :]], label = label, ylabel = "y(t)")
    p3 = plot(times, [test_data[3, :], prediction[3, :]], label = label, ylabel = "z(t)", xlabel = "t * λ_max")
    plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
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