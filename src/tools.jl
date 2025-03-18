"""
    train_val_test_split(data; num_val_months, num_test_months)

Split the given data into training, validation, and test sets.
val_percent: number of time steps wanted in the validation set.
test_percent: number of time steps wanted in the test set.
"""
function train_val_test_split(data::DataFrame; val_percent::Float64=0.15, test_percent::Float64=0.15)
    N = size(data, 1)
    N_val = round(Int, val_percent*N)
    N_test = round(Int, test_percent*N)
    
    ind1 = N - N_test - N_val
    ind2 = N - N_test
    
    train_data = data[1:ind1, :]
    val_data = data[ind1+1:ind2, :]
    test_data = data[ind2+1:end, :]
    
    return train_data, val_data, test_data
end

"""
    plot_data_split_predictions(predictions::Matrix, data::Matrix, splits::Array)

    create one plot containing all subplots of different data splits.
    Nrows corresponds to number of splits, Ncols is length of data to predict.
    Splits: Array of integers corresponding to percentual splits
"""
function plot_data_split_predictions(predictions::Matrix, data::Matrix, splits::Array=[20,40,50,60,70,80])
    subplots = []
    label = ["actual" "predicted"]
    times =  collect(0:size(data,2))[1:end-1]
    for i in eachindex(predictions[:,1])
        s = splits[i]
        push!(subplots, plot(times, [data[1,:], predictions[i, :]], label=label,  title="Data Split $s %"))
    end
    plot(subplots..., layout=(2,3), size=(1200,800))
end

"""
    plot_loss_evolution(loss::Vector, splits::Vector)

    Bar plot of the training data in % versus the validation loss.
    Splits contains the percentages to be considered on the x-axis
"""

function plot_loss_evolution(loss::Vector, splits::Vector)
    bar(splits, loss, xlabel= "Training Data in %", ylabel="Validation Loss", label ="", title="Validation Loss vs. Training Data Size")
end

"""
    plot_error_curve(errors::Matrix, threshold::Float64)

    given the errors between predictions and dataset, as well as an error threshold, plot the error curve
"""

function plot_error_curve(errors::Matrix, threshold::Float64)
    plot(errors[1,:], xlabel="Months", ylabel="Error Curve", label="Error", title="Error between Prediction and True Data")
    hline!([threshold], linestyle=:dash, color=:red, label="Threshold")
end


"""
    forecast_δ_1D(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. 
Returns an `N_t` long array, judging how accurate the prediction is. 
Adapted such that it is possible to only consider the prediction's and truth's first dimension for calculation.

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
* `"abs`: absolute difference between arrays
"""
function forecast_δ_1D(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, type::String, mode::String="norm") where {T,N}

    if !(mode in ["mean","largest","both","norm", "abs"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    if type == "1D"
        prediction = prediction[1,:]
        truth = truth[1,:]
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return Statistics.mean(δ, dims=1:(N-1))
    elseif mode == "maximum"
        return maximum(δ, dims=1:(N-1))
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=(1:(N-1))))./sqrt(mean(sum(abs2, truth, dims=(1:(N-1)))))
    elseif mode =="abs"
        return abs.(prediction .- truth)
    else
        return (Statistics.mean(δ, dims=1:(N-1)), maximum(δ, dims=1))
    end
end

"""
    forecast_lengths(model, t::AbstractArray{T,1}, input_data::AbstractArray{T,S}, N_t::Integer=300; λ_max=0, mode="norm", threshold=0.4, output_data::Union{Nothing, AbstractArray{T,S}}=nothing)

Returns the forecast lengths of predictions on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. 
    `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
function forecast_lengths(model, t::AbstractArray{T,1}, data::AbstractArray{T,S},type::String, N_t::Integer; λ_max=0, mode="norm", threshold=0.4, output_data::Union{Nothing, AbstractArray{T,S}}=nothing) where {T,S}
    
    N = length(t) - N_t
    @assert N >= 1 

    if isnothing(output_data)
        output_data = data 
    end

    @assert size(data)[end] == size(output_data)[end]

    forecasts = zeros(N)

    if typeof(t) <: AbstractRange 
        dt = step(t)
    else 
        dt = t[2] - t[1]
    end
    
    for i=1:N 
        δ = forecast_δ_1D(model((t[i:i+N_t], data[..,i:i])), output_data[..,i:i+N_t], type, mode) # adapt to only first dimension
        δ = δ[:] # return a 1x1...xN_t array, so we flatten it here
        first_ind = findfirst(δ .> threshold) 

        if isnothing(first_ind) # in case no element of δ is larger than the threshold
            @warn "Prediction error smaller than threshold for the full predicted range, consider increasing N_t"
            first_ind = N_t + 1
        end 

        if λ_max == 0
            forecasts[i] = first_ind 
        else 
            forecasts[i] = first_ind * dt * λ_max
        end
    end 
    
    return forecasts
end
forecast_lengths(model, valid, type, N_t::Integer=300; kwargs...)  = forecast_lengths(model, valid.t, valid.data, type, N_t; kwargs...) 

"""
    average_forecast_length(predict, valid::NODEDataloader,N_t; λmax=0, mode="norm")

Returns the average forecast length on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. 
`N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
average_forecast_length(args...; kwargs...)  = Statistics.mean(forecast_lengths(args...; kwargs...))