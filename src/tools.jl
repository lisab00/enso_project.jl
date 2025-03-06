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
    forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="both") where {T,N}

Assumes that the last dimension of the input arrays is the time dimension and `N_t` long. 
Returns an `N_t` long array, judging how accurate the prediction is. 

Supported modes: 
* `"mean"`: mean between the arrays
* `"maximum"`: maximum norm 
* `"norm"`: normalized, similar to the metric used in Pathak et al 
"""
function forecast_δ(prediction::AbstractArray{T,N}, truth::AbstractArray{T,N}, mode::String="norm") where {T,N}

    if !(mode in ["mean","largest","both","norm"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return Statistics.mean(δ, dims=1:(N-1))
    elseif mode == "maximum"
        return maximum(δ, dims=1:(N-1))
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=(1:(N-1))))./sqrt(mean(sum(abs2, truth, dims=(1:(N-1)))))
    else
        return (Statistics.mean(δ, dims=1:(N-1)), maximum(δ, dims=1))
    end
end

"""
    forecast_lengths(model, t::AbstractArray{T,1}, input_data::AbstractArray{T,S}, N_t::Integer=300; λ_max=0, mode="norm", threshold=0.4, output_data::Union{Nothing, AbstractArray{T,S}}=nothing)

Returns the forecast lengths of predictions on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. 
    `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
function forecast_lengths(model, t::AbstractArray{T,1}, data::AbstractArray{T,S}, N_t::Integer; λ_max=0, mode="norm", threshold=0.4, output_data::Union{Nothing, AbstractArray{T,S}}=nothing) where {T,S}
    
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
        δ = forecast_δ(model((t[i:i+N_t], data[..,i:i])), output_data[..,i:i+N_t], mode)
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
forecast_lengths(model, valid, N_t::Integer=300; kwargs...)  = forecast_lengths(model, valid.t, valid.data, N_t; kwargs...) 

"""
    average_forecast_length(predict, valid::NODEDataloader,N_t; λmax=0, mode="norm")

Returns the average forecast length on a NODEDataloader set (should be valid or test set) given a `(t, u0) -> prediction` function. `N_t` is the length of each forecast, has to be larger than the expected forecast length. If a `λmax` is given, the results are scaled with it (and `dt``)
"""
average_forecast_length(args...; kwargs...)  = Statistics.mean(forecast_lengths(args...; kwargs...))