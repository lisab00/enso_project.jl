"""
    function tde_with_negative_shift(data_1D)

Create TDE with negative time shift.
First use optimal_separated_de method, then manually embed data by -\tau.
"""
function tde_with_negative_shift(data_1D)
    D, τ, E = optimal_separated_de(data_1D)
    D = Matrix(D)
    emb_dim = size(D, 2)
    data_emb = embed(data_1D, emb_dim, -τ)
    data_emb = Matrix(data_emb)
    shift = (emb_dim-1)*τ
    return data_emb[shift+1:end-τ,:], τ
end

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
    train_val_test_split(data; num_val_months, num_test_months)

Split the given data into training, validation, and test sets.
val_percent: number of time steps wanted in the validation set.
test_percent: number of time steps wanted in the test set.
"""
function train_val_test_split(data::AbstractMatrix; val_percent::Float64=0.15, test_percent::Float64=0.15)
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


# error metricsa for ENSO
"""
    function hss(predictions::AbstractMatrix, test_data::AbstractMatrix)

Measures accuracy of predictions (wrt randomly generated forecast). ENSO predicted if abs(ONI)>0.5. Compute ratio out of TP,TN,FP,FN.
Is considered good if >0.5.
Formula from paper "Long-term ENSO prediction with echo-state networks" by Hassanibesheli F. et al.

# Arguments:
    - `predicitons::AbstractMatrix`:  predictions, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data::AbstractMatrix`: test data for each sample, NxL matrix.

# Returns:
    - `Vector`: HSS for each lead time, vector of length L
"""
function hss(predictions::AbstractMatrix, test_data::AbstractMatrix)

    # predict event if indec > 0.5
    events_pred = abs.(Int.(round.(predictions)))
    events_true = abs.(Int.(round.(test_data)))

    # compute TN
    TN_mat = events_pred + events_true
    TN = sum(TN_mat .== 0, dims=1)

    # compute FP,FN,TP
    compare = events_pred - events_true # matrix: 0 for TP, TN; 1 for FP, -1 for FN
    FP = sum(compare .== 1, dims=1)
    FN = sum(compare .==-1, dims=1)
    TP = sum(compare .==0, dims=1) .- TN

    N = TN + TP + FN + FP # should be sample size

    # calculate formula
    CRF = ((TP .+ FN) .* (TP .+ FP) .+ (TN .+ FN) .* (TN .+ FP)) ./ N
    HSS = (TP .+ TN .- CRF) ./ (N .- CRF)
    return HSS[1,:]
end

"""
    function pcc(predictions::AbstractMatrix, test_data::AbstractMatrix)

Compute the Pearson-Correlation-Coefficient between sample and test data for each lead time considered. I.e., PCC is computed between respective data columns.
Is considered good, if > 0.5.

# Arguments:
    - `predicitons::AbstractMatrix`:  predictions, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data::AbstractMatrix`: test data for each sample, NxL matrix.

# Returns:
    - `Vector`: PCC for each lead time, vector of length L
"""
function pcc(predictions::AbstractMatrix, test_data::AbstractMatrix)
    L = size(predictions, 2)
    return [cor(predictions[:,i], test_data[:,i]) for i in 1:L]
end

"""
    function rmse(predictions::AbstractMatrix, test_data::AbstractMatrix)
       
compute the rmse between predicitons and test data for each lead time. Is considered good if smaller 1.4.
Version for 1D datasets (ENSO)

# Arguments:
    - `predicitons::AbstractMatrix`: predictions, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data::AbstractMatrix`: test data for each sample, NxL matrix.

# Returns:
    - `Vector`: RMSE for each lead time, vector of length L
"""
function rmse(predictions::AbstractMatrix, test_data::AbstractMatrix)
    N, L = size(predictions, 1), size(predictions,2)
    sse_vals = zeros(N,L)
    sse_vals = (predictions .- test_data).^2
    rmse = sqrt.(sum(sse_vals, dims=1) ./N)
    return rmse[1,:]
end


# error metrics for MJO
"""
    function rmse(predictions::AbstractMatrix, test_data::AbstractMatrix, predictions2::AbstractMatrix, test_data2::AbstractMatrix)
       
compute the rmse between predicitons and test data for each lead time. Is considered good if smaller 1.4.
Version for 2D data sets (MJO)

# Arguments:
    - `predicitons::AbstractMatrix`: predictions, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data::AbstractMatrix`: test data for each sample, NxL matrix.
    - `predicitons2::AbstractMatrix`: predictions of second component, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data2::AbstractMatrix`: test data of second component for each sample, NxL matrix.

# Returns:
    - `Vector`: RMSE for each lead time, vector of length L
"""
function rmse(predictions::AbstractMatrix, test_data::AbstractMatrix, predictions2::AbstractMatrix, test_data2::AbstractMatrix)
    N, L = size(predictions, 1), size(predictions,2)
    sse_vals = zeros(N,L)
    sse_vals = (predictions .- test_data).^2 .+ (predictions2 .- test_data2).^2
    rmse = sqrt.(sum(sse_vals, dims=1) ./N)
    return rmse[1,:]
end

"""
    function bivariate_corr(predictions::AbstractMatrix, test_data::AbstractMatrix, predictions2::AbstractMatrix, test_data2::AbstractMatrix)

Compute the bivariate correlation coefficient between sample and test data for each lead time considered.
Inputs are N×L matrices: rows = samples, cols = lead times.
Formula from paper: "Improving the Predictability of the Madden-Julian Oscillation at Subseasonal Scales With Gaussian Process Models" by Chen H. et al.

# Arguments:
    - `predicitons::AbstractMatrix`:  predictions, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data::AbstractMatrix`: test data for each sample, NxL matrix.
    - `predicitons2::AbstractMatrix`:  predictions of second component, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `test_data2::AbstractMatrix`: test data of second component for each sample, NxL matrix.

# Returns:
    - `Vector`: PCC for each lead time, vector of length L
"""
function bivariate_corr(predictions::AbstractMatrix, test_data::AbstractMatrix, predictions2::AbstractMatrix, test_data2::AbstractMatrix)

    L = size(test_data, 2)
    corrs = zeros(L) # obtain correlation for each lead time

    for l in 1:L
        # vectors of data at lead time l
        z1 = test_data[:,l]
        z2 = test_data2[:,l]
        zh1 = predictions[:,l]
        zh2 = predictions2[:,l]

        num   = sum(z1 .* zh1 .+ z2 .* zh2) # element-wise product summed over sample N
        denom = sqrt(sum(z1.^2 .+ z2.^2)) * sqrt(sum(zh1.^2 .+ zh2.^2))

        corrs[l] = num / denom
    end

    return corrs
end

"""
    function hss(pred1::AbstractMatrix, pred2::AbstractMatrix, true1::AbstractMatrix, true2::AbstractMatrix)

Computes hss scores for MJO phase i.
Formula from paper "Improving the Predictability of the Madden-Julian Oscillation at Subseasonal Scales With Gaussian Process Models" by Chen H. et al.

# Arguments:
    - `pred1::AbstractMatrix`:  predictions of pc1, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `pred2::AbstractMatrix`:  predictions of pc2, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `true1::AbstractMatrix`: test data for each sample of pc1, NxL matrix.
    - `true2::AbstractMatrix`: test data for each sample of pc2, NxL matrix.

# Returns:
    - `Vector`: HSS of phase i for each lead time, vector of length L.
"""
function hss_i(i::Int64, pred1::AbstractMatrix, pred2::AbstractMatrix, true1::AbstractMatrix, true2::AbstractMatrix)
    N, L = size(pred1,1), size(pred1, 2)
    hss_scores = zeros(L) # store hss of phase i for each lead time

    for l in 1:L
        # needed to compute hss score of sample
        a, b, c, d = 0, 0, 0, 0
        
        # sample and truth vectors of length N
        z1, z2 = true1[:,l], true2[:,l]
        zh1, zh2 = pred1[:,l], pred2[:,l]

        # quantities needed to determine phases
        angles_true, r_true = atan.(z1, z2), sqrt.(z1.^2 + z2.^2)
        anglesh, rh = atan.(zh1, zh2), sqrt.(zh1.^2 + zh2.^2)

        for n in 1:N # for each sample item
            angle_true = angles_true[n]
            angleh = anglesh[n]

            # check whether phase i is the correct phase
            if i == 0
                ph_i = r_true[n] < 1
                ph_i_pred = rh[n] < 1 
            else 
                left = -π + π/4*(i-1)
                right = -3/4*π + π/4*(i-1)
                ph_i = (angle_true > left) && (angle_true <= right)
                ph_i_pred = (angleh > left) && (angleh <= right)
            end
            a += (ph_i && ph_i_pred) ? 1 : 0 # TP 
            d += (!ph_i && !ph_i_pred) ? 1 : 0 # TN 
            c += (ph_i && !ph_i_pred) ? 1 : 0 # FN 
            b += (!ph_i && ph_i_pred) ? 1 : 0 # FP 
        end
        # now compute HSS for sample by formula
        num = 2*(a*d-b*c)
        denom = (a+b)*(b+d)+(a+c)*(c+d)
        hss_scores[l] = num / denom
        #hss_scores[l] = (denom == 0) ? 0 : num / denom
    end
    return hss_scores
end


"""
    function hss(pred1::AbstractMatrix, pred2::AbstractMatrix, true1::AbstractMatrix, true2::AbstractMatrix)

Measures accuracy of predictions (wrt randomly generated forecast). Is considered good if >0.5.
Gathers hss scores for different MJO phases in a matrix, where phase 0 is the inactive phase.
Formula from paper "Improving the Predictability of the Madden-Julian Oscillation at Subseasonal Scales With Gaussian Process Models" by Chen H. et al.

# Arguments:
    - `pred1::AbstractMatrix`:  predictions of pc1, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `pred2::AbstractMatrix`:  predictions of pc2, NxL matrix. N is sample size per lead time, L is all lead times considered
    - `true1::AbstractMatrix`: test data for each sample of pc1, NxL matrix.
    - `true2::AbstractMatrix`: test data for each sample of pc2, NxL matrix.

# Returns:
    - `Matrics`: HSS for each lead time, matrix of size 9xL. Rows of matrix represent different MJO phases.
"""
function hss(pred1::AbstractMatrix, pred2::AbstractMatrix, true1::AbstractMatrix, true2::AbstractMatrix)
    N, L = size(pred1,1), size(pred1, 2)
    hss_scores = zeros(9,L)
    for i in 0:8
        hss_scores[i+1,:] = hss_i(i, pred1, pred2, true1, true2)
    end
    return hss_scores
end