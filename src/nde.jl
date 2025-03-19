abstract type AbstractChaoticNDEModel end 

# implement a model ChaoticNDE to assist in setting up NDEs based on the ChaoticNDETools.jl package
"""
    ChaoticNDE{P,R,A,K} <: AbstractChaoticNDEModel

Model for setting up and training a Chaotic Neural Differential Equation.

# Fields:

* 'p' parameter vector 
* 'prob' NDEProblem 
* 'alg' Algorithm to use for the 'solve' command 
* 'kwargs' any additional keyword arguments that should be handed over (e.g. `sensealg`)

# Constructor 

'ChaoticNDE(prob; alg=Tsit5(), kwargs...)'
"""
struct ChaoticNDE{P,R,A,K} <: AbstractChaoticNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
end 

function ChaoticNDE(prob; alg=OrdinaryDiffEq.Tsit5(), kwargs...)
    p = prob.p 
    ChaoticNDE{typeof(p), typeof(prob), typeof(alg), typeof(kwargs)}(p, prob, alg, kwargs)
end 

Flux.@functor ChaoticNDE
Optimisers.trainable(m::ChaoticNDE) = (p=m.p,)

function (m::ChaoticNDE)(X, p=m.p)
    (t, x) = X  
    Array(solve(remake(m.prob; tspan=(t[1],t[end]), u0=x[:,1],p=p), m.alg; saveat=t, m.kwargs...)) 
end

# implement an object RandomSampler for hyperparameter sampling from the SlurmHyperopt.jl package
abstract type AbstractHyperparameterSampler end

"""
    RandomSampler(;kwargs...)

Sample randomly a hyperparameters configuration from the 'kwargs'.
"""
struct RandomSampler <: AbstractHyperparameterSampler
    par_dic
    par_names 
end

RandomSampler(;kwargs...) = RandomSampler(kwargs, keys(kwargs))

function (samp::RandomSampler)(results, i)
    Dict([key => rand(samp.par_dic[key]) for key in samp.par_names]...)
end 

"""
    setup_nn(N_weights, N_hidden_layers, act, seed, dimension)

Setup a NDE with input and output layers of 'dim' nodes, with weights 'N_weights', activation function 'act', number of hidden layers 'N_hidden_layers'.
For reproducibility when choosing initial parameters for the network set random 'seed'.
"""
function setup_nn(N_weights::Int64, N_hidden_layers::Int64, activation::String, seed::Int64, dim::Int64)
    if activation == "relu"
        act_func = relu 
    else
        act_func = swish
    end
    Random.seed!(seed)
    hidden_layers = [Flux.Dense(N_weights, N_weights, act_func) for i=1:N_hidden_layers]
    nn = Chain(Flux.Dense(dim, N_weights, act_func), hidden_layers...,  Flux.Dense(N_weights, dim)) 
    p, re_nn = Flux.destructure(nn)
    return p, re_nn
end

"""
    setup_node(pars, re_nn, u0, dt)

Setup NODE problem with parameters 'pars' of the neural network 're_nn' and intial value 'u0', time step 'dt'.
"""
function setup_node(p::Vector{Float32}, re_nn::Any, u0::Vector{Float32}, dt::Float32)
    neural_ode(u, p, t) = re_nn(p)(u)
    basic_tgrad(u,p,t) = zero(u)
    odefunc = SciMLBase.ODEFunction{false}(neural_ode,tgrad=basic_tgrad)
    node_prob = SciMLBase.ODEProblem(odefunc, u0, (Float32(0.),Float32(dt)), p)
    return node_prob
end

"""
    train_node(training_data::NODEData, validation_data::NODEData, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Train the NDE with weights 'N_weights', activation function 'act', until integration length 'τ_max' with learning rate 'η'. 
For reproducibility of the results set random 'seed'.
"""
function train_node(train::Any, valid::Any, 
    N_epochs::Int64, N_weights::Int64, N_hidden_layers::Int64, activation::String, τ_max::Int64, η::Float32, seed::Int64)

    # setup neural network based on network structure defined by the given hyperpars
    p, re_nn = setup_nn(N_weights,N_hidden_layers,activation,seed,length(train.data[:,1]))

    # define NODE problem
    u0 = Vector(train.data[:,1])
    dt = train.t[2] - train.t[1]
    node_prob = setup_node(p,re_nn,u0,dt)
    
    model = ChaoticNDE(node_prob)

    loss = Flux.Losses.mse

    opt = Flux.AdamW(η)
    opt_state = Optimisers.setup(opt, model)

    N_epochs = ceil(N_epochs)

    for i_τ = 2:τ_max
        println("starting training with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - N_hidden_layers=",N_hidden_layers,
         " - activation=",activation, " - τ_max=",τ_max, " - η=",η)
        N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        
        train_i = NODEData.NODEDataloader(train, i_τ)
        for i_e = 1:N_epochs_i

            Flux.train!(model, train_i, opt_state) do m, t, x
                result = m((t,x))
                loss(result, x)
            end 

            if (i_e % 5) == 0  # reduce the learning rate every 5 epochs
                η /= 2
                Flux.adjust!(opt_state, η)
            end
        end
    end

    println("Validation MSE = ",loss(model((valid.t,valid.data)), valid.data)) # record MSE of the hyperpar config
   
    return model.p, loss(model((valid.t,valid.data)), valid.data)  # return optimal parameters, MSE on validation set
end

"""
    train_and_validate_node(training_data::NODEData, validation_data::NODEData, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Sample randomly 'N_samples' hyperparameter configs for the NDE and train the NDE for each config over 'N_epochs' epochs.
For reproducibility of the results set random seed for each hyperparameter sample from given 'seeds'.
"""
function train_and_validate_node(train::Any, valid::Any, 
    N_samples::Int64, N_epochs::Int64, seeds::Vector{Int64}, sampler::RandomSampler)

    opt_ps = [] # collect optimal parameters for each hyperparameter config
    valid_losses = [] # collect MSE error on validation set for each hyperparameter config

    for i in 1:N_samples
        # Sample hyperparameters
        Random.seed!(seeds[i])
        pars = sampler(seeds[i],i)

        # set sampled hyperparameters for iteration i
        N_weights = pars[:N_weights]
        N_hidden_layers = pars[:N_hidden_layers]
        τ_max = pars[:τ_max]
        act = pars[:activation]
        η = pars[:eta]

        # Training 
        mod_p, mse = train_node(train, valid, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seeds[i])
        push!(opt_ps, mod_p)
        push!(valid_losses, mse)
    end

    # Validating
    opt_loss = findmin(valid_losses)

    Random.seed!(seeds[opt_loss[2]])
    opt_hpars = sampler(seeds[opt_loss[2]],opt_loss[2])
    opt_pars = opt_ps[opt_loss[2]]

    return opt_loss, opt_hpars, opt_pars
end

"""
    predict_node(model, test::NODEData, data_name)

Given an optimal NDE 'model', return and plot its prediction on the given test set 'test'.
'data_name' is used to label the plot correctly.
"""
function predict_node(m::ChaoticNDE, test::Any, data_name::String)
    prediction = m((test.t,test.data))
    
    label = ["actual" "predicted"]

    plt = plot(test.t, [test.data[1,:], prediction[1,:]], label=label, ylabel="SST", xlabel="Months", title="Prediction of NDE on $data_name")
    display(plt)
    return prediction[1,:]
end

"""
    retrain_node(training_data::NODEData, validation_data::NODEData, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Rerain the NDE with optimal hyperparameters 'N_weights', 'N_hidden_layers', 'act', until integration length 'τ_max' with learning rate 'η'
on both the training and validation data together.
For reproducibility of the results set random 'seed'.
"""
function retrain_node(train::Any, valid::Any, 
    N_epochs::Int64, N_weights::Int64, N_hidden_layers::Int64, activation::String, τ_max::Int64, η::Float32, seed::Int64)

    # Merge train and validation data
    data = hcat(train.data, valid.data)
    time = Float32.(0:size(data,2)-1)
    trainvalid = NODEDataloader(Array(data), time, 2)

    # Training 
    mod_p, mse = train_node(trainvalid, trainvalid, N_epochs, N_weights, N_hidden_layers, activation, τ_max, η, seed)
    
    return mod_p, mse

end


