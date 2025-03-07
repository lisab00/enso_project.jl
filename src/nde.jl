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
    train_node(training_data::NODEDataloader, validation_data::NODEDataloader, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Train the NDE with weights 'N_weights', activation function 'act', until integration length 'τ_max' with learning rate 'η'. 
For reproducibility of the results set random 'seed'.
"""
function train_node(train::Any, valid::Any, 
    N_epochs::Int64, N_weights::Int64, N_hidden_layers::Int64, activation::Function, τ_max::Int64, η::Float32, seed::Int64)

    Random.seed!(seed)
    u0 = Vector(train.data[:,1])
    dt = train.t[2] - train.t[1]

    hidden_layers = [Flux.Dense(N_weights, N_weights, activation) for i=1:N_hidden_layers]
    nn = Chain(Flux.Dense(5, N_weights, activation), hidden_layers...,  Flux.Dense(N_weights, 5)) 
    p, re_nn = Flux.destructure(nn)

    neural_ode(u, p, t) = re_nn(p)(u)
    
    basic_tgrad(u,p,t) = zero(u)
    odefunc = SciMLBase.ODEFunction{false}(neural_ode,tgrad=basic_tgrad)
    node_prob = SciMLBase.ODEProblem(odefunc, u0, (Float32(0.),Float32(dt)), p)
    
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
    train_and_validate_node(training_data::NODEDataloader, validation_data::NODEDataloader, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

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
        act_func = pars[:activation]
        η = pars[:eta]

        if act_func == "relu"
            activation = relu 
        else
            activation = swish
        end

        # Training 
        mod_p, mse = train_node(train, valid, N_epochs, N_weights, N_hidden_layers, activation, τ_max, η, seeds[i])
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

