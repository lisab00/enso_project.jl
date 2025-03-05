abstract type AbstractChaoticNDEModel end 

"""
    ChaoticNDE{P,R,A,K} <: AbstractChaoticNDEModel

Model for setting up and training Chaotic Neural Differential Equations.

# Fields:

* `p` parameter vector 
* `prob` DEProblem 
* `alg` Algorithm to use for the `solve` command 
* `kwargs` any additional keyword arguments that should be handed over (e.g. `sensealg`)

# Constructor 

`ChaoticNDE(prob; alg=Tsit5(), kwargs...)`
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

# implement RandomSampler for hyperparameter sampling from SlurmHyperopt.jl package
abstract type AbstractHyperparameterSampler end

"""
    RandomSampler(;kwargs...)

Draws a hyperparameters config from the `kwargs` randomly.
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
    plot_node(model, data)

Given a Chaotic Neural Differential Equation, plot the model's output compared to the actual data.
"""
function plot_node(m::ChaoticNDE, ndedata::Any)
    plt = plot(ndedata.t, m((ndedata.t, ndedata[1][2]))', label="Neural ODE", xlabel="Time")
    plot!(plt, ndedata.t, ndedata.data', label="Actual Data",ylims=[-4,4], linestyle = :dash, linecolor = [1 2 3 4 5])
    display(plt)
end

"""
    train_node(training_data, validation_data, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Train the NDE with weights `N_weights`, activation function `act`, until integration length `τ_max` with learning rate `η`. 
For reproducibility of the results set random `seed`.
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
        println("starting training with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - activation=",activation, " - η=",η)
        N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        
        train_i = NODEData.NODEDataloader(train, i_τ)
        for i_e = 1:N_epochs_i

            Flux.train!(model, train_i, opt_state) do m, t, x
                result = m((t,x))
                loss(result, x)
            end 

            if (i_e % 5) == 0  # reduce the learning rate every 5 epochs
                #global η /= 2
                η /= 2
                Flux.adjust!(opt_state, η)
            end
        end
    end

    plot_node(model, valid)

    #return ChaoticNDETools.average_forecast_length(model, valid, 300; λ_max=λ_max), p
    return p, loss(model((valid.t,valid.data)), valid.data) # return optimal parameters and MSE of the model 
end

"""
    train_and_validate_node(training_data, validation_data, N_epochs, N_weights, N_hidden_layers, act, τ_max, η, seed)

Sample randomly `N_samples` hyperparameter configs for the NDE and train the NDE for each config over `N_epochs` epochs.
For reproducibility of the results set random seed for each hyperparameter sample from given `seeds`.
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
        p, mse = train_node(train, valid, N_epochs, N_weights, N_hidden_layers, activation, τ_max, η, seeds[i])
        push!(opt_ps, p)
        push!(valid_losses, mse)
    end

    # Validating
    opt_loss = findmin(valid_losses)
    Random.seed!(seeds[opt_loss[2]])
    opt_hpars = sampler(seeds[opt_loss[2]],opt_loss[2])
    #opt_pars = opt_ps[opt_loss[2]]

    return opt_loss, opt_hpars#, opt_pars
end

