using OrdinaryDiffEq, SciMLSensitivity, Flux

# test with a Lotka-Volterra system
begin
    N_T = 200
    N_SAMPLES = 2
    N_EPOCHS = 10
    τ_MAX = 2
    η = 0.001
    N_WEIGHTS = 5
    N_HIDDEN_LAYERS = 1
    ACT = "swish"
    seed = 1903
    dt = 0.1
end 

begin 
    function lotka_volterra(x,p,t)
        α, β, γ, δ = p 
        [α*x[1] - β*x[1]*x[2], -γ*x[2] + δ*x[1]*x[2]]
    end
    
    α = 1.3
    β = 0.9
    γ = 0.8
    δ = 1.8
    p = [α, β, γ, δ] 
    tspan = (0.,10.)
    
    x0 = [0.44249296, 4.6280594]
    
    prob = ODEProblem(lotka_volterra, x0, tspan, p) 
    sol = solve(prob, Tsit5(), saveat=dt)
end 

t_train = Float32.(0:dt:5.)
train = (t = t_train, data = Float32.(Array(sol(t_train))))
t_test = Float32.(5.:dt:10.)
test = (t = t_test, data = Float32.(Array(sol(t_test))))

# test if everything compiles and runs without errors 
p, re_nn = enso_project.setup_nn(N_WEIGHTS,N_HIDDEN_LAYERS,ACT,seed) # setup neural network 
node_prob = enso_project.setup_node(p,re_nn,x0,dt) # define NODE problem

model = enso_project.ChaoticNDE(node_prob)
model((train.t,train.data))

loss(m, x, y) = Flux.mse(m(x),y) 
loss(model, (train.t,train.data), train.data) 

# check if the gradient works 
g = gradient(model) do m
    loss(m, (train.t,train.data), train.data)
end
pgrad = g[1][:p]

# do a check that the gradient is nonzero, noninf and nonnothing
@test sum(isnan.(pgrad)) == 0
@test sum(isinf.(pgrad)) == 0 
@test sum(isnothing.(pgrad)) == 0 

# check if training procedure is indeed making changes to the model parameters
p_opt, loss_valid = enso_project.train_node(train, valid, N_EPOCHS, N_WEIGHTS, N_HIDDEN_LAYERS, ACT, τ_MAX, η, seed)
@test model.p != p_opt

# check that training loss is indeed decreasing during training
node_prob_opt = enso_project.setup_node(p_opt,re_nn,x0,dt)
model_opt = enso_project.ChaoticNDE(node_prob_opt)
@test loss(model, (train.t,train.data), train.data) >= loss(model_opt, (train.t,train.data), train.data)

# check that nde prediction has correct size
pred = predict_node(model_opt, test, "")
@test size(pred, 2) == size(test.data, 2)

# check that random sampler returns object of correct type and length, and each value is from the appropriate range
sampler = enso_project.RandomSampler(value1=1:5, value2=6:10, value3=11:15, value4=16:20)
sample = sampler(seed,1)
ranges = values(values(sampler.par_dic)[collect(keys(sample))])
@test typeof(sample) <: Dict
@test length(sample) == length(sampler.par_names)
@test sum(.!in.(values(sample),ranges)) == 0