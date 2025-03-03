module enso_project

using Parameters, ReservoirComputing, Plots, Flux, Optimisers, DifferentialEquations, NODEData

# Write your package code here.
include("esn.jl")

include("nde.jl")

end
