module enso_project

using Parameters, ReservoirComputing, Plots, Flux, Optimisers, Random, DataFrames #, NODEData

# Write your package code here.
include("esn.jl")
include("nde.jl")
include("tools.jl")

end
