using enso_project
using Test, ReservoirComputing, OrdinaryDiffEq, SciMLSensitivity, Flux

@testset "enso_project.jl" begin
    # Write your tests here.
    include("test_esn.jl")
    include("test_nde.jl")
end
