# test that created parameter grid is of correct size
@testset "parameter grid" begin
    grid = enso_project.create_param_grid([100, 200], [1.0], [0.1],  [0.1],  [0.01])
    @test length(grid) == 2
end

# test that cross validation returns objects of correct types
@testset "cross validate ESN" begin
    train_data = rand(5,5)
    val_data = rand(5,3)
    grid = create_param_grid([100], [1.0], [0.1],  [0.1],  [0.01])
    esn, W_out = cross_validate_esn(train_data, val_data, grid)

    @test typeof(esn) <: ReservoirComputing.ESN
    @test typeof(W_out) <: ReservoirComputing.OutputLayer

end

# test that training procedure actually changes the output weights
@testset "train ESN" begin
    train_data = rand(5,5)
    esn = ESN(train_data, size(train_data, 1), 100; reservoir=rand_sparse(; radius=1.0, sparsity=0.1),
    input_layer=scaled_rand(; scaling=0.1),)
    esn_2 = deepcopy(esn)
    train_esn!(esn_2, train_data, 0.1)

    @test esn != esn_2
end