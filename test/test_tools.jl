# test that data splits have correct size
@testset "data split" begin
    data = rand(2,100)
    val_percent = 0.2
    test_percent =0.1
    train_data, val_data, test_data = enso_project.train_val_test_split(data; val_percent, test_percent)
    @test size(val_data, 2) == size(data,2)*val_percent
    @test size(test_data, 2) == size(data,2)*test_percent
    @test size(train_data, 2) == size(data,2)*(1-val_percent-test_percent)
end

# test that forecast can both correctly take the first dimension or all dimensions into account when checking the accuracy
@testset "forecast" begin
    prediction = rand(2,100)
    truth = rand(2,100)
    error_1D_true = abs.(prediction[1,:] .- truth[1,:])
    error_all_true = abs.(prediction .- truth)
    error_1D = enso_project.forecast_δ_1D(prediction, truth, "1D", "abs")
    error_all = enso_project.forecast_δ_1D(prediction, truth, "all", "abs")
    @test all(error_1D == error_1D_true)
    @test all(error_all == error_all_true)
    @test size(error_1D, 1) == size(error_all, 1)
    @test size(error_1D, 2) != size(error_all, 2)
end