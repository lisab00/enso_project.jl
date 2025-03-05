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