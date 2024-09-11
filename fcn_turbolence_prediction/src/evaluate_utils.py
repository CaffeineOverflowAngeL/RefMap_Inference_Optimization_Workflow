import numpy as np
from .utils import calculate_mse

def get_mse_per_dim(n_vars_out, Y_pred, Y_test, var_names):
    
    overall_mse_per_dimension = {}
    for i in range(n_vars_out):
        # Calculate overall MSE for each dimension
        overall_mse = calculate_mse(Y_pred[:, i, :, :].ravel(), Y_test[:, i, :, :].ravel())
        overall_mse_per_dimension[var_names[i]] = overall_mse
        
    return overall_mse_per_dimension

def get_mse_stats_per_dim(n_vars_out, Y_pred, Y_test, var_names):
# Calculate MSE and additional statistics for each pair of the second dimension
    mse_results = []
    max_values = []
    quartiles = []

    # Calculate MSE for each sample and each dimension
    mse_per_sample = []
    for i in range(n_vars_out):
        mse = [calculate_mse(a.ravel(), b.ravel()) 
            for a, b in zip(Y_pred[:, i, :, :], Y_test[:, i, :, :])]
        mse_per_sample.append(mse)

    # Calculate the statistical properties for each dimension's MSE
    stats = {}
    for i, mse_values in enumerate(mse_per_sample, start=0):
        max_val = np.max(mse_values)
        q1 = np.percentile(mse_values, 25)
        median = np.percentile(mse_values, 50)
        q3 = np.percentile(mse_values, 75)
        stats[f'{var_names[i]}'] = {'Max': max_val, 'Q1': q1, 'Median': median, 'Q3': q3}
        
    return stats