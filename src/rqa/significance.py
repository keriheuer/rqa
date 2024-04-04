from surrogates import *
from rqa_utils import convert_seconds
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries, EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric, MaximumMetric 
from pyrqa.analysis_type import Classic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from rich.progress import Progress
from time import time

def compute_measure(surrogate, rqa_measure_func, tau, dim, lag_of_interest):
    # This function calculates the RQA measure for a given surrogate series
    return rqa_measure_func(surrogate, tau, dim, lag_of_interest)

def compute_measure_wrapper(args):
    # Unpack the arguments
    surrogate, rqa_measure_func, tau, dim, lag_of_interest = args
    return compute_measure(surrogate, rqa_measure_func, tau, dim, lag_of_interest)

# def test_significance(original_series, tau, dim, surrogates, rqa_measure_func, lag_of_interest):
#     original_measure = rqa_measure_func(original_series, tau, dim, lag_of_interest)

#     # Create a list of tuples, each containing all arguments needed for compute_measure
#     args_list = [(surrogate, rqa_measure_func, tau, dim, lag_of_interest) for surrogate in surrogates]

#     # Compute measures in parallel using ProcessPoolExecutor
#     with ProcessPoolExecutor() as executor:
#         # Map compute_measure_wrapper across args_list
#         surrogate_measures = list(executor.map(compute_measure_wrapper, args_list))

#     # Calculate the empirical p-value (two-tailed test)
#     extreme_values_count = np.sum(np.array(surrogate_measures) >= original_measure) #+ np.sum(np.array(surrogate_measures) <= -original_measure)
#     p_value = extreme_values_count / len(surrogates)

def test_significance(original_series, tau, dim, surrogates, rqa_measure_func, lag_of_interest):
    start = time()
    
    original_measure = rqa_measure_func(original_series, tau, dim, lag_of_interest)

    surrogate_measures = []

    # Set up progress bar
    with Progress() as progress:
        task1 = progress.add_task("[cyan]Computing surrogates...", total=len(surrogates))

        # Compute measures in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            # Submit all the compute tasks
            # futures = [executor.submit(compute_measure_wrapper, args) for args in args_list]
            futures = list(executor.map(compute_measure_wrapper, args_list))

            # As each future completes, update the progress bar
            for future in as_completed(futures):
                result = future.result()
                surrogate_measures.append(result)
                progress.update(task1, advance=1)

    # Calculate the empirical p-value (two-tailed test)
    extreme_values_count = np.sum(np.array(surrogate_measures) >= original_measure)
    p_value = extreme_values_count / len(surrogates)

    elapsed_time = time() - start
    minutes, seconds = convert_seconds(elapsed_time)
    print(f"Took {minutes} minute(s) and {seconds:.2f} second(s)")
    
    return p_value, surrogate_measures

def rqa_measure_func(ts, tau, dim, lag):
    # This should return the RQA measure of interest at the specified lag
    epsilon = find_epsilon(ts, 0.05, tau=tau, dim=dim, theiler=tau*dim)
    
    time_series = TimeSeries(ts, embedding_dimension=dim, time_delay=tau)
    settings = Settings(time_series,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(epsilon),
                        similarity_measure=MaximumMetric,
                        theiler_corrector=tau*dim)
                        
    rqa = RQAComputation.create(settings).run()
    close_returns = rqa.diagonal_frequency_distribution
    frequency = close_returns[lag]
    return frequency

def find_epsilon(time_series_data, target_recurrence_rate, tau=1, dim=1, theiler=1, initial_epsilon=1.0, tolerance=0.0001, max_iterations=100):
    """
    Find the epsilon value that corresponds to a given recurrence rate.
    
    Parameters:
        time_series_data (np.array): The time series data.
        target_recurrence_rate (float): The target recurrence rate.
        initial_epsilon (float): Initial guess for epsilon.
        tolerance (float): Tolerance for the difference between current and target recurrence rates.
        max_iterations (int): Maximum number of iterations to perform.
        
    Returns:
        float: The epsilon value corresponding to the target recurrence rate.
    """
    lower_epsilon = 0
    upper_epsilon = initial_epsilon
    current_epsilon = initial_epsilon
    time_series = TimeSeries(time_series_data, time_delay=tau, embedding_dimension=dim)
    
    for _ in range(max_iterations):
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(current_epsilon),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=tau*dim)
        
        rqa_computation = RQAComputation.create(settings)
        rqa_result = rqa_computation.run()
        current_recurrence_rate = rqa_result.recurrence_rate
        
        if abs(current_recurrence_rate - target_recurrence_rate) < tolerance:
            return current_epsilon  # Found the epsilon corresponding to the target recurrence rate

        # Update the search bounds and current epsilon based on the comparison
        if current_recurrence_rate > target_recurrence_rate:
            upper_epsilon = current_epsilon
            current_epsilon = (lower_epsilon + upper_epsilon) / 2
        else:
            lower_epsilon = current_epsilon
            current_epsilon = (lower_epsilon + upper_epsilon) / 2
    
    return current_epsilon  # Return the best approximation of epsilon after max_iterations

if __name__ == "__main__":
    start = time()
    df = pd.read_csv('1ES_binned_lc.csv')
    ts = df['scaled lumin'].values
    tau, dim = 12, 5
    N = 1000
    create_surrogates = True

    # Create new surrogates
    if create_surrogates:
        surrogates = {}
        for sur_type in ['phase_randomized', 'white_noise', 'correlated_noise', 'IAAFT', 'AAFT', 'time_reversed']:
            surrogates[sur_type] = generate_surrogates(sur_type, ts, N)

        np.savez_compressed(f'../data/1ES_surrogates_{N}.npz', **surrogates)
    else:     
        data = np.load('../data/1ES_surrogates.npz', allow_pickle=True)
        surrogates = {key: data[key] for key in data.files}
    
    pvals = {}
    for key in surrogates.keys():
            p_value, frequencies = test_significance(ts, tau, dim, surrogates[key], rqa_measure_func, lag_of_interest=10)
            print(f"{key} surrogates P-value: {p_value}")
            pvals[key] = frequencies
    np.savez_compressed(f'../data/1ES_surrogates_{N}_close_returns_lag=10.npz', **pvals)
    
    elapsed_time = time() - start
    minutes, seconds = convert_seconds(elapsed_time)
    print(f"Took {minutes} minute(s) and {seconds:.2f} second(s)")
    