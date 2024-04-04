import numpy as np
import pwlf
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries, EmbeddedSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric, MaximumMetric 
from pyrqa.analysis_type import Classic
import pandas as pd
# SNR = median_abs_deviation(y)
import numpy.fft as fft
from eztao.ts.carma_sim import pred_lc

def white_noise_surrogates(ts):
    return np.random.permutation(ts)

def correlated_noise_surrogate(ts):
    
    # Fourier transform of the original time series
    ts_fourier = fft.rfft(ts)
    
    # Random phases
    random_phases = np.exp(2j * np.pi * np.random.random(len(ts_fourier)))
    
    # Surrogate with randomized phases
    surrogate_fourier = np.abs(ts_fourier) * random_phases
    
    # Inverse Fourier transform to obtain the time series
    return fft.irfft(surrogate_fourier, n=len(ts))

def IAAFT_surrogates(ts, max_iter=1000):
    
    # Sort the original time series and generate a random phase structure
    sorted_ts = np.sort(ts)
    surrogate = np.random.permutation(ts)
    
    for i in range(max_iter):
        
        # Match the power spectrum to that of the original signal
        surrogate_fourier = fft.rfft(surrogate)
        surrogate = fft.irfft(np.abs(ts_fourier) * surrogate_fourier / np.abs(surrogate_fourier), n=len(ts))
        
        # Reshuffle the surrogate to match the sorted amplitude distribution of the original signal
        old_surrogate = np.copy(surrogate)
        surrogate = np.interp(np.argsort(np.argsort(surrogate)), np.argsort(np.argsort(sorted_ts)), sorted_ts)
        
        # Convergence check (can be adjusted with a tolerance threshold)
        if np.allclose(old_surrogate, surrogate, atol=1e-6, rtol=0):
            break
    
    return surrogate

# Example usage:
# ts = np.array(time_series_data_here)
# white_surrogates = white_noise_surrogates(ts)
# correlated_surrogates = correlated_noise_surrogate(ts)
# IAAFT_surrogates = IAAFT_surrogates(ts)

def plot_predicted_lc(t, y, yerr, params, p, t_pred, original_params=False, scaling="z_score", ylabel=None, twinx=None, twinxlabel=None, plot_input=True):
    """
    Plot GP predicted time series given best-fit parameters.

    Args:
        t (array(float)): Time stamps of the input time series.
        y (array(float)): y values of the input time series.
        yerr (array(float)): Measurement errors for y values.
        params (array(float)): Best-fit CARMA parameters
        p (int): The AR order (p) of the best-fit model.
        t_pred (array(float)): Time stamps at which to generate predictions.
        plot_input (bool): Whether to plot the input time series. Defaults to True.
    """

    fig, ax = plt.subplots(figsize=(15, 10)) #1, 1, dpi=150, figsize=(8, 4))
        
    # get pred lc
    t_pred, mu, var = pred_lc(t, y, yerr, params, int(p), t_pred)

    # rescale to original scale
    if original_params:
        unscaled_y, unscaled_yerr = unscale_lc(y, yerr, original_params, method=scaling)
        unscale_factor = unscaled_y/y
        # mean_original = original_params['mean']
        # std_dev_original = original_params['std']

        # Rescale the predicted light curve and variance 
        mu = mu * unscale_factor #(mu * std_dev_original) + mean_original
        var = var * unscale_factor #(std_dev_original ** 2)
        ax.set_yscale('log')
        
    # Plot the data
    t_range = t_pred[-1] - t_pred[0]
    color = "#c994c7" #"#ff7f0e"
    ax.plot(t_pred, mu, color=color, label="Mean Prediction", alpha=0.8)

    # if valid variance returned
    if np.median(var) > (np.median(np.abs(yerr)) / 1e10):
        std = np.sqrt(var)
        ax.fill_between(
            t_pred, mu + std, mu - std, color=color, alpha=0.3, edgecolor="none"
        )

    if plot_input:
        ax.errorbar(
            t, y, yerr=yerr, fmt=".k", capsize=2, label="Input Data", markersize=3
        )
        
    # twin x axis for input time
    if twinxlabel and twinx is not None:
        twin_ax = ax.twiny()
        twin_ax.plot(twinx, y, alpha=0)
        twin_ax.set_xlabel(twinxlabel)
        ax.tick_params(which="both", axis="both", bottom=True, labelbottom=True)
    else:
        print("Error: must supply twinxlabel if twinx supplied, not plotting twinx.")

    # other plot settings
    ax.set_xlim(t_pred[0] - t_range / 50, t_pred[-1] + t_range / 50)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(r"Flux (arb. unit)")
    ax.set_xlabel(r"t$_{rest}$ (days)")
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.set_title("Maximum Likelihood Prediction")
    # ax.set_title("Predicted Lightcurve of Best-Fit DHO")
    ax.legend(fancybox=False, edgecolor='k')
    ax.tick_params(which="both", axis="both", top=True, right=False)
    
    fig.tight_layout()
    return fig, mu, var

def scale_lc(luminosity, errors, method='fraction_of_max'):
    """
    Scale the luminosity and errors.

    Args:
    - luminosity (array-like): Luminosity values.
    - errors (array-like): Corresponding errors for the luminosity values.
    - method (str): Method to scale the luminosity and errors ('fraction_of_max', 'z_score', 'logarithmic').

    Returns:
    - scaled_luminosity (np.ndarray): Scaled luminosity.
    - scaled_errors (np.ndarray): Scaled errors.
    """
    
    if method == 'fraction_of_max':
        # Scale by the maximum luminosity value
        max_lumin = np.max(luminosity)
        scaled_luminosity = luminosity / max_lumin
        scaled_errors = errors / max_lumin  # Scale errors in the same way as luminosity
        
    elif method == 'z_score':
        # Scale by z-score (mean=0, std=1)
        mean_lumin = np.mean(luminosity)
        std_lumin = np.std(luminosity)
        scaled_luminosity = (luminosity - mean_lumin) / std_lumin
        scaled_errors = errors / std_lumin  # Errors are scaled by the std of the luminosity
        
    elif method == 'logarithmic':
        # Apply logarithmic scaling
        scaled_luminosity = np.log10(luminosity)
        
        # Scale errors using logarithmic error propagation rule
        scaled_errors = errors / (luminosity * np.log(10))
        
    else:
        raise ValueError("Invalid method. Use 'fraction_of_max', 'z_score', or 'logarithmic'.")
    
    return scaled_luminosity, scaled_errors

def convert_epsilon(time_series, epsilon, theiler=1):
    """ Finds recurrence rate corresponding to threshold value epsilon.
    param: time_series: pyrqa TimeSeries object
    param: epsilon: float (single threshold) or list/array (multiple thresholds)
    param: theiler: int: Theiler corrector
    """
    
    if isinstance(epsilon, (list, np.ndarray)): # list of epsilons supplied
        thresholds = epsilon  
    else: # if single epsilon is supplied
        thresholds = [epsilon]

    # find corresponding RR for each value of epsilon
    recurrence_rates = []
    for epsilon in thresholds:
        settings = Settings(time_series,
                                analysis_type=Classic,
                                neighbourhood=FixedRadius(epsilon),
                                similarity_measure=EuclideanMetric,
                                theiler_corrector=theiler)
            
        rqa = RQAComputation.create(settings).run()
        RR = rqa.recurrence_rate
        recurrence_rates.append(int(np.round(RR*100)))
       
    if isinstance(epsilon, (list, np.ndarray)): # list of epsilons supplied
        return recurrence_rates
    else: # if single epsilon is supplied
        return recurrence_rates[0]
    
    
# SURROGATES
def white_noise_surrogates(ts):
    return np.random.permutation(ts)

def correlated_noise_surrogate(ts):
    # Fourier transform of the original time series
    ts_fourier = fft.rfft(ts)
    # Random phases
    random_phases = np.exp(2j * np.pi * np.random.random(len(ts_fourier)))
    # Surrogate with randomized phases
    surrogate_fourier = np.abs(ts_fourier) * random_phases
    # Inverse Fourier transform to obtain the time series
    return fft.irfft(surrogate_fourier, n=len(ts))

def IAAFT_surrogates(ts, max_iter=1000):
    # Sort the original time series and generate a random phase structure
    sorted_ts = np.sort(ts)
    surrogate = np.random.permutation(ts)
    
    for i in range(max_iter):
        # Match the power spectrum to that of the original signal
        surrogate_fourier = fft.rfft(surrogate)
        surrogate = fft.irfft(np.abs(ts_fourier) * surrogate_fourier / np.abs(surrogate_fourier), n=len(ts))
        
        # Reshuffle the surrogate to match the sorted amplitude distribution of the original signal
        old_surrogate = np.copy(surrogate)
        surrogate = np.interp(np.argsort(np.argsort(surrogate)), np.argsort(np.argsort(sorted_ts)), sorted_ts)
        
        # Convergence check (can be customized with a tolerance threshold)
        if np.allclose(old_surrogate, surrogate, atol=1e-6, rtol=0):
            break
    
    return surrogate

def embed_time_series(ts, dim, tau):
    """
    Embed a time series with given dimension and time delay.
    
    :param ts: 1D NumPy array or list containing the time series.
    :param dim: The embedding dimension.
    :param tau: The time delay (lag).
    :return: 2D NumPy array containing the embedded time series.
    """
    n = len(ts)
    if n < (dim - 1) * tau + 1:
        raise ValueError("Time series is too short for the given dimension and time delay")

    embedded_ts = np.empty((n - (dim - 1) * tau, dim))
    for i in range(dim):
        embedded_ts[:, i] = ts[i * tau: n - (dim - 1) * tau + i * tau]
    return embedded_ts
    
def scale_lc(luminosity, errors, method='fraction_of_max'):
    """
    Scale the luminosity and errors.

    Args:
    - luminosity (array-like): Luminosity values.
    - errors (array-like): Corresponding errors for the luminosity values.
    - method (str): Method to scale the luminosity and errors ('fraction_of_max', 'z_score', 'logarithmic').

    Returns:
    - scaled_luminosity (np.ndarray): Scaled luminosity.
    - scaled_errors (np.ndarray): Scaled errors.
    """
    
    if method == 'fraction_of_max':
        # Scale by the maximum luminosity value
        max_lumin = np.max(luminosity)
        scaled_luminosity = luminosity / max_lumin
        scaled_errors = errors / max_lumin  # Scale errors in the same way as luminosity
        
    elif method == 'z_score':
        # Scale by z-score (mean=0, std=1)
        mean_lumin = np.mean(luminosity)
        std_lumin = np.std(luminosity)
        scaled_luminosity = (luminosity - mean_lumin) / std_lumin
        scaled_errors = errors / std_lumin  # Errors are scaled by the std of the luminosity
        
    elif method == 'logarithmic':
        # Apply logarithmic scaling
        scaled_luminosity = np.log10(luminosity)
        # Scale errors using logarithmic error propagation rule
        scaled_errors = errors / (luminosity * np.log(10))
        
    else:
        raise ValueError("Invalid method. Use 'fraction_of_max', 'z_score', or 'logarithmic'.")
    
    return scaled_luminosity, scaled_errors

def unscale_lc(scaled_luminosity, scaled_errors, original_params, method='fraction_of_max'):
    """
    Reverse the scaling applied to luminosity and errors.

    Args:
    - scaled_luminosity (np.ndarray): Scaled luminosity values.
    - scaled_errors (np.ndarray): Scaled errors corresponding to the luminosity values.
    - original_params (dict): Parameters used for the original scaling, like 'mean', 'std', and 'max_lumin'.
    - method (str): Method that was used to scale the luminosity and errors ('fraction_of_max', 'z_score', 'logarithmic').

    Returns:
    - original_luminosity (np.ndarray): Luminosity values in original scale.
    - original_errors (np.ndarray): Errors in original scale.
    """
    
    if method == 'fraction_of_max':
        max_lumin = original_params['max_lumin']
        original_luminosity = scaled_luminosity * max_lumin
        original_errors = scaled_errors * max_lumin
        
    elif method == 'z_score':
        mean_lumin = original_params['mean']
        std_lumin = original_params['std']
        original_luminosity = scaled_luminosity * std_lumin + mean_lumin
        original_errors = scaled_errors * std_lumin
        
    elif method == 'logarithmic':
        original_luminosity = 10 ** scaled_luminosity
        # Inverse of the logarithmic error propagation
        original_errors = scaled_errors * (10 ** scaled_luminosity * np.log(10))
        
    else:
        raise ValueError("Invalid method. Use 'fraction_of_max', 'z_score', or 'logarithmic'.")
    
    return original_luminosity, original_errors

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
    time_series = TimeSeries(time_series_data, embedding_dimension=dim, time_delay=tau)
    
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

def get_RQA_significance(original_rqa_df, surrogates_rqa_df):
    # Extract unique surrogate types
    surrogate_types = surrogates_rqa_df.index.get_level_values('surrogate_type').unique()

    # Initialize a DataFrame to hold the p-values
    p_values_df = pd.DataFrame(index=surrogate_types, columns=original_rqa_df.columns)

    # Vectorized computation of p-values for each surrogate type
    for sur_type in surrogate_types:
        # Filter the surrogates DataFrame for the current type
        surrogates_of_type = surrogates_rqa_df.xs(sur_type, level='surrogate_type').to_numpy()

        # Get the original metrics as a NumPy array
        original_metrics = original_rqa_df.to_numpy()

        # Assuming larger values are better, adjust if necessary
        # Vectorized comparison to count surrogates with more extreme values
        count_extreme_values = np.sum(surrogates_of_type >= original_metrics, axis=0)

        # Calculate p-values and assign to the appropriate row in the p-values DataFrame
        p_values_df.loc[sur_type] = count_extreme_values / surrogates_of_type.shape[0]

    return p_values_df

def convert_seconds(sec):
    minutes = sec // 60
    seconds = sec % 60
    return int(minutes), seconds

def interpret_pvals(pvals_df):
    # Define the significance thresholds with confidence levels
    significance_thresholds = {
        "Significant": {"threshold": 0.05, "confidence": "95%"},
        "Moderately significant": {"threshold": 0.1, "confidence": "90%"},
        "Not significant": {"threshold": float('inf'), "confidence": ""}
    }
    
    # Iterate over each RQA metric column in the DataFrame
    for metric in pvals_df.columns:
        # Initialize a dictionary to hold surrogate types for each significance level
        significance_results = {sig: [] for sig in significance_thresholds.keys()}
        
        # Go through each surrogate type and its p-value
        for surrogate_type, p_value in pvals_df[metric].items():
            # Determine the significance level of the p-value
            for sig, info in significance_thresholds.items():
                if p_value < info["threshold"]:
                    formatted_surrogate_type = surrogate_type.replace("_", " ").title()
                    significance_results[sig].append(formatted_surrogate_type)
                    break  # Break after assigning to the first matching threshold

        # Print the results for the current metric
        print(f"\n{metric}:")
        for sig, surrogates in significance_results.items():
            if surrogates:  # Only print if there are surrogates at this significance level
                confidence = significance_thresholds[sig]["confidence"]
                # Only add confidence level if it is not empty
                confidence_str = f" ({confidence} confidence)" if confidence else ""
                print(f"    {sig} against {', '.join(surrogates)}{confidence_str}")
