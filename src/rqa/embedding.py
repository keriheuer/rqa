
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

def embed_time_series(ts, tau, dim):
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

def compute_ami(time_series, max_delay):
    """Compute average mutual information for different time delays."""
    ami_values = []
    for delay in range(1, max_delay + 1):
        ami = mutual_info_score(time_series[:-delay], time_series[delay:])
        ami_values.append(ami)
    return np.array(ami_values)

def find_local_minima_ami(time_series, max_delay):
    ami_values = compute_ami(time_series, max_delay)
    
    # Find local minima indices
    local_minima_indices = (np.diff(np.sign(np.diff(ami_values))) > 0).nonzero()[0] + 1
    local_minima_values = ami_values[local_minima_indices]
    
    return local_minima_indices, local_minima_values, ami_values

def compute_derivative(values):
    """Compute the derivative of a numpy array."""
    return np.gradient(values, edge_order=2)

def find_optimal_delay(local_minima_indices, derivative_adaf):
    for index in local_minima_indices:
        if index < len(derivative_adaf) and derivative_adaf[index] < 0:
            return index  # Return the index where the first local minimum has a decreasing derivative of ADAF
    return None  # Return None if no such index is found

def optimal_delay_ami(time_series, max_delay=100, plot=False):
    """Find the optimal time delay using the first local minimum of AMI."""

    local_minima_indices, local_minima_values, ami_values = find_local_minima_ami(time_series, max_delay)
    optimal_delay_index = find_optimal_delay(local_minima_indices, derivative_adaf)
    if optimal_delay_index is not None:
        print(f'Optimal time delay: {optimal_delay_index + 1}')  # Adding 1 because delay starts from 1
        min_delay = optimal_delay_index + 1
    else:
        if len(local_minima_indices) > 1:
            print(f'No optimal time delay found with the given criteria... using first local minima: {local_minima_indices[0]}')
            min_delay = local_minima_indices[0]
        else:
            print(f'No optimal time delay found with the given criteria... using global minimum: {np.argmin(ami_values) + 1}')
            min_delay = np.argmin(ami_values) + 1  # Fallback to global minimum if no local minimum is found
        
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_delay + 1), ami_values, marker='o')
        plt.axvline(x=min_delay, color='r', linestyle='--', label=f'Optimal delay: {min_delay}')
        plt.xlabel('Delay')
        plt.ylabel('Average Mutual Information')
        plt.title('Average Mutual Information vs Delay')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return min_delay

def plot_acf(time_series, nlags=100):
    # Compute the autocorrelation function
    acf_values = acf(time_series, nlags=nlags)
    
    # Find the first zero crossing
    first_zero = np.where(acf_values < 0)[0][0] if any(acf_values < 0) else len(acf_values)
    print(f'First zero in ACF: {first_zero}')
    
    # Find the autocorrelation time (decay to 1/e)
    autocorr_time = np.where(acf_values < 1/np.e)[0][0] if any(acf_values < 1/np.e) else len(acf_values)
    print(f'ACF time: {autocorr_time}')
    
    # Plot the ACF
    plt.figure(figsize=(10, 5))
    plt.plot(acf_values, marker='o', linestyle='-', label='ACF')
    plt.axvline(x=first_zero, color='r', linestyle='--', label='First Zero Crossing')
    plt.axvline(x=autocorr_time, color='g', linestyle='--', label='Autocorrelation Time (1/e decay)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_adaf(time_series, max_delay):
    """Compute ADAF values for different time delays."""
    adaf_values = []
    for delay in range(1, max_delay + 1):
        distances = []
        for i in range(len(time_series) - delay):
            distance = np.abs(i - (i + delay))
            distances.append(distance)
        adaf_values.append(np.mean(distances))
    return np.array(adaf_values)

def compute_derivative_adaf(adaf_values):
    """Compute the derivative of the ADAF values."""
    derivative_adaf = np.diff(adaf_values)
    return derivative_adaf

def find_first_local_minimum(ami_values):
    """Find the first local minimum in the average mutual information values."""
    for i in range(1, len(ami_values) - 1):
        if ami_values[i] < ami_values[i - 1] and ami_values[i] < ami_values[i + 1]:
            return i  # Return the delay corresponding to the first local minimum
    return None  # Return None if no local minimum is found

def false_nearest_neighbors(time_series, max_dim, delay, rt):
    def embed(series, dim, tau):
        """Embed the time series."""
        N = len(series)
        embedded = np.empty((N - (dim - 1) * tau, dim))
        for i in range(dim):
            embedded[:, i] = series[i * tau:N - (dim - 1) * tau + i * tau]
        return embedded
    
    def find_nearest_neighbor(embedded, index):
        """Find the nearest neighbor of a point, excluding itself."""
        distances = np.linalg.norm(embedded - embedded[index], axis=1)
        distances[index] = np.inf  # Exclude the point itself
        return np.argmin(distances), np.min(distances)

    fnn_ratio = []
    for dim in range(1, max_dim + 1):
        embedded = embed(time_series, dim, delay)
        false_neighbors = 0
        for i in range(embedded.shape[0] - delay):  # Exclude last points which do not have a future value
            neighbor_index, neighbor_distance = find_nearest_neighbor(embedded, i)
            if neighbor_index + delay < embedded.shape[0]:  # Check if future index is in bounds
                next_future_value = time_series[neighbor_index + delay * dim]
                current_future_value = time_series[i + delay * dim]
                distance_increase = abs(next_future_value - current_future_value)
                if distance_increase > rt * neighbor_distance:
                    false_neighbors += 1

        # The ratio of false neighbors to total points considered
        fnn_ratio.append(false_neighbors / (len(time_series) - (dim - 1) * delay))

    return fnn_ratio


def false_nearest_neighbors(time_series, max_dim, delay, rt):
    def embed(series, dim, tau):
        """Embed the time series."""
        N = len(series)
        embedded = np.empty((N - (dim - 1) * tau, dim))
        for i in range(dim):
            embedded[:, i] = series[i * tau:N - (dim - 1) * tau + i * tau]
        return embedded
    
    def find_nearest_neighbor(embedded, index):
        """Find the nearest neighbor of a point, excluding itself."""
        distances = np.linalg.norm(embedded - embedded[index], axis=1)
        distances[index] = np.inf  # Exclude the point itself
        return np.argmin(distances), np.min(distances)

    fnn_ratio = []
    for dim in range(1, max_dim + 1):
        embedded = embed(time_series, dim, delay)
        false_neighbors = 0
        for i in range(embedded.shape[0] - delay):  # Exclude last points which do not have a future value
            neighbor_index, neighbor_distance = find_nearest_neighbor(embedded, i)
            if neighbor_index + delay < embedded.shape[0]:  # Check if future index is in bounds
                next_future_value = time_series[neighbor_index + delay * dim]
                current_future_value = time_series[i + delay * dim]
                distance_increase = abs(next_future_value - current_future_value)
                if distance_increase > rt * neighbor_distance:
                    false_neighbors += 1

        # The ratio of false neighbors to total points considered
        fnn_ratio.append(false_neighbors / (len(time_series) - (dim - 1) * delay))

    return fnn_ratio