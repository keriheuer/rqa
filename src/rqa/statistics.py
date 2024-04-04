import numpy as np

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def time_reversal_asymmetry_statistic(time_series, delay=1):
    n = len(time_series)
    # Ensure we have enough data points for the calculation
    if n <= 2 * delay:
        raise ValueError("Time series is too short for the given delay")

    # Calculate the asymmetry statistic according to the provided equation
    # Typically this involves terms like (y[n] - y[n - delay]) ** 3 and (y[n] - y[n - delay]) ** 2
    sum_num = 0.0
    sum_den = 0.0
    for i in range(delay, n - delay):
        diff = time_series[i] - time_series[i - delay]
        sum_num += (time_series[i + delay] - time_series[i])**3
        sum_den += diff**2

    # Normalize the numerator by the number of terms
    sum_num /= (n - 2 * delay)

    # The denominator is the square of the difference so take the square root after summing
    sum_den = (sum_den / (n - 2 * delay))**1.5

    return sum_num / sum_den if sum_den != 0 else np.nan

def compute_trend(rp, rqa):
    """
    Compute the TREND (tendency of the phases to become more recurrent) from a recurrence matrix.

    Parameters:
    recurrence_matrix (np.array): a square binary matrix representing the recurrence plot

    Returns:
    float: TREND metric value
    """
    
    N = rp.shape[0]
    
    # Create a histogram of the line lengths of diagonal lines in the recurrence plot
    # diag_counts = [np.sum(np.diagonal(rp, offset=i)) for i in range(-N+1, N)]
    diag_counts = rqa.diagonal_frequency_distribution
    
    # Mean recurrence rate
    RR_mean = np.mean(diag_counts)
    
    # Apply TREND formula
    numerator = sum((i - N/2) * (RR_i - RR_mean) for i, RR_i in enumerate(diag_counts, 1))
    denominator = sum((i - N/2)**2 for i in range(1, 2*N))
    
    return numerator / denominator if denominator != 0 else 0

def compute_p_values(original_rqa_df, surrogate_rqa_dfs):
    
    # List of metrics to calculate p vals for
    metrics = original_rqa_df.columns
    
    # Initialize a dictionary to hold p-values for each RQA metric at each recurrence rate
    p_values = {metric: [] for metric in metrics}
    p_values['RR'] = []  # Add a list to hold recurrence rates
    
    # Iterate over each recurrence rate
    for rr in original_rqa_df.index:
        # Get the RQA metrics for the original time series at the current recurrence rate
        original_metrics = original_rqa_df.loc[rr]
        
        # Get the RQA metrics for all surrogates at the current recurrence rate
        surrogate_metrics = surrogate_rqa_dfs.xs(rr, level='RR')
        
        # Compute p-values for each metric
        for metric in metrics:
            # Count the number of surrogates with a more extreme value than the original
            count = np.sum(surrogate_metrics[metric] >= original_metrics[metric]) if larger_is_better_mapping[metric] \
                    else np.sum(surrogate_metrics[metric] <= original_metrics[metric])
            
            # Calculate the empirical p-value
            p_value = count / len(surrogate_metrics)
            p_values[metric].append(p_value)
        
        # Append the current recurrence rate
        p_values['RR'].append(rr)
    
    # Convert the dictionary of p-values into a DataFrame
    p_values_df = pd.DataFrame(p_values)
    p_values_df.set_index('RR', inplace=True)
    
    return p_values_df

def DynK(TS_full, win, value_range=100):
    if len(np.unique(TS_full)) < 3:
        print('A minimum of 3 unique values is required to calculate the dynamic complexity.')
        return [], [], []

    if max(TS_full) - min(TS_full) > value_range:
        print('Please check range - has to be at least max - min of the values of the time series')
        return [], [], []

    if 0 in TS_full:
        TS_full = np.array(TS_full) + 1

    C = []
    F = []
    D = []

    m_end = len(TS_full) - win + 1
    for m in range(m_end):
        TS = np.array(TS_full[m:m+win])

        # Fluctuation intensity
        intervals = []
        current_sign = np.sign(TS[1] - TS[0])
        interval = [TS[0], TS[1]]
        for i in range(2, win):
            new_sign = np.sign(TS[i] - TS[i-1])
            if new_sign == current_sign or TS[i] - TS[i-1] == 0:
                interval.append(TS[i])
            else:
                intervals.append(interval)
                interval = [TS[i-1], TS[i]]
            current_sign = new_sign
        intervals.append(interval)  # add the last interval

        M = [(max(interval) - min(interval)) / (len(interval) - 1) for interval in intervals if len(interval) > 1]
        F_counter = sum(M)
        F_denom = (value_range - 1) * (win - 1)
        F_win = F_counter / F_denom
        F.append(F_win)

        # Distribution
        sTS = sorted(TS)
        S = (value_range - 1) / (win - 1)
        diff_Soll = 0
        diff_Ist = 0

        for c in range(win - 1):
            for d in range(c + 1, win):
                for a in range(c, d):
                    for b in range(a + 1, d):
                        diff_Soll += S * (b - a)
                        if sTS[b] - sTS[a] >= 0:
                            diff_Ist += S * (b - a) - (sTS[b] - sTS[a])

        D_win = 1 - diff_Ist / diff_Soll
        D.append(D_win)

        # Calculate the dynamic complexity for the window
        C_win = D_win * F_win
        C.append(C_win)

    return C, F, D

# # Example usage:
# TS_full = [4, 5, 3, 6, 2, 7, 1, 4, 5, 3, 6, 2, 7, 1]  # Example time series data
# win = 7  # Window size
# value_range = 7  # Range of possible values (e.g., 7-point Likert scale)
# C, F, D = DynK(TS_full, win, value_range)

# print('Dynamic Complexity:', C)
# print('Fluctuation Intensity:', F)
# print('Distribution:', D)


def calculate_renyi_entropy(close_returns, alpha):
    """
    Calculate the Rényi entropy of a given order alpha from close returns.
    
    :param close_returns: A histogram or array-like representing the frequency of close returns.
    :param alpha: The order of the Rényi entropy. Must be a non-negative value, not equal to 1.
    :return: The Rényi entropy of the given order.
    """
    if alpha < 0:
        raise ValueError("Alpha must be non-negative.")
    if alpha == 1:
        raise ValueError("Alpha must not be 1. Use Shannon entropy instead.")

    # Convert close returns to probability distribution
    total_returns = np.sum(close_returns)
    probabilities = close_returns / total_returns
    
    # Filter out zero probabilities for numerical stability
    valid_probabilities = probabilities[probabilities > 0]

    # Calculate Rényi entropy
    if alpha == 0:
        # Max entropy: simply the count of non-zero probabilities
        return np.log(len(valid_probabilities))
    elif alpha == np.inf:
        # Min-entropy: -log(max probability)
        return -np.log(np.max(valid_probabilities))
    else:
        # General case for Rényi entropy
        entropy = (1 / (1 - alpha)) * np.log(np.sum(valid_probabilities ** alpha))

    return entropy

# Example usage:
# close_returns = histogram_of_close_returns_here
# alpha = 2  # Example for alpha=2, but can be any non-negative value, not equal to 1
# renyi_entropy = calculate_renyi_entropy(close_returns, alpha)
# print(renyi_entropy)

def calculate_correlation_dimension(diagonal_lengths, num_segments=2):
    """
    Calculate the correlation dimension (K2) from the distribution of diagonal line lengths
    using a piecewise linear fit.

    :param diagonal_lengths: A histogram (array-like) of diagonal line lengths from a recurrence plot.
    :param num_segments: Number of segments to use in the piecewise linear fit.
    :return: Estimated correlation dimension K2 (slope of the fitted line in the scaling region).
    """
    # Prepare the data for fitting
    # x_data: lengths, y_data: log(frequency) of these lengths
    lengths = np.arange(1, len(diagonal_lengths) + 1)
    frequencies = np.array(diagonal_lengths)
    # Filter out zero frequencies for log scaling
    valid_indices = frequencies > 0
    x_data = lengths[valid_indices]
    y_data = np.log(frequencies[valid_indices])

    # Initialize piecewise linear fit with x and y data
    my_pwlf = pwlf.PiecewiseLinFit(x_data, y_data)

    # Fit the data with the specified number of line segments
    res = my_pwlf.fit(num_segments)

    # Get the slopes of line segments
    slopes = my_pwlf.calc_slopes()

    # Correlation dimension K2 is typically estimated from the first slope
    # (assuming the first segment represents the scaling region)
    K2 = slopes[0]

    return K2, my_pwlf

# Example usage:
# diagonal_lengths = histogram_of_diagonal_line_lengths_here
# K2, pwlf_model = calculate_correlation_dimension(diagonal_lengths)
# print(K2)