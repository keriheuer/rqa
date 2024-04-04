import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from numpy.fft import rfft, irfft, fftfreq

import numpy as np
from numpy.fft import rfft, irfft

def white_noise_surrogates(ts, N=100):
    surrogates = np.empty((N, len(ts)))
    for i in range(N):
        surrogates[i] = np.random.permutation(ts)
    return surrogates

def correlated_noise_surrogates(ts, N=100):
    ts_fourier = rfft(ts)
    magnitude = np.abs(ts_fourier)
    surrogates = np.empty((N, len(ts)))
    for i in range(N):
        random_phases = np.exp(2j * np.pi * np.random.random(len(ts_fourier)))
        surrogate_fourier = magnitude * random_phases
        surrogates[i] = irfft(surrogate_fourier, n=len(ts))
    return surrogates

def IAAFT_surrogates(ts, N=100, max_iter=1000):
    sorted_ts = np.sort(ts)
    ts_fourier = rfft(ts)
    magnitude = np.abs(ts_fourier)
    surrogates = np.empty((N, len(ts)))
    
    for i in range(N):
        surrogate = np.random.permutation(ts)
        for _ in range(max_iter):
            surrogate_fourier = rfft(surrogate)
            phase = surrogate_fourier / np.abs(surrogate_fourier)
            surrogate = irfft(magnitude * phase, n=len(ts))
            surrogate_sorted = np.sort(surrogate)
            surrogate = np.interp(surrogate, surrogate_sorted, sorted_ts)
        
        surrogates[i] = surrogate
        
    return surrogates

def phase_randomized_surrogates(ts, N=100):
    ts_fourier = rfft(ts)
    magnitude = np.abs(ts_fourier)
    surrogates = np.empty((N, len(ts)))
    for i in range(N):
        random_phases = np.exp(2j * np.pi * np.random.random(len(ts_fourier)))
        surrogate_fourier = magnitude * random_phases
        surrogates[i] = irfft(surrogate_fourier, n=len(ts))
    return surrogates

def time_reversed_surrogates(ts, N=100):
    surrogates = np.empty((N, len(ts)))
    reversed_ts = ts[::-1]
    for i in range(N):
        surrogates[i] = reversed_ts
    return surrogates

def AAFT_surrogates(ts, N=100):
    ts_sorted = np.sort(ts)
    ts_fourier = rfft(ts)
    magnitude = np.abs(ts_fourier)
    surrogates = np.empty((N, len(ts)))
    for i in range(N):
        surrogate = np.random.permutation(ts)
        surrogate_fourier = rfft(surrogate)
        phase = surrogate_fourier / np.abs(surrogate_fourier)
        surrogate = irfft(magnitude * phase, n=len(ts))
        surrogate_sorted = np.sort(surrogate)
        surrogates[i] = np.interp(surrogate, surrogate_sorted, ts_sorted)
    return surrogates



# # Define the surrogate functions
# def white_noise_surrogates(ts):
#     return np.random.permutation(ts)

# def correlated_noise_surrogate(ts):
#     ts_fourier = rfft(ts)
#     random_phases = np.exp(2j * np.pi * np.random.random(len(ts_fourier)))
#     surrogate_fourier = np.abs(ts_fourier) * random_phases
#     return irfft(surrogate_fourier, n=len(ts))

# def IAAFT_surrogates(ts, max_iter=1000):
#     sorted_ts = np.sort(ts)
#     surrogate = np.random.permutation(ts)
#     ts_fourier = rfft(ts)
#     for i in range(max_iter):
#         surrogate_fourier = rfft(surrogate)
#         surrogate = irfft(np.abs(ts_fourier) * surrogate_fourier / np.abs(surrogate_fourier), n=len(ts))
#         surrogate = np.interp(np.argsort(np.argsort(surrogate)), np.argsort(np.argsort(sorted_ts)), sorted_ts)
#     return surrogate

# Plotting function
def plot_time_series_and_spectrum(ts, surrogates):
    f, axes = plt.subplots(2, 3, figsize=(15, 6))

    # Original time series
    axes[0, 0].plot(ts, label='Original')
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].legend()

    # Original power spectrum
    f_original, Pxx_original = welch(ts)
    axes[1, 0].semilogy(f_original, Pxx_original, label='Original')
    axes[1, 0].set_title('Power Spectrum (Original)')
    axes[1, 0].legend()

    # White noise surrogate
    axes[0, 1].plot(surrogates['white'], label='White Noise')
    axes[0, 1].set_title('White Noise Surrogate')
    axes[0, 1].legend()

    # White noise power spectrum
    f_white, Pxx_white = welch(surrogates['white'])
    axes[1, 1].semilogy(f_white, Pxx_white, label='White Noise')
    axes[1, 1].set_title('Power Spectrum (White Noise)')
    axes[1, 1].legend()

    # Correlated noise surrogate
    axes[0, 2].plot(surrogates['correlated'], label='Correlated Noise')
    axes[0, 2].set_title('Correlated Noise Surrogate')
    axes[0, 2].legend()

    # Correlated noise power spectrum
    f_corr, Pxx_corr = welch(surrogates['correlated'])
    axes[1, 2].semilogy(f_corr, Pxx_corr, label='Correlated Noise')
    axes[1, 2].set_title('Power Spectrum (Correlated Noise)')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

# Plotting function
def plot_amplitude_and_spectrum(ts, surrogates):
    fig, axes = plt.subplots(2, len(surrogates), figsize=(18, 8))

    # Set titles for columns
    titles = ['White Noise Surrogates', 'Correlated Noise Surrogates', 'IAAFT Surrogates']

    for i, (title, surrogate) in enumerate(surrogates.items()):
        # Amplitude distribution (histogram)
        axes[0, i].hist(surrogate, bins=30, density=True, alpha=0.7, label=title)
        axes[0, i].hist(ts, bins=30, density=True, histtype='step', color='black', linewidth=2, label='Original')
        axes[0, i].set_title(title)
        axes[0, i].set_xlabel('Amplitude')
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()

        # Power spectrum
        f_surrogate, Pxx_surrogate = welch(surrogate)
        axes[1, i].semilogy(f_surrogate[f_surrogate > 0], Pxx_surrogate[f_surrogate > 0], label=title, alpha=0.7)
        f_original, Pxx_original = welch(ts)
        axes[1, i].semilogy(f_original[f_original > 0], Pxx_original[f_original > 0], label='Original', color='black', linestyle='--', linewidth=2)
        axes[1, i].set_title('Power Spectrum (' + title + ')')
        axes[1, i].set_xlabel('Frequency')
        axes[1, i].set_ylabel('PSD')
        axes[1, i].legend()

    plt.tight_layout()
    plt.show()
    
# Function to compute and normalize autocorrelation
def autocorrelation(ts):
    result = correlate(ts, ts, mode='full')
    result = result / np.max(result)  # Normalize the result
    return result[result.size // 2:]  # Return one side

def generate_surrogates(surrogate_type, ts, N):
    surrogate_functions = {
        'white_noise': white_noise_surrogates,
        'correlated_noise': correlated_noise_surrogates,
        'IAAFT': IAAFT_surrogates,
        'AAFT': AAFT_surrogates,
        'phase_randomized': phase_randomized_surrogates,
        'time_reversed': time_reversed_surrogates
    }
    return surrogate_functions[surrogate_type](ts, N)
