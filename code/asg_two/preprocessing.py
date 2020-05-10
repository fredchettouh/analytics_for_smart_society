from numpy import fft
from scipy import signal
import numpy as np
from scipy import stats


def fourier_transform(data_window):
    n_vals = len(data_window)
    fft_vals = fft.fft(data_window)
    magnitude = 2 / n_vals * abs(fft_vals[:n_vals // 2])
    return magnitude

def select_top_fourier(data_window, n_vals=3):
    fourier_vals = fourier_transform(data_window)
    top_indexes = fourier_vals.argsort()[-n_vals:][::-1]
    return fourier_vals [top_indexes]

def sample_ft(data_window, threshold=2, num_vals=2):
    fornier_vals = fourier_transform(data_window)
    all_peaks, _ = signal.find_peaks(fornier_vals)
    all_prominences = signal.peak_prominences(fornier_vals, all_peaks)[0]
    try:
        sorted_all_prominences = sorted(all_prominences, reverse=True)
        threshold = sorted_all_prominences[threshold]
        peaks, _ = signal.find_peaks(fornier_vals, prominence=threshold)
        result = np.random.choice(peaks, num_vals)
    except IndexError:
        result = [1000] * num_vals
    finally:
        return result


def power_spectral_entropy(data_window, fs=16, window='boxcar'):
    freq, psd = signal.welch(data_window, fs=fs, window=window)
    sum_psd = sum(psd)
    normalized_psd = psd / sum_psd
    return stats.entropy(normalized_psd)






