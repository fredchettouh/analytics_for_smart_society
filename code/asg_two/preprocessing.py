from numpy import fft
from scipy import signal
import numpy as np
from scipy import stats
from copy import deepcopy


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


def get_spectral_energy(data):
    data_copy = deepcopy(data)
    for col in data_copy.columns:
        if "ft" in col:
            data_copy[f'{col}_2'] = data_copy[col]**2

    return data_copy


def add_features(df):
    mode_results = apply_to_window(df, len(set(df['window_id'])), 'label', stats.mode)
    data = map_back_to_window(df, {key: int(mode_results[key][0][0])for key in mode_results}, 'freq_label')

    for variable in COLNAMES_TRAIN[0:5]:

        mean_results = apply_to_window(data, len(set(df['window_id'])), variable, np.mean)

        data = map_back_to_window(data, {key: mean_results[key]for key in mean_results}, 'mean_'+variable)



        std_results = apply_to_window(data, len(set(df['window_id'])), variable, np.std)
        data = map_back_to_window(data, {key: std_results[key]for key in std_results}, 'std_'+variable)

        mad_results = apply_to_window(data, len(set(df['window_id'])), variable, stats.median_absolute_deviation)
        data = map_back_to_window(data, {key: mad_results[key]for key in mad_results}, 'mad_'+variable)

        min_results = apply_to_window(df, len(set(df['window_id'])), variable, min)
        data = map_back_to_window(data, {key: min_results[key]for key in min_results}, 'min_'+variable)
        max_results = apply_to_window(df, len(set(df['window_id'])), variable, max)
        data = map_back_to_window(data, {key: max_results[key]for key in max_results}, 'max_'+variable)


        var_results = apply_to_window(data, len(set(df['window_id'])), variable, np.var)
        data = map_back_to_window(data, {key: var_results[key]for key in var_results}, 'var_'+variable)

        median_results = apply_to_window(data, len(set(df['window_id'])), variable, np.median)
        data = map_back_to_window(data, {key: median_results[key]for key in median_results}, 'median_'+variable)

    return data



def add_features_test(df):
    #mode_results = apply_to_window(df, len(set(df['window_id'])), 'label', stats.mode)
    #data = map_back_to_window(df, {key: int(mode_results[key][0][0])for key in mode_results}, 'freq_label')

    for variable in COLNAMES_TRAIN[0:5]:

        mean_results = apply_to_window(df, len(set(df['window_id'])), variable, np.mean)

        data = map_back_to_window(df, {key: mean_results[key]for key in mean_results}, 'mean_'+variable)



        std_results = apply_to_window(data, len(set(df['window_id'])), variable, np.std)
        data = map_back_to_window(data, {key: std_results[key]for key in std_results}, 'std_'+variable)

        mad_results = apply_to_window(data, len(set(df['window_id'])), variable, stats.median_absolute_deviation)
        data = map_back_to_window(data, {key: mad_results[key]for key in mad_results}, 'mad_'+variable)

        min_results = apply_to_window(df, len(set(df['window_id'])), variable, min)
        data = map_back_to_window(data, {key: min_results[key]for key in min_results}, 'min_'+variable)
        max_results = apply_to_window(df, len(set(df['window_id'])), variable, max)
        data = map_back_to_window(data, {key: max_results[key]for key in max_results}, 'max_'+variable)


        var_results = apply_to_window(data, len(set(df['window_id'])), variable, np.var)
        data = map_back_to_window(data, {key: var_results[key]for key in var_results}, 'var_'+variable)

        median_results = apply_to_window(data, len(set(df['window_id'])), variable, np.median)
        data = map_back_to_window(data, {key: median_results[key]for key in median_results}, 'median_'+variable)

    return data


def segmentation(data, window_size=16):
    data['index_col'] = data.index
    counter = 0
    data['window_id'] = np.nan

    for (start, end) in windows2(data["index_col"], window_size):

        if (len(data["index_col"][start:end]) == window_size):
            data['window_id'][start:end] = counter
            counter = counter + 1

    data.fillna(method='ffill', inplace=True)
    data.drop(columns='index_col', inplace=True)

    return data


def windows2(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)
