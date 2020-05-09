from numpy import fft
from copy import deepcopy

from scipy import signal
from scipy.fftpack import rfft, fftfreq

# ## Fourier Transformation and derived features
#
# The Fourier transformation allows us to convert a timeseries from the time
# domain to the frequency domain. The resulting series shows amplitude as a
# function of frequency. Thus it shows all frequencies that contribute to the
# series. For real input the real part of the result holds the scaling for a
# cosine at the particular frequency, the imaginary part of the result has
# the scaling for a sine at that frequency. The Magnitude of the signal at
# any given point is the absolute value of the transformed value, or the
# power. The other part is the phase part.


def fourier_transform_magnitude(data):
    data_copy = deepcopy(data)
    for col in data_copy.columns[:-2]:
        temp_results = []
        for idx in range(0, 8):
            temp_ft = abs(fft.fft(data_copy[data_copy['id'] == idx][col]) /
                          len(data_copy[data_copy['id'] == idx][col]))
            temp_results.extend(temp_ft)
        data_copy[f'{col}_ft'] = temp_results
    return data_copy

# The second frequency-domain feature set was chosen to be spectral energy,
# which is defined to be the sum of the squared FFT coefficients. Preece,
# Stephen J., John Yannis Goulermas, Laurence PJ Kenney, and David Howard. "A
# comparison of feature extraction methods for the classification of dynamic
# activities from accelerometer data." IEEE Transactions on Biomedical
# Engineering 56, no. 3 (2008): 871-879.


def get_spectral_energy(data):
    data_copy = deepcopy(data)
    for col in data_copy.columns:
        if "ft" in col:
            data_copy[f'{col}_2'] = data_copy[col]**2

    return data_copy



