import numpy as np

from scipy.signal import detrend

from sklearn.preprocessing import MinMaxScaler

from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec


# =====================================================================
# Auxiliary functions
# =====================================================================
def Energy(x):
    """Computes the energy of a signal. The energy is essentially the
    magnitude of the signal (inner product of x with itself).

    @param x Input signal as numpy array (1D)

    @return The energy of the input signal x
    """
    return np.dot(x, x)


def EuclideanNorm(x):
    """! Computes the Euclidean norm (p-norm with p=2) of the input
    1D vector (signal) x.

    @param x The input signal (numpy float 1D ndarray)

    @return The norm of the input signal as a float scaler
    """
    return np.linalg.norm(x)


def mad(x):
    """! Estimates the Median Absolute Deviation (MAD). MAD is defined to be
    the median of the absolute difference between the input X and median(X).

    @param x The input signal (1D ndarray)
    @return The median absolute deviation of the input signal

    @note More details on the MAD can be found on the Wikipedia page:
    please see https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def meanad(x):
    """! Estimates the Mean Absolute Deviation (MeanAD). MeanAD is defined to
    be the mean of the absolute difference between the input X and mean(X).

    @param x The input signal (1D ndarray)
    @return The mean absolute deviation of the input signal
    """
    return 1.482579 * np.mean(np.abs(x - np.mean(x)))


def grad_g_fun(x, thr=1):
    return (x >= thr) * 1 + (x <= -thr) * 1 + (np.abs(x) <= thr) * 0


def NearestEvenInteger(n):
    """! Returns the nearest even integer to number n.

    @param n Input number for which one requires the nearest even integer

    @return The even nearest integer to the input number
    """
    if n % 2 == 0:
        res = n
    else:
        res = n-1
    return res


def DyadicLength(x):
    """! Returns the length and the dyadic length of the input 1D array x.

    @param x The input signal (float 1D ndarray)

    @return Returns the length m and the least power of 2 greater than m
    """
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.)).astype('i')
    return m, j


def SoftHardThresholding(x, thr=1, method='s'):
    """! Performs either a soft or hard thresholding on the input signal x.

    @param x The 1D input signal
    @param thr The threshold value (float, default=1)
    @param method A string that indicates if either the soft or the hard
    thresholding is being used (default=soft, s for soft, h for hard)

    @return Returns the thresholded signal
    """
    if method.lower() == 'h':
        res = x * (np.abs(x) > thr)
    elif method.lower() == 's':
        res = ((x >= thr) * (x - thr) + (x <= -thr) * (x + thr)
               + (np.abs(x) <= thr) * 0)
    else:
        print("Thresholding method not found! Choose s (soft) or h (hard)")
        res = None
    return res


# =====================================================================
# Main wavelets denoising functions
# =====================================================================
def preprocess(signal, normalize=False, scaler=None):
    """Removes trends and optionally normalizes the signal."""
    xhat = detrend(signal)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        xhat = scaler.fit_transform(xhat.reshape(-1, 1))[:, 0]
        return xhat, scaler
    return xhat, scaler

def std_coeffs(signal, nlevel, level=None):
    """Estimates the standard deviation for wavelet coefficients."""
    if level is None:
        sigma = np.ones((nlevel, ))
        return sigma
    if level > nlevel:
        print("WARNING: The level you set exceeds the nominal value!")
        print(" Level has been replaced by the largest possible value")
        level = nlevel - 1
    elif level == nlevel:
        sigma = np.array([1.4825 * np.median(np.abs(signal[i])) for i in range(nlevel)])
    else:
        tmp_sigma = 1.4825 * np.median(np.abs(signal[nlevel-1]))
        sigma = np.array([tmp_sigma for _ in range(nlevel)])
    return sigma

def wav_transform(signal, wavelet, nlevel):
    """Performs wavelet multilevel decomposition."""
    filter_ = Wavelet(wavelet)
    size = NearestEvenInteger(signal.shape[0])
    if nlevel == 0:
        nlevel = dwt_max_level(signal.shape[0], filter_len=filter_.dec_len)
    coeffs = wavedec(signal[:size], filter_, level=nlevel)
    return coeffs, filter_, nlevel

def universal_threshold(signal, sigma=True):
    m = signal.shape[0]
    if sigma:
        sd = mad(signal)
    else:
        sd = 1.0
    thr = sd * np.sqrt(2 * np.log(m))
    return thr

def stein_threshold(signal):
    m = signal.shape[0]
    sorted_signal = np.sort(np.abs(signal))**2
    c = np.linspace(m-1, 0, m)
    s = np.cumsum(sorted_signal) + c * sorted_signal
    risk = (m - (2.0 * np.arange(m)) + s) / m
    ibest = np.argmin(risk)
    thr = np.sqrt(sorted_signal[ibest])
    return thr

def heurstein_threshold(signal):
    m, j = DyadicLength(signal)
    magic = np.sqrt(2 * np.log(m))
    eta = (np.linalg.norm(signal)**2 - m) / m
    critical = j**(1.5)/np.sqrt(m)
    if eta < critical:
        thr = magic
    else:
        thr = np.min((stein_threshold(signal), magic))
    return thr

def sqrtlog_threshold(signal):
    m = len(signal)
    thr = np.sqrt(2.0 * np.log(m))
    return thr

def energy_threshold(signal, perc=0.1):
    tmp_signal = np.sort(np.abs(signal))[::-1]
    energy_thr = perc * Energy(tmp_signal)
    energy_tmp = 0
    for sig in tmp_signal:
        energy_tmp += sig**2
        if energy_tmp >= energy_thr:
            thr = sig
            break
    return thr

def determine_threshold(signal, method='universal', energy_perc=0.9):
    if method == 'universal':
        return universal_threshold(signal)
    elif method == 'sqtwolog':
        return universal_threshold(signal, sigma=False)
    elif method == 'stein':
        return stein_threshold(signal)
    elif method == 'heurstein':
        return heurstein_threshold(signal)
    elif method == 'energy':
        return energy_threshold(signal, perc=energy_perc)
    else:
        print("No such method detected! Set back to default (universal thresholding)!")
        return universal_threshold(signal)

def denoise(signal, wavelet='haar', level=1, thr_mode='soft', recon_mode='smooth', selected_level=0, method='universal', energy_perc=0.9, normalize=False):
    # Preprocess
    xhat, scaler = preprocess(signal, normalize=normalize)
    # Wavelet transform
    coeffs, filter_, nlevel = wav_transform(xhat, wavelet, level)
    # Estimate SD
    sigma = std_coeffs(coeffs[1:], nlevel, level=selected_level)
    # Thresholds
    thr = [determine_threshold(coeffs[1+lvl] / sigma[lvl], method=method, energy_perc=energy_perc) * sigma[lvl] for lvl in range(nlevel)]
    # Apply threshold
    coeffs[1:] = [threshold(c, value=thr[i], mode=thr_mode) for i, c in enumerate(coeffs[1:])]
    # Reconstruct
    denoised_signal = waverec(coeffs, filter_, mode=recon_mode)
    # Inverse normalization
    if normalize and scaler is not None:
        denoised_signal = scaler.inverse_transform(denoised_signal.reshape(-1, 1))[:, 0]
    return denoised_signal
