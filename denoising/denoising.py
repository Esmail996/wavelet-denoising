"""Denoising utilities for the Multifrequenz pipeline.

This file extends the original DWT-based denoising functions with two new
functions that implement the parallel denoising branches of the analysis:

    Branch A - `swt_denoise_bayes`
        Stationary (undecimated) wavelet transform with sym6 J=7,
        BayesShrink threshold per detail level, garrote shrinkage,
        optional level-5 attenuation to protect the carrier band.
        Strength: shift-invariance preserves transient shape.

    Branch B - `wpt_denoise_bayes`
        Wavelet Packet Transform with sym6 J=7, BayesShrink per packet,
        garrote shrinkage, optional attenuation on the three carrier
        packets (40, 50, 60 kHz).
        Strength: 7.8 kHz packet bandwidth at fs=2 MHz separates the
        three carriers into distinct packets, sharper SNR at long range.

The original `denoise` function is preserved for back-compat and ablation.
"""
import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec
from pywt import swt, iswt        # SWT functions added for Branch A
from pywt import WaveletPacket    # WPT for Branch B


EPS = 1e-12


# =====================================================================
# Auxiliary functions  (UNCHANGED from your original)
# =====================================================================
def Energy(x):
    """Computes the energy of a signal."""
    return np.dot(x, x)


def EuclideanNorm(x):
    """Computes the Euclidean norm (p-norm with p=2)."""
    return np.linalg.norm(x)


def mad(x):
    """Estimates the Median Absolute Deviation. The 1.482579 factor is
    the asymptotic conversion to a Gaussian std."""
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def NearestEvenInteger(n):
    if n % 2 == 0:
        return n
    return n - 1


def DyadicLength(x):
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.)).astype('i')
    return m, j


# =====================================================================
# Original DWT-based denoising functions  (UNCHANGED)
# =====================================================================
def preprocess(signal, normalize=False, scaler=None):
    """Removes trends and optionally normalizes the signal."""
    xhat = signal - np.mean(signal)
    xhat = detrend(xhat)
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        xhat = scaler.fit_transform(xhat.reshape(-1, 1))[:, 0]
        return xhat, scaler
    return xhat, scaler


def std_coeffs(signal, nlevel, level=None):
    if level is None:
        return np.ones((nlevel,))
    if level > nlevel:
        print("WARNING: The level you set exceeds the nominal value!")
        print(" Level has been replaced by the largest possible value")
        level = nlevel - 1
    if level == nlevel:
        return np.array([1.4825 * np.median(np.abs(signal[i])) for i in range(nlevel)])
    tmp_sigma = 1.4825 * np.median(np.abs(signal[nlevel - 1]))
    return np.array([tmp_sigma for _ in range(nlevel)])


def wav_transform(signal, wavelet, nlevel):
    filter_ = Wavelet(wavelet)
    size = NearestEvenInteger(signal.shape[0])
    if nlevel == 0:
        nlevel = dwt_max_level(signal.shape[0], filter_len=filter_.dec_len)
    coeffs = wavedec(signal[:size], filter_, level=nlevel)
    return coeffs, filter_, nlevel


def universal_threshold(signal, sigma=True):
    m = signal.shape[0]
    sd = mad(signal) if sigma else 1.0
    return sd * np.sqrt(2 * np.log(m))


def stein_threshold(signal):
    m = signal.shape[0]
    sorted_signal = np.sort(np.abs(signal)) ** 2
    c = np.linspace(m - 1, 0, m)
    s = np.cumsum(sorted_signal) + c * sorted_signal
    risk = (m - (2.0 * np.arange(m)) + s) / m
    ibest = np.argmin(risk)
    return np.sqrt(sorted_signal[ibest])


def heurstein_threshold(signal):
    m, j = DyadicLength(signal)
    magic = np.sqrt(2 * np.log(m))
    eta = (np.linalg.norm(signal) ** 2 - m) / m
    critical = j ** (1.5) / np.sqrt(m)
    if eta < critical:
        return magic
    return np.min((stein_threshold(signal), magic))


def sqrtlog_threshold(signal):
    return np.sqrt(2.0 * np.log(len(signal)))


def energy_threshold(signal, perc=0.1):
    tmp_signal = np.sort(np.abs(signal))[::-1]
    energy_thr = perc * Energy(tmp_signal)
    energy_tmp = 0.0
    thr = 0.0
    for sig in tmp_signal:
        energy_tmp += sig ** 2
        if energy_tmp >= energy_thr:
            thr = sig
            break
    return thr


def determine_threshold(signal, method='universal', energy_perc=0.9):
    if method == 'universal':
        return universal_threshold(signal)
    if method == 'sqtwolog':
        return sqrtlog_threshold(signal)
    if method == 'stein':
        return stein_threshold(signal)
    if method == 'heurstein':
        return heurstein_threshold(signal)
    if method == 'energy':
        return energy_threshold(signal, perc=energy_perc)
    print("No such method detected! Set back to default (universal thresholding)!")
    return universal_threshold(signal)


def denoise(signal, wavelet='haar', level=1, thr_mode='soft',
            recon_mode='smooth', selected_level=0, method='universal',
            energy_perc=0.9, normalize=False):
    """Original DWT-based denoising. Preserved for ablation and back-compat."""
    xhat, scaler = preprocess(signal, normalize=normalize)
    coeffs, filter_, nlevel = wav_transform(xhat, wavelet, level)
    sigma = std_coeffs(coeffs[1:], nlevel, level=selected_level)
    thr = [
        determine_threshold(coeffs[1 + lvl] / sigma[lvl], method=method,
                            energy_perc=energy_perc) * sigma[lvl]
        for lvl in range(nlevel)
    ]
    coeffs[1:] = [
        threshold(c, value=thr[i], mode=thr_mode) for i, c in enumerate(coeffs[1:])
    ]
    denoised_signal = waverec(coeffs, filter_, mode=recon_mode)
    if normalize and scaler is not None:
        denoised_signal = scaler.inverse_transform(denoised_signal.reshape(-1, 1))[:, 0]
    return denoised_signal


# =====================================================================
# Branch A: SWT + BayesShrink + garrote
# =====================================================================
def _bayes_threshold(detail, sigma_n):
    """BayesShrink per-subband threshold (Chang, Yu, Vetterli 2000).

    sigma_y2 = mean(d^2)                              total subband variance
    sigma_x  = sqrt(max(0, sigma_y2 - sigma_n^2))     signal std
    lambda   = sigma_n^2 / sigma_x

    When the subband contains no signal (sigma_x -> 0), the threshold
    blows up and forces the entire subband to zero, which is what we want.
    We cap the returned threshold at 100 * max|d| so log printouts stay
    readable; the cap never affects shrinkage on coefficients that would
    survive anyway.
    """
    sigma_y2 = float(np.mean(detail ** 2))
    sigma_x = np.sqrt(max(sigma_y2 - sigma_n ** 2, 1e-18))
    lam = sigma_n ** 2 / max(sigma_x, 1e-12)
    return min(lam, 100.0 * float(np.max(np.abs(detail))) + 1e-9)


def _garrote_shrink(w, lam):
    """Non-negative garrote shrinkage (Gao 1998):
        shrink(w) = (w - lambda^2 / w) * 1{|w| > lambda}
    Coefficients above the threshold are shrunk by an amount that decays
    to zero as |w| grows large. Below the threshold they are killed.
    Continuous and asymptotically unbiased — preferred over soft and hard
    for transient ultrasonic echoes.

    PyWavelets >= 1.1 has 'garrote' built into pywt.threshold, so we
    delegate to that for numerical consistency. If you have an older
    version, the inline implementation below is mathematically identical:

        out = np.zeros_like(w)
        mask = np.abs(w) > lam
        out[mask] = w[mask] - (lam ** 2) / w[mask]
        return out
    """
    return threshold(w, value=lam, mode='garrote')


def swt_denoise_bayes(signal, wavelet='sym6', level=7,
                      noise_level=4, level5_attenuation=0.5,
                      level_scale=None,
                      sigma_n_fixed=None,
                      return_diagnostics=False):
    """Branch A denoiser: SWT with BayesShrink and garrote shrinkage.

    Parameters
    ----------
    signal : 1-D array-like
        Input signal, typically the wideband-filtered trial after the
        Butterworth-6 25-75 kHz front-end.
    wavelet : str
        PyWavelets wavelet name. Default 'sym6'. Use 'db6' if you prefer
        the CSIC Madrid convention (matches Pardo et al. ultrasonic NDE
        literature). The two perform within 1-2 dB of each other.
    level : int
        Number of SWT decomposition levels. Default 7. At fs = 2 MHz this
        gives band 5 = 31.25-62.5 kHz, which contains all three carriers
        (40, 50, 60 kHz).
    noise_level : int
        Which SWT level to use for the MAD-based noise estimator. Default 4.
        After a 25-75 kHz wideband front-end, levels 1-3 (covering 250 kHz
        to 1 MHz) are essentially empty so MAD gives 0 there. Level 4
        covers 62.5-125 kHz, inside the passband but outside the carrier
        band, so it contains noise but no signal.
    level5_attenuation : float
        Multiplier applied to the BayesShrink threshold at level 5 only.
        Ignored when `level_scale` is provided.
    level_scale : dict[int, float] | None
        Per-level threshold scale factors: ``{physical_level: scale}``.  The
        effective threshold becomes ``scale * BayesShrink_lambda``.  Levels
        absent from the dict default to 1.0 (no scaling).  When provided,
        this takes precedence over `level5_attenuation`.
        Example:  ``{4: 0.8, 5: 0.5, 6: 0.3}``
    sigma_n_fixed : float | None
        If provided, this fixed noise standard deviation is used directly
        for BayesShrink thresholding and the MAD/noise-level estimate is
        skipped.
    return_diagnostics : bool
        If True, returns (denoised, diagnostics_dict) instead of just
        the denoised signal. Useful for thesis figures and ablations.

    Returns
    -------
    denoised : 1-D ndarray
        Denoised signal, same length as input.
    diagnostics : dict (only if return_diagnostics=True)
        Contains 'sigma_n' (estimated noise std), 'thresholds' (per-level
        lambda values used), and 'wavelet' for record-keeping.

    Notes
    -----
    PyWavelets `swt` requires the input length to be divisible by 2^level.
    For an 18000-sample trial at level=7, the next multiple is 18048, so
    we pad with 48 zero samples at the end and trim them off after
    reconstruction. The padding does not contaminate the output because
    the trim happens before any downstream processing sees the signal.
    """
    x = np.asarray(signal, dtype=np.float64)
    n_orig = len(x)

    # SWT length requirement: x must be divisible by 2^level
    pad = (-n_orig) % (2 ** level)
    if pad:
        x_padded = np.concatenate([x, np.zeros(pad)])
    else:
        x_padded = x

    # SWT decomposition. PyWavelets `swt` with trim_approx=True returns
    # [cA_J, cD_J, cD_{J-1}, ..., cD_1] -- approximation first, then
    # detail levels from coarsest to finest.
    coeffs = swt(x_padded, wavelet, level=level, trim_approx=True, norm=True)

    # Identify the indices in the coeffs list. coeffs[0] is the approximation;
    # coeffs[1] is cD_J (coarsest detail); coeffs[-1] is cD_1 (finest detail).
    # We need to map "physical level k" to the right list index:
    #   physical level k  -->  index = level - k + 1
    #   level 1 (finest)  -->  index = level
    #   level 7 (coarsest)-->  index = 1
    cA = coeffs[0]
    detail_indices = list(range(1, level + 1))  # 1..level

    # Use caller-provided fixed sigma_n when available. Otherwise fall back
    # to MAD on the selected noise level.
    noise_idx = level - noise_level + 1
    if sigma_n_fixed is not None:
        sigma_n = float(sigma_n_fixed)
    else:
        sigma_n = mad(coeffs[noise_idx])

    # Build per-level thresholds and apply garrote shrinkage
    thresholds = {}
    new_coeffs = [cA]  # approximation passes through unchanged
    for idx in detail_indices:
        physical_level = level - idx + 1
        d = coeffs[idx]
        lam = _bayes_threshold(d, sigma_n)
        if level_scale is not None:
            lam = lam * level_scale.get(physical_level, 1.0)
        elif physical_level == 5:
            lam = lam * level5_attenuation
        thresholds[physical_level] = lam
        new_coeffs.append(_garrote_shrink(d, lam))

    # Reconstruct via inverse SWT
    rec = iswt(new_coeffs, wavelet, norm=True)

    # Trim back to original length
    out = np.asarray(rec)[:n_orig]

    if return_diagnostics:
        diag = {
            'sigma_n': float(sigma_n),
            'sigma_n_source': 'fixed' if sigma_n_fixed is not None else 'mad_noise_level',
            'noise_idx': int(noise_idx),
            'thresholds': {k: float(v) for k, v in thresholds.items()},
            'wavelet': wavelet,
            'level': level,
            'level5_attenuation': level5_attenuation,
            'level_scale': level_scale,
        }
        return out, diag
    return out


# =====================================================================
# Branch B: Wavelet Packet Transform + BayesShrink + garrote
# =====================================================================
def _natural_to_freq_order(level):
    """Convert WPT natural-order index to frequency-ascending order.

    PyWavelets `WaveletPacket.get_level(level, order='natural')` returns
    leaves in binary-tree natural order. This is not strictly frequency-
    ascending because high-pass branches invert ordering recursively.

    Frequency ordering corresponds to Gray-code indices:
        gray(k) = k XOR (k >> 1)
    """
    pairs = []
    n_leaves = 2 ** level
    for natural_idx in range(n_leaves):
        gray = natural_idx ^ (natural_idx >> 1)
        pairs.append((gray, natural_idx))
    pairs.sort()
    return pairs


def _packet_paths_in_freq_order(level):
    """Return WPT leaf node paths sorted by ascending centre frequency."""
    pairs = _natural_to_freq_order(level)
    paths = []
    for _, natural_idx in pairs:
        bits = format(natural_idx, f'0{level}b')
        path = ''.join('a' if b == '0' else 'd' for b in bits)
        paths.append(path)
    return paths


def _carrier_packet_indices(carriers_hz, fs, level):
    """Return frequency-ordered packet indices that contain each carrier."""
    pkt_bw = fs / (2 ** (level + 1))
    out = {}
    for f0 in carriers_hz:
        idx = int(round(f0 / pkt_bw - 0.5))
        idx = max(0, min(2 ** level - 1, idx))
        out[f0] = idx
    return out


def wpt_denoise_bayes(
    signal,
    wavelet='sym6',
    level=7,
    fs=2_000_000.0,
    carriers_hz=(40_000.0, 50_000.0, 60_000.0),
    carrier_attenuation=0.5,
    noise_packet_idx=None,
    return_diagnostics=False,
):
    """Branch B denoiser: WPT with BayesShrink and garrote shrinkage."""
    x = np.asarray(signal, dtype=np.float64)
    n_orig = len(x)
    pad = (-n_orig) % (2 ** level)
    if pad:
        x_padded = np.concatenate([x, np.zeros(pad)])
    else:
        x_padded = x

    wp = WaveletPacket(data=x_padded, wavelet=wavelet, mode='symmetric', maxlevel=level)

    freq_paths = _packet_paths_in_freq_order(level)
    n_pkts = len(freq_paths)

    carrier_idx = _carrier_packet_indices(carriers_hz, fs, level)
    carrier_idx_set = set(carrier_idx.values())

    if noise_packet_idx is None:
        pkt_bw = fs / (2 ** (level + 1))
        candidates = [
            i for i in range(n_pkts)
            if 25_000 <= (i + 0.5) * pkt_bw <= 75_000 and i not in carrier_idx_set
        ]
        noise_packet_idx = candidates[-1] if candidates else 9

    noise_packet_idx = int(max(0, min(n_pkts - 1, int(noise_packet_idx))))
    noise_data = wp[freq_paths[noise_packet_idx]].data
    sigma_n = mad(noise_data)

    per_packet_thresholds = {}
    for freq_idx, path in enumerate(freq_paths):
        d = wp[path].data
        lam = _bayes_threshold(d, sigma_n)
        if freq_idx in carrier_idx_set:
            lam = lam * carrier_attenuation
        per_packet_thresholds[freq_idx] = lam
        wp[path].data = _garrote_shrink(d, lam)

    rec = wp.reconstruct(update=False)
    out = np.asarray(rec)[:n_orig]

    if return_diagnostics:
        carrier_thresholds = {
            f'{f / 1000:.0f}kHz_pkt{i}': per_packet_thresholds[i]
            for f, i in carrier_idx.items()
        }
        diag = {
            'sigma_n': float(sigma_n),
            'noise_packet_idx': int(noise_packet_idx),
            'carrier_packet_indices': {f'{f / 1000:.0f}kHz': i for f, i in carrier_idx.items()},
            'carrier_packet_thresholds': {k: float(v) for k, v in carrier_thresholds.items()},
            'all_packet_thresholds': {k: float(v) for k, v in per_packet_thresholds.items()},
            'wavelet': wavelet,
            'level': level,
            'packet_bandwidth_hz': float(fs / (2 ** (level + 1))),
            'carrier_attenuation': float(carrier_attenuation),
        }
        return out, diag
    return out

