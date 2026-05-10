"""
features_family1.py — Family 1 statistical features per (Mic, Tx) ROI.

Designed to run on your machine where pywt is installed. A numpy-only
fallback is provided for sandbox verification.

For each (object, distance, angle, trial), and for each (Mic, Tx) channel
pairing, this produces a fixed-length feature vector containing:

    1. Per-SWT-subband statistics: energy, log-energy, kurtosis,
       Shannon entropy, sparsity (||c||_1 / ||c||_2 ratio), peak
       amplitude, peak position normalised.
    2. Hilbert envelope features: peak amplitude, peak time relative
       to ROI start, FWHM, rise time (10%->90% of peak), decay time
       (peak->50%), envelope kurtosis.
    3. Spectral features (FFT on ROI): spectral centroid, spread,
       skewness, kurtosis, dominant frequency, energy in carrier band
       (±2kHz), out-of-band energy ratio.

Total per (Mic, Tx) ≈ 6 SWT levels × 7 stats + 6 Hilbert + 7 spectral
                    = 42 + 6 + 7 = 55 features.
With 2 mics × 3 Tx = 6 (Mic, Tx) cells → 330 features per trial.
(If you also use Mic3 → 9 cells → ~500 features per trial.)

USAGE (your machine):
    from features_family1 import extract_family1_features
    feats = extract_family1_features(
        roi_signal,            # 1D numpy array of denoised waveform
        fs=2_000_000,
        carrier=50_000,        # for spectral band-power
    )

The signature is per-channel. For a full trial with 6 channels, call this
6 times and concatenate the 6 vectors with channel-prefixed names.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

from roi_preprocessing import preprocess_roi


# -----------------------------------------------------------------
# Wavelet decomposition — production uses pywt; fallback for sandbox
# -----------------------------------------------------------------

def _swt_pywt(x: np.ndarray, wavelet: str = "sym6", level: int = 5):
    """SWT via pywt — preferred on your machine."""
    import pywt
    # pywt SWT requires the signal length to be divisible by 2**level.
    n = len(x)
    pad_n = ((n + 2**level - 1) // 2**level) * 2**level
    if pad_n != n:
        x_pad = np.zeros(pad_n)
        x_pad[:n] = x
    else:
        x_pad = x
    coeffs = pywt.swt(x_pad, wavelet, level=level, trim_approx=False, norm=False)
    # pywt returns [(cA_J, cD_J), ..., (cA_1, cD_1)]
    # Trim back to original length
    out = []
    for cA, cD in coeffs:
        out.append((cA[:n], cD[:n]))
    return out


def _swt_atrous_numpy(x: np.ndarray, wavelet_name: str = "sym6", level: int = 5):
    """à trous SWT in pure numpy — fallback for sandbox verification.

    Uses a fixed sym6-like filter (Daubechies symlets-6 low-pass coefficients).
    Not bit-identical to pywt; for development/verification only.
    """
    sym6_lo = np.array([
         0.01540410932703,  0.00349071208422, -0.11799011114820,
        -0.04831174450184,  0.49105594192674,  0.78764114443247,
         0.33792942172298, -0.07263752278647, -0.02106139571175,
         0.04472490177067,  0.00176815715318, -0.00782787221008,
    ])
    lo = sym6_lo / np.sqrt((sym6_lo**2).sum())
    hi = lo[::-1] * (-1)**np.arange(len(lo))
    n = len(x)
    coeffs = []
    a = x.copy()
    for j in range(level):
        # Upsample filters with à trous holes for level j
        gap = 2**j
        f_lo = np.zeros((len(lo)-1) * gap + 1)
        f_lo[::gap] = lo
        f_hi = np.zeros((len(hi)-1) * gap + 1)
        f_hi[::gap] = hi
        a_new = np.convolve(a, f_lo, mode="same")
        d = np.convolve(a, f_hi, mode="same")
        coeffs.append((a_new, d))
        a = a_new
    coeffs.reverse()  # match pywt order: deepest first
    return coeffs


def _swt(x, wavelet="sym6", level=5):
    try:
        return _swt_pywt(x, wavelet, level)
    except ImportError:
        return _swt_atrous_numpy(x, wavelet, level)


# -----------------------------------------------------------------
# Per-array statistical helpers
# -----------------------------------------------------------------

def _shannon_entropy(c: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of normalised squared coefficients."""
    p = c**2
    p_sum = p.sum()
    if p_sum < eps:
        return 0.0
    p = p / p_sum
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def _sparsity(c: np.ndarray, eps: float = 1e-12) -> float:
    """L1/L2 ratio (sqrt(N) for uniform; 1 for impulse). Normalised to [0,1]
    by dividing by sqrt(N) so 1=dense, 0=spiky."""
    l1 = np.abs(c).sum()
    l2 = np.sqrt((c**2).sum() + eps)
    return float(l1 / (l2 * np.sqrt(len(c))))


def _kurtosis(c: np.ndarray) -> float:
    return float(scipy_kurtosis(c, fisher=True, bias=False))


def _peak_features(c: np.ndarray) -> tuple[float, float]:
    """Peak amplitude and peak position normalised to [0,1]."""
    if len(c) == 0:
        return 0.0, 0.0
    idx = int(np.argmax(np.abs(c)))
    return float(np.abs(c[idx])), float(idx) / max(1, len(c) - 1)


def _swt_subband_features(coeffs) -> dict[str, float]:
    """For each (cA, cD) level: 7 stats per detail subband and approx subband.
    Returns flat dict keyed level/type/stat."""
    out = {}
    for j, (cA, cD) in enumerate(coeffs):
        for label, c in [(f"cD{j+1}", cD), (f"cA{j+1}", cA)] if j == 0 else [(f"cD{j+1}", cD)]:
            energy = float((c**2).sum())
            out[f"{label}_energy"] = energy
            out[f"{label}_logenergy"] = float(np.log1p(energy))
            out[f"{label}_kurt"] = _kurtosis(c)
            out[f"{label}_shannon"] = _shannon_entropy(c)
            out[f"{label}_sparsity"] = _sparsity(c)
            amp, pos = _peak_features(c)
            out[f"{label}_peakamp"] = amp
            out[f"{label}_peakpos"] = pos
    return out


# -----------------------------------------------------------------
# Hilbert envelope features
# -----------------------------------------------------------------

def _envelope_features(x: np.ndarray, fs: float) -> dict[str, float]:
    """Peak time, FWHM, 10–90 rise, peak–50% decay, envelope kurtosis."""
    env = np.abs(hilbert(x))
    if env.max() < 1e-12:
        return {f"env_{k}": 0.0 for k in ("peakamp", "peaktime", "fwhm", "rise", "decay", "kurt")}
    peak_idx = int(np.argmax(env))
    peak_amp = float(env[peak_idx])
    n = len(env)

    half = peak_amp / 2.0
    above = env >= half
    if above.any():
        i_l = int(np.argmax(above))
        i_r = n - 1 - int(np.argmax(above[::-1]))
        fwhm_samp = max(0, i_r - i_l)
    else:
        fwhm_samp = 0

    p10 = peak_amp * 0.10
    p90 = peak_amp * 0.90
    above10_pre = env[:peak_idx + 1] >= p10
    above90_pre = env[:peak_idx + 1] >= p90
    rise_samp = 0
    if above10_pre.any() and above90_pre.any():
        i10 = int(np.argmax(above10_pre))
        i90 = int(np.argmax(above90_pre))
        rise_samp = max(0, i90 - i10)

    p50 = peak_amp * 0.50
    above50_post = env[peak_idx:] >= p50
    decay_samp = 0
    if above50_post.any():
        below_after = ~above50_post
        if below_after.any():
            decay_samp = int(np.argmax(below_after))

    return {
        "env_peakamp": peak_amp,
        "env_peaktime_us": peak_idx / fs * 1e6,
        "env_fwhm_us": fwhm_samp / fs * 1e6,
        "env_rise_us": rise_samp / fs * 1e6,
        "env_decay_us": decay_samp / fs * 1e6,
        "env_kurt": _kurtosis(env),
    }


# -----------------------------------------------------------------
# Spectral features
# -----------------------------------------------------------------

def _spectral_features(x: np.ndarray, fs: float, carrier_hz: float,
                       band_hz: float = 2000.0) -> dict[str, float]:
    """Spectral moments + carrier-band power ratio."""
    n = len(x)
    if n < 4:
        return {k: 0.0 for k in ("centroid_hz", "spread_hz", "skew", "kurt",
                                 "dominant_hz", "carrier_band_frac",
                                 "out_of_band_frac")}
    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(X)
    p = mag**2
    if p.sum() < 1e-18:
        return {k: 0.0 for k in ("centroid_hz", "spread_hz", "skew", "kurt",
                                 "dominant_hz", "carrier_band_frac",
                                 "out_of_band_frac")}
    p_norm = p / p.sum()
    centroid = float((f * p_norm).sum())
    spread = float(np.sqrt(((f - centroid)**2 * p_norm).sum()))
    if spread > 1e-9:
        sk = float(((f - centroid)**3 * p_norm).sum() / spread**3)
        kt = float(((f - centroid)**4 * p_norm).sum() / spread**4) - 3.0
    else:
        sk, kt = 0.0, 0.0
    dominant_hz = float(f[int(np.argmax(p))])

    band_mask = (f >= carrier_hz - band_hz) & (f <= carrier_hz + band_hz)
    carrier_frac = float(p[band_mask].sum() / p.sum())
    bp_mask = (f >= 25_000) & (f <= 75_000)
    out_of_band_frac = float(p[~bp_mask].sum() / p.sum())

    return {
        "centroid_hz": centroid,
        "spread_hz": spread,
        "skew": sk,
        "kurt": kt,
        "dominant_hz": dominant_hz,
        "carrier_band_frac": carrier_frac,
        "out_of_band_frac": out_of_band_frac,
    }


# -----------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------

def extract_family1_features(
    roi: np.ndarray,
    fs: float,
    carrier_hz: float,
    swt_levels: int = 5,
    swt_wavelet: str = "sym6",
) -> dict[str, float]:
    """Extract ~55 features from one ROI signal.

    Parameters
    ----------
    roi : 1D numpy array of denoised waveform (float64 recommended)
    fs : sample rate in Hz
    carrier_hz : nominal carrier of the Tx generating this echo (40k/50k/60k)
    swt_levels : SWT decomposition depth
    swt_wavelet : pywt wavelet name (sym6 default to match pipeline)

    Returns
    -------
    dict of {feature_name: float}
    """
    x = np.asarray(roi, dtype=np.float64)
    coeffs = _swt(x, swt_wavelet, swt_levels)
    out = {}
    out.update(_swt_subband_features(coeffs))
    out.update(_envelope_features(x, fs))
    out.update(_spectral_features(x, fs, carrier_hz))
    return out


# -----------------------------------------------------------------
# Per-trial driver: extract for all (Mic, Tx) channels and concatenate
# -----------------------------------------------------------------

def extract_family1_for_trial(
    trial_signals: dict[str, np.ndarray],   # {Mic1: array, Mic2: array, Mic3: array}
    tof_per_pair_s: dict[tuple[str, str], float],  # {(Mic, Tx): tof_corrected_s}
    fs: float,
    carrier_per_tx: dict[str, float],
    tx_half_bw_hz: dict[str, float] | None = None,
    roi_half_us: float = 200.0,
    mics: Sequence[str] = ("Mic1", "Mic2"),
    txs: Sequence[str] = ("Tx1", "Tx5", "Tx8"),
    realign: bool = True,
    normalise: bool = True,
) -> dict[str, float]:
    """For one trial, extract Family 1 features per (Mic, Tx) channel and
    concatenate with channel-prefixed names.

    The ROI for each (Mic, Tx) is centred on the corrected TOF arrival,
    width ±roi_half_us. ROI is band-passed through the carrier ±3 kHz to
    suppress out-of-band content of unrelated Tx.

    If `realign`, each ROI is re-centred on its envelope peak before
    feature extraction (eliminates residual TOF mis-alignment).
    If `normalise`, each ROI is divided by its RMS energy (removes the
    1/R² distance amplitude effect; features encode SHAPE only).
    """
    out = {}
    n_samp = len(next(iter(trial_signals.values())))
    half_samp = int(roi_half_us * 1e-6 * fs)
    tx_half_bw_hz = tx_half_bw_hz or {}

    for mic in mics:
        sig_full = np.asarray(trial_signals[mic], dtype=np.float64)
        for tx in txs:
            tof = tof_per_pair_s[(mic, tx)]
            fc = carrier_per_tx[tx]
            half_bw = float(tx_half_bw_hz.get(tx, 3000.0))
            # Band-pass around each Tx carrier using configurable half-bandwidth
            sos = butter(4, [fc - half_bw, fc + half_bw], btype="band",
                         fs=fs, output="sos")
            sig = sosfiltfilt(sos, sig_full)
            # ROI window centred on TOF
            centre = int(round(tof * fs))
            lo = max(0, centre - half_samp)
            hi = min(n_samp, centre + half_samp)
            roi = sig[lo:hi]
            # Apply preprocessing
            roi = preprocess_roi(roi, realign=realign, normalise=normalise)
            feats = extract_family1_features(roi, fs=fs, carrier_hz=fc)
            for k, v in feats.items():
                out[f"{mic}_{tx}_{k}"] = v
    return out


# -----------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------

if __name__ == "__main__":
    fs = 2_000_000
    np.random.seed(0)
    t = np.arange(800) / fs
    burst = np.sin(2 * np.pi * 50_000 * t) * np.hanning(800)
    sig = np.zeros(2000)
    sig[800:1600] = burst
    sig += 0.05 * np.random.randn(2000)
    feats = extract_family1_features(sig, fs=fs, carrier_hz=50_000)
    print(f"Extracted {len(feats)} features:")
    for k, v in list(feats.items())[:20]:
        print(f"  {k}: {v:.4f}")
    print("...")
