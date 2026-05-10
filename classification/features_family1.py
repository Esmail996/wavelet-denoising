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
# CHANGE (post-92.76% baseline): added find_peaks — required by _tail_features (Group 1, added later)
from scipy.signal import hilbert, butter, sosfiltfilt, find_peaks
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


# CHANGE (post-92.76% baseline): entire _tail_features function is NEW.
# Motivation: Glas misclassified as Box on ~47% of trials in classification_f2.
# Hypothesis: Glas has longer ringdown / secondary reflections; Box echo decays
# faster and more cleanly. These 3 features probe exactly that:
#   - late_to_peak_ratio : how much energy remains 50-250 us after the main peak
#   - n_envelope_peaks   : count of secondary echo bumps (reverb structure)
#   - decay_slope_db_us  : dB/us slope of the envelope after peak (steep = fast decay)
# Result (v3/v4/v5): feature added correctly but Glas->Box error unchanged at 394.
# Net accuracy with this block: 92.4-92.76% (no regression, no improvement on Glas).
def _tail_features(x: np.ndarray, fs: float, eps: float = 1e-12) -> dict[str, float]:
    """Tail-shape features targeted at Glas-vs-Box discrimination.

    Features:
      - late_to_peak_ratio: E[peak+50us : peak+250us] / E[peak-25us : peak+25us]
      - n_envelope_peaks: count of secondary envelope peaks >=25% main peak
                          in [peak-300us : peak+300us]
      - decay_slope_db_us: linear slope of 20*log10(envelope) in [peak : peak+150us]
    """
    x = np.asarray(x, dtype=np.float64)
    # Validate: no NaN/inf, has sufficient length, non-zero energy
    if len(x) < 8 or not np.isfinite(x).all():
        return {
            "late_to_peak_ratio": 0.0,
            "n_envelope_peaks": 0.0,
            "decay_slope_db_us": 0.0,
        }
    
    # Compute envelope safely
    try:
        env = np.abs(hilbert(x))
    except (ValueError, RuntimeError, TypeError):
        # If Hilbert fails (e.g., all zeros), return zeros
        return {
            "late_to_peak_ratio": 0.0,
            "n_envelope_peaks": 0.0,
            "decay_slope_db_us": 0.0,
        }
    
    n = len(env)
    if float(env.max()) < eps:
        return {
            "late_to_peak_ratio": 0.0,
            "n_envelope_peaks": 0.0,
            "decay_slope_db_us": 0.0,
        }

    peak_idx = int(np.argmax(env))
    peak_amp = float(env[peak_idx])

    def _us_to_samp(us: float) -> int:
        return int(round(us * 1e-6 * fs))

    # Main-echo energy around peak (+/-25 us)
    main_lo = max(0, peak_idx - _us_to_samp(25.0))
    main_hi = min(n, peak_idx + _us_to_samp(25.0) + 1)
    main_e = float(np.sum(env[main_lo:main_hi] ** 2))

    # Late-tail energy (peak+50 us to peak+250 us)
    late_lo = max(0, peak_idx + _us_to_samp(50.0))
    late_hi = min(n, peak_idx + _us_to_samp(250.0) + 1)
    late_e = float(np.sum(env[late_lo:late_hi] ** 2)) if late_hi > late_lo else 0.0
    late_to_peak_ratio = float(late_e / (main_e + eps))

    # Count secondary envelope peaks >= 25% main peak around +/-300 us
    search_lo = max(0, peak_idx - _us_to_samp(300.0))
    search_hi = min(n, peak_idx + _us_to_samp(300.0) + 1)
    if search_hi - search_lo >= 3:
        local = env[search_lo:search_hi]
        peaks, _ = find_peaks(local, height=0.25 * peak_amp)
        peaks_abs = peaks + search_lo
        # Exclude the main peak itself to measure secondary structure
        n_secondary = int(np.sum(peaks_abs != peak_idx))
    else:
        n_secondary = 0

    # Decay slope in dB/us from peak to peak+150 us
    dec_lo = peak_idx
    dec_hi = min(n, peak_idx + _us_to_samp(150.0) + 1)
    if dec_hi - dec_lo >= 3:
        y = 20.0 * np.log10(env[dec_lo:dec_hi] + eps)
        x_us = (np.arange(dec_lo, dec_hi) - peak_idx) / fs * 1e6
        slope, _ = np.polyfit(x_us, y, 1)
        decay_slope_db_us = float(slope)
    else:
        decay_slope_db_us = 0.0

    return {
        "late_to_peak_ratio": late_to_peak_ratio,
        "n_envelope_peaks": float(n_secondary),
        "decay_slope_db_us": decay_slope_db_us,
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
    # CHANGE (post-92.76% baseline): include_tail flag added so the per-trial
    # driver can call this function without tail features when it will merge
    # tail features computed on a separate (raw) ROI via Option 2.
    include_tail: bool = True,
) -> dict[str, float]:
    """Extract Family 1 features from one ROI signal.

    Parameters
    ----------
    roi : 1D numpy array of denoised waveform (float64 recommended)
    fs : sample rate in Hz
    carrier_hz : nominal carrier of the Tx generating this echo (40k/50k/60k)
    swt_levels : SWT decomposition depth
    swt_wavelet : pywt wavelet name (sym6 default to match pipeline)
    include_tail : if True, include tail morphology features (Group 1);
                   if False, omit them (for use when computing separately on raw ROI)

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
    # CHANGE (post-92.76% baseline): tail features only appended when include_tail=True.
    # When tail_realign=False in the per-trial driver, this is called with
    # include_tail=False and tail features are merged separately from the raw ROI.
    if include_tail:
        out.update(_tail_features(x, fs))
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
    # CHANGE (post-92.76% baseline): tail_realign parameter added for Option 2
    # experiment. Default False = tail computed on raw TOF-anchored ROI (Option 2).
    # True = tail computed on the same realigned ROI as all other features (Option 1).
    # Tested in v5; result: Glas->Box stayed at 394 regardless of this flag.
    tail_realign: bool = False,
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

    If `tail_realign=False`, Group 1 tail morphology features are computed
    on the raw (unrealigned) TOF-anchored ROI. This preserves box-specific
    tail structure that may be smoothed by realign-to-peak preprocessing.
    All other features use the realigned ROI.
    """
    out = {}
    channel_summary: dict[tuple[str, str], dict[str, float]] = {}
    channel_tail_feats: dict[tuple[str, str], dict[str, float]] = {}
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

            # CHANGE (post-92.76% baseline): Option 2 branch.
            # When tail_realign=False (default), tail features are extracted here
            # from the raw bandpass-filtered, TOF-windowed ROI BEFORE realignment.
            # This preserves absolute timing and amplitude context of the tail.
            # **OPTION 2**: Compute tail features on raw (non-realigned) ROI
            if not tail_realign:
                tail_feats = _tail_features(roi, fs)
                channel_tail_feats[(mic, tx)] = tail_feats

            # Apply preprocessing (realign + normalise) for all other features
            roi_processed = preprocess_roi(roi, realign=realign, normalise=normalise)
            # CHANGE (post-92.76% baseline): include_tail=tail_realign so that when
            # Option 2 is active (tail_realign=False) the tail is NOT recomputed here.
            feats = extract_family1_features(roi_processed, fs=fs, carrier_hz=fc,
                                             include_tail=tail_realign)
            channel_summary[(mic, tx)] = {
                "env_peaktime_us": float(feats.get("env_peaktime_us", 0.0)),
                "centroid_hz": float(feats.get("centroid_hz", 0.0)),
            }
            for k, v in feats.items():
                out[f"{mic}_{tx}_{k}"] = v

            # CHANGE (post-92.76% baseline): merge raw-ROI tail features into
            # the output dict, overwriting any tail keys from the preprocessed path.
            # Merge tail features computed on raw ROI (if tail_realign=False)
            if not tail_realign:
                for k, v in tail_feats.items():
                    out[f"{mic}_{tx}_{k}"] = v

    # CHANGE (post-92.76% baseline): entire Group 2 block is NEW.
    # Adds cross-mic timing and frequency centroid offsets per Tx.
    # Hypothesis: off-axis material response is angle-asymmetric; the difference
    # between Mic1 and Mic2 arrival times / spectral centroids encodes geometry
    # that helps disambiguate Glas (more specular, angle-sensitive) from Box.
    # Net effect on 92.76% run: included by default; did not close Glas->Box gap.
    # Group 2: cross-mic off-axis signatures per Tx
    if len(mics) >= 2:
        if "Mic1" in mics and "Mic2" in mics:
            m_a, m_b = "Mic1", "Mic2"
        else:
            m_a, m_b = mics[0], mics[1]

        for tx in txs:
            a = channel_summary.get((m_a, tx))
            b = channel_summary.get((m_b, tx))
            if a is None or b is None:
                continue
            # CHANGE (post-92.76% baseline): these two cross-mic delta features
            # are new additions (Group 2). They did not exist in the original
            # baseline feature set that achieved 93.3%.
            out[f"peak_time_offset_M1_vs_M2_{tx}_us"] = (
                a["env_peaktime_us"] - b["env_peaktime_us"]
            )
            out[f"freq_centroid_shift_M1_vs_M2_{tx}_hz"] = (
                a["centroid_hz"] - b["centroid_hz"]
            )
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
