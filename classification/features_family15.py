"""
features_family15.py — Family 1.5 features.

Two new feature groups designed to capture material-discriminating
information that Family 1 misses:

    GROUP A — Cepstral coefficients (per channel).
        13 real-cepstrum coefficients per (Mic, Tx). The cepstrum
        captures the spectral envelope shape independent of overall
        amplitude — it is the standard tool in audio classification
        for distinguishing materials by their formant-like resonances.

        13 × 6 channels = 78 features per trial.

    GROUP B — Cross-channel ratios.
        For each Tx, the log-ratios of energy and envelope-peak amplitude
        across the two mics carry angular signature.
        For each Mic, the log-ratios across the three Tx carry
        frequency-dependent material signature.

        2 ratios per Tx × 3 Tx (Mic1/Mic2) = 6
        + 2 stats per Mic × 2 Mics × 3 Tx-pairs = 12
        Total ≈ 18 features per trial.

Combined Family 1.5: ~96 new features that complement Family 1's
within-channel statistics with cross-channel and spectral-envelope info.

USAGE:
    from features_family15 import (
        extract_cepstral_features,
        extract_cross_channel_features,
        extract_family15_for_trial,
    )

    f15 = extract_family15_for_trial(
        trial_signals, tof_per_pair_s, fs, carrier_per_tx,
    )

The signature matches features_family1.extract_family1_for_trial so
the driver can call them in parallel.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from scipy.signal import butter, sosfiltfilt, hilbert

from roi_preprocessing import preprocess_roi


# =================================================================
# GROUP A — Cepstral coefficients
# =================================================================

def real_cepstrum(x: np.ndarray, n_coeffs: int = 13,
                  eps: float = 1e-10) -> np.ndarray:
    """Return the first n_coeffs of the real cepstrum of x.

    Real cepstrum: c[n] = IFFT(log(|FFT(x)|))[n]
    Captures spectral envelope shape independent of amplitude and phase.
    The first coefficient (c[0]) is the log-energy; coefficients 1..n
    encode the slow-varying spectral envelope.
    """
    n = len(x)
    if n < 2 * n_coeffs:
        return np.zeros(n_coeffs, dtype=np.float64)
    # Apply window to reduce spectral leakage
    w = np.hanning(n) * x
    # Spectrum and log-magnitude
    spec = np.fft.rfft(w)
    log_mag = np.log(np.abs(spec) + eps)
    # Symmetrise to full spectrum and IFFT
    full = np.concatenate([log_mag, log_mag[-2:0:-1]])
    cep = np.real(np.fft.ifft(full))
    return cep[:n_coeffs].astype(np.float64)


def extract_cepstral_features(
    roi: np.ndarray,
    n_coeffs: int = 13,
) -> dict[str, float]:
    """Real cepstrum coefficients labelled c00..c12."""
    cep = real_cepstrum(roi, n_coeffs=n_coeffs)
    return {f"cep_{i:02d}": float(cep[i]) for i in range(n_coeffs)}


# =================================================================
# GROUP B — Cross-channel ratios
# =================================================================

def _channel_summary(roi: np.ndarray) -> dict[str, float]:
    """Two amplitude summary statistics per channel: log-energy and log-peak."""
    energy = float((roi ** 2).sum())
    if len(roi) == 0:
        return {"log_energy": 0.0, "log_peak": 0.0}
    env = np.abs(hilbert(roi))
    peak = float(env.max())
    return {
        "log_energy": float(np.log(energy + 1e-12)),
        "log_peak":   float(np.log(peak + 1e-12)),
    }


def extract_cross_channel_features(
    rois_per_pair: dict[tuple[str, str], np.ndarray],
    mics: Sequence[str] = ("Mic1", "Mic2"),
    txs: Sequence[str] = ("Tx1", "Tx5", "Tx8"),
) -> dict[str, float]:
    """Compute cross-channel ratios from already-extracted ROI signals.

    INPUT
    -----
    rois_per_pair : dict {(mic, tx): roi_array}
        ROI signals keyed by (Mic, Tx). All ROIs must already be at the
        same length and have been preprocessed identically.

    OUTPUT
    ------
    dict of cross-channel features:

      For each Tx:
        Mic1_vs_Mic2_<tx>_dlog_energy   : log_energy(Mic1) - log_energy(Mic2)
        Mic1_vs_Mic2_<tx>_dlog_peak     : log_peak(Mic1)   - log_peak(Mic2)

      For each Mic:
        Tx1_vs_Tx8_<mic>_dlog_energy
        Tx1_vs_Tx8_<mic>_dlog_peak
        Tx5_vs_Tx1_<mic>_dlog_energy
        Tx5_vs_Tx1_<mic>_dlog_peak

    Plus one global combined ratio that is signed in the angle:
        signed_amp_ratio  : (Mic1_vs_Mic2_Tx1) - (Mic1_vs_Mic2_Tx8)
    """
    out: dict[str, float] = {}

    # Per-channel summaries
    summaries = {(m, t): _channel_summary(rois_per_pair[(m, t)])
                 for m in mics for t in txs if (m, t) in rois_per_pair}

    # Mic-vs-Mic ratios per Tx (only meaningful with 2+ mics)
    if len(mics) >= 2:
        m_a, m_b = mics[0], mics[1]
        for tx in txs:
            sa = summaries.get((m_a, tx)); sb = summaries.get((m_b, tx))
            if sa is None or sb is None:
                continue
            out[f"{m_a}_vs_{m_b}_{tx}_dlog_energy"] = sa["log_energy"] - sb["log_energy"]
            out[f"{m_a}_vs_{m_b}_{tx}_dlog_peak"]   = sa["log_peak"]   - sb["log_peak"]

    # Tx-vs-Tx ratios per Mic
    tx_pairs = [(txs[0], txs[-1]),     # Tx1 vs Tx8 — opposite ends
                (txs[1], txs[0])]      # Tx5 vs Tx1 — middle vs left
    for tx_a, tx_b in tx_pairs:
        for mic in mics:
            sa = summaries.get((mic, tx_a)); sb = summaries.get((mic, tx_b))
            if sa is None or sb is None:
                continue
            out[f"{tx_a}_vs_{tx_b}_{mic}_dlog_energy"] = sa["log_energy"] - sb["log_energy"]
            out[f"{tx_a}_vs_{tx_b}_{mic}_dlog_peak"]   = sa["log_peak"]   - sb["log_peak"]

    # Signed amplitude ratio — combination that should be monotone in angle
    if len(mics) >= 2 and txs[0] in [t[1] for t in summaries.keys()] and txs[-1] in [t[1] for t in summaries.keys()]:
        m_a, m_b = mics[0], mics[1]
        try:
            r_left  = (summaries[(m_a, txs[0])]["log_energy"]  - summaries[(m_b, txs[0])]["log_energy"])
            r_right = (summaries[(m_a, txs[-1])]["log_energy"] - summaries[(m_b, txs[-1])]["log_energy"])
            out["signed_amp_ratio"] = float(r_left - r_right)
        except KeyError:
            pass

    return out


# =================================================================
# Per-trial driver
# =================================================================

def extract_family15_for_trial(
    trial_signals: dict[str, np.ndarray],
    tof_per_pair_s: dict[tuple[str, str], float],
    fs: float,
    carrier_per_tx: dict[str, float],
    tx_half_bw_hz: dict[str, float] | None = None,
    roi_half_us: float = 200.0,
    realign: bool = True,
    normalise: bool = True,
    n_cep_coeffs: int = 13,
    mics: Sequence[str] = ("Mic1", "Mic2"),
    txs: Sequence[str] = ("Tx1", "Tx5", "Tx8"),
) -> dict[str, float]:
    """Extract Family 1.5 features per trial.

    Returns a dict mapping cep_<channel>_<i> and cross-channel ratio names
    to floats.
    """
    out: dict[str, float] = {}
    n_samp = len(next(iter(trial_signals.values())))
    half_samp = int(roi_half_us * 1e-6 * fs)
    tx_half_bw_hz = tx_half_bw_hz or {}

    rois: dict[tuple[str, str], np.ndarray] = {}
    for mic in mics:
        sig_full = np.asarray(trial_signals[mic], dtype=np.float64)
        for tx in txs:
            tof = tof_per_pair_s[(mic, tx)]
            fc = carrier_per_tx[tx]
            half_bw = float(tx_half_bw_hz.get(tx, 3000.0))
            sos = butter(4, [fc - half_bw, fc + half_bw], btype="band",
                         fs=fs, output="sos")
            sig = sosfiltfilt(sos, sig_full)
            centre = int(round(tof * fs))
            lo = max(0, centre - half_samp)
            hi = min(n_samp, centre + half_samp)
            roi = sig[lo:hi]
            roi = preprocess_roi(roi, realign=realign, normalise=normalise)
            rois[(mic, tx)] = roi

            # Per-channel cepstral coefficients
            cep = extract_cepstral_features(roi, n_coeffs=n_cep_coeffs)
            for k, v in cep.items():
                out[f"{mic}_{tx}_{k}"] = v

    # Cross-channel ratios from preprocessed ROIs
    cross = extract_cross_channel_features(rois, mics=mics, txs=txs)
    out.update(cross)
    return out


# =================================================================
# Self-test
# =================================================================

if __name__ == "__main__":
    fs = 2_000_000

    np.random.seed(0)
    t = np.arange(800) / fs
    sig = np.sin(2 * np.pi * 50_000 * t) * np.hanning(800)
    sig += 0.05 * np.random.randn(800)

    cep = extract_cepstral_features(sig, n_coeffs=13)
    print(f"Cepstral features ({len(cep)}):")
    for k, v in cep.items():
        print(f"  {k}: {v:+.4f}")

    # Cross-channel test with synthetic
    rois = {('Mic1','Tx1'): sig*1.2, ('Mic2','Tx1'): sig*1.0,
            ('Mic1','Tx5'): sig*1.1, ('Mic2','Tx5'): sig*0.9,
            ('Mic1','Tx8'): sig*0.95,('Mic2','Tx8'): sig*1.0}
    cross = extract_cross_channel_features(rois)
    print(f"\nCross-channel features ({len(cross)}):")
    for k, v in cross.items():
        print(f"  {k}: {v:+.4f}")
