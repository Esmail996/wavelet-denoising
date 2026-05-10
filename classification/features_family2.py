"""
features_family2.py — Family 2 Wavelet Scattering Transform features.

Production version uses Kymatio (CPU/GPU). A simplified numpy fallback is
provided for sandbox verification. Both produce log(1 + |·|)-compressed
scattering coefficients suitable for downstream classifiers.

PRODUCTION USAGE (your machine, with kymatio installed):
    pip install kymatio

    from features_family2 import build_scattering, extract_family2_for_trial

    scattering = build_scattering(N=1024, J=8, Q=8)   # build once

    # For each trial, pass the dict of (Mic, Tx) ROI arrays of length N
    feats = extract_family2_for_trial(rois, scattering)

The scattering operator is order-2 by default, returning S0, S1, S2 paths.
With N=1024, J=8, Q=8 this typically gives ~150–250 paths per channel,
times 6 channels = 900–1500 features per trial. That matches the V3-deck plan.

Key parameters:
    N : ROI length in samples (must be power of 2; we pad to 1024 by default)
    J : maximum scale (2^J samples). With fs=2 MHz and N=1024, J=8 covers
        the full ROI.
    Q : wavelets per octave; Q=8 gives high frequency resolution at low scales,
        important for the carrier-band content.

The output is log(1 + |coefficients|) (standard scattering normalisation),
which compresses dynamic range and makes features approximately Gaussian
for downstream classifiers.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence
from scipy.signal import butter, sosfiltfilt

from roi_preprocessing import preprocess_roi


# =================================================================
# Production path: Kymatio
# =================================================================

def build_scattering(N: int = 1024, J: int = 8, Q: int = 8, T=None):
    """Build a Kymatio Scattering1D object.

    Parameters
    ----------
    N : signal length (samples). Must be ≥ 2^J. 1024 is a good default.
    J : maximum scale (output averaging is 2^J).
    Q : wavelets per octave (Q1=Q for first order, Q2=1 default).

    Returns
    -------
    Kymatio Scattering1D object that you call as `S(x)` on a 1D numpy array.
    """
    try:
        from kymatio.numpy import Scattering1D
    except Exception:
        # New SciPy versions may break kymatio.numpy due to optional 3D imports.
        # Import the 1D frontend directly as a compatible fallback.
        try:
            from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D
        except Exception as e:
            raise ImportError(
                "kymatio Scattering1D is unavailable. Install/update compatible kymatio and scipy."
            ) from e

    kwargs = {"J": J, "shape": (N,), "Q": Q}
    if T is not None:
        kwargs["T"] = T
    S = Scattering1D(**kwargs)
    return S


def extract_family2_features_kymatio(
    roi: np.ndarray,
    scattering,
    pad_to: int = 1024,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Compute scattering coefficients on one ROI signal.

    Returns a 1D numpy array of feature values (log-compressed scattering
    coefficients, time-averaged).
    """
    x = np.asarray(roi, dtype=np.float32)
    n = len(x)
    if n < pad_to:
        x_pad = np.zeros(pad_to, dtype=np.float32)
        start = (pad_to - n) // 2
        x_pad[start:start + n] = x
        x = x_pad
    elif n > pad_to:
        # crop centrally
        start = (n - pad_to) // 2
        x = x[start:start + pad_to]
    Sx = scattering(x)             # shape (n_paths, T)
    # Average over time → one value per scattering path
    Sx_mean = Sx.mean(axis=-1)
    # Log compression
    return np.log(np.abs(Sx_mean) + log_eps).astype(np.float32)


# =================================================================
# Sandbox fallback: simplified numpy scattering
# =================================================================

def _morlet_filter(N: int, sigma: float, xi: float) -> np.ndarray:
    """Morlet wavelet of length N centred at xi (Hz / sample) with bandwidth σ."""
    t = np.arange(N) - N // 2
    psi = np.exp(1j * 2 * np.pi * xi * t / N) * np.exp(-(t**2) / (2 * (sigma * N)**2))
    psi = psi - psi.mean()
    return psi.astype(np.complex64)


def _build_filter_bank_simple(N: int, J: int, Q: int) -> list[np.ndarray]:
    """A simple Morlet filter bank in the frequency domain. Not bit-identical
    to Kymatio's filter bank — for sandbox verification only."""
    filters = []
    for j in range(J * Q):
        scale = 2 ** (j / Q)
        xi = 0.4 / scale
        sigma = 0.1 * scale
        psi = _morlet_filter(N, sigma, xi * N)
        filters.append(np.fft.fft(psi))
    return filters


def extract_family2_features_numpy(
    roi: np.ndarray,
    N: int = 1024,
    J: int = 4,
    Q: int = 4,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Simplified order-2 scattering in pure numpy. Use Kymatio in production.

    Steps:
      1. Pad ROI to N
      2. FFT → multiply by each |psi_lambda|² → IFFT → |·| → time-avg → S1
      3. For each S1 channel: same operation again → S2
      4. log(1 + |·|) compression
    """
    x = np.asarray(roi, dtype=np.float32)
    n = len(x)
    if n < N:
        xp = np.zeros(N, dtype=np.float32)
        xp[(N - n) // 2:(N - n) // 2 + n] = x
        x = xp
    elif n > N:
        start = (n - N) // 2
        x = x[start:start + N]
    filters = _build_filter_bank_simple(N, J, Q)
    X = np.fft.fft(x)
    # Order 0: just the average
    S0 = np.abs(x.mean())
    # Order 1: |x * psi_lambda1|, time-averaged
    S1_paths = []
    U1_signals = []
    for psi in filters:
        U1 = np.abs(np.fft.ifft(X * psi))
        U1_signals.append(U1)
        S1_paths.append(U1.mean())
    # Order 2: |U1_lambda1 * psi_lambda2| time-averaged, only when lambda2<lambda1
    S2_paths = []
    for i, U1 in enumerate(U1_signals):
        U1_fft = np.fft.fft(U1)
        for j in range(i + 1, len(filters)):
            U2 = np.abs(np.fft.ifft(U1_fft * filters[j]))
            S2_paths.append(U2.mean())
    feats = np.concatenate([[S0], S1_paths, S2_paths]).astype(np.float32)
    return np.log(np.abs(feats) + log_eps)


# =================================================================
# Per-trial driver
# =================================================================

def extract_family2_for_trial(
    trial_signals: dict[str, np.ndarray],
    tof_per_pair_s: dict[tuple[str, str], float],
    fs: float,
    carrier_per_tx: dict[str, float],
    tx_half_bw_hz: dict[str, float] | None = None,
    scattering=None,
    roi_n: int = 1024,
    mics: Sequence[str] = ("Mic1", "Mic2"),
    txs: Sequence[str] = ("Tx1", "Tx5", "Tx8"),
    use_kymatio: bool = True,
    realign: bool = True,
    normalise: bool = True,
) -> dict[str, float]:
    """Extract Family-2 scattering features for all (Mic, Tx) channels of one trial.

    The ROI is centred on the corrected TOF arrival, length roi_n samples
    (default 1024 = 512 µs at 2 MHz).

    If `realign`, each ROI is re-centred on its envelope peak before
    scattering. If `normalise`, ROI is divided by RMS energy.
    """
    out = {}
    half = roi_n // 2
    tx_half_bw_hz = tx_half_bw_hz or {}
    for mic in mics:
        sig_full = np.asarray(trial_signals[mic], dtype=np.float64)
        for tx in txs:
            tof = tof_per_pair_s[(mic, tx)]
            fc = carrier_per_tx[tx]
            half_bw = float(tx_half_bw_hz.get(tx, 3000.0))
            sos = butter(4, [fc - half_bw, fc + half_bw], btype="band",
                         fs=fs, output="sos")
            sig = sosfiltfilt(sos, sig_full).astype(np.float32)
            centre = int(round(tof * fs))
            lo = max(0, centre - half)
            hi = min(len(sig), lo + roi_n)
            lo = hi - roi_n   # back-fill if at end
            if lo < 0:
                roi = np.zeros(roi_n, dtype=np.float32)
                roi[:hi] = sig[:hi]
            else:
                roi = sig[lo:hi]
            roi = preprocess_roi(roi.astype(np.float64),
                                 realign=realign, normalise=normalise).astype(np.float32)
            if use_kymatio and scattering is not None:
                fv = extract_family2_features_kymatio(roi, scattering)
            else:
                fv = extract_family2_features_numpy(roi, N=roi_n, J=4, Q=4)
            for i, v in enumerate(fv):
                out[f"{mic}_{tx}_S{i:04d}"] = float(v)
    return out


# =================================================================
# Self-test
# =================================================================

if __name__ == "__main__":
    fs = 2_000_000
    np.random.seed(0)
    t = np.arange(1024) / fs
    sig = np.sin(2 * np.pi * 50_000 * t) * np.hanning(1024)
    sig += 0.1 * np.random.randn(1024)

    print("Trying Kymatio path...")
    try:
        S = build_scattering(N=1024, J=8, Q=8)
        feats = extract_family2_features_kymatio(sig, S)
        print(f"  Kymatio: {len(feats)} features, mean={feats.mean():.3f}, std={feats.std():.3f}")
    except ImportError as e:
        print(f"  (skipped) {e}")

    print("\nNumpy fallback...")
    feats = extract_family2_features_numpy(sig, N=1024, J=4, Q=4)
    print(f"  numpy: {len(feats)} features, mean={feats.mean():.3f}, std={feats.std():.3f}")
