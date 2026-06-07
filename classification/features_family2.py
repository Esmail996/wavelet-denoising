"""features_family2.py — wavelet-scattering features on the carrier-band ROI.

Family 2 keeps the SAME two structural controls as features_family.py, realised
differently because scattering is broadband by construction:

  (1) TIME control — the scattering is computed on the SAME fixed ROI window
      [peak - WIN_BEFORE, peak + WIN_AFTER] about the per-trial TOF peak.
  (2) SPECTRAL control — a sharp ZERO-PHASE band-pass (Butterworth, order
      `BP_ORDER`, applied with `sosfiltfilt`) restricts the signal to the
      carrier band BAND = 35-65 kHz BEFORE scattering. Being zero-phase, it adds
      no group delay and does not move or skew the echo inside the window, so the
      scattering shape stays faithful. The filter is applied to the whole trace
      and the ROI is then cut out, so there are no edge transients in the window.

The scattering transform itself (kymatio Scattering1D, complex Morlet) gives a
time-shift-stable description: first order ~ the carrier-band envelope (as the
CWT block of Family 1/3 does), second order ~ the amplitude modulations within
the band that a single CWT scale cannot see. Coefficients are averaged over the
(short) ROI; orders 1 and 2 are divided by the order-0 term so the descriptor is
amplitude/distance-robust and encodes shape. order-0 is kept separately and is
the only amplitude (range-dependent) term.

One vector per (Mic, trial). No Cohen's d, no leakage flagging here.
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Sequence
import numpy as np
from scipy.signal import butter, sosfiltfilt

def _import_scattering1d():
    """Return the kymatio Scattering1D class, importing it robustly.

    `from kymatio.numpy import Scattering1D` breaks on current SciPy (it pulls in
    the 3D frontend, which imports the removed `scipy.special.sph_harm`). The 1D
    frontend can be imported directly, which is what we use."""
    try:
        from kymatio.numpy import Scattering1D
        return Scattering1D
    except Exception:
        from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D
        return ScatteringNumPy1D


try:
    _SCATTERING1D = _import_scattering1d()
    _HAVE_KYMATIO = True
except Exception:
    _SCATTERING1D = None
    _HAVE_KYMATIO = False

FS_DEFAULT = 2_000_000.0
BAND = (35_000.0, 65_000.0)    # sharp carrier-band gate (covers 40/50/60 kHz)
BP_ORDER = 8                   # zero-phase via sosfiltfilt -> ~order 16, steep, no phase distortion
WIN_BEFORE, WIN_AFTER = 256, 768            # ROI; before+after = 1024 = 2**10
SCAT_J = 8                     # scattering depth (carriers ~ octave 5 at fs=2 MHz)
SCAT_Q = 8                     # wavelets/octave (resolves 40/50/60 within the band)

_SCAT_CACHE: dict[tuple, "Scattering1D"] = {}


def _bandpass(x, fs, band=BAND, order=BP_ORDER):
    """Zero-phase Butterworth band-pass. Maximally flat passband (no ripple), so
    the echo magnitude shape in the window is preserved; sosfiltfilt removes the
    group delay so the echo does not shift."""
    sos = butter(order, [band[0], band[1]], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _get_scattering(T, J=SCAT_J, Q=SCAT_Q):
    key = (int(T), int(J), int(Q))
    if key not in _SCAT_CACHE:
        if not _HAVE_KYMATIO:
            raise ImportError("kymatio is required for features_family2 (pip install kymatio)")
        _SCAT_CACHE[key] = _SCATTERING1D(J=J, shape=(int(T),), Q=Q)
    return _SCAT_CACHE[key]


def extract_features2(
    waveform: np.ndarray,
    fs: float = FS_DEFAULT,
    window: tuple[int, int] | None = None,
    peak: int | None = None,
    before: int = WIN_BEFORE,
    after: int = WIN_AFTER,
    band=BAND,
    J: int = SCAT_J,
    Q: int = SCAT_Q,
) -> "OrderedDict[str, float]":
    """One (Mic, trial) waveform -> scattering feature dict on the band-passed ROI.

    Pass `peak` (per-trial TOF sample); the window is [peak-before, peak+after].
    The band-pass is applied to the whole trace first, then the window is cut, so
    the window length T = before + after is fixed (1024 by default) and there are
    no filter edge transients inside it.
    """
    x = np.asarray(waveform, dtype=np.float64).ravel()
    n = x.size
    if peak is not None:
        window = (int(peak) - before, int(peak) + after)
    lo, hi = (0, n) if window is None else (max(0, int(window[0])), min(n, int(window[1])))

    xb = _bandpass(x, fs, band)
    seg = xb[lo:hi]
    T = before + after if peak is not None else (hi - lo)
    # enforce the fixed length T (zero-pad if a window ran past the trace edge)
    if seg.size < T:
        seg = np.concatenate([seg, np.zeros(T - seg.size)])
    elif seg.size > T:
        seg = seg[:T]

    S = _get_scattering(T, J, Q)
    Sx = S(seg)                                   # (n_coeffs, t_out)
    meta = S.meta()
    order = np.asarray(meta["order"]).ravel()
    coeff = Sx.mean(axis=-1)                       # time-average over the ROI

    f = OrderedDict()
    s0 = float(coeff[order == 0][0]) if np.any(order == 0) else float(coeff[0])
    f["ff2_s0"] = s0                               # order-0 (amplitude / range term)
    denom = abs(s0) + 1e-12
    i1 = 0
    for j, o in enumerate(order):
        if o == 1:
            f[f"ff2_s1_{i1:02d}"] = float(coeff[j] / denom)   # 1st order / s0 (shape)
            i1 += 1
    i2 = 0
    for j, o in enumerate(order):
        if o == 2:
            f[f"ff2_s2_{i2:02d}"] = float(coeff[j] / denom)   # 2nd order / s0 (modulation shape)
            i2 += 1
    return f


def extract_features2_for_trial(
    trial_signals: dict[str, np.ndarray],
    window: tuple[int, int] | None = None,
    fs: float = FS_DEFAULT,
    mics: Sequence[str] = ("Mic1", "Mic2", "Mic3"),
    peaks: dict[str, int] | None = None,
    before: int = WIN_BEFORE,
    after: int = WIN_AFTER,
    band=BAND,
    J: int = SCAT_J,
    Q: int = SCAT_Q,
) -> "OrderedDict[str, float]":
    out = OrderedDict()
    for m in mics:
        if m not in trial_signals:
            continue
        pk = None if peaks is None else peaks.get(m)
        feats = extract_features2(trial_signals[m], fs=fs, window=window, peak=pk,
                                  before=before, after=after, band=band, J=J, Q=Q)
        for k, v in feats.items():
            out[f"{m}_{k}"] = v
    return out


if __name__ == "__main__":
    import pickle
    with open("data_full/Multifrequenz/Box/75cm_0Grad.pickle", "rb") as fh:
        df = pickle.load(fh)
    x = np.asarray(df["Mic2"].iloc[0], dtype=float)
    feats = extract_features2(x, window=(8624, 9648))
    n1 = sum(k.startswith("ff2_s1_") for k in feats)
    n2 = sum(k.startswith("ff2_s2_") for k in feats)
    print(f"scattering: {len(feats)} coeffs (1 x s0, {n1} x s1, {n2} x s2), band {BAND} Hz, J={SCAT_J} Q={SCAT_Q}")
    for k, v in list(feats.items())[:12]:
        print(f"  {k:14s} {v:+.5g}")
