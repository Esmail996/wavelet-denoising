"""Time-of-flight estimation — echo timing only.

This module turns a denoised microphone waveform into a raw time of flight
(seconds) and the envelope peak index. It does NOT convert ToF to distance and
it does NOT apply any per-channel delay calibration; that lives in
``calibration.py``. Keeping the two apart means the picker can be changed and
re-run without touching the calibration constants, and the calibration can be
re-fit without re-running the picker.

What changed from the gated version:
  * The search interval is no longer a fixed +/- GATE_CM gate centred on the
    geometric arrival. It is a per-distance window measured offline by the
    short-time-Fourier analysis (``stft_windows.json``), selected at run time by
    the protocol distance. The window already excludes the transmit ringdown and
    the far reflections, so the picker no longer has to defend against them.
  * Inside the window the peak is the maximum of the envelope (``picker="max"``).
    The old top-3-nearest-nominal rule is kept as ``picker="nearest"`` for
    comparison, but the default is the plain maximum.
  * Sub-sample parabolic interpolation of the peak is retained.

Geometry constants (PCB, carriers, sound speed, sample rate) live here because
both the picker and the calibration need them; calibration imports them from
this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pywt
from scipy.signal import argrelmax, butter, correlate, hilbert, sosfiltfilt


FS_HZ = 2_000_000.0
SOUND_SPEED_M_S = 343.2
GATE_CM = 15.0  # fallback half-width only, used when a distance has no measured window
MATCHED_FILTER_CYCLES = 10
MORLET_WAVELET = "cmor1.5-1.0"

# === PCB GEOMETRY — engineer email, manufacturer specification =======
# Tx centre-to-centre spacing: 4.08 mm (+/- 0.05 mm).
# Mic 5 mm above Tx row.
# Tx5 placed at origin; Tx1 to the left, Tx8 to the right.
# Mics directly above their respective Tx.
# Units: metres. Frame: +x = right, +y = forward toward target.
PCB = {
    "Tx1":  (-0.00408, 0.000),
    "Tx5":  ( 0.00000, 0.000),
    "Tx8":  (+0.00408, 0.000),
    "Mic1": (-0.00408, 0.005),
    "Mic2": ( 0.00000, 0.005),
    "Mic3": (+0.00408, 0.005),
}

CARRIERS_HZ = {"Tx1": 40_000.0, "Tx5": 50_000.0, "Tx8": 60_000.0}
# ====================================================================


# === STFT SEARCH WINDOWS =============================================
# Absolute sample bounds [lo, hi] in the 18000-sample record, one per protocol
# distance (cm). Measured offline over the whole dataset (all angles, objects,
# microphones, carriers) on db6/level-6 BayesShrink + 25-75 kHz band-pass; see
# stft_search_windows.py. The protocol distance selects the window at run time.
#
# These embedded values are the fallback. If a file named ``stft_windows.json``
# is found next to this module, in the project root, or in the working
# directory, it overrides them — regenerate that file to update the table
# without editing code.
_DEFAULT_SEARCH_WINDOWS: dict[int, tuple[int, int]] = {
    25:  (2610, 3498),
    50:  (5461, 6184),
    75:  (8437, 9120),
    100: (11298, 12123),
    125: (14063, 15135),
}


def _load_search_windows() -> dict[int, tuple[int, int]]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "stft_windows.json",
        here.parents[1] / "stft_windows.json" if len(here.parents) > 1 else None,
        Path.cwd() / "stft_windows.json",
    ]
    for cand in candidates:
        if cand is None:
            continue
        try:
            if cand.is_file():
                raw = json.loads(cand.read_text())
                return {int(k): (int(v[0]), int(v[1])) for k, v in raw.items()}
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return dict(_DEFAULT_SEARCH_WINDOWS)


SEARCH_WINDOWS: dict[int, tuple[int, int]] = _load_search_windows()
# ====================================================================


def _as_1d_float64(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {x.shape}.")
    return x


def _validate_keys(mic_name: str, tx_name: str, geom: dict[str, tuple[float, float]]) -> None:
    if tx_name not in CARRIERS_HZ:
        raise ValueError(f"Unsupported tx_name '{tx_name}'. Expected one of {sorted(CARRIERS_HZ)}.")
    if tx_name not in geom:
        raise KeyError(f"Missing transmitter geometry for '{tx_name}'.")
    if mic_name not in geom:
        raise KeyError(f"Missing microphone geometry for '{mic_name}'.")


def _hann_burst(fc_hz: float, n_cycles: int = MATCHED_FILTER_CYCLES, fs_hz: float = FS_HZ) -> np.ndarray:
    duration_s = float(n_cycles) / float(fc_hz)
    n_samples = int(round(duration_s * float(fs_hz)))
    t = np.arange(n_samples, dtype=np.float64) / float(fs_hz)
    return np.hanning(n_samples) * np.sin(2.0 * np.pi * float(fc_hz) * t)


def _nominal_arrival_index(
    distance_m: float,
    mic_name: str,
    tx_name: str,
    geom: dict[str, tuple[float, float]],
    c_m_s: float,
    fs_hz: float,
) -> int:
    """Geometric round-trip arrival sample for a reflector on-axis at depth distance_m.

    Used only to size the fallback gate when a distance has no measured window.
    """
    _validate_keys(mic_name=mic_name, tx_name=tx_name, geom=geom)
    tx_xy = np.asarray(geom[tx_name], dtype=np.float64)
    mic_xy = np.asarray(geom[mic_name], dtype=np.float64)
    reflector_xy = np.array([0.0, float(distance_m)], dtype=np.float64)
    d_nom_m = np.linalg.norm(tx_xy - reflector_xy) + np.linalg.norm(reflector_xy - mic_xy)
    return int(round(d_nom_m / float(c_m_s) * float(fs_hz)))


def _resolve_window(
    distance_m: float,
    n_nom: int,
    n_samples: int,
    fs_hz: float,
    c_m_s: float,
    windows: dict[int, tuple[int, int]],
    fallback_gate_cm: float,
) -> tuple[int, int]:
    """Pick the search interval for this distance.

    Uses the measured per-distance window when the protocol distance is in the
    table; otherwise falls back to a +/- fallback_gate_cm gate around the
    geometric arrival, so the picker still runs on distances not characterised.
    """
    dist_cm = int(round(float(distance_m) * 100.0))
    win = windows.get(dist_cm)
    if win is not None:
        lo, hi = int(win[0]), int(win[1])
    else:
        gate = int(round(float(fallback_gate_cm) * 1e-2 / float(c_m_s) * float(fs_hz)))
        lo, hi = int(n_nom) - gate, int(n_nom) + gate
    lo = max(0, lo)
    hi = min(int(n_samples), hi)
    if hi <= lo:
        raise ValueError(f"Empty search window ({lo}, {hi}) for distance {dist_cm} cm.")
    return lo, hi


def _window_peak(
    env: np.ndarray,
    idx_lo: int,
    idx_hi: int,
    n_nom: int,
    fc_hz: float,
    fs_hz: float = FS_HZ,
    picker: str = "max",
    top_n: int = 3,
) -> int:
    """Return the peak index inside the window [idx_lo, idx_hi).

    picker="max":     the largest envelope sample in the window. With the
                      measured window the ringdown and far reflections are
                      already excluded, so the maximum is the echo.
    picker="nearest": legacy behaviour — among the top-N local maxima in the
                      window, the one closest to the geometric arrival n_nom.
                      Kept only for comparison.
    """
    a = max(0, int(idx_lo))
    b = min(env.size, int(idx_hi))
    if b <= a:
        raise ValueError(f"Invalid window bounds ({idx_lo}, {idx_hi}) for signal length {env.size}.")

    if picker == "max":
        return a + int(np.argmax(env[a:b]))

    if picker == "nearest":
        order = max(1, int(round(0.5 * fs_hz / fc_hz)))
        local_idxs = argrelmax(env[a:b], order=order)[0]
        if len(local_idxs) == 0:
            return a + int(np.argmax(env[a:b]))
        abs_idxs = local_idxs + a
        amps = env[abs_idxs]
        top_cut = min(top_n, len(abs_idxs))
        top_abs = abs_idxs[np.argsort(amps)[::-1][:top_cut]]
        return int(top_abs[np.argmin(np.abs(top_abs - n_nom))])

    raise ValueError(f"Unknown picker '{picker}'. Use 'max' or 'nearest'.")


def _parabolic_subsample(r: np.ndarray, k: int) -> float:
    """Sub-sample offset of the peak at index k from a 3-point parabola fit.

    Returns a value in [-0.5, 0.5] samples. The envelope around the echo is a
    broad smooth lobe, so the three points at the apex form a good local
    parabola. This refines the peak below the 0.5 us sample grid; it does not
    change which lobe was selected.
    """
    if k <= 0 or k >= (len(r) - 1):
        return 0.0
    denom = float(r[k - 1] - 2.0 * r[k] + r[k + 1])
    if abs(denom) < 1e-18:
        return 0.0
    raw = float((r[k - 1] - r[k + 1]) / (2.0 * denom))
    return max(-0.5, min(0.5, raw))


def tof_path_A(
    x_denoised: np.ndarray,
    distance_m: float,
    mic_name: str,
    tx_name: str,
    geom: dict[str, tuple[float, float]] = PCB,
    c_m_s: float = SOUND_SPEED_M_S,
    fs_hz: float = FS_HZ,
    windows: dict[int, tuple[int, int]] | None = None,
    fallback_gate_cm: float = GATE_CM,
    picker: str = "max",
    use_bandpass: bool = True,
) -> tuple[float, float, int]:
    """Matched-filter path: band-pass, correlate with a 10-cycle Hann burst,
    Hilbert envelope, then peak the envelope inside the measured window.

    Returns (tof_s, envelope_amplitude, peak_index). The window was characterised
    on the Path B (CWT) envelope; the +/-150-sample margin covers the small
    difference in where the matched-filter envelope peaks.
    """
    x = _as_1d_float64(x_denoised)
    fc_hz = CARRIERS_HZ[tx_name]

    if use_bandpass:
        sos = butter(2, [fc_hz - 3_000.0, fc_hz + 3_000.0], btype="band", fs=float(fs_hz), output="sos")
        x_band = sosfiltfilt(sos, x)
    else:
        x_band = x

    reference = _hann_burst(fc_hz=fc_hz, n_cycles=MATCHED_FILTER_CYCLES, fs_hz=fs_hz)
    matched = correlate(x_band, reference, mode="same")
    envelope = np.abs(hilbert(matched))

    n_nom = _nominal_arrival_index(distance_m, mic_name, tx_name, geom, c_m_s, fs_hz)
    lo, hi = _resolve_window(
        distance_m, n_nom, envelope.size, fs_hz, c_m_s,
        windows if windows is not None else SEARCH_WINDOWS, fallback_gate_cm,
    )
    k = _window_peak(envelope, lo, hi, n_nom=n_nom, fc_hz=fc_hz, fs_hz=fs_hz, picker=picker)

    delta = _parabolic_subsample(envelope, k)
    tof_s = (k - delta) / float(fs_hz)
    return float(tof_s), float(envelope[k]), int(k)


def tof_path_B(
    x_denoised: np.ndarray,
    distance_m: float,
    mic_name: str,
    tx_name: str,
    geom: dict[str, tuple[float, float]] = PCB,
    c_m_s: float = SOUND_SPEED_M_S,
    fs_hz: float = FS_HZ,
    windows: dict[int, tuple[int, int]] | None = None,
    fallback_gate_cm: float = GATE_CM,
    picker: str = "max",
) -> tuple[float, float, int]:
    """Morlet-CWT path (production): single-scale complex CWT at the carrier,
    magnitude envelope, then peak the envelope inside the measured window.

    Returns (tof_s, envelope_amplitude, peak_index).
    """
    x = _as_1d_float64(x_denoised)
    fc_hz = CARRIERS_HZ[tx_name]
    scale = float(fs_hz) / float(fc_hz)

    cwt_coeffs, _ = pywt.cwt(x, [scale], MORLET_WAVELET, sampling_period=1.0 / float(fs_hz))
    envelope = np.abs(cwt_coeffs[0])

    n_nom = _nominal_arrival_index(distance_m, mic_name, tx_name, geom, c_m_s, fs_hz)
    lo, hi = _resolve_window(
        distance_m, n_nom, envelope.size, fs_hz, c_m_s,
        windows if windows is not None else SEARCH_WINDOWS, fallback_gate_cm,
    )
    k = _window_peak(envelope, lo, hi, n_nom=n_nom, fc_hz=fc_hz, fs_hz=fs_hz, picker=picker)

    delta = _parabolic_subsample(envelope, k)
    tof_s = (k - delta) / float(fs_hz)
    return float(tof_s), float(envelope[k]), int(k)
