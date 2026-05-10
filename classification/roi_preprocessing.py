"""
roi_preprocessing.py — shared ROI utilities used by all feature families.

Two preprocessing operations:

    realign_roi_on_envelope_peak
        Re-centres a ROI on its Hilbert envelope peak. Removes the residual
        inter-channel TOF mis-alignment that survives Sarabia calibration
        (~5–10 mm = 30–60 samples at 2 MHz).

    normalise_roi_energy
        Divides the ROI by its RMS energy. Removes the 1/R² distance amplitude
        effect so that downstream features encode SHAPE only, not amplitude.

These are toggled via flags in the extraction driver. Default for the
new pipeline: both ON.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def realign_roi_on_envelope_peak(
    roi: np.ndarray,
    target_offset_samples: int | None = None,
) -> np.ndarray:
    """Roll the ROI so the Hilbert envelope peak sits at `target_offset_samples`.

    Default target is the centre of the ROI. The roll is circular, so a small
    amount of wraparound can occur — for typical envelope peaks well within
    the ROI this is harmless.

    If the envelope is essentially zero (silent ROI), the ROI is returned
    unchanged.
    """
    n = len(roi)
    if n < 4:
        return roi.copy()
    env = np.abs(hilbert(roi))
    if env.max() < 1e-12:
        return roi.copy()
    peak_idx = int(np.argmax(env))
    target = n // 2 if target_offset_samples is None else int(target_offset_samples)
    shift = target - peak_idx
    return np.roll(roi, shift)


def normalise_roi_energy(roi: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide ROI by its RMS so that ‖roi‖₂ / √N = 1.

    Preserves shape; removes amplitude. Critical when the dataset spans
    multiple distances since echo amplitude scales as 1/R².
    """
    rms = np.sqrt((roi ** 2).mean())
    if rms < eps:
        return roi.copy()
    return roi / rms


def preprocess_roi(
    roi: np.ndarray,
    realign: bool = True,
    normalise: bool = True,
) -> np.ndarray:
    """Convenience: apply optional realignment and energy normalisation."""
    out = roi
    if realign:
        out = realign_roi_on_envelope_peak(out)
    if normalise:
        out = normalise_roi_energy(out)
    return out
