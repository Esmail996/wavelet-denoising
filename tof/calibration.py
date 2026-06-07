"""Range calibration — ToF to distance.

This module is the second half of the pipeline. It takes a raw time of flight
(seconds) produced by ``TOF_estimation.py`` and returns a distance, optionally
subtracting a fitted per-channel delay. Nothing here touches waveforms or the
picker; it only inverts the geometry and applies delay constants.

Three layers:
  * ``tof_to_distance_m`` — pure bistatic inversion, no delay. ToF -> distance.
  * ``tof_to_distance_calibrated`` — subtract the 9-parameter empirical delay
    table (production calibration) before inverting.
  * ``tof_to_distance_structural`` — subtract the 4-parameter structural
    (resonator) delay model before inverting. Physical reference / consistency
    check.

The ground-truth housing offset (``GT_OFFSET_MM``) and the delay tables live
here as well, since they belong to the ranging step, not to timing. Geometry
constants are imported from ``TOF_estimation`` so there is a single source of
truth for the PCB layout and the carriers.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from TOF_estimation import PCB, SOUND_SPEED_M_S, CARRIERS_HZ, _validate_keys


# Engineer-documented ground-truth correction (housing exterior to acoustic
# path). Email: 2 mm housing + 1.6 mm PCB = 3.6 mm added to every
# protocol-recorded distance before fitting tau.
GT_OFFSET_MM = 3.6


# === DELAY CALIBRATION ===============================================
#
# Two model variants are maintained:
#
#  (A) DELAY_CALIBRATION_EMPIRICAL_S — 9-parameter per-(method, Tx, Mic)
#      median fit on the manufacturer geometry + GT+3.6 mm correction.
#      Production calibration. Headline accuracy 3.51 mm consensus
#      median (Path B, db6-denoised, Mic1+Mic2 multi-carrier).
#
#  (B) DELAY_CALIBRATION_STRUCTURAL_PARAMS — 4-parameter structural
#      reference model. tau_{m,t} = tau_drive + a*tau_g(f_t; f0, Q) +
#      delta_m, with delta_Mic1 = 0 (gauge). Accuracy 3.66 mm
#      consensus median. Used to assign physical interpretation to the
#      empirical fit and as the consistency-check baseline.
#
# Re-fit when the picker, denoiser, or matched filter change. Note the picker
# changed (max-in-window instead of top-3-nearest on a +/-15 cm gate), so these
# constants must be re-fit on the new ToF output before they are trusted.
# Units: seconds. Key: (method, tx_name, mic_name).

DELAY_CALIBRATION_EMPIRICAL_S: dict[tuple[str, str, str], float] = {
    # Path B (Morlet CWT) on db6-denoised data, manufacturer geometry,
    # GT+3.6 mm corrected. Re-fit when the picker, denoiser or
    # matched filter change.
    ("path_b", "Tx1", "Mic1"): 19.214e-6,
    ("path_b", "Tx1", "Mic2"): 1.670e-6,
    ("path_b", "Tx1", "Mic3"): 41.676e-6,
    ("path_b", "Tx5", "Mic1"): -6.850e-6,
    ("path_b", "Tx5", "Mic2"): -7.927e-6,
    ("path_b", "Tx5", "Mic3"): 69.813e-6,
    ("path_b", "Tx8", "Mic1"): -1.429e-6,
    ("path_b", "Tx8", "Mic2"): -5.856e-6,
    ("path_b", "Tx8", "Mic3"): 66.746e-6,
    # Path A entries deliberately omitted; Path B is the production path.
}

DELAY_CALIBRATION_STRUCTURAL_PARAMS: dict[str, float] = {
    # Re-fit on max-picker peaks from
    # outputs/tof_matched/multifrequenz_denoised_branchA_bayes_garrote_peaks.csv
    # using tof/refit_structural_calibration.py (Path B fit, 2026-06-01).
    "tau_drive_s": 40.262e-6,
    "a": 0.654593,
    "f0_hz": 53_779.125303,
    "Q": 7.068344,
    "delta_Mic1_s": 0.0,
    "delta_Mic2_s": 20.328e-6,
    "delta_Mic3_s": 39.961e-6,
}

# Back-compat alias for existing call sites.
DELAY_CALIBRATION_S = DELAY_CALIBRATION_EMPIRICAL_S
# ====================================================================


def tof_to_distance_m(
    tof_s: float,
    mic_name: str,
    tx_name: str,
    geom: dict[str, tuple[float, float]] = PCB,
    c_m_s: float = SOUND_SPEED_M_S,
    d_min_m: float = 0.001,
    d_max_m: float = 10.0,
) -> float:
    """Invert a measured ToF to object distance (metres) using the full bistatic
    path model: path = ||Tx - reflector|| + ||reflector - Mic||.

    The reflector is assumed to lie on the sensor axis (x=0) at depth d along y.
    When Tx and Mic are co-located this reduces to the monostatic formula
    d = tof*c/2 (corrected for the sensor's y-offset from the origin).

    Solves f(d) = 0 numerically via Brent's method.
    """
    _validate_keys(mic_name=mic_name, tx_name=tx_name, geom=geom)
    tx_xy = np.asarray(geom[tx_name], dtype=np.float64)
    mic_xy = np.asarray(geom[mic_name], dtype=np.float64)
    total_path_m = float(tof_s) * float(c_m_s)

    def _residual(d: float) -> float:
        reflector = np.array([0.0, d])
        return float(np.linalg.norm(tx_xy - reflector) + np.linalg.norm(reflector - mic_xy)) - total_path_m

    try:
        return float(brentq(_residual, d_min_m, d_max_m, xtol=1e-6))
    except ValueError:
        # ToF outside plausible range — fall back to monostatic approximation
        return total_path_m / 2.0


def tof_to_distance_calibrated(
    tof_s: float,
    mic_name: str,
    tx_name: str,
    method: str,
    geom: dict[str, tuple[float, float]] = PCB,
    c_m_s: float = SOUND_SPEED_M_S,
    calibration: dict[tuple[str, str, str], float] = DELAY_CALIBRATION_S,
) -> float:
    """Like tof_to_distance_m() but subtracts the fitted per-(method, Tx, Mic)
    delay constant before inverting. method must be 'path_a' or 'path_b'.
    If the exact key is not found (e.g. path_a constants are commented out),
    falls back to the corresponding path_b constant, then to zero.
    """
    key = (method, tx_name, mic_name)
    fallback = ("path_b", tx_name, mic_name)
    tau_s = calibration.get(key, calibration.get(fallback, 0.0))
    return tof_to_distance_m(tof_s - tau_s, mic_name, tx_name, geom, c_m_s)


def _resonator_group_delay_s(f_hz: float, f0_hz: float, Q: float) -> float:
    """Second-order resonator group delay (seconds) at frequency f_hz.

    For H(jw) = w0^2 / (w0^2 - w^2 + j*w*w0/Q), the group delay is
        tau_g(w) = (w0/Q)(w0^2 + w^2) / ((w0^2 - w^2)^2 + (w*w0/Q)^2)

    Used as a fixed basis function in the structural calibration model.
    """
    w = 2.0 * np.pi * float(f_hz)
    w0 = 2.0 * np.pi * float(f0_hz)
    numerator = (w0 / float(Q)) * (w0**2 + w**2)
    denominator = (w0**2 - w**2)**2 + (w * w0 / float(Q))**2
    return float(numerator / denominator)


def structural_tau_s(
    mic_name: str,
    tx_name: str,
    params: dict[str, float] = DELAY_CALIBRATION_STRUCTURAL_PARAMS,
) -> float:
    """Predict tau (seconds) from the 4-parameter structural model.

    tau_{m,t} = tau_drive + a * tau_g(f_t; f0, Q) + delta_m

    Parameters keyed in `params`: tau_drive_s, a, f0_hz, Q,
                                  delta_Mic1_s, delta_Mic2_s, delta_Mic3_s.
    """
    _validate_keys(mic_name=mic_name, tx_name=tx_name, geom=PCB)
    fc = CARRIERS_HZ[tx_name]
    g = _resonator_group_delay_s(fc, params["f0_hz"], params["Q"])
    delta_key = f"delta_{mic_name}_s"
    return (
        params["tau_drive_s"]
        + params["a"] * g
        + params.get(delta_key, 0.0)
    )


def tof_to_distance_structural(
    tof_s: float,
    mic_name: str,
    tx_name: str,
    geom: dict[str, tuple[float, float]] = PCB,
    c_m_s: float = SOUND_SPEED_M_S,
    params: dict[str, float] = DELAY_CALIBRATION_STRUCTURAL_PARAMS,
) -> float:
    """Same as tof_to_distance_calibrated, but uses the 4-parameter
    structural model for tau instead of the 9-element empirical table.
    Returns distance in metres.
    """
    tau_s = structural_tau_s(mic_name, tx_name, params)
    return tof_to_distance_m(tof_s - tau_s, mic_name, tx_name, geom, c_m_s)
