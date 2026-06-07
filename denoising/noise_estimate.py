from __future__ import annotations

from pathlib import Path

import numpy as np

from helpers.data_rw import load_trials
from .preprocess import preprocess_signal_for_denoising

MAD_TO_STD = 1.0 / 0.6744897501960817  # converts MAD to Gaussian-equivalent sigma

# ---------------------------------------------------------------------------
# Edit these before running: python noise_estimate.py
# ---------------------------------------------------------------------------
REFERENCE_PICKLE = r"Multifrequenz Dataset/Multifrequenz/referenz/referenz.pickle"
MIC              = "Mic2"   # "Mic1", "Mic2", or "Mic3"
GATE_START       = 2400     # samples before this index are ignored (ringdown)
GATE_END         = 17000    # tail blanking to avoid end-of-recording artefacts
# Bandpass preprocessing (applied before sigma estimation)
FS_HZ            = 2_000_000
BP_CENTER_HZ     = 50_000
BP_BW_HZ         = 25_000
BP_METHOD        = "iir"
BP_IIR_ORDER     = 2
# ---------------------------------------------------------------------------


def mad_std(x: np.ndarray) -> float:
    """Robust Gaussian-equivalent sigma from MAD."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return float(mad * MAD_TO_STD) if mad > 0.0 else float(np.std(arr))


def gate(signal: np.ndarray, start: int, end: int | None = None) -> np.ndarray:
    """Return signal[start:end] (the post-ringdown noise region)."""
    x = np.asarray(signal, dtype=float).ravel()
    return x[int(start): end]


def estimate_sigma(
    pickle_path: str | Path,
    mic: str,
    gate_start: int,
    gate_end: int | None = None,
    fs_hz: float = 2_000_000,
    bp_center_hz: float = 50_000,
    bp_bw_hz: float = 25_000,
    bp_method: str = "iir",
    iir_order: int = 2,
) -> dict:
    """
    Load all trials from a reference pickle, bandpass-preprocess them,
    gate the post-ringdown region, and return the MAD-based sigma
    (median across trials).
    """
    trials = load_trials(pickle_path)
    if mic not in trials:
        raise KeyError(f"Mic '{mic}' not found. Available: {list(trials.keys())}")

    per_trial: list[float] = []
    for signal in trials[mic]:
        preprocessed = preprocess_signal_for_denoising(
            signal=signal,
            preprocessing="bandpassed",
            fs_hz=fs_hz,
            band_center_hz=bp_center_hz,
            bw_hz=bp_bw_hz,
            bp_method=bp_method,
            iir_order=iir_order,
        )
        seg = gate(preprocessed, gate_start, gate_end)
        if seg.size > 0:
            per_trial.append(mad_std(seg))

    sigma_median = float(np.median(per_trial)) if per_trial else float("nan")
    return {
        "sigma_median": sigma_median,
        "per_trial_sigma": per_trial,
        "n_trials": len(per_trial),
        "mic": mic,
        "gate_start": gate_start,
        "gate_end": gate_end,
        "pickle": str(pickle_path),
    }


if __name__ == "__main__":
    result = estimate_sigma(
        REFERENCE_PICKLE, MIC, GATE_START, GATE_END,
        fs_hz=FS_HZ, bp_center_hz=BP_CENTER_HZ, bp_bw_hz=BP_BW_HZ,
        bp_method=BP_METHOD, iir_order=BP_IIR_ORDER,
    )
    print(f"sigma_n ({result['mic']}, median over {result['n_trials']} trials): {result['sigma_median']:.12e}")
    for i, s in enumerate(result["per_trial_sigma"]):
        print(f"  trial {i:3d}: {s:.6e}")


# Known values from reference dataset:
# sigma Mic1 = 4.696011894698e+00
# sigma Mic2 = 3.179713290847e+00
# sigma Mic3 = 1.008575153711e+00
