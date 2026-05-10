"""
Branch A denoising runner.

Loads every .pickle from an input dataset folder, denoises each trial
with SWT + BayesShrink (Branch A), and saves the output pickles to an
output folder along with a summary CSV.

Constants at the top define all algorithm settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_rw import load_trials, iter_dataset
from preprocess import preprocess_signal_for_denoising
from denoising import swt_denoise_bayes

# ---------------------------------------------------------------------------
# Constants — edit these to change the run configuration
# ---------------------------------------------------------------------------

# Bandpass front-end
FS_HZ = 2_000_000          # sampling rate
BP_CENTER_HZ = 50_000      # bandpass centre frequency
BP_BW_HZ = 25_000          # half-bandwidth → passes 25–75 kHz
BP_METHOD = "iir"
BP_IIR_ORDER = 4
GATE_START_SAMPLE = 2000   # zero samples before this index after preprocessing

# SWT decomposition
SWT_WAVELET = "sym6"
SWT_LEVEL = 7
SWT_NOISE_LEVEL = 4        # which SWT level to use for MAD noise estimate
# Per-level threshold scale factors: lambda_eff = scale * BayesShrink_lambda
# Levels not listed default to 1.0 (no scaling). D = detail level index.
LEVEL_SCALE: dict[int, float] = {
    4: 1,
    5: 0.5,
    6: 1,
}

# Fixed noise sigma per mic from reference dataset
SIGMA_N = {
    "Mic1": 4.155096851585,
    "Mic2": 3.004820656408,
    "Mic3": 9.819104075190,
}

# ---------------------------------------------------------------------------
# Core: denoise one pickle file
# ---------------------------------------------------------------------------

def _gate_signal(signal: np.ndarray, gate_start_sample: int) -> np.ndarray:
    gated = np.asarray(signal, dtype=float).copy()
    if gate_start_sample > 0:
        gated[:gate_start_sample] = 0.0
    return gated

def _denoise_pickle(pickle_path: Path) -> tuple[dict, list[dict]]:
    """Process one pickle: returns (denoised_dict, summary_rows)."""
    trials = load_trials(pickle_path)
    denoised: dict[str, list[np.ndarray]] = {}
    summary_rows: list[dict] = []

    for key, signals in trials.items():
        sigma_n = SIGMA_N.get(key)
        denoised[key] = []
        for trial_idx, signal in enumerate(signals):
            preprocessed = preprocess_signal_for_denoising(
                signal=signal,
                preprocessing="bandpassed",
                fs_hz=FS_HZ,
                band_center_hz=BP_CENTER_HZ,
                bw_hz=BP_BW_HZ,
                bp_method=BP_METHOD,
                iir_order=BP_IIR_ORDER,
            )
            gated = _gate_signal(preprocessed, GATE_START_SAMPLE)
            result, diag = swt_denoise_bayes(
                signal=gated,
                wavelet=SWT_WAVELET,
                level=SWT_LEVEL,
                noise_level=SWT_NOISE_LEVEL,
                level_scale=LEVEL_SCALE,
                sigma_n_fixed=sigma_n,
                return_diagnostics=True,
            )
            denoised[key].append(result)
            summary_rows.append({
                "file": pickle_path.name,
                "mic": key,
                "trial": trial_idx,
                "sigma_n": diag.get("sigma_n"),
                "sigma_n_source": diag.get("sigma_n_source"),
            })
    return denoised, summary_rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(input_dir: Path, output_dir: Path) -> None:
    all_rows = iter_dataset(input_dir, output_dir, _denoise_pickle)
    csv_path = Path(output_dir) / "denoising_summary.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nDone. Summary: {csv_path} ({len(all_rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Branch A SWT denoising runner")
    parser.add_argument(
        "--input-dir", type=str,
        default="Multifrequenz Dataset/Multifrequenz_mean25",
        help="Folder with raw .pickle files",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="Multifrequenz Dataset/Multifrequenz_mean25_denoised_branchA_sym6",
        help="Folder to write denoised .pickle files",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(Path(args.input_dir), Path(args.output_dir))


#python run_wavelet_denoising.py --input-dir "Multifrequenz Dataset/Multifrequenz" --output-dir "Multifrequenz Dataset/Multifrequenz_denoised_branchA_sym6" 2>&1