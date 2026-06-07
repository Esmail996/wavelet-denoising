"""Global denoising runner for Path A (SWT) and Path B (WPT).

This script processes every .pickle file in a dataset directory, denoises each
trial for all microphones, stores denoised pickles while preserving the
original payload structure, and writes denoising_summary.csv with diagnostics.

How to run
----------
Run Path A (SWT):
    python -m denoising.runners.run_global_denoising --path A

Run Path B (WPT):
    python -m denoising.runners.run_global_denoising --path B

Run Path A with fixed sigma (closest to legacy run_wavelet_denoising.py):
    python -m denoising.runners.run_global_denoising --path A --a-use-fixed-sigma

Run with explicit dataset folders:
    python -m denoising.runners.run_global_denoising --path A \
        --input-dir "Multifrequenz Dataset/Multifrequenz" \
        --output-dir "Multifrequenz Dataset/Multifrequenz_denoised_pathA_global"

CLI options
-----------
Required:
    --path {A,B}
        Select denoising branch.

I/O:
    --input-dir STR
        Input root searched recursively for .pickle files.
        Default: Multifrequenz Dataset/Multifrequenz
    --output-dir STR
        Output root for denoised .pickle files and denoising_summary.csv.
        Default: path-dependent auto value
            A -> Multifrequenz Dataset/Multifrequenz_denoised_pathA_global
            B -> Multifrequenz Dataset/Multifrequenz_denoised_pathB_global

Shared preprocessing:
    --preprocessing {bandpassed,detrended_only}
        Pre-denoising frontend mode. Default: bandpassed
    --bp-center-hz FLOAT
        Bandpass center frequency in Hz. Default: 50000
    --bp-bw-hz FLOAT
        Bandpass bandwidth in Hz. Default: 25000
    --bp-method {iir,fir}
        Bandpass implementation. Default: iir
    --bp-iir-order INT
        IIR order when --bp-method iir. Default: 4

Path A (SWT) options:
    --a-wavelet STR
        SWT wavelet. Default: db6
    --a-level INT
        SWT decomposition level. Default: 6
    --a-noise-level INT
        SWT level used for MAD noise estimate. Default: 4
    --a-use-fixed-sigma
        Use fixed per-microphone sigma values instead of MAD estimate.

Path B (WPT) options:
    --b-wavelet STR
        WPT wavelet. Default: sym6
    --b-level INT
        WPT max level. Default: 7
    --b-carrier-attenuation FLOAT
        Threshold scaling for carrier packets. Default: 0.5
    --b-noise-packet-idx INT
        Optional fixed packet index for noise sigma estimation.
        Default: auto-selected from passband candidates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from denoising import swt_denoise_bayes, wpt_denoise_bayes
from denoising.preprocess import preprocess_signal_for_denoising
from helpers.data_rw import iter_dataset, load_trials

FS_HZ = 2_000_000

# Shared preprocessing defaults.
BP_CENTER_HZ = 50_000
BP_BW_HZ = 25_000
BP_METHOD = "iir"
BP_IIR_ORDER = 4

# Path A defaults.
A_WAVELET = "db6"
A_LEVEL = 6
A_NOISE_LEVEL = 4
A_LEVEL_SCALE: dict[int, float] = {}
A_SIGMA_N = {
    "mic1": 4.155096851585,
    "mic2": 3.004820656408,
    "mic3": 9.819104075190,
}

# Path B defaults.
B_WAVELET = "sym6"
B_LEVEL = 7
B_CARRIERS_HZ = (40_000.0, 50_000.0, 60_000.0)
B_CARRIER_ATTENUATION = 0.5


def _normalize_mic_name(name: str) -> str:
    return name.strip().lower()


def _default_output_dir_for_path(path_choice: str) -> str:
    if path_choice == "A":
        return "Multifrequenz Dataset/Multifrequenz_denoised_pathA_global"
    return "Multifrequenz Dataset/Multifrequenz_denoised_pathB_global"


def _build_process_fn(args: argparse.Namespace):
    path_choice = args.path

    def _process_pickle(pickle_path: Path) -> tuple[dict, list[dict]]:
        trials = load_trials(pickle_path)
        denoised: dict[str, list[np.ndarray]] = {}
        summary_rows: list[dict] = []

        for mic_name, signals in trials.items():
            denoised[mic_name] = []
            mic_key = _normalize_mic_name(mic_name)

            for trial_idx, signal in enumerate(signals):
                preprocessed = preprocess_signal_for_denoising(
                    signal=signal,
                    preprocessing=args.preprocessing,
                    fs_hz=FS_HZ,
                    band_center_hz=args.bp_center_hz,
                    bw_hz=args.bp_bw_hz,
                    bp_method=args.bp_method,
                    iir_order=args.bp_iir_order,
                )

                if path_choice == "A":
                    sigma_n = A_SIGMA_N.get(mic_key) if args.a_use_fixed_sigma else None
                    result, diag = swt_denoise_bayes(
                        signal=preprocessed,
                        wavelet=args.a_wavelet,
                        level=args.a_level,
                        noise_level=args.a_noise_level,
                        level_scale=A_LEVEL_SCALE,
                        sigma_n_fixed=sigma_n,
                        return_diagnostics=True,
                    )
                    summary_rows.append(
                        {
                            "file": pickle_path.name,
                            "path": "A",
                            "mic": mic_name,
                            "trial": trial_idx,
                            "sigma_n": diag.get("sigma_n"),
                            "sigma_n_source": diag.get("sigma_n_source"),
                            "wavelet": diag.get("wavelet"),
                            "level": diag.get("level"),
                        }
                    )
                else:
                    result, diag = wpt_denoise_bayes(
                        signal=preprocessed,
                        wavelet=args.b_wavelet,
                        level=args.b_level,
                        fs=FS_HZ,
                        carriers_hz=B_CARRIERS_HZ,
                        carrier_attenuation=args.b_carrier_attenuation,
                        noise_packet_idx=args.b_noise_packet_idx,
                        return_diagnostics=True,
                    )
                    summary_rows.append(
                        {
                            "file": pickle_path.name,
                            "path": "B",
                            "mic": mic_name,
                            "trial": trial_idx,
                            "sigma_n": diag.get("sigma_n"),
                            "noise_packet_idx": diag.get("noise_packet_idx"),
                            "wavelet": diag.get("wavelet"),
                            "level": diag.get("level"),
                        }
                    )

                denoised[mic_name].append(result)

        return denoised, summary_rows

    return _process_pickle


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_fn = _build_process_fn(args)
    all_rows = iter_dataset(Path(args.input_dir), output_dir, process_fn)

    csv_path = output_dir / "denoising_summary.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nDone. Path {args.path} summary: {csv_path} ({len(all_rows)} rows)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Global denoising runner for Path A (SWT) or Path B (WPT)."
    )
    parser.add_argument(
        "--path",
        choices=["A", "B"],
        required=True,
        help="Denoising path to run: A (SWT) or B (WPT).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="Multifrequenz Dataset/Multifrequenz",
        help="Folder with raw .pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Folder to write denoised .pickle files. If omitted, a path-based default is used.",
    )

    # Shared preprocessing options.
    parser.add_argument(
        "--preprocessing",
        choices=["bandpassed", "detrended_only"],
        default="bandpassed",
        help="Signal preprocessing mode before denoising.",
    )
    parser.add_argument("--bp-center-hz", type=float, default=BP_CENTER_HZ)
    parser.add_argument("--bp-bw-hz", type=float, default=BP_BW_HZ)
    parser.add_argument("--bp-method", choices=["iir", "fir"], default=BP_METHOD)
    parser.add_argument("--bp-iir-order", type=int, default=BP_IIR_ORDER)

    # Path A options.
    parser.add_argument("--a-wavelet", type=str, default=A_WAVELET)
    parser.add_argument("--a-level", type=int, default=A_LEVEL)
    parser.add_argument("--a-noise-level", type=int, default=A_NOISE_LEVEL)
    parser.add_argument(
        "--a-use-fixed-sigma",
        action="store_true",
        help="Use fixed per-microphone sigma for Path A instead of MAD estimate.",
    )

    # Path B options.
    parser.add_argument("--b-wavelet", type=str, default=B_WAVELET)
    parser.add_argument("--b-level", type=int, default=B_LEVEL)
    parser.add_argument("--b-carrier-attenuation", type=float, default=B_CARRIER_ATTENUATION)
    parser.add_argument(
        "--b-noise-packet-idx",
        type=int,
        default=None,
        help="Optional fixed WPT noise packet index for Path B.",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = _default_output_dir_for_path(args.path)
    run(args)


if __name__ == "__main__":
    main()
